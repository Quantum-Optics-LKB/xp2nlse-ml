#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from NLSE import NLSE
import numpy as np
import cupy as cp
from scipy.constants import c, epsilon_0
import gc
from cupyx.scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from tqdm import tqdm
from engine.noise_generator import line_noise, salt_and_pepper_noise


def add_model_noise(beam, poisson_noise_lam, normal_noise_sigma):
        
    poisson_noise = np.random.poisson(lam=poisson_noise_lam, size=(beam.shape))*poisson_noise_lam*0.75
    normal_noise = np.random.normal(0, normal_noise_sigma, (beam.shape))

    total_noise = normal_noise + poisson_noise
    noisy_beam = np.real(beam) + total_noise + 1j * np.imag(beam)


    noisy_beam = noisy_beam.astype(np.complex64)
    return noisy_beam

def data_creation(
    numbers: tuple,
    cameras: tuple,
    saving_path: str = "",
    ) -> np.ndarray:
    
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = numbers
    resolution_in, window_in, window_out, resolution_training = cameras

    crop = int(0.5*(window_in - window_out)*resolution_in/window_in)
  
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    isat = isat[:, np.newaxis, np.newaxis]

    X, delta_X = np.linspace(-window_in / 2, window_in / 2,
            num=resolution_in,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
    Y, delta_Y = np.linspace(-window_in / 2, window_in / 2,
            num=resolution_in,
            endpoint=False,
            retstep=True,
            dtype=np.float32,
        )
    XX, YY = np.meshgrid(X, Y)
    

    beam = np.ones((number_of_isat, resolution_in, resolution_in), dtype=np.complex64)*np.exp(-(XX**2 + YY**2) / waist**2)
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = add_model_noise(beam, poisson_noise_lam, normal_noise_sigma)
    E = np.zeros((number_of_n2*number_of_isat,3, resolution_training, resolution_training), dtype=np.float16)
      

    for index, n2_value in tqdm(enumerate(n2),desc=f"NLSE", 
                                total=number_of_n2, unit="n2"):

      simu = NLSE(power=in_power, alpha=alpha, window=window_in, n2=n2_value, 
                     V=None, L=length, NX=resolution_in, NY=resolution_in, 
                     Isat=isat, nl_length=nl_length)
      
      if nl_length != 0:
        simu.nl_profile =  simu.nl_profile[np.newaxis, :,:]
      simu.delta_z = delta_z
      A = simu.out_field(beam, z=length, verbose=False, plot=False, normalize=True, precision="single")
    
      density = np.abs(A)**2 * c * epsilon_0 / 2
      phase = np.angle(A)
      uphase = unwrap_phase(phase)
        
      density = density[:,crop:-crop,crop:-crop]
      phase = phase[:,crop:-crop,crop:-crop] 
      uphase = uphase[:,crop:-crop,crop:-crop] 

      zoom_factor = resolution_training / phase.shape[-1]
      density_cp = zoom(cp.asarray(density), (1, zoom_factor, zoom_factor),order=3)
      density = normalize_data(density_cp.get()).astype(np.float16)  

      phase_cp = zoom(cp.asarray(phase), (1, zoom_factor, zoom_factor),order=3)
      phase = normalize_data(phase_cp.get()).astype(np.float16)

      uphase_cp = zoom(cp.asarray(uphase), (1, zoom_factor, zoom_factor),order=3)
      uphase = normalize_data(uphase_cp.get()).astype(np.float16)

      del density_cp
      del phase_cp
      del uphase_cp

      gc.collect()
      cp.get_default_memory_pool().free_all_blocks()

      start_index = number_of_n2 * index
      end_index = number_of_n2 * (index + 1)
      E[start_index:end_index,0,:,:] = density
      E[start_index:end_index,1,:,:] = phase
      E[start_index:end_index,2,:,:] = uphase
    
    if saving_path != "":
        np.save(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}', E)
    return E

def generate_labels(
        n2: float,
        isat: float,
        )-> tuple:

    N2_labels, ISAT_labels = np.meshgrid(n2, isat) 

    n2_labels = N2_labels.reshape(-1)
    isat_labels = ISAT_labels.reshape(-1)

    labels = (len(n2), n2_labels, len(isat), isat_labels)
    return labels

def data_augmentation(
    E: np.ndarray,
    in_power: float,
    expansion: bool,
    path: str, 
    labels: tuple,
    ) -> np.ndarray:

    number_of_n2, n2_labels, number_of_isat, isat_labels = labels

    angles = np.linspace(0, 90, 5)
    noises = [0.01, 0.1] 
    lines = [20, 50, 100]
    augmentation = len(noises) + len(lines) * len(noises) * len(angles) + 1

    n2_labels = np.repeat(n2_labels, augmentation)
    isat_labels = np.repeat(isat_labels, augmentation)

    labels = (number_of_n2, n2_labels, number_of_isat, isat_labels)
    if expansion:
        
        print("---- EXPANSION ----")

        augmented_data = np.zeros((augmentation*E.shape[0], E.shape[1], E.shape[2],E.shape[3]), dtype=np.float32)

        for channel in range(E.shape[1]):
            index = 0
            for image_index in range(E.shape[0]):
                image_at_channel = normalize_data(E[image_index,channel,:,:]).astype(np.float32)
                augmented_data[index,channel ,:, :] = normalize_data(image_at_channel).astype(np.float32)
                index += 1  
                for noise in noises:
                    augmented_data[index,channel ,:, :] = normalize_data(salt_and_pepper_noise(image_at_channel, noise)).astype(np.float32)
                    index += 1
                    for angle in angles:
                        for num_lines in lines:
                            augmented_data[index,channel ,:, :] = normalize_data(line_noise(image_at_channel, num_lines, np.max(image_at_channel)*noise,angle)).astype(np.float32)
                            index += 1

        np.save(f'{path}/Es_w{augmented_data.shape[-1]}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}_extended', augmented_data.astype(np.float16))
        return augmented_data, labels
    
    else:
        augmented_data = np.load(f'{path}/Es_w{E.shape[-1]}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}_extended.npy')
        
        return augmented_data, labels

def normalize_data(
        data: np.ndarray,
        ) -> np.ndarray: 
    data -= np.min(data, axis=(-2, -1), keepdims=True)
    data /= np.max(data, axis=(-2, -1), keepdims=True)
    return data