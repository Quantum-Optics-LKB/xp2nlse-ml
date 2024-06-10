#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import NLSE.nlse as nlse
import numpy as np
import cupy as cp
from scipy.constants import c, epsilon_0
import gc
from cupyx.scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from engine.noise_generator import line_noise, salt_and_pepper_noise


def add_model_noise(beam, poisson_noise_lam, normal_noise_sigma):
        
    poisson_noise = np.random.poisson(lam=poisson_noise_lam, size=(beam.shape))*poisson_noise_lam*0.75
    normal_noise = np.random.normal(0, normal_noise_sigma, (beam.shape))

    total_noise = normal_noise + poisson_noise
    noisy_beam = np.real(beam) + total_noise + 1j * (np.imag(beam) + total_noise)

    noisy_beam = normalize_data(noisy_beam)

    noisy_beam = noisy_beam.astype(np.complex64)
    return noisy_beam

def data_creation(
    numbers: tuple,
    cameras: tuple,
    delta_z: float, 
    length: float,
    saving_path: str,
    ) -> np.ndarray:
    
    #NLSE parameters
    n2, in_power, alpha, isat, waist, nl_length = numbers
    resolution_in, window_in, window_out, resolution_training = cameras

    crop = int(0.5*(window_in - window_out)*resolution_in/window_in)
  
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    n2 = cp.asarray(n2)
    isat = cp.asarray(isat)

    n2 = n2[:, cp.newaxis, cp.newaxis, cp.newaxis]
    isat = isat[cp.newaxis, :, cp.newaxis, cp.newaxis]

    simu = nlse.NLSE(power=in_power, alpha=alpha, window=window_in, n2=n2, 
                     V=None, L=length, NX=resolution_in, NY=resolution_in, 
                     Isat=isat, nl_length=nl_length)
    
    simu.nl_profile =  simu.nl_profile[np.newaxis, np.newaxis, :,:]
    simu.delta_z = delta_z

    beam = np.ones((number_of_n2, number_of_isat, simu.NX, simu.NY), dtype=np.complex64)*np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = add_model_noise(beam, poisson_noise_lam, normal_noise_sigma)
    beam = cp.asarray(beam)

    A_gpu = simu.out_field(beam, z=length, verbose=True, plot=False, normalize=True, precision="single")
    A = A_gpu.reshape((number_of_n2*number_of_isat, simu.NX, simu.NY)).get()
    
    E = np.zeros((number_of_n2*number_of_isat,3, resolution_training, resolution_training), dtype=np.float16)

    del A_gpu
      
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

    E[:,0,:,:] = density
    E[:,1,:,:] = phase
    E[:,2,:,:] = uphase

    np.save(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}', E)
    return E

def data_augmentation(
    number_of_n2: int, 
    number_of_isat: int,
    in_power: float,
    E: np.ndarray,
    noise_level: float,
    path: str, 
    ) -> np.ndarray:
    """
    Applies data augmentation techniques to a set of input data arrays representing optical fields. 
    This augmentation includes adding salt-and-pepper noise, line noise at various angles, and 
    creating multiple instances with different noise intensities. The function aims to increase 
    the robustness of machine learning models by providing a more diverse training dataset.

    Parameters:
    - number_of_n2 (int): The number of different nonlinear refractive index (n2) values used in the simulation.
    - number_of_isat (int): The number of different saturation intensities (Isat) used in the simulation.
    - number_of_power (int): he number of different power (power) used in the simulation
    - E (np.ndarray): The input data array to be augmented. Expected shape is 
      [n2*power*Isat, channels, resolution, resolution], where channels typically include amplitude, 
      phase, and unwrapped phase information.
    - noise_level (float): The base level of noise to be applied. This function will also apply a 
      noise level ten times greater as part of the augmentation.
    - path (str): The file path for saving the augmented data arrays

    Returns:
    - np.ndarray: An augmented data array with an increased number of samples due to applied 
      augmentations. The shape of the array is determined by the augmentation factors applied to 
      the original dataset.

    The function iterates over each channel in the input data array, applying noise and line noise 
    augmentations at various intensities and angles. The augmented data significantly increases 
    the dataset size, enhancing the potential diversity for training purposes. The augmentation 
    parameters include a fixed set of angles (0 to 90 degrees at intervals), two levels of 
    salt-and-pepper noise based on the provided noise_level, and line noise with a predefined 
    set of line densities. If a save path is provided, the augmented dataset is saved using numpy's 
    .npy format, with filenames that reflect the simulation and augmentation parameters.
    """
    angles = np.linspace(0, 90, 5)
    noises = [noise_level, noise_level*10] 
    lines = [20, 50, 100]
    augmentation = len(noises) + len(lines) * len(noises) * len(angles) + 1

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
    return augmentation, augmented_data

def normalize_data(
        data: np.ndarray,
        ) -> np.ndarray:
    """
    Normalizes the data in each channel of a multi-channel dataset to a range of [0, 1]. 
    This is done individually for each data array (e.g., image or field) across all channels, 
    by subtracting the minimum value and dividing by the range of the data.

    Parameters:
    - data (np.ndarray): A multi-dimensional numpy array where the first dimension is considered 
      as different data samples (e.g., images), and the second dimension as different channels 
      (e.g., RGB channels, amplitude/phase in optical fields, etc.). The last two dimensions are 
      considered the spatial dimensions of the data.

    Returns:
    - np.ndarray: A numpy array of the same shape as the input, containing the normalized data. 
      Each element in the array is a float64 representing the normalized value of that data point, 
      ensuring that each channel of each data sample has values scaled between 0 and 1.

    The normalization process enhances the numerical stability of algorithms processing the data 
    and is often a prerequisite step for machine learning model inputs. This function ensures that 
    each channel of each data sample is independently normalized, making it suitable for diverse 
    datasets with varying ranges of values across samples or channels.
    """    
    data -= np.min(data, axis=(-2, -1), keepdims=True)
    data /= np.max(data, axis=(-2, -1), keepdims=True)
    return data