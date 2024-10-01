#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import gc
import cupy as cp
import numpy as np
from tqdm import tqdm
from NLSE import NLSE
from cupyx.scipy.ndimage import zoom
from scipy.constants import c, epsilon_0
from engine.utils import experiment_noise, set_seed
set_seed(10)

def data_creation(
    nlse_settings: tuple,
    cameras: tuple,
    device: int,
    saving_path: str = "",
    ) -> np.ndarray:
    """
    Create dataset based on Nonlinear Schrödinger Equation (NLSE) simulations.

    Args:
        nlse_settings (tuple): A tuple containing the NLSE settings in the following order:
            (n2, in_power, alpha, isat, waist, nl_length, delta_z, length).
            - n2 (np.ndarray): Array of nonlinear refractive indices.
            - input_power (float): Input power of the beam.
            - alpha (np.ndarray): Array of alpha values.
            - isat (np.ndarray): Array of saturation intensities.
            - waist (float): Beam waist.
            - non_locality_length (float): Nonlocality length.
            - delta_z (float): Step size for the z-axis.
            - length (float): Total propagation length.
        cameras (tuple): A tuple containing camera settings in the following order:
            (resolution_in, window_in, window_out, resolution_training).
            - resolution_in (int): Input resolution of the camera.
            - window_in (float): Input window size.
            - window_out (float): Output window size.
            - resolution_training (int): Resolution for training data.
        device (int): The device ID for GPU computation.
        saving_path (str, optional): The path to save the generated dataset. Defaults to "".

    Returns:
        np.ndarray: A 4D numpy array containing the augmented dataset with dimensions 
                    (number_of_samples, channels, resolution_training, resolution_training).

    Description:
        This function performs the following steps to create and augment the dataset:
        1. Extracts NLSE and camera settings from the input tuples.
        2. Initializes variables for augmentation, including random noise settings.
        3. Generates the initial beam profile.
        4. Iterates through each n2 value and performs NLSE simulation.
        5. Calculates the density, phase, and unwrapped phase of the output field.
        6. Crops and resizes the simulated data to the desired training resolution.
        7. Applies normalization and Gaussian blur to the unwrapped phase.
        8. Saves the generated dataset to the specified path if provided.
    """
    
    n2, input_power, alpha, isat, waist, non_locality_length, delta_z, length = nlse_settings
    resolution_in, window_in, window_out, resolution_training = cameras
  
    crop = int(0.5*(window_in - window_out)*resolution_in/window_in)
  
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)
    
    alpha = alpha[:, np.newaxis, np.newaxis]

    X = np.linspace(-window_in / 2, window_in / 2, num=resolution_in, endpoint=False, dtype=np.float64)
    Y = np.linspace(-window_in / 2, window_in / 2, num=resolution_in, endpoint=False, dtype=np.float64)
    XX, YY = np.meshgrid(X, Y)
    
    beam = np.ones((number_of_alpha, resolution_in, resolution_in), dtype=np.complex64)*np.exp(-(XX**2 + YY**2) / waist**2)
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = experiment_noise(beam, poisson_noise_lam, normal_noise_sigma)
    E = np.zeros((number_of_n2*number_of_isat*number_of_alpha,2, resolution_training, resolution_training), dtype=np.float64)

    for n2_index, n2_value in tqdm(enumerate(n2),desc=f"NLSE", total=number_of_n2, unit="n2"):
      for isat_index, isat_value in enumerate(isat):

        simu = NLSE(power=input_power, alpha=alpha, window=window_in, n2=n2_value, 
                      V=None, L=length, NX=resolution_in, NY=resolution_in, 
                      Isat=isat_value, nl_length=non_locality_length)
        
        if non_locality_length != 0:
          simu.nl_profile =  simu.nl_profile[np.newaxis, np.newaxis, :,:]
        simu.delta_z = delta_z
        A = simu.out_field(beam, z=length, verbose=False, plot=False, normalize=True, precision="single")

        density = np.abs(A)**2 * c * epsilon_0 / 2
        phase = np.angle(A)
        
        if crop != 0:
          density = density[:,crop:-crop,crop:-crop]
          phase = phase[:,crop:-crop,crop:-crop] 

        zoom_factor = resolution_training / phase.shape[-1]
        density_cp = zoom(cp.asarray(density), (1, zoom_factor, zoom_factor),order=3)
        density = density_cp.get()

        phase_cp = zoom(cp.asarray(phase), (1, zoom_factor, zoom_factor),order=3)
        phase = phase_cp.get()

        del density_cp
        del phase_cp

        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        start_index = number_of_alpha * number_of_isat * n2_index + number_of_alpha * isat_index
        end_index = number_of_alpha * number_of_isat * (n2_index) + number_of_alpha * (isat_index + 1)
        E[start_index:end_index,0,:,:] = density
        E[start_index:end_index,1,:,:] = phase

    if saving_path != "":
      np.save(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}', E)
    
    return E

def generate_labels(
      n2: np.ndarray, 
      isat: np.ndarray,
      alpha: np.ndarray
      ) -> tuple:
  """
    Generate labels for the dataset based on the provided n2, isat, and alpha values.

    Args:
        n2 (np.ndarray): Array of nonlinear refractive indices.
        isat (np.ndarray): Array of saturation intensities.
        alpha (np.ndarray): Array of alpha values.

    Returns:
        tuple: A tuple containing the label counts and flattened label arrays in the 
               following format:
               (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels).

    Description:
        This function creates a grid of labels using `np.meshgrid` and reshapes them 
        to 1D arrays for compatibility with the dataset.
    """
  
  N2_labels, ISAT_labels, ALPHA_labels = np.meshgrid(n2, isat, alpha) 

  n2_labels = N2_labels.reshape(-1)
  isat_labels = ISAT_labels.reshape(-1)
  alpha_labels = ALPHA_labels.reshape(-1)

  labels = (len(n2), n2_labels, len(isat), isat_labels, len(alpha), alpha_labels)

  return labels