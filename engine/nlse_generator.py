#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from NLSE import NLSE
from tqdm import tqdm
import numpy as np
import cupy as cp
from scipy.constants import c, epsilon_0
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from engine.noise_generator import line_noise, salt_and_pepper_noise


def data_creation(
    input_field: np.ndarray,
    window: float,
    n2_values: np.ndarray,
    power_values: np.ndarray,
    isat_values: np.ndarray,
    resolution_in: int,
    resolution_out: int,
    delta_z:float,
    trans: float,
    L: float,
    path: str = None,
    ) -> np.ndarray:
    """
    Simulates the nonlinear Schrödinger equation (NLSE package) propagation of an input field over a distance L, 
    considering various nonlinear refractive index (n2), power, and saturation intensity (Isat) values. 
    It generates and optionally saves the output field's amplitude and phase (both wrapped and unwrapped) 
    at a specified resolution.

    Parameters:
    - input_field (np.ndarray): The input optical field as a 2D numpy array.
    - window (float): The spatial extent of the simulation window in meters.
    - n2_values (np.ndarray): An array of nonlinear refractive index (n2) values to simulate.
    - power_values (np.ndarray): An array of optical power values to simulate.
    - isat_values (np.ndarray): An array of saturation intensities (Isat) to simulate.
    - resolution_in (int): The resolution of the input field (assumed to be square).
    - resolution_out (int): The desired resolution of the output fields (assumed to be square).
    - delta_z (float): The step size in meters for the simulation propagation.
    - trans (float): The transmission coefficient for the medium.
    - L (float): The total propagation distance in meters.
    - path (str, optional): The file path for saving the output arrays. If None, arrays are not saved.

    Returns:
    - np.ndarray: A 4D numpy array containing the simulation results for all combinations of n2, power, 
      and Isat values. The dimensions are [n2*power*Isat, 3, resolution_out, resolution_out], where 
      the second dimension corresponds to the output amplitude, phase, and unwrapped phase fields.

    This function uses GPU acceleration (via CuPy) for NLSE simulation and post-processing. It calculates 
    the nonlinear propagation using specified parameters, then resizes the output to the desired resolution, 
    and normalizes the data. If a path is provided, it saves the output arrays in the specified directory 
    with a naming convention that reflects the simulation parameters.
    """
    
    #NLSE parameters
    number_of_power = len(power_values)
    number_of_n2 = len(n2_values)
    number_of_isat = len(isat_values)

    power_list = cp.array(power_values)
    n2_list = cp.array(n2_values)
    Isat_list = cp.array(isat_values)

    n2 =  cp.zeros((n2_list.size ,1 ,1 ,1 ,1))
    n2[:, 0, 0, 0, 0] = n2_list

    power =  cp.zeros((1, power_list.size ,1 ,1 ,1))
    power[0, :, 0, 0, 0] = power_list.T

    Isat =  cp.zeros((1, 1, Isat_list.size ,1 ,1 ))
    Isat[0, 0, :, 0, 0] = Isat_list.T

    alpha = -cp.log(trans)/L
    zoom_factor = resolution_out / resolution_in

    #Data generation using NLSE
    simu = NLSE.nlse.NLSE(alpha, power, window, n2, None, L, NX=resolution_in, NY=resolution_in, Isat=Isat)
    simu.delta_z = delta_z
    A = simu.out_field(cp.array(input_field), L, verbose=True, plot=False, normalize=True, precision="single").get()

    for index_puiss in range(number_of_power):
        E_init = A[:,index_puiss, :, :, :].reshape((number_of_n2*number_of_isat, resolution_in, resolution_in))

        out_resized_Pha = zoom(np.angle(E_init), (1, zoom_factor, zoom_factor),order=5)
        out_resized_Pha_unwrap = zoom(unwrap_phase(np.angle(E_init)), (1, zoom_factor, zoom_factor),order=5)
        out_resized_Amp = zoom( np.abs(E_init)**2 * c * epsilon_0 / 2, (1, zoom_factor, zoom_factor),order=5)

        E = np.zeros((number_of_n2*number_of_isat,3, resolution_out, resolution_out))
        E[:,0,:,:] = out_resized_Amp
        E[:,1,:,:] = out_resized_Pha
        E[:,2,:,:] = out_resized_Pha_unwrap
        
        E = normalize_data(E).astype(np.float16)
        if not (path == None):
            np.save(f'{path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power_values[index_puiss])[:4]}_amp_pha_pha_unwrap', E)
        
    E_all = np.zeros((number_of_n2*number_of_power*number_of_isat ,3 , resolution_out, resolution_out))
    E_out = A.reshape((number_of_n2*number_of_power*number_of_isat,resolution_in, resolution_in)) 

    out_resized_Pha = zoom(np.angle(E_out), (1, zoom_factor, zoom_factor),order=5)
    out_resized_Pha_unwrap = zoom(unwrap_phase(np.angle(E_out)), (1, zoom_factor, zoom_factor),order=5)
    out_resized_Amp = zoom( np.abs(E_out)**2 * c * epsilon_0 / 2, (1, zoom_factor, zoom_factor),order=5)

    E_all[:,0,:,:] = out_resized_Amp
    E_all[:,1,:,:] = out_resized_Pha
    E_all[:,2,:,:] = out_resized_Pha_unwrap
    
    E_all = normalize_data(E_all).astype(np.float16)
    #Data saving
    if not (path == None):
        np.save(f'{path}/Es_w{E_all.shape[-1]}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all', E_all)
    return E_all

def data_augmentation(
    number_of_n2: int, 
    number_of_power: int,
    number_of_isat: int,
    power: int,
    E: np.ndarray,
    noise_level: float,
    path: str = None, 
    ) -> np.ndarray:
    """
    Applies data augmentation techniques to a set of input data arrays representing optical fields. 
    This augmentation includes adding salt-and-pepper noise, line noise at various angles, and 
    creating multiple instances with different noise intensities. The function aims to increase 
    the robustness of machine learning models by providing a more diverse training dataset.

    Parameters:
    - number_of_n2 (int): The number of different nonlinear refractive index (n2) values used in the simulation.
    - number_of_power (int): The number of different power levels used in the simulation.
    - number_of_isat (int): The number of different saturation intensities (Isat) used in the simulation.
    - power (int): The specific power value for which the augmentation is being done. If set to 0, 
      it assumes augmentation across all power levels.
    - E (np.ndarray): The input data array to be augmented. Expected shape is 
      [n2*power*Isat, channels, resolution, resolution], where channels typically include amplitude, 
      phase, and unwrapped phase information.
    - noise_level (float): The base level of noise to be applied. This function will also apply a 
      noise level ten times greater as part of the augmentation.
    - path (str, optional): The file path for saving the augmented data arrays. If None, data arrays 
      are not saved. The saved filename reflects the augmentation parameters.

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

    augmented_data = np.zeros((augmentation*E.shape[0], E.shape[1], E.shape[2],E.shape[3]), dtype=np.float16)
  
    for channel in range(E.shape[1]):
        index = 0
        for image_index in tqdm(range(E.shape[0]), position=4,desc="Iteration", leave=False):
            image_at_channel = E[image_index,channel,:,:]
            augmented_data[index,channel ,:, :] = image_at_channel
            index += 1  
            for noise in noises:
                augmented_data[index,channel ,:, :] = salt_and_pepper_noise(image_at_channel, noise)
                index += 1
                for angle in angles:
                    for num_lines in lines:
                        augmented_data[index,channel ,:, :] = line_noise(image_at_channel, num_lines, np.max(image_at_channel)*noise,angle)
                        index += 1
    augmented_data = normalize_data(augmented_data)
    if not (path == None):
        if power != 0:
            np.save(f'{path}/Es_w{augmented_data.shape[-1]}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap_extended', augmented_data)
        else:
            np.save(f'{path}/Es_w{augmented_data.shape[-1]}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all_extended', augmented_data)
    return augmented_data, augmentation

def normalize_data(
        data: np.ndarray
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
    
    min_vals = np.min(data, axis=(2, 3), keepdims=True)
    max_vals = np.max(data, axis=(2, 3), keepdims=True)

    normalized_data = (data - min_vals) / (max_vals - min_vals)

    normalized_data = normalized_data.astype(np.float16)

    return normalized_data