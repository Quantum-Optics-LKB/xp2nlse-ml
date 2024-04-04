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

from noise_generator import line_noise, salt_and_pepper_noise


def create(
    input_field: np.ndarray,
    window: float,
    n2_values: np.ndarray,
    power_values: np.ndarray,
    isat_values: np.ndarray,
    resolution_in: int,
    resolution_out: int,
    delta_z:float,
    path: str = None,
    ) -> np.ndarray:
    """
    Generates a sequence of a certain number of frames (number_of_frames) for each n2 (number_of_n2)
    from [-1e-8 ; -1e-9] where each frame a certain resolution (resolution) and stores the data 
    in an array of shape (n2_num, frame_num, resolution)

    Args:
        number_of_n2 (int): number of different n2.
        resolution (int): resolution of the 1d array.
        full (bool): checks if we want the full sequence or just beginning and start. (Default to True)
        number_of_frames (int): number of frames per n2. (Default to 2 when full == False)
        extend (bool): checks if we want to add data with noise. (Default to False)
        extension (int): number of extensions per n2 (Default to 0)
        path (str): path of directory in which the data is saved. (Default to None)
    Returns:
        The complete data set
    """
    #NLSE parameters
    trans = 0.01
    L = 20e-2

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
            np.save(f'{path}/Es_w{resolution_out}_n2{number_of_n2}_Isat{number_of_isat}_power{1}_at{str(power_values[index_puiss])[:4]}_amp_pha_pha_unwrap', E)
        
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
        np.save(f'{path}/Es_w{E_all.shape[-1]}_n2{number_of_n2}_Isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all', E_all)
    return E_all

def expend(
    number_of_n2: int, 
    number_of_power: int,
    number_of_isat: int,
    power: int,
    E: np.ndarray,
    noise_level: float,
    path: str = None, 
    ) -> np.ndarray:
    """
    From data in a array of shape (n2_num, frame_num, resolution), take the sequence for each n2
    and add a certain amount of noisy extra datasets for each n2 (number_of_extra) and return it
    or save.

    Args:
        number_of_n2 (int): number of different n2.
        number_of_frames (int): number of frames per n2.
        resolution (int): resolution of the 1d array.
        number_of_extra (int): number of extra datasets for each n2,
        noise_level (float): amount of noise added,
        path (str): path of directory in which the data is saved. (Default to None)
        data (np.ndarray): an array of shape (n2_num, frame_num, resolution) (Default to None)
    Returns:
        The complete dataset
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
    if not (path == None):
        if power != 0:
            np.save(f'{path}/Es_w{augmented_data.shape[-1]}_n2{number_of_n2}_Isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap_extended', augmented_data)
        else:
            np.save(f'{path}/Es_w{augmented_data.shape[-1]}_n2{number_of_n2}_Isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all_extended', augmented_data)
    return augmented_data

def normalize_data(data):
    """
    Normalize the data by applying channel-wise normalization.

    Parameters:
    data (np.ndarray): The input data array of shape (N, C, H, W) where
                       N is the number of images,
                       C is the number of channels (expected to be 4 in this case),
                       H is the height,
                       W is the width.

    Returns:
    np.ndarray: The normalized data with the same shape as input.
    """

    normalized_data = np.zeros_like(data, dtype=np.float64)
    
    for ch in range(data.shape[1]):
        for i in range(data.shape[0]):
            channel_data = data[i, ch, :, :]
            normalized_data[i, ch, :, :] = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data))

    return normalized_data