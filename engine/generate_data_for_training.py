#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from engine.nlse_generator import data_creation, data_augmentation
from PIL import Image
from engine.waist_fitting import waist_computation
from scipy.ndimage import zoom

def from_input_image(
        path: str,
        number_of_power: int,
        number_of_n2: int,
        number_of_isat: int,
        resolution_in: int
        ) -> np.ndarray:
    """
    Loads an input image from the specified path and prepares it for nonlinear Schrödinger equation (NLSE) analysis by
    normalizing and tiling the image array based on specified parameters. The function also computes an approximation
    of the beam waist within the image.

    Parameters:
    - path (str): The file path to the input image. Currently supports TIFF format. Assumed to be square image
    - number_of_power (int): The number of different power levels for which the input field will be tiled.
    - number_of_n2 (int): The number of different nonlinear refractive index (n2) values for tiling.
    - number_of_isat (int): The number of different saturation intensities (Isat) for tiling.
    - resolution_in (float): The spatial resolution of the input image in meters per pixel.

    Returns:
    - input_field_tiled_n2_power_isat (np.ndarray): A 5D numpy array of the tiled input field adjusted for different
      n2, power, and Isat values. The array shape is (number_of_n2, number_of_power, number_of_isat, height, width),
      where height and width correspond to the dimensions of the input image.
    - waist (float): An approximation of the beam waist in meters, calculated based on the input image and resolution.
    """
    print("--- LOAD INPUT IMAGE ---")
    input_tiff = Image.open(path)
    image = np.array(input_tiff, dtype=np.float32)
    
    print("--- FIND WAIST ---")
    window = resolution_in * 5.5e-6
    waist = waist_computation(image, window, image.shape[0], image.shape[0], False)

    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]), order=5)

    print("--- PREPARE FOR NLSE ---")
    input_field = image + 1j * np.zeros_like(image, dtype=np.float32)
    input_field_tiled_n2_power_isat = np.tile(input_field[np.newaxis,np.newaxis, np.newaxis, :,:], (number_of_n2,number_of_power,number_of_isat, 1,1))

    return input_field_tiled_n2_power_isat, waist

def generate_data(
        saving_path: str, 
        image_path: str,
        resolutions: tuple,
        numbers: tuple, 
        generate: bool, 
        visualize: bool, 
        expanded: bool,
        expansion: bool,
        factor_window: int, 
        delta_z: float, 
        length: float, 
        transmission: float,
        device_number: int,
        )-> tuple:
    """
    Generates or loads data for NLSE simulation, optionally performing data augmentation and visualization.

    Parameters:
    - saving_path (str): Path to save generated or loaded data.
    - image_path (str): Path to an input image, if `is_from_image` is True.
    - resolutions (tuple): Tuple of input and output resolutions (resolution_in, resolution_out).
    - numbers (tuple): Tuple of the numbers of n2, power, and isat instances (number_of_n2, number_of_power, number_of_isat).
    - generate (bool): Flag to enable data generation.
    - visualize (bool): Flag to enable visualization of generated data.
    - expansion (bool): Flag to enable data augmentation.
    - factor_window (int): Factor to adjust the simulation window based on the waist.
    - delta_z (float): Step size in the Z-direction for the NLSE simulation.
    - length (float): Length of the propagation medium.
    - transmission (float): Transmission coefficient for the medium.

    Returns:
    - tuple: Contains labels and values for augmented data, if `expansion` is True; otherwise, 
      it returns labels and values for the generated or loaded data.
    """
    resolution_in, resolution_out = resolutions
    number_of_n2, number_of_power, number_of_isat = numbers

    n2_values = np.linspace(-1e-11, -1e-10, number_of_n2)
    n2_labels = np.arange(0, number_of_n2)

    power_values = np.linspace(.02, .5001, number_of_power)
    power_labels = np.arange(0, number_of_power)

    isat_values = np.linspace(1e4, 1e6, number_of_isat)
    isat_labels = np.arange(0, number_of_isat)
    
    
    if generate:
        input_field, waist = from_input_image(image_path, number_of_power, number_of_n2, number_of_isat, resolution_in)
        window = factor_window*waist
        with cp.cuda.Device(device_number):
            print("---- NLSE ----")
            data_creation(input_field, window, n2_values,power_values,isat_values, resolution_in,resolution_out, delta_z,transmission, length,saving_path)

    N2_values_single, ISAT_values_single = np.meshgrid(n2_values, isat_values,) 
    N2_labels_single, ISAT_labels_single = np.meshgrid(n2_labels, isat_labels)

    n2_values_all_single = N2_values_single.flatten()
    isat_values_all_single = ISAT_values_single.flatten()

    n2_labels_all_single = N2_labels_single.reshape((number_of_n2*number_of_isat,))
    isat_labels_all_single = ISAT_labels_single.reshape((number_of_n2*number_of_isat,))

    values_all_single = (n2_values_all_single, isat_values_all_single)
    labels_all_single = (n2_labels_all_single, isat_labels_all_single)

    if visualize:
        print("---- VISUALIZE ----")

        data_types = ["amp", "pha"]
        cmap_types = ["viridis", "twilight_shifted", "viridis"]

        for data_types_index in range(len(data_types)):
            
            for power_index in range(number_of_power):
                counter = 0
                E_power = np.load(f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power_values[power_index])[:4]}_amp_pha_pha_unwrap.npy')

                if number_of_isat > number_of_n2:
                    fig, axs = plt.subplots(number_of_isat,number_of_n2, figsize=(number_of_isat*5, number_of_n2*5))
                else:
                    fig, axs = plt.subplots(number_of_n2, number_of_isat, figsize=(number_of_n2*5, number_of_isat*5))
                
                for n2_index in range(number_of_n2):
                    for isat_index in range(number_of_isat):
                        if number_of_isat == 1 and number_of_n2 == 1:
                            axs.imshow(E_power[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                            axs.set_title(f'power={power_values[power_index]}, n2={n2_values[n2_labels_all_single[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_single[counter]])}')
                        elif number_of_isat == 1:
                            axs[n2_index].imshow(E_power[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                            axs[n2_index].set_title(f'power={power_values[power_index]}, n2={n2_values[n2_labels_all_single[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_single[counter]])}')
                        elif number_of_n2 == 1:
                            axs[isat_index].imshow(E_power[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                            axs[isat_index].set_title(f'power={power_values[power_index]}, n2={n2_values[n2_labels_all_single[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_single[counter]])}')
                        else:
                            axs[n2_index, isat_index].imshow(E_power[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                            axs[n2_index, isat_index].set_title(f'power={power_values[power_index]}, n2={n2_values[n2_labels_all_single[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_single[counter]])}')
                        counter += 1
                plt.tight_layout()
                plt.savefig(f'{saving_path}/{data_types[data_types_index]}_{str(power_values[power_index])[:4]}p_{number_of_n2}n2_{number_of_isat}Isat.png')
                plt.close()

    if expansion:

        print("---- EXPEND ----")
        for power in power_values:
            file = f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap.npy'
            E = np.load(file)

            noise = 0.01
            expansion_factor = data_augmentation(number_of_n2, number_of_isat, power, E, noise, saving_path)

        n2_labels_augmented_single = np.repeat(n2_labels_all_single, expansion_factor)
        isat_labels_augmented_single = np.repeat(isat_labels_all_single, expansion_factor)

        n2_values_augmented_single = np.repeat(n2_values_all_single, expansion_factor)
        isat_values_augmented_single = np.repeat(isat_values_all_single, expansion_factor)

        values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
        labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)
        
        return labels_augmented_single, values_augmented_single
    else:
        if expanded:
            expansion_factor = 33
            n2_labels_augmented_single = np.repeat(n2_labels_all_single, expansion_factor)
            isat_labels_augmented_single = np.repeat(isat_labels_all_single, expansion_factor)

            n2_values_augmented_single = np.repeat(n2_values_all_single, expansion_factor)
            isat_values_augmented_single = np.repeat(isat_values_all_single, expansion_factor)

            values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
            labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)
            
            single_augmented = labels_augmented_single, values_augmented_single
            return single_augmented
        else:
            return labels_all_single, values_all_single