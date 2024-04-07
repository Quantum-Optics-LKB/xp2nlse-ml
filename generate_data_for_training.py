#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from nlse_generator import data_creation, data_augmentation
from PIL import Image
from waist_fitting import waist_computation
from scipy.ndimage import zoom
from matplotlib.colors import LogNorm

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
    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]), order=5)
    print("--- FIND WAIST ---")
    window = resolution_in * 5.5e-6
    waist = waist_computation(image, window, resolution_in, resolution_in, False)

    print("--- PREPARE FOR NLSE ---")
    input_field = image + 1j * np.zeros_like(image, dtype=np.float32)
    input_field_tiled_n2_power_isat = np.tile(input_field[np.newaxis,np.newaxis, np.newaxis, :,:], (number_of_n2,number_of_power,number_of_isat, 1,1))

    return input_field_tiled_n2_power_isat, waist

def from_gaussian(
        NXY: int, 
        window: int,
        numbers: tuple,
        waist: float 
        ) ->  np.ndarray:
    """
    Generates a 2D Gaussian beam profile across a specified meshgrid.

    Parameters:
    - NXY (int): The number of points in both the X and Y dimensions for the meshgrid.
    - window (int): The spatial extent of the square window in which the beam is defined.
    - numbers (tuple): A tuple containing the number of n2, power, and isat values.
    - waist (float): The waist (radius) of the Gaussian beam, determining its spread.

    Returns:
    - np.ndarray: A complex numpy array representing the Gaussian beam profile across the
      meshgrid, tiled according to the specified number of n2, power, and isat values.
    """
    number_of_n2, number_of_power, number_of_isat = numbers
    X, delta_X = np.linspace(
        -window / 2,
        window / 2,
        num=NXY,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )
    Y, delta_Y = np.linspace(
        -window / 2,
        window / 2,
        num=NXY,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )
    number_of_n2, number_of_power, number_of_isat = numbers
    XX, YY = np.meshgrid(X, Y)

    return np.ones((number_of_n2,number_of_power,number_of_isat, NXY, NXY), dtype=np.complex64) * np.exp(-(XX**2 + YY**2) / (waist**2))

def generate_data(
        saving_path: str, 
        image_path: str,
        resolutions: tuple,
        numbers: tuple, 
        is_from_image: bool, 
        generate: bool, 
        visualize: bool, 
        expension: bool,
        single_power: bool, 
        multiple_power: bool, 
        factor_window: int, 
        delta_z: float, 
        length: float, 
        transmission: float
        )-> tuple:
    """
    Generates or loads data for NLSE simulation, optionally performing data augmentation and visualization.

    Parameters:
    - saving_path (str): Path to save generated or loaded data.
    - image_path (str): Path to an input image, if `is_from_image` is True.
    - resolutions (tuple): Tuple of input and output resolutions (resolution_in, resolution_out).
    - numbers (tuple): Tuple of the numbers of n2, power, and isat instances (number_of_n2, number_of_power, number_of_isat).
    - is_from_image (bool): Flag to indicate whether data is generated from an image.
    - generate (bool): Flag to enable data generation.
    - visualize (bool): Flag to enable visualization of generated data.
    - expension (bool): Flag to enable data augmentation.
    - single_power (bool): Flag to indicate single power data generation.
    - multiple_power (bool): Flag to indicate multiple power levels data generation.
    - factor_window (int): Factor to adjust the simulation window based on the waist.
    - delta_z (float): Step size in the Z-direction for the NLSE simulation.
    - length (float): Length of the propagation medium.
    - transmission (float): Transmission coefficient for the medium.

    Returns:
    - tuple: Contains labels and values for augmented data, if `expension` is True; otherwise, 
      it returns labels and values for the generated or loaded data.
    """
    resolution_in, resolution_out = resolutions
    number_of_n2, number_of_power, number_of_isat = numbers

    n2_values = np.linspace(-1e-11, -1e-10, number_of_n2)
    n2_labels = np.arange(0, number_of_n2)

    power_values = np.linspace(.02, 0.5001, number_of_power)
    power_labels = np.arange(0, number_of_power)

    isat_values = np.linspace(1e4, 1e6, number_of_isat)
    isat_labels = np.arange(0, number_of_isat)

    if single_power:

        N2_values_single, ISAT_values_single = np.meshgrid(n2_values, isat_values,) 
        N2_labels_single, ISAT_labels_single = np.meshgrid(n2_labels, isat_labels)

        n2_values_all_single = N2_values_single.reshape((number_of_n2*number_of_isat,))
        isat_values_all_single = ISAT_values_single.reshape((number_of_n2*number_of_isat,))


        n2_labels_all_single = N2_labels_single.reshape((number_of_n2*number_of_isat,))
        isat_labels_all_single = ISAT_labels_single.reshape((number_of_n2*number_of_isat,))

        values_all_single = (n2_values_all_single, isat_values_all_single)
        labels_all_single = (n2_labels_all_single, isat_labels_all_single)
    
    if multiple_power:

        N2_values_multiple, POWER_values_multiple, ISAT_values_multiple = np.meshgrid(n2_values,power_values, isat_values,) 
        N2_labels_multiple, POWER_labels_multiple, ISAT_labels_multiple = np.meshgrid(n2_labels, power_labels, isat_labels)

        power_values_all_multiple = POWER_values_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))
        n2_values_all_multiple = N2_values_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))
        isat_values_all_multiple = ISAT_values_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))


        power_labels_all_multiple = POWER_labels_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))
        n2_labels_all_multiple = N2_labels_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))
        isat_labels_all_multiple = ISAT_labels_multiple.reshape((number_of_power*number_of_n2*number_of_isat,))

        values_all_multiple = (n2_values_all_multiple, power_values_all_multiple, isat_values_all_multiple)
        labels_all_multiple = (n2_labels_all_multiple, power_labels_all_multiple, isat_labels_all_multiple)

        if visualize:
            print("---- VISUALIZE ----")

            data_types = ["amp", "pha", "pha_unwrap"]
            cmap_types = ["viridis", "twilight_shifted", "viridis"]

            for data_types_index in range(len(data_types)):
                counter = 0
                for power_index in range(number_of_power):

                    if number_of_isat > number_of_n2:
                        fig, axs = plt.subplots(number_of_isat,number_of_n2, figsize=(number_of_n2*5, number_of_isat*5))
                    else:
                        fig, axs = plt.subplots(number_of_n2, number_of_isat, figsize=(number_of_n2*5, number_of_isat*5))
                    
                    for n2_index in range(number_of_n2):
                        for isat_index in range(number_of_isat):
                            if number_of_isat == 1 and number_of_n2 == 1:
                                axs.imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                                axs.set_title(f'power={power_values[power_labels_all_multiple[counter]]}, n2={n2_values[n2_labels_all_multiple[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_multiple[counter]])}')
                                plt.axis('off')
                            elif number_of_isat == 1:
                                axs[n2_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                                axs[n2_index].set_title(f'power={power_values[power_labels_all_multiple[counter]]}, n2={n2_values[n2_labels_all_multiple[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_multiple[counter]])}')
                                plt.axis('off')
                            elif number_of_n2 == 1:
                                axs[isat_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                                axs[isat_index].set_title(f'power={power_values[power_labels_all_multiple[counter]]}, n2={n2_values[n2_labels_all_multiple[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_multiple[counter]])}')
                                plt.axis('off')
                            else:
                                axs[n2_index, isat_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                                axs[n2_index, isat_index].set_title(f'power={power_values[power_labels_all_multiple[counter]]}, n2={n2_values[n2_labels_all_multiple[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all_multiple [counter]])}')
                                plt.axis('off')
                            counter += 1
                    plt.tight_layout()
                    plt.savefig(f'{saving_path}/{data_types[data_types_index]}_{str(power_values[power_index])[:4]}p_{number_of_n2}n2_{number_of_isat}Isat.png')
                    plt.close()

    if is_from_image:
        input_field, waist = from_input_image(image_path, number_of_power, number_of_n2, number_of_isat, resolution_in)
        window = factor_window*waist
    else:
        waist = 1e-3
        window = factor_window*waist
        input_field = from_gaussian(resolution_in, window, numbers, waist)

    if generate:
        with cp.cuda.Device(0):
            print("---- NLSE ----")
            E_clean = data_creation(input_field, window, n2_values,power_values,isat_values, resolution_in,resolution_out, delta_z,transmission, length,saving_path)
    else:
        E_clean = np.load(f"{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all.npy")

    if expension:

        if multiple_power and single_power:
            print("---- EXPEND MULTIPLE----")
            noise = 0.01
            power = 0
            E_expend_multiple, expension_factor = data_augmentation(number_of_n2, number_of_power, number_of_isat, power, E_clean, noise, saving_path)
            power_labels_augmented_multiple = np.repeat(power_labels_all_multiple, expension_factor)
            n2_labels_augmented_multiple = np.repeat(n2_labels_all_multiple, expension_factor)
            isat_labels_augmented_multiple = np.repeat(isat_labels_all_multiple, expension_factor)

            power_values_augmented_multiple = np.repeat(power_values_all_multiple, expension_factor)
            n2_values_augmented_multiple = np.repeat(n2_values_all_multiple, expension_factor)
            isat_values_augmented_multiple = np.repeat(isat_values_all_multiple, expension_factor)

            values_augmented_multiple = (n2_values_augmented_multiple, power_values_augmented_multiple, isat_values_augmented_multiple)
            labels_augmented_multiple = (n2_labels_augmented_multiple, power_labels_augmented_multiple, isat_labels_augmented_multiple)


            for power in power_values:
                print("---- EXPEND SINGLE ----")
                file = f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap.npy'
                E = np.load(file)

                noise = 0.01
                E_expend_single, expension_factor = data_augmentation(number_of_n2, number_of_power, number_of_isat, power, E, noise, saving_path)

            n2_labels_augmented_single = np.repeat(n2_labels_all_single, expension_factor)
            isat_labels_augmented_single = np.repeat(isat_labels_all_single, expension_factor)

            n2_values_augmented_single = np.repeat(n2_values_all_single, expension_factor)
            isat_values_augmented_single = np.repeat(isat_values_all_single, expension_factor)

            values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
            labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)

            labels_augmented = (labels_augmented_single, labels_augmented_multiple)
            values_augmented = (values_augmented_single, values_augmented_multiple)
            
            return labels_augmented, values_augmented
        
        elif single_power:
            
            for power in power_values:

                print("---- EXPEND SINGLE ----")
                file = f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap.npy'
                E = np.load(file)

                noise = 0.01
                E_expend_single, expension_factor = data_augmentation(number_of_n2, number_of_power, number_of_isat, power, E, noise, saving_path)

            n2_labels_augmented_single = np.repeat(n2_labels_all_single, expension_factor)
            isat_labels_augmented_single = np.repeat(isat_labels_all_single, expension_factor)

            n2_values_augmented_single = np.repeat(n2_values_all_single, expension_factor)
            isat_values_augmented_single = np.repeat(isat_values_all_single, expension_factor)

            values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
            labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)

            single_augmented = labels_augmented_single, values_augmented_single
            
            return single_augmented
        
        elif multiple_power:
            print("---- EXPEND MULTIPLE----")
            noise = 0.01
            power = 0
            E_expend_multiple, expension_factor = data_augmentation(number_of_n2, number_of_power, number_of_isat, power, E_clean, noise, saving_path)
            power_labels_augmented = np.repeat(power_labels_all_multiple, expension_factor)
            n2_labels_augmented = np.repeat(n2_labels_all_multiple, expension_factor)
            isat_labels_augmented = np.repeat(isat_labels_all_multiple, expension_factor)

            power_values_augmented = np.repeat(power_values_all_multiple, expension_factor)
            n2_values_augmented = np.repeat(n2_values_all_multiple, expension_factor)
            isat_values_augmented = np.repeat(isat_values_all_multiple, expension_factor)

            values_augmented = (n2_values_augmented, power_values_augmented, isat_values_augmented)
            labels_augmented = (n2_labels_augmented, power_labels_augmented, isat_labels_augmented)

            multiple_augmented = labels_augmented, values_augmented
            
            return  multiple_augmented
    else:
        expension_factor = 33
        power_labels_augmented_multiple = np.repeat(power_labels_all_multiple, expension_factor)
        n2_labels_augmented_multiple = np.repeat(n2_labels_all_multiple, expension_factor)
        isat_labels_augmented_multiple = np.repeat(isat_labels_all_multiple, expension_factor)

        power_values_augmented_multiple = np.repeat(power_values_all_multiple, expension_factor)
        n2_values_augmented_multiple = np.repeat(n2_values_all_multiple, expension_factor)
        isat_values_augmented_multiple = np.repeat(isat_values_all_multiple, expension_factor)

        values_augmented_multiple = (n2_values_augmented_multiple, power_values_augmented_multiple, isat_values_augmented_multiple)
        labels_augmented_multiple = (n2_labels_augmented_multiple, power_labels_augmented_multiple, isat_labels_augmented_multiple)
    
        n2_labels_augmented_single = np.repeat(n2_labels_all_single, expension_factor)
        isat_labels_augmented_single = np.repeat(isat_labels_all_single, expension_factor)

        n2_values_augmented_single = np.repeat(n2_values_all_single, expension_factor)
        isat_values_augmented_single = np.repeat(isat_values_all_single, expension_factor)

        values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
        labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)

        if multiple_power and single_power:
            labels_augmented = (labels_augmented_single, labels_augmented_multiple)
            values_augmented = (values_augmented_single, values_augmented_multiple)
            return labels_augmented, values_augmented
        
        elif multiple_power:
            multiple_augmented = labels_augmented_multiple, values_augmented_multiple
            return multiple_augmented
        
        elif single_power:
            single_augmented = labels_augmented_single, values_augmented_single
            return single_augmented