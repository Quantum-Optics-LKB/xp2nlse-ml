#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
import cupy as cp
from engine.nlse_generator import data_creation, data_augmentation
from PIL import Image
from engine.waist_fitting import pinhole
from scipy.ndimage import zoom

def from_input_image(
        path: str,
        number_of_n2: int,
        number_of_isat: int,
        resolution_in: int,
        pinsize: float
        ) -> np.ndarray:
    """
    Loads an input image from the specified path and prepares it for nonlinear Schrödinger equation (NLSE) analysis by
    normalizing and tiling the image array based on specified parameters. The function also computes an approximation
    of the beam waist within the image.

    Parameters:
    - path (str): The file path to the input image. Currently supports TIFF format. Assumed to be square image
    - number_of_n2 (int): The number of different nonlinear refractive index (n2) values for tiling.
    - number_of_isat (int): The number of different saturation intensities (Isat) for tiling.
    - pinsize (float): size of the pinhole

    Returns:
    - input_field_tiled_n2_power_isat (np.ndarray): A 4D numpy array of the tiled input field adjusted for different
      n2, and Isat values. The array shape is (number_of_n2, number_of_isat, height, width),
      where height and width correspond to the dimensions of the input image.
    - waist (float): An approximation of the beam waist in meters, calculated based on the input image and resolution.
    """
    print("---- LOAD INPUT IMAGE ----")
    if path.endswith(".npy"):
        image = np.load(path).astype(np.float32)
    else:
        input_tiff = Image.open(path)
        image = np.array(input_tiff, dtype=np.float32)
    min_image = np.min(image)
    max_image = np.max(image)
    image -= min_image
    image /= max_image - min_image 
    np.sqrt(image, out=image)

    resolution_image = image.shape[0]

    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]))

    print("---- PINHOLE ----")
    window =  resolution_image * 5.5e-6
    image = pinhole(image, window, image.shape[0], image.shape[0], False,pinsize)


    image = image + 1j * 0
    print("---- PREPARE FOR NLSE ----")
    input_field_tiled_n2_power_isat = np.tile(image[np.newaxis, np.newaxis, :,:], (number_of_n2,number_of_isat, 1,1))
    return input_field_tiled_n2_power_isat, window

def generate_data(
        saving_path: str, 
        image_path: str,
        resolutions: tuple,
        numbers: tuple, 
        generate: bool, 
        expansion: bool,
        training: bool,
        delta_z: float, 
        length: float, 
        device_number: int,
        pinsize: float,
        )-> tuple:
    """
    Generates or loads data for NLSE simulation, optionally performing data augmentation and visualization.

    Parameters:
    - saving_path (str): Path to save generated or loaded data.
    - image_path (str): Path to an input image, if `is_from_image` is True.
    - resolutions (tuple): Tuple of input and output resolutions (resolution_in, resolution_out).
    - numbers (tuple): Tuple of the numbers of n2, power, and isat instances (number_of_n2, number_of_power, number_of_isat).
    - generate (bool): Flag to enable data generation.
    - expansion (bool): Flag to enable data augmentation.
    - delta_z (float): Step size in the Z-direction for the NLSE simulation.
    - length (float): Length of the propagation medium.
    - pinsize (float): size of pinhole

    Returns:
    - tuple: Contains labels and values for augmented data, if `expansion` is True; otherwise, 
      it returns labels and values for the generated or loaded data.
    """
    resolution_in, resolution_out = resolutions
    n2, powers, alpha, isat = numbers

    number_of_power = len(powers)
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    N2_single, ISAT_single = np.meshgrid(n2, isat) 

    n2_all_single = N2_single.reshape(-1)
    isat_all_single = ISAT_single.reshape(-1)
    
    if generate and expansion:
        
        input_field, window = from_input_image(image_path, number_of_n2, number_of_isat,resolution_in ,pinsize)
        with cp.cuda.Device(device_number):
            print("---- NLSE ----")
            E = data_creation(input_field, window, numbers, resolution_in,resolution_out, delta_z, length,saving_path)

        noise = 0.01
        expansion_factor, E_augmented = data_augmentation(number_of_n2, number_of_isat, number_of_power, E, noise, saving_path)

        n2_augmented_single = np.repeat(n2_all_single, expansion_factor)
        isat_augmented_single = np.repeat(isat_all_single, expansion_factor)

        values_augmented_single = (n2_augmented_single, isat_augmented_single)        
        
        return values_augmented_single, E_augmented
    
    elif generate:
        input_field, window = from_input_image(image_path, number_of_n2, number_of_isat,resolution_in ,pinsize)
        with cp.cuda.Device(device_number):
            print("---- NLSE ----")
            E = data_creation(input_field, window, numbers, resolution_in,resolution_out, delta_z, length,saving_path)

        values_single = (n2_all_single, isat_all_single)
        return values_single, E
    elif expansion:

        print("---- EXPEND ----")
        file = f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}.npy'
        E = np.load(file)

        noise = 0.01
        expansion_factor, E_augmented = data_augmentation(number_of_n2, number_of_isat, number_of_power, E, noise, saving_path)

        n2_augmented_single = np.repeat(n2_all_single, expansion_factor)
        isat_augmented_single = np.repeat(isat_all_single, expansion_factor)

        values_augmented_single = (n2_augmented_single, isat_augmented_single)
        
        return values_augmented_single, E
    
    elif training:
        
        E = np.load(f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}.npy')
        values_single = (n2_all_single, isat_all_single)
        return values_single, E
    
    else:
        values_single = (n2_all_single, isat_all_single)
        E = np.zeros((number_of_isat*number_of_n2,2*number_of_power,resolution_out, resolution_out), dtype=np.float16)
        return values_single, E