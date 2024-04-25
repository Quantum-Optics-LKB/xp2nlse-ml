#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from matplotlib import pyplot as plt
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
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    image = np.sqrt(image)
    resolution_image = image.shape[0]

    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]))

    print("---- PINHOLE ----")
    window =   resolution_image * 5.5e-6
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
    number_of_n2, power_alpha, number_of_isat = numbers

    power_values, alpha_values = power_alpha
    number_of_power = len(power_values)
    n2_values = np.linspace(-1e-11, -1e-9, number_of_n2)
    n2_labels = np.arange(0, number_of_n2)

    isat_values = np.linspace(1e4, 1e6, number_of_isat)
    isat_labels = np.arange(0, number_of_isat)
    
    
    if generate:
        input_field, window = from_input_image(image_path, number_of_n2, number_of_isat,resolution_in ,pinsize)
        with cp.cuda.Device(device_number):
            print("---- NLSE ----")
            data_creation(input_field, window, n2_values,power_alpha,isat_values, resolution_in,resolution_out, delta_z, length,saving_path)

    N2_values_single, ISAT_values_single = np.meshgrid(n2_values, isat_values,) 
    N2_labels_single, ISAT_labels_single = np.meshgrid(n2_labels, isat_labels)

    n2_values_all_single = N2_values_single.flatten()
    isat_values_all_single = ISAT_values_single.flatten()

    n2_labels_all_single = N2_labels_single.reshape((number_of_n2*number_of_isat,))
    isat_labels_all_single = ISAT_labels_single.reshape((number_of_n2*number_of_isat,))

    if expansion:

        print("---- EXPEND ----")
        file = f'{saving_path}/Es_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}.npy'
        E = np.load(file)

        noise = 0.01
        expansion_factor = data_augmentation(number_of_n2, number_of_isat, number_of_power, E, noise, saving_path)

        n2_labels_augmented_single = np.repeat(n2_labels_all_single, expansion_factor)
        isat_labels_augmented_single = np.repeat(isat_labels_all_single, expansion_factor)

        n2_values_augmented_single = np.repeat(n2_values_all_single, expansion_factor)
        isat_values_augmented_single = np.repeat(isat_values_all_single, expansion_factor)

        values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
        labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)
        
        return labels_augmented_single, values_augmented_single
    else:
        expansion_factor = 33
        n2_labels_augmented_single = np.repeat(n2_labels_all_single, expansion_factor)
        isat_labels_augmented_single = np.repeat(isat_labels_all_single, expansion_factor)

        n2_values_augmented_single = np.repeat(n2_values_all_single, expansion_factor)
        isat_values_augmented_single = np.repeat(isat_values_all_single, expansion_factor)

        values_augmented_single = (n2_values_augmented_single, isat_values_augmented_single)
        labels_augmented_single = (n2_labels_augmented_single, isat_labels_augmented_single)
        
        single_augmented = labels_augmented_single, values_augmented_single
        return single_augmented
        