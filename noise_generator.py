#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np

def salt_and_pepper_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Adds salt and pepper noise to the given data.

    This function randomly flips some pixels of the image to either black (0) or white (1) to 
    create the 'salt and pepper' effect. The number of pixels affected is determined by the 
    noise level.

    Parameters:
        data (np.ndarray): The original dataset to which noise will be added. The data is 
                           expected to be an array of images.
        noise_level (float): A proportion between 0 and 1 indicating the percentage of 
                             pixels in the image to be affected by noise.

    Returns:
        np.ndarray: The dataset with added salt and pepper noise.
    """
    noisy_data = np.copy(data)
    salt_pepper = np.random.choice([0, 1, 2], data.shape, p=[noise_level/2, noise_level/2, 1-noise_level])
    noisy_data[salt_pepper == 0] = 0  # pepper (black)
    noisy_data[salt_pepper == 1] = np.max(data)  # salt (white)
    return noisy_data

def line_noise(image: np.ndarray, num_lines: int, amplitude: float, angle: float) -> np.ndarray:
    """
    Applies a pattern of lines at a specified angle across the image.

    Parameters:
        image (np.ndarray): The original image to which noise will be added.
        num_lines (int): The number of lines (complete sine wave cycles) to be applied.
        amplitude (float): The amplitude of the lines pattern, determining the intensity variation.
        angle (float): The angle of the lines in degrees, measured counter-clockwise from the horizontal axis.

    Returns:
        np.ndarray: The image with the applied lines pattern.
    """
    height, width = image.shape
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle)
    
    # Create a grid of coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Rotate the coordinate system by the specified angle
    X_rotated = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    
    # Calculate the wave frequency for the desired number of lines
    diagonal_length = np.sqrt(width**2 + height**2)
    wave_frequency = (num_lines * 2 * np.pi) / diagonal_length
    
    # Apply the sine function to create the lines pattern
    lines_pattern =  amplitude* np.sin(X_rotated * wave_frequency)
    
    noisy_image = image + lines_pattern
    
    return noisy_image

def gaussian_noise(
    data: np.ndarray, 
    noise_level: float
    ) -> np.ndarray:
    """
    Adds Gaussian noise to the given data.

    This function generates noise following a Gaussian distribution centered at zero with a 
    standard deviation defined by the noise level. The noise is added to the original data, 
    producing a noisy version of the input dataset.

    Parameters:
        data (np.ndarray): The original dataset to which noise will be added. This can be 
                           an array of any shape.
        noise_level (float): The standard deviation of the Gaussian noise to be added. 
                             This determines the level of noise.

    Returns:
        np.ndarray: The dataset with added Gaussian noise.
    """
    
    noisy_data = data.real + np.random.normal(0, noise_level, data.shape) + 1j * (data.imag + np.random.normal(0, noise_level, data.shape))
    return noisy_data.astype(np.complex64)