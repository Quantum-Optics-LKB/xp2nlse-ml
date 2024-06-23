#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.seed_settings import set_seed
set_seed(42)

def salt_and_pepper_noise(image, noise_level):
    # Function to add salt and pepper noise to an image
    noisy = image.copy()
    probs = np.random.rand(*image.shape)
    noisy[probs < noise_level / 2] = 0
    noisy[probs > 1 - noise_level / 2] = 100*np.max(image)
    return noisy

def line_noise(
        image: np.ndarray,
        num_lines: int, 
        amplitude: float, 
        angle: float
        ) -> np.ndarray:
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
    
    noisy_image = image.copy() + lines_pattern
    
    return noisy_image