#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import random
import numpy as np
import torch
from engine.seed_settings import set_seed
import kornia.augmentation as K
set_seed(42)

def augmentation(
        original_height: int, 
        original_width: int
        ) -> torch.nn.Sequential:
    """
    Constructs a data augmentation pipeline with a specific set of transformations tailored for 
    optical field data or similar types of images. This pipeline includes blurring, cropping, and 
    resizing operations to simulate various realistic alterations that data might undergo.

    Parameters:
    - original_height (int): The original height of the images before augmentation.
    - original_width (int): The original width of the images before augmentation.

    Returns:
    torch.nn.Sequential: A Kornia Sequential object that contains a 
    sequence of augmentation transformations to be applied to the images. These transformations 
    include Gaussian blur, motion blur, a slight shift without scaling or rotation, 
    and resizing back to the original dimensions.

    The pipeline is set up to apply these transformations with certain probabilities, allowing for a 
    diversified dataset without excessively distorting the underlying data characteristics. This 
    augmentation strategy is particularly useful for training machine learning models on image data, 
    as it helps to improve model robustness by exposing it to a variety of visual perturbations.

    Example Usage:
        augmentation_pipeline = get_augmentation(256, 256)
        augmented_image = augmentation_pipeline(image=image)
    """
    shift = random.uniform(0.1,0.25)
    shear = random.uniform(20,50)
    direction = random.uniform(-1, 1)
    return torch.nn.Sequential(
        K.RandomMotionBlur(kernel_size=51, angle=random.uniform(0, 360), direction=(direction, direction), border_type='replicate', p=0.2),
        K.RandomGaussianBlur(kernel_size=(51, 51), sigma=(100.0, 100.0), p=0.5),
        K.RandomAffine(degrees=0, translate=(shift, shift), scale=(1.0, 1.0), shear=shear, p=0.2),
        K.Resize((original_height, original_width))
    )

def add_model_noise(
        beam: np.ndarray, 
        poisson_noise_lam: float,
        normal_noise_sigma: float
          )-> np.ndarray:
        
    poisson_noise = np.random.poisson(lam=poisson_noise_lam, size=(beam.shape))*poisson_noise_lam*0.75
    normal_noise = np.random.normal(0, normal_noise_sigma, (beam.shape))

    total_noise = normal_noise + poisson_noise
    noisy_beam = np.real(beam) + total_noise + 1j * np.imag(beam)

    noisy_beam = noisy_beam.astype(np.complex64)
    return noisy_beam

def salt_and_pepper_noise(image, noise_level):
    # Function to add salt and pepper noise to an image
    noisy = image.copy()
    probs = np.random.rand(*image.shape)
    noisy[probs < noise_level / 2] = 0
    noisy[probs > 1 - noise_level / 2] = np.max(image)
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
    lines_pattern =  amplitude*np.sin(X_rotated * wave_frequency)
    
    noisy_image = image.copy() + lines_pattern
    
    return noisy_image