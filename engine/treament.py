#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch
import random
import numpy as np
import kornia.augmentation as K
from engine.seed_settings import set_seed

set_seed(10)

def normalize_data(
        data: np.ndarray,
        ) -> np.ndarray: 
    data -= np.min(data, axis=(-2, -1), keepdims=True)
    data /= np.max(data, axis=(-2, -1), keepdims=True)
    return data

def general_extrema(E):
    if E[E.shape[-2]//2, E.shape[-1]//2] > E[0, 0]:
        E -= np.max(E)
        E = np.abs(E)
    return E

def elastic_saltpepper() -> torch.nn.Sequential:
    
    elastic_sigma = (random.randrange(35, 42, 2), random.randrange(35, 42, 2))
    elastic_alpha = (1, 1)
    salt_pepper = random.uniform(0.01, .11)
    return torch.nn.Sequential(
        K.RandomElasticTransform(kernel_size=51, sigma=elastic_sigma, alpha=elastic_alpha ,p=.5),
        K.RandomSaltAndPepperNoise(amount=salt_pepper,salt_vs_pepper=(.5, .5), p=.2),
    )

def experiment_noise(
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