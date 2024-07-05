#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
from engine.generate import data_creation
from skimage.restoration import unwrap_phase
from engine.utils import plot_sandbox, set_seed
from engine.utils import general_extrema, normalize_data
set_seed(10)

def experiment(
        resolution_training: int,
        exp_image_path: str
        ) -> tuple:
    """
    Process experimental field data to generate density, phase, and unwrapped phase images.

    Args:
        resolution_training (int): The resolution to which the experimental data should be resized.
        exp_image_path (str): The path to the experimental image data (a numpy file).

    Returns:
        tuple: A tuple containing:
            - density_experiment (np.ndarray): Normalized density image.
            - phase_experiment (np.ndarray): Normalized phase image.
            - uphase_experiment (np.ndarray): Normalized unwrapped phase image.

    Description:
        This function loads the experimental field data, computes the density, phase,
        and unwrapped phase, normalizes them, and resizes them to the specified resolution.
    """
    field = np.load(exp_image_path)
    density_experiment = normalize_data(zoom(np.abs(field), 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = np.angle(field)
    uphase_experiment = general_extrema(unwrap_phase(phase_experiment, rng=10))
    uphase_experiment = normalize_data(zoom(uphase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)

    return density_experiment, phase_experiment, uphase_experiment

def sandbox(
        device: int, 
        resolution_input_beam: int,
        window_input: float,
        window_out: float,
        resolution_training: int,
        n2: np.ndarray,
        input_power: float,
        alpha: float,
        isat: float,
        waist_input_beam: float,
        non_locality_length: float,
        delta_z: float,
        cell_length: float, 
        exp_image_path: str,
        saving_path: str,
        ) -> None:
    """
    Perform NLSE data creation and compare with experimental results, plotting the outcome.

    Args:
        device (int): The GPU device ID for computation.
        resolution_input_beam (int): The input resolution of the beam.
        window_input (float): The size of the input window.
        window_out (float): The size of the output window.
        resolution_training (int): The resolution for training data.
        n2 (np.ndarray): Array of nonlinear refractive indices.
        input_power (float): The input power of the beam.
        alpha (float): The alpha parameter.
        isat (float): The saturation intensity.
        waist_input_beam (float): The waist of the input beam.
        non_locality_length (float): The non-locality length.
        delta_z (float): The step size for the z-axis.
        cell_length (float): The total length of the cell.
        exp_image_path (str): The path to the experimental image data (a numpy file).
        saving_path (str): The path to save the generated results.

    Returns:
        None

    Description:
        This function performs the following steps:
        1. Sets up camera and NLSE settings.
        2. Creates the synthetic dataset using the NLSE settings.
        3. Processes the experimental data to generate density, phase, and unwrapped phase images.
        4. Plots the synthetic and experimental data for comparison.
    """

    cameras = resolution_input_beam, window_input, window_out, resolution_training
    nlse_settings = np.array([n2]), input_power, np.array([alpha]), np.array([isat]), waist_input_beam, non_locality_length, delta_z, cell_length
    
    with cp.cuda.Device(device):
        E = data_creation(nlse_settings, cameras, device)

    density_experiment, phase_experiment, uphase_experiment = experiment(resolution_training, exp_image_path)

    plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment, window_out, n2, isat, alpha, input_power, saving_path)  