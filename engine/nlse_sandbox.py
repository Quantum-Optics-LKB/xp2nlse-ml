#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
from engine.engine_dataset import EngineDataset
from engine.generate import simulation
from engine.utils import plot_sandbox, set_seed
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
    experiment_field = np.load(exp_image_path)
    density_experiment = zoom(np.abs(experiment_field), (resolution_training/experiment_field.shape[-2], resolution_training/experiment_field.shape[-1])).astype(np.float64)
    phase_experiment = zoom(np.angle(experiment_field), (resolution_training/experiment_field.shape[-2], resolution_training/experiment_field.shape[-1])).astype(np.float64)

    return density_experiment, phase_experiment

def sandbox(
        device_number: int, 
        resolution_simulation: int,
        window_simulation: float,
        window_training: float,
        resolution_training: int,
        n2_values: np.ndarray,
        input_power: float,
        alpha_values: float,
        isat_values: float,
        waist: float,
        non_locality: float,
        delta_z: float,
        length: float, 
        exp_image_path: str,
        saving_path: str,
        ) -> None:

    dataset = EngineDataset(n2_values=n2_values, alpha_values=alpha_values, isat_values=isat_values, 
                           input_power=input_power, waist=waist, non_locality=non_locality, delta_z=delta_z,
                           length=length, resolution_simulation=resolution_simulation, resolution_training=resolution_training, 
                           window_simulation=window_simulation, window_training=window_training, saving_path=saving_path, learning_rate=0, 
                           batch_size=0, num_epochs=0, accumulator=0, device_number=device_number)
        
    with cp.cuda.Device(device_number):
        simulation(dataset)

    density_experiment, phase_experiment = experiment(resolution_training, exp_image_path)
    plot_sandbox(dataset, density_experiment, phase_experiment)  