#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
from engine.engine_dataset import EngineDataset
from engine.generate import simulation
from engine.utils import apply_hog, plot_sandbox, set_seed
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

    experiment_field = zoom(experiment_field, (resolution_training/experiment_field.shape[-2], resolution_training/experiment_field.shape[-1]), order=5)
    density_experiment = np.abs(experiment_field)
    phase_experiment = (np.angle(experiment_field) + np.pi)/ (2*np.pi)


    density_experiment -= np.min(density_experiment, axis=(-1, -2), keepdims=True)
    density_experiment /= np.max(density_experiment, axis=(-1, -2), keepdims=True)
    
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

    dataset = EngineDataset(n2_values=np.asarray([n2_values]), alpha_values=np.asarray([alpha_values]), isat_values=np.asarray([isat_values]), 
                           input_power=input_power, waist=waist, non_locality=non_locality, delta_z=delta_z,
                           length=length, resolution_simulation=resolution_simulation, resolution_training=resolution_training, 
                           window_simulation=window_simulation, window_training=window_training, saving_path=saving_path, learning_rate=0, 
                           batch_size=0, num_epochs=0, accumulator=0, device_number=device_number)
        
    with cp.cuda.Device(device_number):
        simulation(dataset)

    dataset.field[:,0,:,:] = dataset.field[:,0,:,:] - np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
    dataset.field[:,0,:,:] = dataset.field[:,0,:,:] / np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)

    dataset.field[:, 1, :, :] = (dataset.field[:, 1, :, :] + np.pi) / (2 * np.pi)

    density_experiment, phase_experiment = experiment(resolution_training, exp_image_path)
    plot_sandbox(dataset, density_experiment, phase_experiment)  