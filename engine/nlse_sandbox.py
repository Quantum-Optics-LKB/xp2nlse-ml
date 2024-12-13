#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
from engine.generate import simulation
from engine.engine_dataset import EngineDataset
from engine.utils import plot_sandbox, set_seed
set_seed(10)

def experiment(
        resolution_training: int,
        exp_image_path: str
        ) -> tuple:
    """
    Processes experimental field data to generate normalized density and phase images.

    Parameters:
    -----------
    resolution_training : int
        The resolution to which the experimental data should be resized.
    exp_image_path : str
        Path to the experimental image data (a numpy file).

    Returns:
    --------
    tuple:
        - density_experiment (np.ndarray): Normalized density image, shape [resolution_training, resolution_training].
        - phase_experiment (np.ndarray): Normalized phase image, shape [resolution_training, resolution_training].
    """
    experiment_field = np.load(exp_image_path)

    experiment_field = zoom(experiment_field, (resolution_training/experiment_field.shape[-2], resolution_training/experiment_field.shape[-1]), order=5)
    density_experiment = np.abs(experiment_field)
    phase_experiment = (np.angle(experiment_field) + np.pi)/ (2*np.pi)


    density_experiment -= np.min(density_experiment, axis=(-1, -2), keepdims=True)
    density_experiment /= np.max(density_experiment, axis=(-1, -2), keepdims=True)
    
    return density_experiment, phase_experiment

def sandbox(
        resolution_simulation: int,
        window_simulation: float,
        window_training: float,
        n2_values: np.ndarray,
        input_power: float,
        alpha_values: float,
        isat_values: float,
        waist: float,
        non_locality: float,
        length: float, 
        exp_image_path: str,
        saving_path: str,
        delta_z: float = 1e-4,
        device_number: int = 0, 
        resolution_training: int = 224,
        ) -> None:
    """
    Simulates a dataset and compares the simulated fields with experimental data.

    Parameters:
    -----------
    device_number : int
        GPU device number to use for simulation.
    resolution_simulation : int
        Resolution of the simulation grid.
    window_simulation : float
        Simulation window size in meters.
    window_training : float
        Training window size in meters.
    resolution_training : int
        Resolution of the training grid.
    n2_values : np.ndarray
        Array of n2 parameter values.
    input_power : float
        Input power in watts.
    alpha_values : float
        Alpha parameter value.
    isat_values : float
        Isat parameter value.
    waist : float
        Waist size of the beam in meters.
    non_locality : float
        Non-locality parameter value.
    delta_z : float
        Z-axis step size in meters.
    length : float
        Simulation length in meters.
    exp_image_path : str
        Path to the experimental image data.
    saving_path : str
        Path to save the generated simulation data.

    Returns:
    --------
    None
        This function performs simulations and visualizes comparisons without returning values.
    """

    # Initialize the dataset with provided simulation and training parameters
    dataset = EngineDataset(
        n2_values=np.asarray([n2_values]), 
        alpha_values=np.asarray([alpha_values]), 
        isat_values=np.asarray([isat_values]), 
        input_power=input_power, waist=waist, 
        non_locality=non_locality, 
        delta_z=delta_z,
        length=length, 
        resolution_simulation=resolution_simulation, 
        resolution_training=resolution_training, 
        window_simulation=window_simulation, 
        window_training=window_training, 
        saving_path=saving_path, 
        learning_rate=0, 
        batch_size=0, 
        num_epochs=0, 
        accumulator=0, 
        device_number=device_number
        )
    
    # Perform simulation on the specified GPU device
    with cp.cuda.Device(device_number):
        simulation(dataset)

    dataset.field[:,0,:,:] = dataset.field[:,0,:,:] - np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
    dataset.field[:,0,:,:] = dataset.field[:,0,:,:] / np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)

    dataset.field[:, 1, :, :] = (dataset.field[:, 1, :, :] + np.pi) / (2 * np.pi)

    # Process experimental field data
    density_experiment, phase_experiment = experiment(resolution_training, exp_image_path)

    # Visualize the comparison between simulated and experimental data
    plot_sandbox(dataset, density_experiment, phase_experiment)  