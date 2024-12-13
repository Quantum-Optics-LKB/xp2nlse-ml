#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from engine.use import get_parameters
from engine.generate import simulation
from engine.engine_dataset import EngineDataset
from engine.utils import plot_generated_set, set_seed, shuffle_dataset
from engine.training_manager import manage_training, prepare_training
set_seed(10)

def manager(
        generate: bool, 
        training: bool, 
        create_visual: bool,
        use: bool,
        plot_generate_compare: bool,
        window_training: float,
        n2_values: np.ndarray,
        alpha_values: float,
        isat_values: float,
        input_power: float,
        waist: float,
        length: float, 
        saving_path: str,
        exp_image_path: str,
        resolution_simulation: int,
        window_simulation: float,
        device_number: int = 0, 
        resolution_training: int = 224,
        non_locality: float = 0,
        delta_z: float = 1e-4,
        learning_rate: float = 1e-4, 
        batch_size: int = 128, 
        num_epochs: int = 200, 
        accumulator: int = 32
        ) -> None:
    """
    Manages dataset generation, training, visualization, and usage processes.

    Parameters:
    -----------
    generate : bool
        Flag to generate simulation data.
    training : bool
        Flag to train the dataset.
    create_visual : bool
        Flag to create visualizations of generated datasets.
    use : bool
        Flag to compute and compare parameters with experimental data.
    plot_generate_compare : bool
        Flag to visualize comparisons of generated data.
    window_training : float
        Training dataset window size in meters.
    n2_values : np.ndarray
        Array of n2 parameter values.
    alpha_values : float
        Array of alpha parameter values.
    isat_values : float
        Array of isat parameter values.
    input_power : float
        Input power value in watts.
    waist : float
        Waist size in meters.
    length : float
        Simulation length in meters.
    saving_path : str
        Path to save generated data and results.
    exp_image_path : str
        Path to experimental images for comparison.
    device_number : int, optional
        GPU device number for computation. Default is 0.
    resolution_simulation : int, optional
        Resolution of the simulation grid. Default is 512.
    window_simulation : float, optional
        Simulation window size in meters. Default is 20e-3.
    resolution_training : int, optional
        Resolution of the training grid. Default is 256.
    non_locality : float, optional
        Non-locality parameter value. Default is 0.
    delta_z : float, optional
        Z-axis step size for simulations. Default is 1e-4.
    learning_rate : float, optional
        Learning rate for training. Default is 0.01.
    batch_size : int, optional
        Batch size for training. Default is 100.
    num_epochs : int, optional
        Number of epochs for training. Default is 30.
    accumulator : int, optional
        Accumulator value for simulation steps. Default is 1.

    Returns:
    --------
    None
        This function does not return any value. It manages the dataset and processes.
    """
    # Initialize the dataset object with provided parameters    
    dataset = EngineDataset(n2_values=n2_values, alpha_values=alpha_values, isat_values=isat_values, 
                           input_power=input_power, waist=waist, non_locality=non_locality, delta_z=delta_z,
                           length=length, resolution_simulation=resolution_simulation, resolution_training=resolution_training, 
                           window_simulation=window_simulation, window_training=window_training, saving_path=saving_path, learning_rate=learning_rate, 
                           batch_size=batch_size, num_epochs=num_epochs, accumulator=accumulator, device_number=device_number)

    # If generating or training is requested
    if generate or training:
        if generate:
            # Generate simulation data on the specified GPU device
            with cp.cuda.Device(device_number):
                simulation(dataset)
        else:
            # Load previously generated field data from file
            path = f'{saving_path}/Es_w{resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{input_power:.2f}.npy'
            dataset.field = np.load(path)
        
        if create_visual:
            # Create visualizations for the generated data
            plot_generated_set(dataset)
        
        # Normalize the field data
        min_values = np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.subtract(dataset.field[:,0,:,:], min_values, out=dataset.field[:,0,:,:])

        max_values = np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.divide(dataset.field[:,0,:,:], max_values, out=dataset.field[:,0,:,:])
        
        # Normalize phase values to [0, 1]
        dataset.field[:, 1, :, :] = (dataset.field[:, 1, :, :] + np.pi) / (2 * np.pi)

        # Shuffle the dataset to randomize data order
        shuffle_dataset(dataset)

        if training:
            # If training is requested, prepare and manage the training process
            print("---- TRAINING ----")
            training_set, validation_set, test_set, model_settings = prepare_training(dataset)
            manage_training(dataset, training_set, validation_set, test_set, model_settings)
    
    # If not generating or training but visualizations are requested
    if not generate and not training and create_visual:
        # Load the field data from file and generate visualizations
        path = f'{saving_path}/Es_w{resolution_training}'
        path += f'_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}'
        path += f'_power{input_power:.2f}.npy'
        dataset.field = np.load(path)
        plot_generated_set(dataset)

    # If using the data for parameter computation
    if use:
        print("---- COMPUTING PARAMETERS ----\n")
        get_parameters(exp_image_path, dataset, plot_generate_compare)