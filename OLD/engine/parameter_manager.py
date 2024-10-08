#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from engine.use import get_parameters
from engine.utils import plot_generated_set, set_seed, standardize_data
from engine.generate import data_creation, generate_labels
from engine.training_manager import manage_training, prepare_training
set_seed(10)

def manager(
        generate: bool, 
        training: bool, 
        create_visual: bool,
        use: bool,
        plot_generate_compare: bool,
        window_out: float,
        n2: np.ndarray,
        number_of_n2: int,
        alpha: float,
        number_of_alpha: int,
        isat: float,
        number_of_isat: int,
        input_power: float,
        waist_input_beam: float,
        cell_length: float, 
        saving_path: str,
        exp_image_path: str,
        device: int = 0, 
        resolution_input_beam: int = 512,
        window_input: float = 20e-3,
        resolution_training: int = 256,
        non_locality_length: float = 0,
        delta_z: float = 1e-4,
        learning_rate: float = 0.01, 
        batch_size: int = 100, 
        num_epochs: int = 30, 
        accumulator: int = 1
        ) -> None:
    
    """
    Manage the generation, training and visualization.

    Args:
        generate (bool): Flag to generate new data.
        training (bool): Flag to train a model with generated data.
        create_visual (bool): Flag to create visualizations of the generated data.
        use (bool): Flag to compute parameters from experimental data.
        plot_generate_compare (bool): Flag to plot and compare generated data with experimental data.
        window_out (float): Size of the output window.
        n2 (np.ndarray): Array of nonlinear refractive indices.
        number_of_n2 (int): Number of n2 values.
        alpha (float): The alpha parameter.
        number_of_alpha (int): Number of alpha values.
        isat (float): The saturation intensity.
        number_of_isat (int): Number of isat values.
        input_power (float): The input power of the beam.
        waist_input_beam (float): The waist of the input beam.
        cell_length (float): The total length of the cell.
        saving_path (str): Path to save the generated results and models.
        exp_image_path (str): Path to the experimental image data (a numpy file).
        device (int, optional): GPU device ID for computation. Default is 0.
        resolution_input_beam (int, optional): Input resolution of the beam. Default is 512.
        window_input (float, optional): Size of the input window. Default is 20e-3.
        resolution_training (int, optional): Resolution for training data. Default is 256.
        non_locality_length (float, optional): Non-locality length. Default is 0.
        delta_z (float, optional): Step size for the z-axis. Default is 1e-4.
        learning_rate (float, optional): Learning rate for training. Default is 0.01.
        batch_size (int, optional): Batch size for training. Default is 100.
        num_epochs (int, optional): Number of epochs for training. Default is 30.
        accumulator (int, optional): Gradient accumulation steps for training. Default is 1.

    Returns:
        None

    Description:
        This function performs multiple tasks based on the provided flags:
        1. Generates new data if `generate` is True.
        2. Loads existing data if `generate` is False and `training` or `create_visual` is True.
        3. Creates visualizations of the data if `create_visual` is True.
        4. Augments the data and prepares it for training if `training` is True.
        5. Trains a model with the data if `training` is True.
        6. Computes parameters from experimental data if `use` is True and plots the results if `plot_generate_compare` is True.
    """
    
    cameras = resolution_input_beam, window_input, window_out, resolution_training
    nlse_settings = n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length

    if generate or training:
        if generate:
            with cp.cuda.Device(device):
                E = data_creation(nlse_settings, cameras, device, saving_path)
        else:
            E = np.load(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}.npy')
        if create_visual:
            plot_generated_set(E, saving_path, nlse_settings)

        mean_standard = np.mean(E, axis=(0, 2, 3), keepdims=True)
        std_standard = np.std(E, axis=(0, 2, 3), keepdims=True)
        E = (E - mean_standard)/std_standard
        labels = generate_labels(n2, isat, alpha)

        if training:
            print("---- TRAINING ----")
            fieldset, model_settings, new_path = prepare_training(nlse_settings, labels, E, saving_path, 
                                                                learning_rate, batch_size, num_epochs, 
                                                                accumulator, device, mean_standard, std_standard)
            manage_training(fieldset, model_settings, nlse_settings,
                            new_path, resolution_training, labels)
    
    if not generate and not training and create_visual:
        E = np.load(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}.npy')
        plot_generated_set(E, saving_path, nlse_settings)


    if use:
        print("---- COMPUTING PARAMETERS ----\n")
        get_parameters(exp_image_path, saving_path, resolution_training, nlse_settings, device, cameras, plot_generate_compare, mean_standard, std_standard)