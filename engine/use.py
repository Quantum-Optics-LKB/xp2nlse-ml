#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from scipy.ndimage import zoom
from engine.model import network
from engine.generate import simulation
from engine.engine_dataset import EngineDataset
from engine.utils import plot_results, set_seed
set_seed(10)

def get_parameters(
        exp_path: str,
        dataset: EngineDataset,
        plot_generate_compare: bool
        ) -> tuple:
    """
    Computes the physical parameters (n2, Isat, alpha) using experimental data and a trained model.

    Parameters:
    -----------
    exp_path : str
        Path to the experimental field data file (numpy array).
    dataset : EngineDataset
        The dataset object containing training and simulation parameters.
    plot_generate_compare : bool
        Flag to indicate whether to simulate and compare the results with experimental data.

    Returns:
    --------
    tuple
        A tuple containing the computed parameters (n2, Isat, alpha).
    """

    # Load the computational device (e.g., GPU or CPU)
    device = torch.device(dataset.device_number)

    # Initialize the model and move it to the device
    model = network()
    model.to(device)

    # Load standardization values for parameters
    directory_path = f'{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}'
    standards_lines = open(f"{directory_path}/standardize.txt", "r").readlines()
    
    dataset.n2_max = float(standards_lines[0])
    dataset.n2_min = float(standards_lines[1])
    dataset.isat_max = float(standards_lines[2])
    dataset.isat_min = float(standards_lines[3])
    dataset.alpha_max = float(standards_lines[4])
    dataset.alpha_min = float(standards_lines[5])

    # Load the trained model's weights
    directory_path += f'/n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    model.load_state_dict(torch.load(directory_path, weights_only=True))

    # Load and preprocess experimental field data
    experiment_field = np.load(exp_path)
    experiment_field = zoom(experiment_field, (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1]), order=5)
    
    # Compute intensity (density) and phase of the experimental field
    density_experiment = np.abs(experiment_field)
    phase_experiment = (np.angle(experiment_field) + np.pi)/ (2*np.pi)
    
    # Normalize the intensity
    density_experiment -= np.min(density_experiment, axis=(-1, -2), keepdims=True)
    density_experiment /= np.max(density_experiment, axis=(-1, -2), keepdims=True)
    
    # Prepare the experimental field tensor for the model
    experiment_field = np.zeros((2, 2, dataset.resolution_training, dataset.resolution_training), dtype=np.float64)
    experiment_field[0, 0, :, :] = density_experiment
    experiment_field[0, 1, :, :] = phase_experiment

    # Run the model to predict the physical parameters
    with torch.no_grad():
        images = torch.from_numpy(experiment_field).to(device = device, dtype=torch.float32)
        outputs, cov_outputs = model(images)
    
    # De-normalize the outputs to get the physical parameters
    computed_n2 = outputs[0, 0].cpu().numpy()*(dataset.n2_max - dataset.n2_min) + dataset.n2_min
    computed_isat = outputs[0, 1].cpu().numpy()*(dataset.isat_max - dataset.isat_min) + dataset.isat_min
    computed_alpha = outputs[0, 2].cpu().numpy()*(dataset.alpha_max - dataset.alpha_min) + dataset.alpha_min

    # Print the computed parameters
    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    experiment_field = np.load(exp_path)
    experiment_field = zoom(experiment_field, (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1]), order=5)
    density_experiment = np.abs(experiment_field)
    phase_experiment = np.angle(experiment_field)

    # Optionally, simulate and compare with experimental data
    if plot_generate_compare:
        # Update dataset parameters with computed values
        dataset.n2_values = np.array([computed_n2])
        dataset.alpha_values =np.array([computed_alpha])
        dataset.isat_values =np.array([computed_isat])

        # Temporarily disable saving for simulation
        temp_saving_path = dataset.saving_path
        dataset.saving_path = ""
        simulation(dataset)

        # Plot the results
        dataset.saving_path = temp_saving_path 
        plot_results(dataset, density_experiment, phase_experiment)
    
    # Return the computed parameters
    return computed_n2, computed_isat, computed_alpha