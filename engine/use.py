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

    device = torch.device(dataset.device_number)
    model = network().double()
    model.to(device)

    directory_path = f'{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}/'
    directory_path += f'n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    model.load_state_dict(torch.load(directory_path))
    
    experiment_field = np.load(exp_path)

    density_experiment = zoom(np.abs(experiment_field), (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1])).astype(np.float64)
    phase_experiment = zoom(np.angle(experiment_field), (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1])).astype(np.float64)
    
    experiment_field = np.zeros((1, 2, dataset.resolution_training, dataset.resolution_training), dtype=np.float64)
    experiment_field[0, 0, :, :] = density_experiment
    experiment_field[0, 1, :, :] = phase_experiment

    experiment_field -= dataset.mean_standard 
    experiment_field /= dataset.std_standard
    
    with torch.no_grad():
        images = torch.from_numpy(experiment_field).float().to(device)
        outputs = model(images)
    
    computed_n2 = outputs[0,0].cpu().numpy()*(dataset.n2_max_standard - dataset.n2_min_standard) + dataset.n2_min_standard
    computed_isat = outputs[0,1].cpu().numpy()*(dataset.isat_max_standard - dataset.isat_min_standard) + dataset.isat_min_standard
    computed_alpha = outputs[0,2].cpu().numpy()*(dataset.alpha_max_standard - dataset.alpha_min_standard) + dataset.alpha_min_standard

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    if plot_generate_compare:
        dataset.n2_values = np.array([computed_n2])
        dataset.alpha_values =np.array([computed_alpha])
        dataset.isat_values =np.array([computed_isat])
        simulation(dataset)
        plot_results(dataset, density_experiment, phase_experiment)
    
    return computed_n2, computed_isat, computed_alpha