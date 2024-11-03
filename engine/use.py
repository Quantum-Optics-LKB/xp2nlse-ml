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
    model = network()
    model.to(device)

    directory_path = f'{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}'
    standards_lines = open(f"{directory_path}/standardize.txt", "r").readlines()
    
    # dataset.mean_standard = np.asarray(standards_lines[0].split("\n")[0].split(";"), dtype=np.float64)[np.newaxis,:,np.newaxis, np.newaxis]
    # dataset.std_standard = np.asarray(standards_lines[1].split("\n")[0].split(";"), dtype=np.float64)[np.newaxis,:,np.newaxis, np.newaxis]
    dataset.n2_max_standard = float(standards_lines[0])
    dataset.n2_min_standard = float(standards_lines[1])
    dataset.isat_max_standard = float(standards_lines[2])
    dataset.isat_min_standard = float(standards_lines[3])
    dataset.alpha_max_standard = float(standards_lines[4])
    dataset.alpha_min_standard = float(standards_lines[5])
    directory_path += f'/n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    model.load_state_dict(torch.load(directory_path))

    experiment_field = np.load(exp_path)
    experiment_field = zoom(experiment_field, (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1]), order=5)
    density_experiment = np.abs(experiment_field)
    phase_experiment = (np.angle(experiment_field) + np.pi)/ (2*np.pi)
    
    density_experiment -= np.min(density_experiment, axis=(-1, -2), keepdims=True)
    density_experiment /= np.max(density_experiment, axis=(-1, -2), keepdims=True)
    
    experiment_field = np.zeros((2, 2, dataset.resolution_training, dataset.resolution_training), dtype=np.float64)
    experiment_field[0, 0, :, :] = density_experiment
    experiment_field[0, 1, :, :] = phase_experiment
    
    with torch.no_grad():
        images = torch.from_numpy(experiment_field).to(device = device, dtype=torch.float32)
        outputs, cov_outputs = model(images)
    
    computed_n2 = outputs[0, 0].cpu().numpy()*(dataset.n2_max_standard - dataset.n2_min_standard) + dataset.n2_min_standard
    computed_isat = outputs[0, 1].cpu().numpy()*(dataset.isat_max_standard - dataset.isat_min_standard) + dataset.isat_min_standard
    computed_alpha = outputs[0, 2].cpu().numpy()*(dataset.alpha_max_standard - dataset.alpha_min_standard) + dataset.alpha_min_standard

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    experiment_field = np.load(exp_path)
    experiment_field = zoom(experiment_field, (dataset.resolution_training/experiment_field.shape[-2], dataset.resolution_training/experiment_field.shape[-1]), order=5)
    density_experiment = np.abs(experiment_field)
    phase_experiment = np.angle(experiment_field)

    if plot_generate_compare:
        dataset.n2_values = np.array([computed_n2])
        dataset.alpha_values =np.array([computed_alpha])
        dataset.isat_values =np.array([computed_isat])
        temp_saving_path = dataset.saving_path
        dataset.saving_path = ""
        simulation(dataset)
        dataset.saving_path = temp_saving_path 
        plot_results(dataset, density_experiment, phase_experiment)
    
    return computed_n2, computed_isat, computed_alpha