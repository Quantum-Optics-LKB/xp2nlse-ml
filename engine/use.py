#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from scipy.ndimage import zoom
from engine.utils import set_seed
from matplotlib import pyplot as plt
from engine.generate import data_creation
from engine.model import Inception_ResNetv2
from skimage.restoration import unwrap_phase
from engine.utils import scientific_formatter
from engine.utils import general_extrema, normalize_data
set_seed(10)

def get_parameters(
        exp_path: str,
        saving_path: str, 
        resolution_out: int, 
        nlse_settings: tuple, 
        device_number: int, 
        cameras: tuple, 
        plot_generate_compare: bool
        ) -> tuple:
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = nlse_settings
    _, _, _, resolution_training = cameras
    
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)

    min_n2 = n2.min()
    max_isat = isat.max()
    max_alpha = alpha.max()

    device = torch.device(f"cuda:{device_number}")
    cnn = Inception_ResNetv2(in_channels=3)
    cnn.to(device)

    directory_path = f'{saving_path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}/'
    directory_path += f'n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}.pth'
    cnn.load_state_dict(torch.load(directory_path))
    
    field = np.load(exp_path)

    density_experiment = normalize_data(zoom(np.abs(field), 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = np.angle(field)
    uphase_experiment = general_extrema(unwrap_phase(phase_experiment, rng=10))
    uphase_experiment = normalize_data(zoom(uphase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    
    E = np.zeros((1, 3, 256, 256), dtype=np.float16)
    E[0, 0, :, :] = density_experiment
    E[0, 1, :, :] = phase_experiment
    E[0, 2, :, :] = uphase_experiment
    
    with torch.no_grad():
        images = torch.from_numpy(E).float().to(device)
        outputs_n2, outputs_isat, outputs_alpha = cnn(images)
    
    computed_n2 = outputs_n2[0,0].cpu().numpy()*min_n2
    computed_isat = outputs_isat[0,0].cpu().numpy()*max_isat
    computed_alpha = outputs_alpha[0,0].cpu().numpy()*max_alpha

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    if plot_generate_compare:

        numbers = np.array([computed_n2]), in_power, np.array([computed_alpha]), np.array([computed_isat]), waist, nl_length, delta_z, length
        E = data_creation(numbers, cameras, device_number)
        plot_results(E, density_experiment, phase_experiment, uphase_experiment,numbers, cameras, number_of_n2, number_of_isat, number_of_alpha, saving_path)
    
    return computed_n2, computed_isat, computed_alpha

