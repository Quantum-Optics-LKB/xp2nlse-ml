#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from engine.seed_settings import set_seed
from engine.generate import data_creation
from engine.model import Inception_ResNetv2
from skimage.restoration import unwrap_phase
from engine.treament import general_extrema, normalize_data
set_seed(10)

def get_parameters(
        exp_path: str,
        saving_path: str, 
        resolution_out: int, 
        nlse_settings: tuple, 
        device_number: int, 
        cameras: tuple, 
        plot_generate_compare: bool):
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = nlse_settings
    resolution_in, window_in, window_out, resolution_training = cameras
    
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)

    min_n2 = n2.min()
    max_isat = isat.max()
    max_alpha = alpha.max()

    device = torch.device(f"cuda:{device_number}")
    cnn = Inception_ResNetv2(in_channels=3)
    cnn.to(device)
    cnn.load_state_dict(torch.load(f'{saving_path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}/n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}.pth'))
    
    field = np.load(exp_path)
    if len(field.shape) == 2: 

        density = normalize_data(zoom(np.abs(field), 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase = np.angle(field)
        uphase = general_extrema(unwrap_phase(phase, rng=10))
        uphase = normalize_data(zoom(uphase, 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase = normalize_data(zoom(phase, 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        
        E = np.zeros((1, 3, 256, 256), dtype=np.float16)
        E[0, 0, :, :] = density
        E[0, 1, :, :] = phase
        E[0, 2, :, :] = uphase

    else:
        density = normalize_data(zoom(np.abs(field), 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase = np.angle(field)
        uphase = general_extrema(unwrap_phase(phase))
        uphase = normalize_data(zoom(uphase, 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase = normalize_data(zoom(phase, 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        
        E = np.zeros((field.shape[0], 3, 256, 256), dtype=np.float16)
        E[:, 0, :, :] = density
        E[:, 1, :, :] = phase
        E[:, 2, :, :] = uphase
    
    with torch.no_grad():
        images = torch.from_numpy(E).float().to(device)
        outputs_n2, outputs_isat, outputs_alpha = cnn(images)
    
    computed_n2 = outputs_n2[0,0].cpu().numpy()*min_n2
    computed_isat = outputs_isat[0,0].cpu().numpy()*max_isat
    computed_alpha = outputs_alpha[0,0].cpu().numpy()*max_alpha

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    if plot_generate_compare:
        plt.rcParams['font.family'] = 'DejaVu Serif'
        plt.rcParams['font.size'] = 10

        numbers = np.array([computed_n2]), in_power, alpha, np.array([computed_isat]), waist, nl_length, delta_z, length
        E = data_creation(numbers, cameras, device_number)
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle(f'Results: {puiss_str} = {in_power:.2e} {puiss_u},\n {n2_str} = {computed_n2} {n2_u}, {isat_str} = {computed_isat} {isat_u}, {alpha_str} = {computed_alpha} {alpha_u}')
        
        axes[0, 0].imshow(density, cmap='viridis')
        axes[0, 0].set_title(f'Experimental Density')

        axes[1, 0].imshow(phase, cmap='twilight_shifted')
        axes[1, 0].set_title(f'Experimental Phase')

        axes[2, 0].imshow(uphase, cmap='viridis')
        axes[2, 0].set_title(f'Experimental Unwrapped Phase')


        axes[0, 1].imshow(E[0, 0, :, :], cmap='viridis')
        axes[0, 1].set_title(f'Predicted Density')

        axes[1, 1].imshow(E[0, 1, :, :], cmap='twilight_shifted')
        axes[1, 1].set_title(f'Predicted Phase')

        axes[2, 1].imshow(E[0, 2, :, :], cmap='viridis')
        axes[2, 1].set_title(f'Predicted Unwrapped Phase')


        plt.tight_layout()
        plt.savefig(f"{saving_path}/prediction_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power}.png")
    
    return computed_n2, computed_isat, computed_alpha
