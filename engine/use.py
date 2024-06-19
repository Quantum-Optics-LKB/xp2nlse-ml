#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.ndimage import zoom
from engine.generate_augment import data_creation
from engine.model import Inception_ResNetv2
from skimage.restoration import unwrap_phase

def get_parameters(exp_path, saving_path, resolution_out, nlse_settings, device_number, cameras, plot_generate_compare):
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = nlse_settings
    resolution_in, window_in, window_out, resolution_training = cameras
    
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    min_n2 = n2.min()
    max_isat = isat.max()

    device = torch.device(f"cuda:{device_number}")
    
    field = np.load(exp_path)

    density = zoom(np.abs(field), 
                   (resolution_training/field.shape[-2], resolution_training/field.shape[-1])).astype(np.float16)
    phase = np.angle(field)
    uphase = zoom(unwrap_phase(phase), 
                  (resolution_training/field.shape[-2], resolution_training/field.shape[-1])).astype(np.float16)
    phase = zoom(phase, 
                 (resolution_training/field.shape[-2], resolution_training/field.shape[-1])).astype(np.float16)
    
    E = np.zeros((1, 3, 256, 256), dtype=np.float16)
    E[0, 0, :, :] = density
    E[0, 1, :, :] = phase
    E[0, 2, :, :] = uphase

    cnn = Inception_ResNetv2(in_channels=E.shape[1])
    cnn.to(device)
    cnn.load_state_dict(torch.load(f'{saving_path}/training_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}/n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}.pth'))
    
    with torch.no_grad():
        images = torch.from_numpy(E).float().to(device)
        outputs_n2, outputs_isat = cnn(images)
    
    computed_n2 = outputs_n2[0,0].cpu().numpy()*min_n2
    computed_isat = outputs_isat[0,0].cpu().numpy()*max_isat

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")

    if plot_generate_compare:
        plt.rcParams['font.family'] = 'DejaVu Serif'
        plt.rcParams['font.size'] = 10
        n2_str = r"$n_2$"
        n2_u = r"$m^2$/$W$"
        isat_str = r"$I_{sat}$"
        isat_u = r"$W$/$m^2$"
        puiss_str = r"$p$"
        puiss_u = r"$W$"

        numbers = np.array([computed_n2]), in_power, alpha, np.array([computed_isat]), waist, nl_length, delta_z, length
        E = data_creation(numbers, cameras)
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle(f'Results:\n {puiss_str} = {in_power:.2e} {puiss_u}, {n2_str} = {computed_n2} {n2_u}, {isat_str} = {computed_isat} {isat_u}')
        
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
        plt.savefig(f"{saving_path}/prediction_n2{number_of_n2}_isat{number_of_isat}_power{in_power}.png")
