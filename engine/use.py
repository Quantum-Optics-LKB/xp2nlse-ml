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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    else:
        density_experiment = normalize_data(zoom(np.abs(field), 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase_experiment = np.angle(field)
        uphase_experiment = general_extrema(unwrap_phase(phase_experiment))
        uphase_experiment = normalize_data(zoom(uphase_experiment, 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        phase_experiment = normalize_data(zoom(phase_experiment, 
                    (1, resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
        
        E = np.zeros((field.shape[0], 3, 256, 256), dtype=np.float16)
        E[:, 0, :, :] = density_experiment
        E[:, 1, :, :] = phase_experiment
        E[:, 2, :, :] = uphase_experiment
    
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

        numbers = np.array([computed_n2]), in_power, np.array([computed_alpha]), np.array([computed_isat]), waist, nl_length, delta_z, length
        E = data_creation(numbers, cameras, device_number)
        plot_results(E, density_experiment, phase_experiment, uphase_experiment,numbers, cameras, number_of_n2, number_of_isat, number_of_alpha, saving_path)
    
    return computed_n2, computed_isat, computed_alpha

def plot_results(E, density_experiment, phase_experiment, uphase_experiment, numbers, cameras, number_of_n2, number_of_isat, number_of_alpha, saving_path):

    computed_n2, in_power, computed_alpha, computed_isat, waist, nl_length, delta_z, length = numbers 
    n2 = computed_n2[0]
    alpha = computed_alpha[0]
    isat = computed_isat[0]
    resolution_in, window_in, window_out, resolution_training = cameras
    
    output_shape = (resolution_training, resolution_training)
    label_x = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)
    label_y = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 10
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle(f'Results: {puiss_str} = {in_power:.2e} {puiss_u},\n {n2_str} = {computed_n2} {n2_u}, {isat_str} = {computed_isat} {isat_u}, {alpha_str} = {computed_alpha} {alpha_u}')

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    im1 = axs[0, 0].imshow(E[0, 0, :, :], cmap="viridis")
    im2 = axs[1, 0].imshow(E[0, 1, :, :], cmap="twilight_shifted")
    im3 = axs[2, 0].imshow(E[0, 2, :, :], cmap="viridis")
    im5 = axs[0, 1].imshow(density_experiment, cmap="viridis")
    im6 = axs[1, 1].imshow(phase_experiment, cmap="twilight_shifted")
    im7 = axs[2, 1].imshow(uphase_experiment, cmap="viridis")


    divider1 = make_axes_locatable(axs[0, 0])
    divider2 = make_axes_locatable(axs[1, 0])
    divider3 = make_axes_locatable(axs[2, 0])
    divider5 = make_axes_locatable(axs[0, 1])
    divider6 = make_axes_locatable(axs[1, 1])
    divider7 = make_axes_locatable(axs[2, 1])

    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    cax6 = divider6.append_axes("right", size="5%", pad=0.05)
    cax7 = divider7.append_axes("right", size="5%", pad=0.05)


    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar5 = plt.colorbar(im5, cax=cax5)
    cbar6 = plt.colorbar(im6, cax=cax6)
    cbar7 = plt.colorbar(im7, cax=cax7)

    axs[0, 0].set_title("Density")
    axs[0, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[0, 0].set_xticklabels(label_x)
    axs[0, 0].set_xlabel(r"x (mm)")
    axs[0, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[0, 0].set_yticklabels(label_y)
    axs[0, 0].set_ylabel(r"y (mm)")

    axs[1, 0].set_title("Phase")
    axs[1, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[1, 0].set_xticklabels(label_x)
    axs[1, 0].set_xlabel(r"x (mm)")
    axs[1, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[1, 0].set_yticklabels(label_y)
    axs[1, 0].set_ylabel(r"y (mm)")

    axs[2, 0].set_title("Phase Unwrap")
    axs[2, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[2, 0].set_xticklabels(label_x)
    axs[2, 0].set_xlabel(r"x (mm)")
    axs[2, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[2, 0].set_yticklabels(label_y)
    axs[2, 0].set_ylabel(r"y (mm)")

    axs[0, 1].set_title("Experimental Density")
    axs[0, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[0, 1].set_xticklabels(label_x)
    axs[0, 1].set_xlabel(r"x (mm)")
    axs[0, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[0, 1].set_yticklabels(label_y)
    axs[0, 1].set_ylabel(r"y (mm)")

    axs[1, 1].set_title("Experimental Phase")
    axs[1, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[1, 1].set_xticklabels(label_x)
    axs[1, 1].set_xlabel(r"x (mm)")
    axs[1, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[1, 1].set_yticklabels(label_y)
    axs[1, 1].set_ylabel(r"y (mm)")

    axs[2, 1].set_title("Experimental Phase Unwrap")
    axs[2, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
    axs[2, 1].set_xticklabels(label_x)
    axs[2, 1].set_xlabel(r"x (mm)")
    axs[2, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
    axs[2, 1].set_yticklabels(label_y)
    axs[2, 1].set_ylabel(r"y (mm)")



    axs[0, 0].tick_params(axis='both', which='major', pad=15)
    axs[1, 0].tick_params(axis='both', which='major', pad=15)
    axs[2, 0].tick_params(axis='both', which='major', pad=15)
    axs[0, 1].tick_params(axis='both', which='major', pad=15)
    axs[1, 1].tick_params(axis='both', which='major', pad=15)
    axs[2, 1].tick_params(axis='both', which='major', pad=15)

    plt.tight_layout()
    plt.savefig(f"{saving_path}/prediction_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power}.png")