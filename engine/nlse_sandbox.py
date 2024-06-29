#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from engine.generate import data_creation
from skimage.restoration import unwrap_phase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from engine.treament import general_extrema, normalize_data


def experiment(
        resolution_training: int,
        exp_image_path: str
        ):
    field = np.load(exp_image_path)
    density_experiment = normalize_data(zoom(np.abs(field), 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = np.angle(field)
    uphase_experiment = general_extrema(unwrap_phase(phase_experiment))
    uphase_experiment = normalize_data(zoom(uphase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)

    return density_experiment, phase_experiment, uphase_experiment


def plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment, resolution_training, window_out,
                 n2, isat, alpha, input_power, non_locality_length, saving_path):
    
    output_shape = (resolution_training, resolution_training)
    label_x = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)
    label_y = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 10
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    nl_str = r"$nl$"
    nl_u = r"$m$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    title = f"{n2_str} = {n2:.2e}{n2_u}, {isat_str} = {isat:.2e}{isat_u}, {puiss_str} = {input_power:.2e}{puiss_u}, {alpha_str} = {alpha:.2e}{alpha_u}, {nl_str} = {non_locality_length:.2e}{nl_u}"
    fig.suptitle(title)

    fig.suptitle(title)

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
    plt.savefig(f"{saving_path}/sandbox.png")

def sandbox(device: int, 
            resolution_input_beam: int,
            window_input: float,
            window_out: float,
            resolution_training: int,
            n2: np.ndarray,
            input_power: float,
            alpha: float,
            isat: float,
            waist_input_beam: float,
            non_locality_length: float,
            delta_z: float,
            cell_length: float, 
            exp_image_path: str,
            saving_path: str,
            ) -> None:

    cameras = resolution_input_beam, window_input, window_out, resolution_training
    nlse_settings = np.array([n2]), input_power, np.array([alpha]), np.array([isat]), waist_input_beam, non_locality_length, delta_z, cell_length
    
    with cp.cuda.Device(device):
        E = data_creation(nlse_settings, cameras, device)

    density_experiment, phase_experiment, uphase_experiment = experiment(resolution_training, exp_image_path)

    plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment,
                 resolution_training, window_out, n2, isat, alpha, input_power, 
                 non_locality_length, saving_path)  