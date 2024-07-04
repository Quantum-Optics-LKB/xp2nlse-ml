#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from engine.seed_settings import set_seed
from engine.generate import data_creation
from skimage.restoration import unwrap_phase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from engine.treament import general_extrema, normalize_data
set_seed(10)

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 10

def scientific_formatter(x):
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${}\times 10^{{{}}}$".format(a, b)

def experiment(
        resolution_training: int,
        exp_image_path: str
        ):
    field = np.load(exp_image_path)
    density_experiment = normalize_data(zoom(np.abs(field), 
                    (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = np.angle(field)
    uphase_experiment = general_extrema(unwrap_phase(phase_experiment, rng=10))
    uphase_experiment = normalize_data(zoom(uphase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)

    return density_experiment, phase_experiment, uphase_experiment


def plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment, resolution_training, window_out,
                 n2, isat, alpha, input_power, saving_path):
    extent = [-window_out/2*1e3, window_out/2*1e3, -window_out/2*1e3, window_out/2*1e3]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    title = f"{n2_str} = {scientific_formatter(n2)}{n2_u},"
    title += f" {isat_str} = {scientific_formatter(isat)}{isat_u},"
    title += f" {puiss_str} = {scientific_formatter(input_power)}{puiss_u},"
    title+= f"{alpha_str} = {scientific_formatter(alpha)}{alpha_u}"

    fig, axs = plt.subplots(3, 2, figsize=(10, 15), layout="tight")
    fig.suptitle(title)
    ims = []
    ims.append(axs[0, 0].imshow(E[0, 0, :, :], cmap="viridis", extent=extent))
    ims.append(axs[0, 1].imshow(density_experiment, cmap="viridis", extent=extent))
    ims.append(axs[1, 0].imshow(E[0, 1, :, :], cmap="twilight_shifted", extent=extent))
    ims.append(axs[1, 1].imshow(phase_experiment, cmap="twilight_shifted", extent=extent))
    ims.append(axs[2, 0].imshow(E[0, 2, :, :], cmap="viridis", extent=extent))
    ims.append(axs[2, 1].imshow(uphase_experiment, cmap="viridis", extent=extent))
    dividers = []
    for ax in axs.flatten():
        dividers.append(make_axes_locatable(ax))
    caxes = []
    for divider in dividers:
        caxes.append(divider.append_axes("right", size="5%", pad=0.05))
    for im, cax in zip(ims, caxes):
        fig.colorbar(im, cax=cax)

    axs[0, 0].set_title("Density")
    axs[1, 0].set_title("Normalized phase")
    axs[2, 0].set_title("Normalized unwrapped phase")
    axs[0, 1].set_title("Experimental density")
    axs[1, 1].set_title("Experimental normalized phase")
    axs[2, 1].set_title("Experimental unwrapped phase")
    for ax in axs.flatten():
        ax.set_xlabel(r"x (mm)")
        ax.set_ylabel(r"y (mm)")



    # axs[0, 0].tick_params(axis='both', which='major', pad=15)
    # axs[1, 0].tick_params(axis='both', which='major', pad=15)
    # axs[2, 0].tick_params(axis='both', which='major', pad=15)
    # axs[0, 1].tick_params(axis='both', which='major', pad=15)
    # axs[1, 1].tick_params(axis='both', which='major', pad=15)
    # axs[2, 1].tick_params(axis='both', which='major', pad=15)
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
                 resolution_training, window_out, n2, isat, alpha, input_power, saving_path)  

