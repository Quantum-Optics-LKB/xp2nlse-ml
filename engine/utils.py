#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import torch
import random
import numpy as np
import kornia.augmentation as K
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def scientific_formatter(x):
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${}\times 10^{{{}}}$".format(a, b)

def normalize_data(
        data: np.ndarray,
        ) -> np.ndarray: 
    data -= np.min(data, axis=(-2, -1), keepdims=True)
    data /= np.max(data, axis=(-2, -1), keepdims=True)
    return data

def general_extrema(E):
    if E[E.shape[-2]//2, E.shape[-1]//2] > E[0, 0]:
        E -= np.max(E)
    elif E[E.shape[-2]//2, E.shape[-1]//2] < 0:
        E -= np.min(E)
    E = np.abs(E)
    return E

def elastic_saltpepper() -> torch.nn.Sequential:
    
    elastic_sigma = (random.randrange(35, 42, 2), random.randrange(35, 42, 2))
    elastic_alpha = (1, 1)
    salt_pepper = random.uniform(0.01, .11)
    return torch.nn.Sequential(
        K.RandomElasticTransform(kernel_size=51, sigma=elastic_sigma, alpha=elastic_alpha ,p=.5),
        K.RandomSaltAndPepperNoise(amount=salt_pepper,salt_vs_pepper=(.5, .5), p=.2),
    )

def experiment_noise(
        beam: np.ndarray, 
        poisson_noise_lam: float,
        normal_noise_sigma: float
          )-> np.ndarray:
        
    poisson_noise = np.random.poisson(lam=poisson_noise_lam, size=(beam.shape))*poisson_noise_lam*0.75
    normal_noise = np.random.normal(0, normal_noise_sigma, (beam.shape))

    total_noise = normal_noise + poisson_noise
    noisy_beam = np.real(beam) + total_noise + 1j * np.imag(beam)

    noisy_beam = noisy_beam.astype(np.complex64)
    return noisy_beam

def line_noise(
        image: np.ndarray,
        num_lines: int, 
        amplitude: float, 
        angle: float
        ) -> np.ndarray:
    height, width = image.shape
    angle_rad = np.radians(angle)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    X_rotated = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    diagonal_length = np.sqrt(width**2 + height**2)
    wave_frequency = (num_lines * 2 * np.pi) / diagonal_length
    lines_pattern =  amplitude*np.sin(X_rotated * wave_frequency)
    noisy_image = image.copy() + lines_pattern
    
    return noisy_image

def set_seed(
        seed: int
        ) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def data_split(
        E: np.ndarray, 
        n2_labels: np.ndarray,
        isat_labels: np.ndarray,
        alpha_labels: np.ndarray,
        train_ratio: float = 0.8, 
        validation_ratio: float = 0.1, 
        test_ratio: float = 0.1
        ) -> tuple:
    assert train_ratio + validation_ratio + test_ratio == 1
    
    indices = np.arange(E.shape[0])
    train_index = int(len(indices) * train_ratio)
    validation_index = int(len(indices) * (train_ratio + validation_ratio))

    training_indices = indices[:train_index]
    validation_indices = indices[train_index:validation_index]
    test_indices = indices[validation_index:]

    train = E[training_indices,:,:,:]
    validation = E[validation_indices,:,:,:]
    test = E[test_indices,:,:,:]

    train_n2 = n2_labels[training_indices]
    validation_n2 = n2_labels[validation_indices]
    test_n2 = n2_labels[test_indices]

    train_isat = isat_labels[training_indices]
    validation_isat = isat_labels[validation_indices]
    test_isat = isat_labels[test_indices]

    train_alpha = alpha_labels[training_indices]
    validation_alpha = alpha_labels[validation_indices]
    test_alpha = alpha_labels[test_indices]

    return (train, train_n2, train_isat, train_alpha), (validation, validation_n2, validation_isat, validation_alpha), (test, test_n2, test_isat, test_alpha)

def plot_loss(
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    path: str, 
    resolution: int, 
    number_of_n2: int, 
    number_of_isat: int,
    number_of_alpha: int,
    ) -> None:
    
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 12

    ax.plot(np.log(y_train), label="Training Loss", marker='^', linestyle='-', color='blue', mfc='lightblue', mec='indigo', markersize=10, mew=2)
    ax.plot(np.log(y_val), label="Validation Loss", marker='^', linestyle='-', color='orange', mfc='#FFEDA0', mec='darkorange', markersize=10, mew=2)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Log Loss")
    fig.suptitle("Training and Validation Log Losses")
    ax.legend()
    fig.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}.png")
    plt.close()

def plot_generated_set(
        data: np.ndarray, 
        saving_path: str, 
        nlse_settings: tuple
        ) -> None:
    
    n2, input_power, alpha, isat, _, _, _, _ = nlse_settings
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)

    field = data.copy().reshape(number_of_n2, number_of_isat, number_of_alpha, 3, data.shape[-2], data.shape[-2])
    density_channels = field[:,  :, :, 0, :, :]
    phase_channels = field[:, :, :, 1, :, :]
    uphase_channels = field[:, :, :, 2, :, :]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 50

    for alpha_index, alpha_value in enumerate(alpha):
        
        fig_density, axes_density = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
        fig_density.suptitle(f'Density Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_density if number_of_n2 == 1 and number_of_isat == 1 else (axes_density[n2_index, isat_index] if number_of_n2 > 1 and number_of_n2 > 1 else (axes_density[n2_index] if number_of_n2 > 1 else axes_density[isat_index]))
                ax.imshow(density_channels[n2_index, isat_index, alpha_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/density_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_density) 

        fig_phase, axes_phase = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
        fig_phase.suptitle(f'Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_phase if number_of_n2 == 1 and number_of_isat == 1 else (axes_phase[n2_index, isat_index] if number_of_n2 > 1 and number_of_isat > 1 else (axes_phase[n2_index] if number_of_n2 > 1 else axes_phase[isat_index]))
                ax.imshow(phase_channels[n2_index, isat_index, alpha_index, :, :], cmap='twilight_shifted')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/phase_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_phase)

        fig_uphase, axes_uphase = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
        fig_uphase.suptitle(f'Unwrapped Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_uphase if number_of_n2 == 1 and number_of_isat == 1 else (axes_uphase[n2_index, isat_index] if number_of_n2 > 1 and number_of_isat > 1 else (axes_uphase[n2_index] if number_of_n2 > 1 else axes_uphase[isat_index]))
                ax.imshow(uphase_channels[n2_index, isat_index, alpha_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/unwrapped_phase_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_phase)

def plot_results(E, density_experiment, phase_experiment, uphase_experiment, 
                 numbers, cameras, number_of_n2, number_of_isat, number_of_alpha, saving_path):

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    
    computed_n2, input_power, computed_alpha, computed_isat, _, _, _, _ = numbers 
    _, _, window_out, resolution_training = cameras
    extent = [-window_out/2*1e3, window_out/2*1e3, -window_out/2*1e3, window_out/2*1e3]

    computed_n2 = computed_n2[0]
    computed_alpha = computed_alpha[0]
    computed_isat = computed_isat[0]

    title = f"Results: {n2_str} = {scientific_formatter(computed_n2)}{n2_u},"
    title += f" {isat_str} = {scientific_formatter(computed_isat)}{isat_u},"
    title += f" {puiss_str} = {scientific_formatter(input_power)}{puiss_u},"
    title+= f"{alpha_str} = {scientific_formatter(computed_alpha)}{alpha_u}"

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 10

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
    
    plt.savefig(f"{saving_path}/prediction_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power}.png")

def plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment, window_out,
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

    plt.savefig(f"{saving_path}/sandbox.png")