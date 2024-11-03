#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import torch
import torch.nn as nn
import random
import numpy as np
import kornia.augmentation as K
from matplotlib import pyplot as plt
from tqdm import tqdm
from engine.engine_dataset import EngineDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CircularFilterAugmentation(nn.Module):
    def __init__(self, radius_range: tuple, p: float = 0.5):
        super(CircularFilterAugmentation, self).__init__()
        self.radius_range = radius_range
        self.p = p

    def forward(self, images):
        B, H, W = images.shape
        device = images.device

        # Randomly decide for each image if augmentation should be applied
        apply_mask = torch.rand(B, device=device) < self.p

        # Generate radii for each image where mask will be applied
        random_radii = (torch.rand(B, device=device) * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]) * min(H, W)
        random_radii = random_radii.int()//2

        # Prepare coordinate grids outside loop for efficiency
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        center_y, center_x = H // 2, W // 2
        distance_from_center = (x - center_x) ** 2 + (y - center_y) ** 2

        # Apply circular masks for each image in batch
        mask_batch = distance_from_center[None, :, :] <= random_radii[:, None, None] ** 2

        # Apply mask only for images where apply_mask is True
        images_augmented = torch.where(
            apply_mask[:, None, None],  # Check which images need augmentation
            images * mask_batch,  # Apply mask
            images  # Leave image unmodified if not applying mask
        )

        return images_augmented


class RandomPhaseShift(nn.Module):
    def __init__(self, shift_range=(-np.pi, np.pi), p=0.5):
        super(RandomPhaseShift, self).__init__()
        self.shift_range = shift_range
        self.p = p

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # Apply phase shift with a probability
        apply_shift = torch.rand(B, device=device) < self.p

        # Scale images to range [-pi, pi]
        x = x * 2 * np.pi - np.pi

        # Generate random shifts for each image in the batch
        phase_shifts = torch.empty(B, device=device).uniform_(*self.shift_range)
        
        # Apply unique phase shifts to each image
        shifted_x = x + phase_shifts[:, None, None]

        # Wrap phase values to be in the range [-pi, pi]
        shifted_x = torch.fmod(shifted_x + np.pi, 2 * np.pi) - np.pi

        # Rescale back to the range [0, 1] and apply conditionally
        x = (shifted_x + np.pi) / (2 * np.pi)
        return torch.where(apply_shift[:, None, None], x, x)


def sigmospace(array, a):
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))
    normalized_array = normalized_array**(a) / (normalized_array**(a) + np.abs(1-normalized_array)**a)
    
    return normalized_array * (np.max(array) - np.min(array)) + np.min(array)

def scientific_formatter(
        x: float
        ) -> str:
    """
    Format a number in scientific notation for LaTeX display.

    Parameters:
    - x (float): Number to be formatted.

    Returns:
    - str: Formatted string in LaTeX format with scientific notation.
    """
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${}\times 10^{{{}}}$".format(a, b)

def standardize_data(
        data: np.ndarray,
        ) -> np.ndarray: 
    """
    Normalize data to the range [0, 1] across the last two dimensions.

    Parameters:
    - data (np.ndarray): Input data array.

    Returns:
    - np.ndarray: Normalized data array.
    """
    data -= np.mean(data,axis=(0, -2, -1), keepdims=True) 
    data /= np.std(data, axis=(0, -2, -1), keepdims=True)
    return data

def augmentation_density(rotation_degrees) -> torch.nn.Sequential:
    """
    Create a sequential transformation pipeline for elastic and salt-pepper noise.

    Returns:
    - torch.nn.Sequential: Sequential transformation pipeline.
    """    
    return torch.nn.Sequential(
        K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=.75, keepdim=True),
    )
def augmentation_phase(rotation_degrees) -> torch.nn.Sequential:
    """
    Create a sequential transformation pipeline for elastic and salt-pepper noise.

    Returns:
    - torch.nn.Sequential: Sequential transformation pipeline.
    """
    return torch.nn.Sequential(
        RandomPhaseShift(shift_range=(0, np.pi/2), p=.5),
        CircularFilterAugmentation(radius_range=(.15, .75), p=.75),
        K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=.75, keepdim=True),
    )

def shuffle_dataset(
        dataset: EngineDataset
        ) -> None:
    indices = np.arange(len(dataset.alpha_labels))
    np.random.shuffle(indices)

    dataset.field = dataset.field[indices, :, :, :]
    dataset.n2_labels = dataset.n2_labels[indices]
    dataset.isat_labels = dataset.isat_labels[indices]
    dataset.alpha_labels = dataset.alpha_labels[indices]

def experiment_noise(
        beam: np.ndarray, 
        poisson_noise_lam: float,
        normal_noise_sigma: float
          )-> np.ndarray:
    """
    Add Poisson and normal noise to an input beam array.

    Parameters:
    - beam (np.ndarray): Input beam array.
    - poisson_noise_lam (float): Lambda parameter for Poisson noise.
    - normal_noise_sigma (float): Standard deviation for normal noise.

    Returns:
    - np.ndarray: Noisy beam array.
    """
        
    poisson_noise = np.random.poisson(lam=poisson_noise_lam, size=(beam.shape))*poisson_noise_lam*0.75
    normal_noise = np.random.normal(0, normal_noise_sigma, (beam.shape))

    total_noise = normal_noise + poisson_noise
    noisy_beam = np.real(beam) + total_noise + 1j * np.imag(beam)

    noisy_beam = noisy_beam.astype(np.complex64)
    return noisy_beam

def set_seed(
        seed: int
        ) -> None:
    """
    Set random seed for reproducibility in NumPy, Torch, and CUDA.

    Parameters:
    - seed (int): Seed value.

    Returns:
    - None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def data_split(
        indices: np.ndarray,
        train_ratio: float = 0.8, 
        validation_ratio: float = 0.1, 
        test_ratio: float = 0.1
        ) -> tuple:
    
    """
    Split data and labels into training, validation, and test sets.

    Parameters:
    - E (np.ndarray): Input data array.
    - n2_labels (np.ndarray): Labels for n2 parameter.
    - isat_labels (np.ndarray): Labels for isat parameter.
    - alpha_labels (np.ndarray): Labels for alpha parameter.
    - train_ratio (float): Ratio of training data (default: 0.8).
    - validation_ratio (float): Ratio of validation data (default: 0.1).
    - test_ratio (float): Ratio of test data (default: 0.1).

    Returns:
    - tuple: Tuple of training, validation, and test data and labels.
    """
    assert train_ratio + validation_ratio + test_ratio == 1
    
    train_index = int(len(indices) * train_ratio)
    validation_index = int(len(indices) * (train_ratio + validation_ratio))

    return train_index, validation_index

def plot_loss(
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    path: str, 
    resolution: int, 
    number_of_n2: int, 
    number_of_isat: int,
    number_of_alpha: int,
    ) -> None:
    """
    Plot training and validation loss curves and save the figure.

    Parameters:
    - y_train (np.ndarray): Training loss data.
    - y_val (np.ndarray): Validation loss data.
    - path (str): Path to save the plot.
    - resolution (int): Resolution of the plot.
    - number_of_n2 (int): Number of n2 parameters.
    - number_of_isat (int): Number of isat parameters.
    - number_of_alpha (int): Number of alpha parameters.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 12

    ax.plot(y_train, label="Training Loss", marker='^', linestyle='-', color='blue', mfc='lightblue', mec='indigo', markersize=10, mew=2)
    ax.plot(y_val, label="Validation Loss", marker='^', linestyle='-', color='orange', mfc='#FFEDA0', mec='darkorange', markersize=10, mew=2)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.suptitle("Training and Validation Losses")
    ax.legend()
    fig.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}.png")
    plt.close()

def plot_generated_set(
        dataset: EngineDataset
        ) -> None:
    """
    Plot and save generated sets of density, phase, and unwrapped phase channels.

    Parameters:
    - data (np.ndarray): Generated data array.
    - saving_path (str): Path to save the plots.
    - nlse_settings (tuple): Settings tuple containing n2, input_power, alpha, isat, etc.

    Returns:
    - None
    """

    field = dataset.field.copy().reshape(dataset.number_of_alpha, dataset.number_of_n2, dataset.number_of_isat, 2, dataset.field.shape[-2], dataset.field.shape[-2])
    density_channels = field[:,  :, :, 0, :, :]
    phase_channels = field[:, :, :, 1, :, :]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    plt.rcParams['font.family'] = 'DejaVu Serif'

    progress_bar = tqdm(enumerate(dataset.alpha_values),desc=f"Plotting", total=len(dataset.alpha_values), unit="alpha")
    for alpha_index, alpha_value in progress_bar:
        
        fig_density, axes_density = plt.subplots(dataset.number_of_n2, dataset.number_of_isat, figsize=(25, 25), layout="tight")
        fig_density.suptitle(f'Density Channels - {puiss_str} = {dataset.input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(dataset.n2_values):
            for isat_index, isat_value in enumerate(dataset.isat_values):
                ax = axes_density if dataset.number_of_n2 == 1 and dataset.number_of_isat == 1 else (axes_density[n2_index, isat_index] if dataset.number_of_n2 > 1 and dataset.number_of_n2 > 1 else (axes_density[n2_index] if dataset.number_of_n2 > 1 else axes_density[isat_index]))
                ax.imshow(density_channels[alpha_index, n2_index, isat_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{dataset.saving_path}/density_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_{alpha_value}_power{dataset.input_power:.2f}.png')
        plt.close(fig_density) 

        fig_phase, axes_phase = plt.subplots(dataset.number_of_n2, dataset.number_of_isat, figsize=(25, 25), layout="tight")
        fig_phase.suptitle(f'Phase Channels - {puiss_str} = {dataset.input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(dataset.n2_values):
            for isat_index, isat_value in enumerate(dataset.isat_values):
                ax = axes_phase if dataset.number_of_n2 == 1 and dataset.number_of_isat == 1 else (axes_phase[n2_index, isat_index] if dataset.number_of_n2 > 1 and dataset.number_of_isat > 1 else (axes_phase[n2_index] if dataset.number_of_n2 > 1 else axes_phase[isat_index]))
                ax.imshow(phase_channels[alpha_index, n2_index, isat_index, :, :], cmap='twilight_shifted')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.savefig(f'{dataset.saving_path}/phase_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_{alpha_value}_power{dataset.input_power:.2f}.png')
        plt.close(fig_phase)

def plot_results(
        dataset: EngineDataset,
        density_experiment: np.ndarray, 
        phase_experiment: np.ndarray
        ) -> None:

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    

    extent = [-dataset.window_training/2*1e3, dataset.window_training/2*1e3, -dataset.window_training/2*1e3, dataset.window_training/2*1e3]

    computed_n2 = dataset.n2_values[0]
    computed_alpha = dataset.alpha_values[0]
    computed_isat = dataset.isat_values[0]

    title = f"Results: {n2_str} = {scientific_formatter(computed_n2)}{n2_u},"
    title += f" {isat_str} = {scientific_formatter(computed_isat)}{isat_u},"
    title += f" {puiss_str} = {scientific_formatter(dataset.input_power)}{puiss_u},"
    title+= f"{alpha_str} = {scientific_formatter(computed_alpha)}{alpha_u}"

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 10

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="tight")
    fig.suptitle(title)

    ims = []
    ims.append(axs[0, 0].imshow(dataset.field[0, 0, :, :], cmap="viridis", extent=extent))
    ims.append(axs[0, 1].imshow(density_experiment, cmap="viridis", extent=extent))
    ims.append(axs[1, 0].imshow(dataset.field[0, 1, :, :], cmap="twilight_shifted", extent=extent))
    ims.append(axs[1, 1].imshow(phase_experiment, cmap="twilight_shifted", extent=extent))
    
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
    axs[0, 1].set_title("Experimental density")
    axs[1, 1].set_title("Experimental normalized phase")
    for ax in axs.flatten():
        ax.set_xlabel(r"x (mm)")
        ax.set_ylabel(r"y (mm)")
    
    plt.savefig(f"{dataset.saving_path}/prediction_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power}.png")

def plot_sandbox(
        dataset: EngineDataset,
        density_experiment: np.ndarray, 
        phase_experiment: np.ndarray):
    """
    Plot and save sandbox experimental results of density, phase, and unwrapped phase.

    Parameters:
    - E (np.ndarray): Experimental data.
    - density_experiment (np.ndarray): Experimental density data.
    - phase_experiment (np.ndarray): Experimental phase data.
    - uphase_experiment (np.ndarray): Experimental unwrapped phase data.
    - window_out (float): Window size.
    - n2 (float): n2 parameter.
    - isat (float): isat parameter.
    - alpha (float): alpha parameter.
    - input_power (float): Input power parameter.
    - saving_path (str): Path to save the plots.

    Returns:
    - None
    """
    extent = [-dataset.window_training/2*1e3, dataset.window_training/2*1e3, -dataset.window_training/2*1e3, dataset.window_training/2*1e3]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    
    title = f"{n2_str} = {scientific_formatter(dataset.n2_values[0])}{n2_u},"
    title += f" {isat_str} = {scientific_formatter(dataset.isat_values[0])}{isat_u},"
    title += f" {puiss_str} = {scientific_formatter(dataset.input_power)}{puiss_u},"
    title+= f"{alpha_str} = {scientific_formatter(dataset.alpha_values[0])}{alpha_u}"

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="tight")
    fig.suptitle(title)
    ims = []
    ims.append(axs[0, 0].imshow(dataset.field[0, 0, :, :], cmap="viridis", extent=extent))
    ims.append(axs[0, 1].imshow(density_experiment, cmap="viridis", extent=extent))
    ims.append(axs[1, 0].imshow(dataset.field[0, 1, :, :], cmap="twilight_shifted", extent=extent))
    ims.append(axs[1, 1].imshow(phase_experiment, cmap="twilight_shifted", extent=extent))

    dividers = []
    for ax in axs.flatten():
        dividers.append(make_axes_locatable(ax))
    caxes = []
    for divider in dividers:
        caxes.append(divider.append_axes("right", size="5%", pad=0.05))
    for im, cax in zip(ims, caxes):
        fig.colorbar(im, cax=cax)

    axs[0, 0].set_title("Density")
    axs[1, 0].set_title("phase")
    axs[0, 1].set_title("Experimental density")
    axs[1, 1].set_title("Experimental phase")
    for ax in axs.flatten():
        ax.set_xlabel(r"x (mm)")
        ax.set_ylabel(r"y (mm)")

    plt.savefig(f"{dataset.saving_path}/sandbox.png")

def plot_prediction(true_values, predictions, path):
    
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 10

    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    variables = [n2_str, isat_str, alpha_str]
    units = [n2_u, isat_u, alpha_u]
    labels = ["n2", "isat", "alpha"]

    for i in range(3):
        plt.figure(figsize=(10, 10))
        plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)
        plt.plot(np.linspace(0, 1,100),np.linspace(0, 1,100), 'r')  # Assuming normalized targets
        plt.xlabel(f'True {variables[i]} ({units[i]})')
        plt.ylabel(f'Predicted {variables[i]} ({units[i]})')
        plt.title(f'Predicted vs True for {variables[i]}')
        plt.savefig(f"{path}/predictedvstrue_{labels[i]}.png")