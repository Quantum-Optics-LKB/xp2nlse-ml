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
from engine.engine_dataset import EngineDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.feature import hog


class RandomPhaseShift(nn.Module):
    def __init__(self, shift_range=(-np.pi, np.pi), p=0.5):
        """
        Random Phase Shift Augmentation for Kornia.

        Parameters:
        - shift_range (tuple): Range from which to sample the phase shift value.
        - p (float): Probability of applying the phase shift.
        """
        super(RandomPhaseShift, self).__init__()
        self.shift_range = shift_range
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            x = x * np.pi
            phase_shift = torch.FloatTensor(1).uniform_(*self.shift_range).to(x.device)
            # Add the phase shift to all images in the batch
            x = x + phase_shift
            # Wrap phase values to be in the range [-pi, pi]
            x = torch.fmod(x + np.pi, 2 * np.pi) - np.pi
            x = x/np.pi
        return x

def apply_hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), orientations=9, channels=False):
    if channels:
        for i in range(image.shape[0]):
            hog_features, image[i, :, :] = hog(image[i, :, :], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, 
                                        orientations=orientations, visualize=True, block_norm='L2-Hys')

    else:
        hog_features, image = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, 
                                    orientations=orientations, visualize=True, block_norm='L2-Hys')
    return image


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
    # data -= np.min(data, axis=(-2, -1), keepdims=True)
    # data /= np.max(data, axis=(-2, -1), keepdims=True)
    data -= np.mean(data,axis=(0, -2, -1), keepdims=True) 
    data /= np.std(data, axis=(0, -2, -1), keepdims=True)
    return data

def augmentation_density(elastic_sigma, elastic_alpha, rotation_degrees) -> torch.nn.Sequential:
    """
    Create a sequential transformation pipeline for elastic and salt-pepper noise.

    Returns:
    - torch.nn.Sequential: Sequential transformation pipeline.
    """
    gamma = np.random.uniform(1, 4.5, 1)[0]
    
    return torch.nn.Sequential(
         K.RandomGaussianBlur(kernel_size=51,sigma=(1, 1), p=.5, keepdim=True),
        #  K.RandomGamma(gamma=(gamma, gamma), p=.15, keepdim=True),
        #  K.RandomElasticTransform(kernel_size=51, sigma=elastic_sigma, alpha=elastic_alpha ,p=.15, keepdim=True),
        #  K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=.5, keepdim=True),
    )
def augmentation_phase(elastic_sigma, elastic_alpha, rotation_degrees) -> torch.nn.Sequential:
    """
    Create a sequential transformation pipeline for elastic and salt-pepper noise.

    Returns:
    - torch.nn.Sequential: Sequential transformation pipeline.
    """
    return torch.nn.Sequential(
         RandomPhaseShift(shift_range=(0, np.pi/2), p=1),
        #  K.RandomElasticTransform(kernel_size=51, sigma=elastic_sigma, alpha=elastic_alpha ,p=.15, keepdim=True),
        #  K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=.25, keepdim=True),
    )

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

    # train_indices = indices[:train_index]
    # validation_indices = indices[train_index:validation_index]
    # test_indices = indices[validation_index:]

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

    field = dataset.field.copy().reshape(dataset.number_of_n2, dataset.number_of_isat, dataset.number_of_alpha, 2, dataset.field.shape[-2], dataset.field.shape[-2])
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

    for alpha_index, alpha_value in enumerate(dataset.alpha_values):
        
        fig_density, axes_density = plt.subplots(dataset.number_of_n2, dataset.number_of_isat, figsize=(25, 25), layout="tight")
        fig_density.suptitle(f'Density Channels - {puiss_str} = {dataset.input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(dataset.n2_values):
            for isat_index, isat_value in enumerate(dataset.isat_values):
                ax = axes_density if dataset.number_of_n2 == 1 and dataset.number_of_isat == 1 else (axes_density[n2_index, isat_index] if dataset.number_of_n2 > 1 and dataset.number_of_n2 > 1 else (axes_density[n2_index] if dataset.number_of_n2 > 1 else axes_density[isat_index]))
                ax.imshow(density_channels[n2_index, isat_index, alpha_index, :, :], cmap='viridis')
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
                ax.imshow(phase_channels[n2_index, isat_index, alpha_index, :, :], cmap='twilight_shifted')
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