#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import kornia.augmentation as K
from matplotlib import pyplot as plt
from engine.engine_dataset import EngineDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CircularFilterAugmentation(nn.Module):
    """
    Applies a circular mask with random radii to images with a probability `p`.

    Parameters:
    -----------
    radius_range : tuple
        Minimum and maximum radii as a fraction of the image size.
    p : float, optional
        Probability of applying the augmentation, default is 0.5.

    Methods:
    --------
    forward(images):
        Applies the circular filter augmentation to a batch of images.
    """
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
    """
    Randomly shifts the phase of an image within a specified range.

    Parameters:
    -----------
    shift_range : tuple, optional
        The range of phase shifts to apply, default is (-pi, pi).
    p : float, optional
        Probability of applying the phase shift, default is 0.5.

    Methods:
    --------
    forward(x):
        Applies the phase shift to a batch of images.
    """
    def __init__(self, shift_range=(-np.pi, np.pi), p=0.5):
        super(RandomPhaseShift, self).__init__()
        self.shift_range = shift_range
        self.p = p

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # Apply phase shift with a probability
        apply_shift = torch.rand(B, device=device) < self.p

        # Scale images from range [0, 1] to [-pi, pi] consistently
        x = x * (2 * np.pi) - np.pi

        # Generate random shifts for each image in the batch
        phase_shifts = torch.empty(B, device=device).uniform_(*self.shift_range)

        # Apply unique phase shifts to each image, ensuring consistent broadcasting
        shifted_x = x + phase_shifts.view(B, 1, 1)

        # Wrap phase values to be in the range [-pi, pi]
        shifted_x = torch.remainder(shifted_x + np.pi, 2 * np.pi) - np.pi

        # Rescale back to [0, 1] and apply only where specified by apply_shift
        x = (shifted_x + np.pi) / (2 * np.pi)
        
        # Apply the shift conditionally based on `apply_shift`
        return torch.where(apply_shift.view(B, 1, 1), x, x)


def sigmospace(array, a):
    """
    Apply a sigmoidal transformation to enhance the dynamic range of an array.

    Parameters:
    -----------
    array : np.ndarray
        Input array to transform.
    a : float
        Exponent controlling the steepness of the sigmoid.

    Returns:
    --------
    np.ndarray
        The transformed array with enhanced dynamic range.
    """
    normalized_array = (array - np.min(array))/(np.max(array) - np.min(array))
    normalized_array = normalized_array**(a) / (normalized_array**(a) + np.abs(1-normalized_array)**a)
    
    return normalized_array * (np.max(array) - np.min(array)) + np.min(array)

def scientific_formatter(
        x: float
        ) -> str:
    """
    Format a number in scientific notation for LaTeX-compatible display.

    Parameters:
    -----------
    x : float
        The number to format.

    Returns:
    --------
    str
        The formatted number in LaTeX-compatible scientific notation.
    """
    a, b = "{:.2e}".format(x).split("e")
    b = int(b)
    return r"${}\times 10^{{{}}}$".format(a, b)

def augmentation_density(rotation_degrees, shear) -> torch.nn.Sequential:
    """
    Creates a sequential transformation pipeline for elastic and affine transformations.

    Parameters:
    -----------
    rotation_degrees : float
        Maximum rotation angle for the affine transformation.
    shear : float
        Shear factor for the affine transformation.

    Returns:
    --------
    torch.nn.Sequential
        A sequence of random elastic and affine transformations.
    """
    alpha = (np.random.uniform(1, 2, 1)[0], np.random.uniform(1, 2, 1)[0])
    return torch.nn.Sequential(
        K.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=alpha, p=0.25),
        K.RandomAffine(degrees=0, shear=shear, p=0.25, keepdim=True, padding_mode="border"),
        K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=0.5, keepdim=True, padding_mode="border"),
    )

def augmentation_phase(rotation_degrees, shear) -> torch.nn.Sequential:
    """
    Creates a sequential transformation pipeline with random phase shifts and affine transformations.

    Parameters:
    -----------
    rotation_degrees : float
        Maximum rotation angle for the affine transformation.
    shear : float
        Shear factor for the affine transformation.

    Returns:
    --------
    torch.nn.Sequential
        A sequence of phase shifts, circular masks, and affine transformations.
    """
    return torch.nn.Sequential(
        RandomPhaseShift(shift_range=(np.pi/6, np.pi/2), p=0.75),
        CircularFilterAugmentation(radius_range=(.5, .75), p=.25),
        K.RandomAffine(degrees=0, shear=shear, p=.25, keepdim=True, padding_mode="border"),
        K.RandomAffine(degrees=rotation_degrees, translate=(.15, .15), p=.5, keepdim=True, padding_mode="border"),
    )

def shuffle_dataset(
        dataset: EngineDataset
        ) -> None:
    """
    Randomizes the order of the dataset fields and corresponding labels.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset object containing fields and labels to be shuffled.

    Returns:
    --------
    None
        This function modifies the dataset in place by shuffling its fields and labels.
    """

    # Generate an array of indices equal to the length of alpha labels
    indices = np.arange(len(dataset.alpha_labels))

    # Shuffle these indices to create a random order
    np.random.shuffle(indices)

    # Reorder the dataset's fields and labels using the shuffled indices
    dataset.field = dataset.field[indices, :, :, :] # Shuffle the field data
    dataset.n2_labels = dataset.n2_labels[indices] # Shuffle n2 labels
    dataset.isat_labels = dataset.isat_labels[indices] # Shuffle isat labels
    dataset.alpha_labels = dataset.alpha_labels[indices] # Shuffle alpha labels

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
    Set a global random seed for reproducibility in random number generation.

    Parameters:
    -----------
    seed : int
        The random seed value to be used.

    Returns:
    --------
    None
        This function does not return any value. It configures the random seed for all relevant libraries.
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
    Splits data indices into training, validation, and test subsets.

    Parameters:
    -----------
    indices : np.ndarray
        Array of dataset indices to split.
    train_ratio : float, optional
        Proportion of data to allocate to the training set. Default is 0.8.
    validation_ratio : float, optional
        Proportion of data to allocate to the validation set. Default is 0.1.
    test_ratio : float, optional
        Proportion of data to allocate to the test set. Default is 0.1.

    Returns:
    --------
    tuple
        - train_index (int): Index marking the end of the training subset.
        - validation_index (int): Index marking the end of the validation subset (start of the test subset).
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
    Plots and saves training and validation loss curves.

    Parameters:
    -----------
    y_train : np.ndarray
        Array of training loss values over epochs.
    y_val : np.ndarray
        Array of validation loss values over epochs.
    path : str
        Directory path to save the loss plot.
    resolution : int
        Image resolution for the plot.
    number_of_n2 : int
        Number of n2 parameters in the dataset.
    number_of_isat : int
        Number of Isat parameters in the dataset.
    number_of_alpha : int
        Number of alpha parameters in the dataset.

    Returns:
    --------
    None
        Saves the plot as a PNG file in the specified path.
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
    Visualizes and saves density and phase channel plots for a dataset.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset object containing field data and simulation parameters.

    Returns:
    --------
    None
        This function saves the generated plots to the dataset's saving path.
    """

    # Reshape the dataset field to separate alpha, n2, isat, and channel dimensions
    field = dataset.field.copy().reshape(dataset.number_of_alpha,
                                          dataset.number_of_n2, 
                                          dataset.number_of_isat, 
                                          2, 
                                          dataset.field.shape[-2], 
                                          dataset.field.shape[-2])
    density_channels = field[:,  :, :, 0, :, :] # Density channel
    phase_channels = field[:, :, :, 1, :, :] # Phase channel
    
    # Set up unit and parameter labels for the plots
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    plt.rcParams['font.family'] = 'DejaVu Serif'

    # Loop over alpha values to generate plots for each alpha level
    progress_bar = tqdm(enumerate(dataset.alpha_values),
                        desc=f"Plotting", 
                        total=len(dataset.alpha_values), 
                        unit="alpha")
    
    for alpha_index, alpha_value in progress_bar:
        
        # Plot density channels
        fig_density, axes_density = plt.subplots(dataset.number_of_n2, 
                                                 dataset.number_of_isat, 
                                                 figsize=(25, 25), 
                                                 layout="tight")
        fig_density.suptitle(
            f'Density Channels - {puiss_str} = {dataset.input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}'
            )

        # Loop over n2 and isat values for density plots
        for n2_index, n2_value in enumerate(dataset.n2_values):
            for isat_index, isat_value in enumerate(dataset.isat_values):

                # Handle subplots based on the number of n2 and isat values
                ax = axes_density if dataset.number_of_n2 == 1 and dataset.number_of_isat == 1 else (
                    axes_density[n2_index, isat_index] if dataset.number_of_n2 > 1 and dataset.number_of_n2 > 1 else (
                        axes_density[n2_index] if dataset.number_of_n2 > 1 else axes_density[isat_index]
                        )
                    )
                ax.imshow(density_channels[alpha_index, n2_index, isat_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{dataset.saving_path}/density_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_{alpha_value}_power{dataset.input_power:.2f}.png')
        plt.close(fig_density) 
        
        # Plot phase channels
        fig_phase, axes_phase = plt.subplots(dataset.number_of_n2, dataset.number_of_isat, figsize=(25, 25), layout="tight")
        fig_phase.suptitle(
            f'Phase Channels - {puiss_str} = {dataset.input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}'
            )

        # Loop over n2 and isat values for phase plots
        for n2_index, n2_value in enumerate(dataset.n2_values):
            for isat_index, isat_value in enumerate(dataset.isat_values):
                ax = axes_phase if dataset.number_of_n2 == 1 and dataset.number_of_isat == 1 else (
                    axes_phase[n2_index, isat_index] if dataset.number_of_n2 > 1 and dataset.number_of_isat > 1 else (
                        axes_phase[n2_index] if dataset.number_of_n2 > 1 else axes_phase[isat_index]
                        )
                    )
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
    axs[1, 0].set_title("phase")
    axs[0, 1].set_title("Experimental density")
    axs[1, 1].set_title("Experimental phase")
    for ax in axs.flatten():
        ax.set_xlabel(r"x (mm)")
        ax.set_ylabel(r"y (mm)")
    
    plt.savefig(f"{dataset.saving_path}/prediction_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power}.png")

def plot_sandbox(
        dataset: EngineDataset,
        density_experiment: np.ndarray, 
        phase_experiment: np.ndarray):
    """
    Plot and save comparisons of simulated and experimental density and phase data.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset containing simulated field data and associated metadata.
    density_experiment : np.ndarray
        The experimental density data, shape [resolution_training, resolution_training].
    phase_experiment : np.ndarray
        The experimental phase data, shape [resolution_training, resolution_training].

    Returns:
    --------
    None
        Saves the generated plot as "sandbox.png" in the dataset's saving path.
    """
    extent = [-dataset.window_training/2*1e3, dataset.window_training/2*1e3, 
              -dataset.window_training/2*1e3, dataset.window_training/2*1e3]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"
    
    # Generate title with formatted physical parameters
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
    axs[1, 0].set_title("Phase")
    axs[0, 1].set_title("Experimental density")
    axs[1, 1].set_title("Experimental phase")
    for ax in axs.flatten():
        ax.set_xlabel(r"x (mm)")
        ax.set_ylabel(r"y (mm)")

    plt.savefig(f"{dataset.saving_path}/sandbox.png")

def plot_prediction(true_values, predictions, path):
    """
    Plot and save scatter plots of true vs. predicted values for n2, Isat, and alpha.

    Parameters:
    -----------
    true_values : np.ndarray
        Array of true parameter values, shape [num_samples, 3].
    predictions : np.ndarray
        Array of predicted parameter values, shape [num_samples, 3].
    path : str
        Path to save the generated plots.

    Returns:
    --------
    None
        Saves scatter plots for each parameter as PNG files in the specified path.
    """
    
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