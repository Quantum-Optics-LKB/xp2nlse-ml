#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from tqdm import tqdm
from NLSE import NLSE
from cupyx.scipy.ndimage import zoom
from scipy.constants import c, epsilon_0
from engine.engine_dataset import EngineDataset
from engine.utils import experiment_noise, set_seed
set_seed(10)

def simulation(
    dataset: EngineDataset, 
    ) -> np.ndarray:
    """
    Simulates beam propagation using the Nonlinear Schrödinger Equation (NLSE) 
    and stores the results in the given dataset.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset object containing simulation parameters and storage fields.

    Returns:
    --------
    np.ndarray
        The simulated dataset field with intensity (density) and phase values.

    Workflow:
    ---------
    1. Initializes beam parameters and adds experimental noise.
    2. Solves the NLSE for each combination of alpha and n2 parameter values.
    3. Crops and rescales the resulting field to the training resolution.
    4. Computes and stores the density (intensity) and phase of the beam.
    5. Optionally saves the dataset to disk if a saving path is specified.

    Notes:
    ------
    - The NLSE solver uses GPU computations (via `cupy`) for performance.
    - Experimental noise is applied to the initial beam profile for realism.

    """
    # Define cropping size based on the simulation and training window difference
    crop = int(0.5*(dataset.window_simulation - dataset.window_training)
               * dataset.resolution_simulation/dataset.window_simulation)
    isat = dataset.isat_values[:, np.newaxis, np.newaxis]

    # Create spatial grid for the beam profile
    X = np.linspace(-dataset.window_simulation/ 2, dataset.window_simulation / 2, num=dataset.resolution_simulation, endpoint=False, dtype=np.float64)
    Y = np.linspace(-dataset.window_simulation/ 2, dataset.window_simulation / 2, num=dataset.resolution_simulation, endpoint=False, dtype=np.float64)
    XX, YY = np.meshgrid(X, Y)
    
    # Initialize Gaussian beam profile
    beam = np.ones((dataset.number_of_isat, dataset.resolution_simulation, dataset.resolution_simulation), dtype=np.complex64)*np.exp(-(XX**2 + YY**2) / dataset.waist**2)
    
    # Add experimental noise to the beam
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = experiment_noise(beam, poisson_noise_lam, normal_noise_sigma)

    # Iterate over alpha and n2 values to simulate the beam propagation
    for alpha_index, alpha_value in tqdm(enumerate(dataset.alpha_values),desc=f"NLSE", total=len(dataset.alpha_values), unit="alpha"):
      for n2_index, n2_value in enumerate(dataset.n2_values):
        
        # Initialize NLSE solver
        simu = NLSE(power=dataset.input_power, alpha=alpha_value, window=dataset.window_simulation, n2=n2_value, 
                      V=None, L=dataset.length, NX=dataset.resolution_simulation, NY=dataset.resolution_simulation, 
                      Isat=isat, nl_length=dataset.non_locality)
        
        # Adjust for non-locality if specified
        if dataset.non_locality != 0:
          simu.nl_profile =  simu.nl_profile[np.newaxis, np.newaxis, :,:]
        simu.delta_z = dataset.delta_z

        # Solve the NLSE for the given beam
        A = simu.out_field(beam, z=dataset.length, verbose=False, plot=False, normalize=True, precision="single")

        # Crop the field to match training window size
        if crop != 0:
          A = A[:,crop:-crop,crop:-crop]

        # Rescale the field to training resolution
        zoom_factor = dataset.resolution_training / A.shape[-1]
        A = zoom(cp.asarray(A), (1, zoom_factor, zoom_factor),order=5).get()

        # Compute density (intensity) and phase
        density = np.abs(A)**2 * c * epsilon_0 / 2
        phase = np.angle(A)
        
        # Store results in the dataset field
        start_index = dataset.number_of_isat * dataset.number_of_n2 * alpha_index + dataset.number_of_isat * n2_index
        end_index = dataset.number_of_isat * dataset.number_of_n2 * (alpha_index) + dataset.number_of_isat * (n2_index + 1)
        dataset.field[start_index:end_index,0,:,:] = density
        dataset.field[start_index:end_index,1,:,:] = phase
    
    # Save the dataset field to the specified path
    if dataset.saving_path != "":
      path = f'{dataset.saving_path}/Es_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}'
      np.save(path, dataset.field)