#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from scipy.ndimage import zoom
from engine.generate import data_creation
from skimage.restoration import unwrap_phase
from engine.utils import plot_sandbox, set_seed
from engine.utils import general_extrema, normalize_data
set_seed(10)

def experiment(
        resolution_training: int,
        exp_image_path: str
        ) -> tuple:
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

    plot_sandbox(E, density_experiment, phase_experiment, uphase_experiment, window_out, n2, isat, alpha, input_power, saving_path)  