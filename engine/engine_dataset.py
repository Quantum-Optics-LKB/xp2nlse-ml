#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from torch.utils.data import Dataset

class EngineDataset(Dataset):
    """
    Defines a dataset for simulation and training in engine computations.

    This class organizes and initializes key parameters for simulations and training, 
    including physical properties, grid resolutions, labels, and training settings.

    Attributes:
    -----------
    n2_values : np.ndarray
        Array of n2 parameter values.
    alpha_values : np.ndarray
        Array of alpha parameter values.
    isat_values : np.ndarray
        Array of isat parameter values.
    input_power : float
        Input power value in watts.
    waist : float
        Beam waist in meters.
    non_locality : float
        Non-locality parameter value.
    delta_z : float
        Step size along the Z-axis for simulation.
    length : float
        Length of the simulation in meters.
    resolution_simulation : int
        Resolution of the simulation grid.
    resolution_training : int
        Resolution of the training grid.
    window_simulation : float
        Simulation window size in meters.
    window_training : float
        Training dataset window size in meters.
    saving_path : str
        Path to save generated data and results.
    learning_rate : list
        List of learning rates for training.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of epochs for training.
    accumulator : int
        Accumulator value for simulation steps.
    device_number : int
        GPU device number used for computations.
    field : np.ndarray
        Array to store the field data for simulations and training.
    n2_labels : np.ndarray
        Flattened array of n2 labels for the dataset.
    alpha_labels : np.ndarray
        Flattened array of alpha labels for the dataset.
    isat_labels : np.ndarray
        Flattened array of isat labels for the dataset.
    n2_min, n2_max : float
        Placeholder values for standardization of n2.
    alpha_min, alpha_max : float
        Placeholder values for standardization of alpha.
    isat_min, isat_max : float
        Placeholder values for standardization of isat.
    number_of_n2, number_of_alpha, number_of_isat : int
        Number of unique n2, alpha, and isat parameter values.
    """

    def __init__(
            self, 
            n2_values: np.ndarray, 
            alpha_values: np.ndarray, 
            isat_values: np.ndarray,
            input_power: float,
            waist: float,
            non_locality: float,
            delta_z: float,
            length: float,
            resolution_simulation: int,
            resolution_training: int,
            window_simulation: float,
            window_training: float,
            saving_path: str,
            learning_rate: list, 
            batch_size: int, 
            num_epochs: int, 
            accumulator: int,
            device_number: int,
            ) -> None:
        """
        Initializes the EngineDataset object with simulation and training parameters.

        Parameters:
        -----------
        n2_values : np.ndarray
            Array of n2 parameter values.
        alpha_values : np.ndarray
            Array of alpha parameter values.
        isat_values : np.ndarray
            Array of isat parameter values.
        input_power : float
            Input power value in watts.
        waist : float
            Beam waist in meters.
        non_locality : float
            Non-locality parameter value.
        delta_z : float
            Step size along the Z-axis for simulation.
        length : float
            Length of the simulation in meters.
        resolution_simulation : int
            Resolution of the simulation grid.
        resolution_training : int
            Resolution of the training grid.
        window_simulation : float
            Simulation window size in meters.
        window_training : float
            Training dataset window size in meters.
        saving_path : str
            Path to save generated data and results.
        learning_rate : list
            List of learning rates for training.
        batch_size : int
            Batch size for training.
        num_epochs : int
            Number of epochs for training.
        accumulator : int
            Accumulator value for simulation steps.
        device_number : int
            GPU device number used for computations.

        Returns:
        --------
        None
            This method initializes the class and its attributes.
        """
        self.n2_values = n2_values
        self.alpha_values = alpha_values
        self.isat_values = isat_values

        self.n2_min = 0
        self.n2_max = 0

        self.alpha_min = 0
        self.alpha_max = 0

        self.isat_min = 0
        self.isat_max = 0

        self.number_of_n2 = len(n2_values)
        self.number_of_alpha = len(alpha_values)
        self.number_of_isat = len(isat_values)

        ALPHA_labels, N2_labels, ISAT_labels = np.meshgrid(alpha_values, n2_values, isat_values) 

        self.n2_labels = N2_labels.reshape(-1)
        self.isat_labels = ISAT_labels.reshape(-1)
        self.alpha_labels = ALPHA_labels.reshape(-1)

        self.input_power = input_power
        self.waist = waist
        self.non_locality = non_locality
        self.delta_z = delta_z
        self.length = length
        self.resolution_simulation = resolution_simulation
        self.resolution_training = resolution_training
        self.window_simulation = window_simulation
        self.window_training = window_training
        self.saving_path = saving_path

        self.field = np.zeros((self.number_of_n2*self.number_of_isat*self.number_of_alpha,2, resolution_training, resolution_training), dtype=np.float32)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.accumulator = accumulator
        self.device_number =  device_number