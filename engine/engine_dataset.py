#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from torch.utils.data import Dataset

class EngineDataset(Dataset):
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
            learning_rate: float, 
            batch_size: int, 
            num_epochs: int, 
            accumulator: int,
            device_number: int,
            ) -> None:
        
        self.n2_values = n2_values
        self.alpha_values = alpha_values
        self.isat_values = isat_values

        self.n2_min_standard = 0
        self.n2_max_standard = 0

        self.alpha_min_standard = 0
        self.alpha_max_standard = 0

        self.isat_min_standard = 0
        self.isat_max_standard = 0

        self.number_of_n2 = len(n2_values)
        self.number_of_alpha = len(alpha_values)
        self.number_of_isat = len(isat_values)

        N2_labels, ISAT_labels, ALPHA_labels = np.meshgrid(n2_values, isat_values, alpha_values) 

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

        self.field = np.zeros((self.number_of_n2*self.number_of_isat*self.number_of_alpha,2, resolution_training, resolution_training), dtype=np.float64)
        self.mean_standard = 0
        self.std_standard = 0

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.accumulator = accumulator
        self.device_number =  device_number