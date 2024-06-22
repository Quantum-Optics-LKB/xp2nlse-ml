#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from torch.utils.data import Dataset
import torch
import numpy as np
from engine.seed_settings import set_seed
set_seed(42)

class FieldDataset(Dataset):
    """
    A custom dataset class for handling datasets that include features such as 
    optical field data and values for various parameters like n2, power, and 
    saturation intensity (isat), with support for data augmentation during training.

    Attributes:
    - device (torch.device): The computing device where the dataset will be processed.
    - training (bool): Specifies whether the dataset is used for training, affecting data augmentation.
    - n2_values (torch.Tensor): Tensor of n2 values.
    - isat_values (torch.Tensor): Tensor of isat values.
    - data (torch.Tensor): The main dataset containing the optical field data.

    Parameters:
    - data (np.ndarray): Optical field data, shaped as [num_samples, num_channels, height, width].
    - n2_values (np.ndarray): values for n2 values, one for each data sample.
    - isat_values (np.ndarray): values for isat values, one for each data sample.
    - training (bool): Flag to enable data augmentation if True.
    - device (torch.device): The device to perform computations on, default is CPU.
    """
    def __init__(
            self, 
            data: np.ndarray, 
            n2_values: np.ndarray, 
            isat_values: np.ndarray):
        
        self.n2_values = torch.from_numpy(n2_values).to(torch.float16).unsqueeze(1)
        self.isat_values = torch.from_numpy(isat_values).to(torch.float16).unsqueeze(1)
        self.data = torch.from_numpy(data).to(torch.float16)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
        int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a single data sample and its associated values by index.

        Parameters:
        - idx (int): Index of the data sample to retrieve.

        Returns:
        tuple: A tuple containing the data sample and its values (data_item, power_value, power_label, n2_label, isat_label),
               where `data_item` is the optical field data, `n2_label`, `isat_label` are the respective values for the sample.
        """
        data_item = self.data[idx,:,:,:]
        n2_label = self.n2_values[idx]
        isat_label = self.isat_values[idx]

        return  data_item, n2_label, isat_label