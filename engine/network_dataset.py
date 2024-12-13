#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from engine.utils import set_seed
from torch.utils.data import Dataset
set_seed(10)

class NetworkDataset(Dataset):
    """
    A custom dataset class for managing field data and associated labels 
    for training and evaluation of neural networks.

    Parameters:
    -----------
    set : np.ndarray
        The dataset containing field data, shape [num_samples, channels, height, width].
    n2_labels : np.ndarray
        Labels for n2, shape [num_samples].
    isat_labels : np.ndarray
        Labels for Isat, shape [num_samples].
    alpha_labels : np.ndarray
        Labels for alpha, shape [num_samples].

    Methods:
    --------
    __len__():
        Returns the total number of samples in the dataset.
    __getitem__(idx):
        Retrieves a sample and its corresponding labels by index.
    """
    def __init__(
            self, 
            set: np.ndarray, 
            n2_labels: np.ndarray, 
            isat_labels: np.ndarray,
            alpha_labels: np.ndarray,
            ) -> None:
        
        self.n2_labels = torch.from_numpy(n2_labels).to(torch.float32).unsqueeze(1)
        self.isat_labels = torch.from_numpy(isat_labels).to(torch.float32).unsqueeze(1)
        self.alpha_labels = torch.from_numpy(alpha_labels).to(torch.float32).unsqueeze(1)
        self.set = torch.from_numpy(set).to(torch.float32)

    def __len__(
            self
            ) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
        int
            The number of samples in the dataset.
        """
        return len(self.n2_labels)
        

    def __getitem__(
            self, 
            idx: int
            ) -> tuple:
        """
        Retrieves a sample and its associated labels by index.

        Parameters:
        -----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        --------
        tuple
            A tuple containing:
            - set_item (torch.Tensor): The field data for the sample, shape [channels, height, width].
            - n2_label (torch.Tensor): The n2 label for the sample, shape [1].
            - isat_label (torch.Tensor): The Isat label for the sample, shape [1].
            - alpha_label (torch.Tensor): The alpha label for the sample, shape [1].
        """
        
        set_item = self.set[idx,:,:,:]
        n2_label = self.n2_labels[idx]
        isat_label = self.isat_labels[idx]
        alpha_label = self.alpha_labels[idx]

        return  set_item, n2_label, isat_label, alpha_label