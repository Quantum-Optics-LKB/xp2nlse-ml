#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from engine.utils import set_seed
from torch.utils.data import Dataset
set_seed(10)

class NetworkDataset(Dataset):
    def __init__(
            self, 
            set: np.ndarray, 
            n2_labels: np.ndarray, 
            isat_labels: np.ndarray,
            alpha_labels: np.ndarray,
            ) -> None:

        self.n2_labels = torch.from_numpy(n2_labels).to(torch.float64).unsqueeze(1)
        self.isat_labels = torch.from_numpy(isat_labels).to(torch.float64).unsqueeze(1)
        self.alpha_labels = torch.from_numpy(alpha_labels).to(torch.float64).unsqueeze(1)
        self.set = torch.from_numpy(set).to(torch.float64)

    def __len__(
            self
            ) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.n2_labels)
        

    def __getitem__(
            self, 
            idx: int
            ) -> tuple:
        """
        Retrieves the image data and associated labels for the given index.
        
        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            tuple: A tuple containing the image data (torch.Tensor) and the associated 
                   n2 (torch.Tensor), isat (torch.Tensor), and alpha (torch.Tensor) labels.
        """
        
        set_item = self.set[idx,:,:,:]
        n2_label = self.n2_labels[idx]
        isat_label = self.isat_labels[idx]
        alpha_label = self.alpha_labels[idx]

        return  set_item, n2_label, isat_label, alpha_label