#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
# from engine.utils import set_seed
from torch.utils.data import Dataset
# set_seed(10)

class FieldDataset(Dataset):
    def __init__(
            self, 
            n2_values: np.ndarray, 
            alpha_values: np.ndarray, 
            isat_values: np.ndarray,
            ) -> None:
        
        self.n2_values = n2_values
        self.alpha_values = alpha_values
        self.isat_values = isat_values

        N2_labels, ISAT_labels, ALPHA_labels = np.meshgrid(n2_values, isat_values, alpha_values) 

        self.n2_labels = N2_labels.reshape(-1)
        self.isat_labels = ISAT_labels.reshape(-1)
        self.alpha_labels = ALPHA_labels.reshape(-1)


    def __len__(
            self
            ) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        if self.flag == "training":
            return len(self.training_indices)
        elif self.flag == "validation":
            return len(self.validation_indices)
        elif self.flag == "test":
            return len(self.test_indices)
        

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
        if self.flag == "training":
            idx = self.training_indices[idx].item()
        elif self.flag == "validation":
            idx = self.validation_indices[idx].item()
        elif self.flag == "test":
            idx = self.test_indices[idx].item()
            
        
        data_item = self.data[idx,:,:,:]
        n2_label = self.n2_values[idx]
        isat_label = self.isat_values[idx]
        alpha_label = self.alpha_values[idx]

        return  data_item, n2_label, isat_label, alpha_label