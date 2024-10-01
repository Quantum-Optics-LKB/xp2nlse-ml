#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from engine.utils import set_seed
from torch.utils.data import Dataset
set_seed(10)

class FieldDataset(Dataset):
    """
    Custom PyTorch Dataset for handling field data along with associated n2, isat, 
    and alpha values.

    Args:
        data (np.ndarray): A 4D numpy array containing the image data with dimensions 
            (number_of_images, channels, height, width).
        n2_values (np.ndarray): A 1D numpy array containing the n2 values associated 
            with each image.
        isat_values (np.ndarray): A 1D numpy array containing the isat values associated 
            with each image.
        alpha_values (np.ndarray): A 1D numpy array containing the alpha values associated 
            with each image.

    Attributes:
        n2_values (torch.Tensor): A 2D tensor containing the n2 values with shape 
            (number_of_images, 1).
        isat_values (torch.Tensor): A 2D tensor containing the isat values with shape 
            (number_of_images, 1).
        alpha_values (torch.Tensor): A 2D tensor containing the alpha values with shape 
            (number_of_images, 1).
        data (torch.Tensor): A 4D tensor containing the image data with shape 
            (number_of_images, channels, height, width).

    Methods:
        __len__() -> int:
            Returns the total number of images in the dataset.
        
        __getitem__(idx: int) -> tuple:
            Returns a tuple containing the image data and the associated n2, isat, 
            and alpha labels for the given index.
    """
    def __init__(
            self, 
            data: np.ndarray, 
            training_indices: np.ndarray,
            validation_indices: np.ndarray,
            test_indices: np.ndarray,
            batch_size: int,
            n2_values: np.ndarray, 
            isat_values: np.ndarray,
            alpha_values: np.ndarray
            ) -> None:
        """
        Initializes the FieldDataset with image data and associated labels.
        
        Args:
            data (np.ndarray): A 4D numpy array containing the image data.
            n2_values (np.ndarray): A 1D numpy array containing the n2 values.
            isat_values (np.ndarray): A 1D numpy array containing the isat values.
            alpha_values (np.ndarray): A 1D numpy array containing the alpha values.
        """
        self.flag = ""
        self.batch_size = batch_size
        self.training_indices = torch.from_numpy(training_indices).to(torch.int)
        self.validation_indices = torch.from_numpy(validation_indices).to(torch.int)
        self.test_indices = torch.from_numpy(test_indices).to(torch.int)

        self.n2_values = torch.from_numpy(n2_values).to(torch.float64).unsqueeze(1)
        self.isat_values = torch.from_numpy(isat_values).to(torch.float64).unsqueeze(1)
        self.alpha_values = torch.from_numpy(alpha_values).to(torch.float64).unsqueeze(1)
        self.data = torch.from_numpy(data).to(torch.float64)

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