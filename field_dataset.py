#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from torch.utils.data import Dataset
import torch
import numpy as np
import albumentations as A

def get_augmentation(
        original_height: int, 
        original_width: int
        ) -> A.Compose:
    """
    Constructs a data augmentation pipeline with a specific set of transformations tailored for 
    optical field data or similar types of images. This pipeline includes blurring, cropping, and 
    resizing operations to simulate various realistic alterations that data might undergo.

    Parameters:
    - original_height (int): The original height of the images before augmentation.
    - original_width (int): The original width of the images before augmentation.

    Returns:
    albumentations.core.composition.Compose: An Albumentations Compose object that contains a 
    sequence of augmentation transformations to be applied to the images. These transformations 
    include Gaussian blur, motion blur, glass blur, a slight shift without scaling or rotation, 
    random cropping to 3/4 of the original dimensions, and resizing back to the original dimensions.

    The pipeline is set up to apply these transformations with certain probabilities, allowing for a 
    diversified dataset without excessively distorting the underlying data characteristics. This 
    augmentation strategy is particularly useful for training machine learning models on image data, 
    as it helps to improve model robustness by exposing it to a variety of visual perturbations.

    Example Usage:
        augmentation_pipeline = get_augmentation(256, 256)
        augmented_image = augmentation_pipeline(image=image)['image']
    """
    return A.Compose([
        A.GaussianBlur(blur_limit=(3, 11), p=0.25),
        A.MotionBlur(blur_limit=(3, 11), p=0.25),
        A.GlassBlur(sigma=0.1, max_delta=4, iterations=2, p=0.25),
        A.ShiftScaleRotate(shift_limit=30/original_height, scale_limit=0, rotate_limit=0, p=0.5),
        A.RandomCrop(height=original_height*3//4, width=original_height*3//4, p=0.5), 
        A.Resize(height=original_height, width=original_width)
    ])

class FieldDataset(Dataset):
    """
    A custom dataset class for handling datasets that include features such as 
    optical field data and labels for various parameters like n2, power, and 
    saturation intensity (isat), with support for data augmentation during training.

    Attributes:
    - device (torch.device): The computing device where the dataset will be processed.
    - training (bool): Specifies whether the dataset is used for training, affecting data augmentation.
    - power_labels (torch.Tensor): Tensor of power labels.
    - power_values (torch.Tensor): Tensor of continuous power values.
    - n2_labels (torch.Tensor): Tensor of n2 labels.
    - isat_labels (torch.Tensor): Tensor of isat labels.
    - data (torch.Tensor): The main dataset containing the optical field data.

    Parameters:
    - data (np.ndarray): Optical field data, shaped as [num_samples, num_channels, height, width].
    - n2_labels (np.ndarray): Labels for n2 values, one for each data sample.
    - power_labels (np.ndarray): Labels for power levels, one for each data sample.
    - isat_labels (np.ndarray): Labels for isat values, one for each data sample.
    - power_values (np.ndarray): Continuous power values, one for each data sample.
    - training (bool): Flag to enable data augmentation if True.
    - device (torch.device): The device to perform computations on, default is CPU.
    """
    def __init__(
            self, 
            data: np.ndarray, 
            n2_labels: np.ndarray, 
            power_labels: np.ndarray, 
            isat_labels: np.ndarray,
            power_values: np.ndarray, 
            training: bool, 
            device=torch.device("cpu")):
        """
        Initializes the FieldDataset instance, setting up the tensors for data and labels, and preparing
        data augmentation if in training mode.

        Parameters:
        - data (np.ndarray): The dataset containing the optical field data. The data is expected
          to be in the format of [num_samples, num_channels, height, width].
        - n2_labels (np.ndarray): An array of labels for the nonlinear refractive index (n2), 
          with each label corresponding to a sample in the dataset.
        - power_labels (np.ndarray): An array of categorical labels for the laser power, 
          where each label corresponds to a sample.
        - isat_labels (np.ndarray): An array of labels for the saturation intensities (Isat),
          with each label corresponding to a sample.
        - power_values (np.ndarray): An array of continuous values representing the power associated
          with each sample.
        - training (bool): A boolean flag that indicates whether the dataset is being used for training.
          When True, data augmentation is applied.
        - device (torch.device): The computing device (CPU, GPU) on which the data will be processed and stored.

        This method initializes the dataset by converting numpy arrays into PyTorch tensors and moving
        them to the specified device. It also prepares the data augmentation process if the dataset is
        initialized for training purposes.
        """
        self.device = device
        self.training = training
        print(power_labels)
        self.power_labels = torch.from_numpy(power_labels).long().to(self.device)  # Convert power values to tensor and move to device
        self.power_values = torch.from_numpy(power_values).float().to(self.device)  # Convert power values to tensor and move to device
        self.n2_labels = torch.from_numpy(n2_labels).long().to(self.device)
        self.isat_labels = torch.from_numpy(isat_labels).long().to(self.device)
        self.augmentation = get_augmentation(data.shape[-2], data.shape[-1])
        self.data = torch.from_numpy(data).float().to(self.device)

        if self.training:
            # Split channels for amplitude and phase
            for i in range(data.shape[0]):
                if self.device.type == 'cpu':
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).numpy().astype(np.float16) # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
                else:
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).cpu().numpy().astype(np.float16)  # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
            self.data.astype(np.float16)
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
        int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a single data sample and its associated labels by index.

        Parameters:
        - idx (int): Index of the data sample to retrieve.

        Returns:
        tuple: A tuple containing the data sample and its labels (data_item, power_value, power_label, n2_label, isat_label),
               where `data_item` is the optical field data, `power_value` is the continuous power value, and
               `power_label`, `n2_label`, `isat_label` are the respective labels for the sample.
        """
        data_item = self.data[idx,:,:,:]
        power_value =self.power_values[idx]
        power_label =self.power_labels[idx]
        n2_label = self.n2_labels[idx]
        isat_label = self.isat_labels[idx]

        return  data_item, power_value, power_label, n2_label, isat_label