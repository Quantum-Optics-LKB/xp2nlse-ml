#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from concurrent.futures import ThreadPoolExecutor
import os
import random
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
    shift = random.uniform(0.01,0.05)
    return A.Compose([
        A.MotionBlur(blur_limit=1001,  p=0.5),
        A.GlassBlur(sigma=.5, max_delta=1, iterations=500, p=0.25),
        A.ShiftScaleRotate(shift_limit=shift, scale_limit=0, rotate_limit=0, p=0.2),  # Shift without scale or rotation
        A.Resize(height=original_height, width=original_width, p=1)
    ])

def augment_image(data, augmentation):
    augmented = np.transpose(data, (1, 2, 0)).astype(np.float32)
    augmented = augmentation(image=augmented)["image"]
    return np.transpose(augmented, (2, 0, 1)).astype(np.float16)

def process_images(data, augmentation):
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(augment_image, data[i], augmentation) for i in range(data.shape[0])]
        results = [f.result() for f in futures]
    return np.array(results)

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
            isat_values: np.ndarray,
            training: bool):
        
        self.training = training
        self.n2_values = torch.from_numpy(n2_values).float().unsqueeze(1)
        self.isat_values = torch.from_numpy(isat_values).float().unsqueeze(1)
        self.augmentation = get_augmentation(data.shape[-2], data.shape[-1])
        
        if self.training:
            data = process_images(data, self.augmentation)
        
        self.data = torch.from_numpy(data).float()

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