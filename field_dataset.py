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
    A custom Dataset class for handling optical field data in PyTorch, designed to accommodate
    datasets where each sample includes image data alongside laser power values, and is associated
    with multiple labels for multi-output modeling tasks. This class facilitates the integration
    of complex datasets with PyTorch's data handling and model training pipelines, including support
    for on-the-fly data augmentation during training.

    The class assumes data loading and processing on a specific computing device, which can be specified
    (e.g., CPU or GPU), and is structured to work with multi-dimensional data that represents
    different characteristics of optical fields (such as amplitude and phase) and their corresponding
    physical parameters like laser power and n2 values.

    Attributes:
        device (torch.device): The computing device (CPU or GPU) where the dataset tensors are stored.
        training (bool): Flag indicating whether the dataset is in training mode, which influences
                         whether data augmentation is applied.
        power_label (torch.Tensor): Tensor containing categorical labels for laser power, intended
                                    for classification tasks.
        power (torch.Tensor): Tensor containing continuous power values for each sample, useful
                              for regression tasks.
        n2 (torch.Tensor): Tensor containing labels for the nonlinear refractive index values,
                           used for categorization or parameter estimation.
        data (torch.Tensor): The primary dataset tensor containing the field data, structured as
                             [num_samples, num_channels, height, width].

    Initialization Parameters:
        data (np.ndarray): The dataset containing the optical field data, with shape
                           [num_samples, num_channels, height, width].
        power (np.ndarray): One-dimensional array of laser power values, shape [num_samples].
        power_lab (np.ndarray): One-dimensional array of categorical labels for the laser power,
                                shape [num_samples].
        n2 (np.ndarray): One-dimensional array of n2 values, shape [num_samples].
        training (bool): Indicates if the dataset is initialized in training mode.
        device (torch.device, optional): Computing device for the tensors, default is CPU.

    Methods:
        __len__() -> int: Returns the total number of samples in the dataset.
        __getitem__(idx: int) -> tuple: Retrieves a data sample and its labels by index.

    The dataset class supports indexing for direct access to individual samples, facilitating easy
    batch loading through PyTorch's DataLoader. It is designed to be flexible, allowing for easy adaptation
    to various types of optical field data and their associated parameters.
    """
    
    def __init__(
            self, 
            data: np.ndarray, 
            power: np.ndarray, 
            power_lab: np.ndarray, 
            n2: np.ndarray, 
            training: bool, 
            device=torch.device("cpu")):
        """
        Initializes an instance of the dataset class, preparing field data and corresponding labels for 
        use in machine learning models, specifically designed for PyTorch. This includes setting up 
        tensors for the field data, laser power, n2 values, and applying data augmentation if in training mode.

        Parameters:
        - data (np.ndarray): The dataset containing the optical field data. Expected to be in the format 
        [num_samples, num_channels, height, width], where 'num_channels' typically corresponds to different 
        types of data (e.g., amplitude, phase) associated with each field.
        - power (np.ndarray): A one-dimensional numpy array containing laser power values for each sample in 
        the dataset, with the shape [num_samples].
        - power_lab (np.ndarray): A one-dimensional numpy array of categorical labels for the laser power, 
        intended for classification tasks, with the shape [num_samples].
        - n2 (np.ndarray): A one-dimensional numpy array containing n2 (nonlinear refractive index) values 
        associated with each sample, used for categorization or parameter estimation, with the shape [num_samples].
        - training (bool): Indicates whether the dataset is being initialized for training purposes. When True, 
        data augmentation is applied to the field data.
        - device (torch.device, optional): Specifies the computing device where the tensors will be allocated. 
        Defaults to CPU. Can be set to a CUDA device for GPU acceleration.

        This constructor method converts the input numpy arrays into PyTorch tensors and moves them to the specified 
        computing device. If the instance is marked for training, it applies specified data augmentation techniques 
        to the field data. The method assumes that a function named 'get_augmentation' is defined elsewhere to obtain 
        the augmentation procedure based on the data dimensions.
        """
        self.device = device
        self.training = training
        self.power_label = torch.from_numpy(power_lab).long().to(self.device)  # Convert power values to tensor and move to device
        self.power = torch.from_numpy(power).float().to(self.device)  # Convert power values to tensor and move to device
        self.n2 = torch.from_numpy(n2).long().to(self.device)
        self.augmentation = get_augmentation(data.shape[-1], data.shape[-1])
        self.data = torch.from_numpy(data).float().to(self.device)

        if self.training:
            # Split channels for amplitude and phase
            for i in range(data.shape[0]):
                if self.device.type == 'cpu':
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).numpy()  # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
                else:
                    channels = torch.from_numpy(data)[i,:, :, :].permute(1, 2, 0).cpu().numpy()  # Replace with your actual amplitude channels
                    # Apply augmentations
                    augmented = self.augmentation(image=channels)['image']
                    self.data[i,:,:,:] = torch.from_numpy(augmented).float().permute(2, 0, 1).to(self.device)
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        This method is a required implementation for any PyTorch dataset class. It allows PyTorch's 
        DataLoader to know the size of the dataset for batching purposes.

        Returns:
        int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a data sample and its associated labels by index.

        This method is crucial for PyTorch's DataLoader to iterate over the dataset. It allows direct 
        access to data points and their labels, making it straightforward to load batches of data 
        during the training or evaluation of models.

        Parameters:
        - idx (int): The index of the data item to be retrieved. This should be within the range 
        of 0 to the length of the dataset minus one.

        Returns:
        tuple: A tuple containing the data item and its associated labels. Specifically, the tuple 
        structure is (data_item, power_value, power_labels, n2_label), where:
            - data_item (torch.Tensor): The field data for the given index, maintaining the shape 
            as defined during the dataset initialization (e.g., [num_channels, height, width]).
            - power_value (torch.Tensor): The continuous power value associated with this data item, 
            useful for regression tasks.
            - power_labels (torch.Tensor): The categorical label for the power, intended for 
            classification tasks.
            - n2_label (torch.Tensor): The label for the nonlinear refractive index (n2) value, 
            which could be used for categorization or parameter estimation.

        This method supports indexing into the dataset to retrieve individual samples along with 
        their corresponding labels, formatted as PyTorch tensors. The tensors are already moved to 
        the appropriate device (e.g., CPU, GPU) as specified during the dataset's initialization.
        """

        data_item = self.data[idx,:,:,:]
        power_value =self.power[idx]
        power_labels =self.power_label[idx]
        labels = self.n2[idx]

        return  data_item, power_value, power_labels, labels