#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from engine.field_dataset import FieldDataset
from torch.utils.data import DataLoader
import torch.optim 

def data_split(
        E: np.ndarray, 
        n2_labels: np.ndarray, 
        isat_labels: np.ndarray,
        train_ratio: float = 0.8, 
        validation_ratio: float = 0.1, 
        test_ratio: float = 0.1
        ) -> tuple:
    """
    Splits the dataset into training, validation, and testing subsets based on specified ratios.

    This function randomly shuffles the dataset and then splits it according to the provided 
    proportions for training, validation, and testing. It ensures that the split is performed 
    across all provided arrays (E, n2_labels, power_labels, power_values) to maintain consistency 
    between data points and their corresponding labels.

    Parameters:
    - E (np.ndarray): The dataset array containing the features, expected to be of shape 
      [num_samples, num_channels, height, width].
    - n2_labels (np.ndarray): Array of n2 labels for each data sample.
    - isat_labels (np.ndarray): Array of isat labels for each data sample.
    - train_ratio (float, optional): Proportion of the dataset to include in the train split. 
      Default is 0.8.
    - validation_ratio (float, optional): Proportion of the dataset to include in the validation 
      split. Default is 0.1.
    - test_ratio (float, optional): Proportion of the dataset to include in the test split. 
      Default is 0.1.

    Returns:
    tuple of tuples: Each tuple contains four np.ndarrays corresponding to one of the splits 
    (train, validation, test). Each of these tuples contain the split's data (E), n2 labels, 
    power labels (power_labels), and power values (power_values), in that order.

    Example Usage:
        (train_data, train_n2, train_power_label, train_power_value), 
        (val_data, val_n2, val_power_label, val_power_value), 
        (test_data, test_n2, test_power_label, test_power_value) = data_split(E, n2_labels, power_labels, power_values)

    Raises:
    AssertionError: If the sum of train_ratio, validation_ratio, and test_ratio does not equal 1.
    """
    # Ensure the ratios sum to 1
    assert train_ratio + validation_ratio + test_ratio == 1
    
    np.random.seed(0)
    indices = np.arange(E.shape[0])
    np.random.shuffle(indices)
    
    # Calculate split indices
    train_index = int(len(indices) * train_ratio)
    validation_index = int(len(indices) * (train_ratio + validation_ratio))

    training_indices = indices[:train_index]
    validation_indices = indices[train_index:validation_index]
    test_indices = indices[validation_index:]
    
    return (E[training_indices], n2_labels[training_indices],isat_labels[training_indices]), (E[validation_indices], n2_labels[validation_indices],isat_labels[validation_indices]),(E[test_indices], n2_labels[test_indices],isat_labels[test_indices])

def data_treatment(
        myset: np.ndarray, 
        n2label: np.ndarray,
        isatlabel: np.ndarray,
        batch_size: int, 
        device: torch.device,
        training: bool):
    """
    Prepares a PyTorch DataLoader for the given dataset, facilitating batch-wise data loading
    for neural network training or evaluation. This function encapsulates the dataset within
    a custom FieldDataset class and then provides a DataLoader for efficient and scalable data
    handling.

    Parameters:
    - myset (np.ndarray): The dataset containing the optical field data, structured as
      [num_samples, num_channels, height, width].
    - n2label (np.ndarray): The labels for the nonlinear refractive index (n2), with shape
      [num_samples].
    - isatlabel (np.ndarray): The categorical labels for the saturation intensity, intended for
      classification tasks, with shape [num_samples].
    - batch_size (int): The number of samples to load per batch.
    - device (torch.device): The computing device (CPU or GPU) where the dataset tensors
      are stored and operations are performed.
    - training (bool): A flag indicating whether the dataset is being used for training
      or evaluation/testing. This influences data augmentation and other preprocessing steps.

    Returns:
    torch.utils.data.DataLoader: A DataLoader object ready to be used in a training or
    evaluation loop, providing batches of data and corresponding labels.

    The returned DataLoader handles the shuffling and batching of the data, making it suitable
    for direct use in training loops or for evaluating model performance. The custom FieldDataset
    class is used to encapsulate the data and perform any necessary preprocessing, such as data
    augmentation if the dataset is marked for training use.
    """
    fieldset = FieldDataset(myset, n2label, isatlabel, training)
    fieldloader = DataLoader(fieldset, batch_size=batch_size, shuffle=True)

    return fieldloader