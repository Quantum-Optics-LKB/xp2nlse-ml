#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from field_dataset import FieldDataset
from torch.utils.data import DataLoader
import torch.optim 

def data_split(
        E: np.ndarray, 
        n2_labels: np.ndarray, 
        puiss_labels: np.ndarray, 
        puiss_values: np.ndarray, 
        train_ratio: float = 0.8, 
        validation_ratio: float = 0.1, 
        test_ratio: float = 0.1
        ) -> tuple:
    """
    Splits the dataset into training, validation, and testing subsets based on specified ratios.

    This function randomly shuffles the dataset and then splits it according to the provided 
    proportions for training, validation, and testing. It ensures that the split is performed 
    across all provided arrays (E, n2_labels, puiss_labels, puiss_values) to maintain consistency 
    between data points and their corresponding labels.

    Parameters:
    - E (np.ndarray): The dataset array containing the features, expected to be of shape 
      [num_samples, num_channels, height, width].
    - n2_labels (np.ndarray): Array of n2 labels for each data sample.
    - puiss_labels (np.ndarray): Array of power (puissance) labels for each data sample.
    - puiss_values (np.ndarray): Array of power values for each data sample.
    - train_ratio (float, optional): Proportion of the dataset to include in the train split. 
      Default is 0.8.
    - validation_ratio (float, optional): Proportion of the dataset to include in the validation 
      split. Default is 0.1.
    - test_ratio (float, optional): Proportion of the dataset to include in the test split. 
      Default is 0.1.

    Returns:
    tuple of tuples: Each tuple contains four np.ndarrays corresponding to one of the splits 
    (train, validation, test). Each of these tuples contain the split's data (E), n2 labels, 
    power labels (puiss_labels), and power values (puiss_values), in that order.

    Example Usage:
        (train_data, train_n2, train_puiss_label, train_puiss_value), 
        (val_data, val_n2, val_puiss_label, val_puiss_value), 
        (test_data, test_n2, test_puiss_label, test_puiss_value) = data_split(E, n2_labels, puiss_labels, puiss_values)

    Raises:
    AssertionError: If the sum of train_ratio, validation_ratio, and test_ratio does not equal 1.
    """
    # Ensure the ratios sum to 1
    assert train_ratio + validation_ratio + test_ratio == 1
    
    np.random.seed(0)
    indices = np.arange(E.shape[0])
    np.random.shuffle(indices)
        
    input = E[indices,:,:,:]
    n2label = n2_labels[indices]
    puisslabel = puiss_labels[indices]
    puissvalues= puiss_values[indices]
    
    # Calculate split indices
    train_index = int(len(indices) * train_ratio)
    validation_index = int(len(indices) * (train_ratio + validation_ratio))
    
    # Split the datasets
    train = input[:train_index]
    validation = input[train_index:validation_index]
    test = input[validation_index:]
    
    train_n2_label = n2label[:train_index]
    validation_n2_label = n2label[train_index:validation_index]
    test_n2_label = n2label[validation_index:]

    train_puiss_label = puisslabel[:train_index]
    validation_puiss_label = puisslabel[train_index:validation_index]
    test_puiss_label = puisslabel[validation_index:]

    train_puiss_value = puissvalues[:train_index]
    validation_puiss_value = puissvalues[train_index:validation_index]
    test_puiss_value = puissvalues[validation_index:]

    return (train, train_n2_label, train_puiss_label,train_puiss_value), (validation, validation_n2_label, validation_puiss_label,validation_puiss_value), (test, test_n2_label, test_puiss_label,test_puiss_value)

def data_treatment(
        myset: np.ndarray, 
        n2label: np.ndarray,
        puissvalue: np.ndarray, 
        puisslabel: np.ndarray, 
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
    - puissvalue (np.ndarray): The continuous values for the laser power associated with each
      sample in the dataset, with shape [num_samples].
    - puisslabel (np.ndarray): The categorical labels for the laser power, intended for
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

    Example Usage:
        field_loader = data_treatment(my_data, my_n2_labels, my_power_values, my_power_labels,
                                      batch_size=32, device=torch.device('cuda'), training=True)
        for batch in field_loader:
            # Process each batch
    """
    fieldset = FieldDataset(myset,puissvalue, puisslabel, n2label, training, device)
    fieldloader = DataLoader(fieldset, batch_size=batch_size, shuffle=True)

    return fieldloader