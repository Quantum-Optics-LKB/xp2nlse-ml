#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from io import TextIOWrapper
from torch.utils.data import DataLoader
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from engine.utils import plot_prediction, set_seed


set_seed(10)

def exam(
        model_settings: tuple,
        test_set: NetworkDataset,
        dataset: EngineDataset,
        file: TextIOWrapper,
        ) -> None:
    """
    Evaluates the model by analyzing its parameters and testing it on the provided dataset.

    Parameters:
    -----------
    model_settings : tuple
        A tuple containing the model and related settings (e.g., optimizer, device, etc.).
    test_set : NetworkDataset
        The dataset used for testing the model.
    dataset : EngineDataset
        Contains simulation parameters and metadata, including batch size.
    file : TextIOWrapper
        A writable file object to log results.

    Returns:
    --------
    None
        Logs evaluation results to the file and prints them to the console.
    """
    
    # Unpack model settings
    model, _, _, _, _, _, _, _ = model_settings

    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_set, batch_size=dataset.batch_size, shuffle=True)
    
    print("---- MODEL ANALYSIS ----")
    # Print model parameter statistics
    count_parameters(model, file)

    print("---- MODEL TESTING ----")
    # Perform model testing
    test_model(model_settings, test_loader, dataset, file)


def test_model(
        model_settings: tuple,
        test_loader: DataLoader,
        file: TextIOWrapper,
        ) -> None:
    """
    Tests the model on the test dataset, calculates performance metrics, and logs the results.

    Parameters:
    -----------
    model_settings : tuple
        A tuple containing the model and related settings (e.g., optimizer, device, etc.).
    test_loader : DataLoader
        DataLoader object for the test dataset.
    file : TextIOWrapper
        A writable file object to log results.

    Returns:
    --------
    None
        Logs performance metrics to the file and prints them to the console.
    """
   
    model, _, _, _, device, new_path, _, _ = model_settings

    # Define loss functions
    mse_loss = torch.nn.MSELoss(reduction='mean')
    mae_loss = torch.nn.L1Loss(reduction='mean')
    
    mse = 0
    mae = 0
    predictions = []
    true_labels = []
    
    # Iterate over the test data
    with torch.no_grad():
        for images, n2_labels, isat_labels, alpha_labels in test_loader:
            images = images.to(device = device)
            
            n2_labels = n2_labels.to(device = device)
            isat_labels = isat_labels.to(device = device)
            alpha_labels = alpha_labels.to(device = device)
            labels = torch.cat((n2_labels,isat_labels, alpha_labels), dim=1)

            # Get predictions from the model
            outputs, cov_outputs = model(images)
            
            
            # Collect predictions and true labels
            predictions.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            
            # Compute loss for this batch        
            mse += mse_loss(outputs, labels).item()
            mae += mae_loss(outputs, labels).item()
        
        # Concatenate all predictions and true labels
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
    
    # Calculate average metrics
    average_mse = mse / len(test_loader)
    average_mae = mae / len(test_loader)

    # Log and print the results
    file.write(f"Average MSE: {average_mse:.4f}\n")
    file.write(f"Average MAE: {average_mae:.4f}\n")
    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average MAE: {average_mae:.4f}") 

    # Visualize true vs predicted values  
    plot_prediction(true_labels, predictions, new_path)

def count_parameters(
        model: torch.nn.Module,
        file: TextIOWrapper,
        ) -> int:
    """
    Counts and logs the number of trainable parameters in the model.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to count parameters for.
    file : TextIOWrapper
        A writable file object to log results.

    Returns:
    --------
    int
        The total number of trainable parameters in the model.

    Description:
    ------------
    - Iterates over the parameters of the model.
    - Counts only the trainable parameters.
    - Logs the total count to the file and console.
    """
    data = []
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        data.append([name, params])
        total_params += params

    # Log the total number of trainable parameters
    file.write(f"Total Trainable Params: {total_params}\n")
    print(f"Total Trainable Params: {total_params}")
    return total_params