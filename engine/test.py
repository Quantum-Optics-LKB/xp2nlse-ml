#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from torch.utils.data import DataLoader
from engine.field_dataset import FieldDataset
from engine.utils import plot_prediction, set_seed


set_seed(10)

def exam(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        path: str
        ) -> None:
    """
    Conducts an examination of the given model, including parameter counting and testing.

    Args:
        model (torch.nn.Module): The model to be analyzed and tested.
        loader (DataLoader): DataLoader for loading the test dataset.
        device (int): The device to run the model on (e.g., GPU id).

    Returns:
        None

    Description:
        This function performs two main tasks:
        1. Counts and prints the number of trainable parameters in the model.
        2. Tests the model using the provided DataLoader and prints the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for the model's predictions on `n2`, `isat`, and `alpha` values.
    """
    print("---- MODEL ANALYSIS ----")
    count_parameters_pandas(model)

    print("---- MODEL TESTING ----")
    test_model(loader, model, device, path)


def test_model(
        fieldset: FieldDataset, 
        net: torch.nn.Module,
        device: int,
        path: str
        ) -> None:
    """
    Tests the model on a given dataset and computes evaluation metrics.

    Args:
        totalloader (DataLoader): DataLoader for loading the test dataset.
        net (torch.nn.Module): The model to be tested.
        device (int): The device to run the model on (e.g., GPU id).

    Returns:
        None

    Description:
        This function tests the model using the provided DataLoader and computes the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for the model's predictions on `n2`, `isat`, and `alpha` values. It prints the average MSE and MAE for these predictions.
    """
    
    mse_loss = torch.nn.MSELoss(reduction='mean')
    mae_loss = torch.nn.L1Loss(reduction='mean')
    
    mse = 0
    mae = 0
    mae_n2 = 0.0
    mae_isat = 0.0
    mae_alpha = 0.0
    predictions = []
    true_values = []
    fieldset.flag = "test"
    test_loader = DataLoader(fieldset, batch_size=fieldset.batch_size, shuffle=True)
    with torch.no_grad():
        for images, n2_values, isat_values, alpha_values in test_loader:
            images = images.to(device = device, dtype=torch.float64)
            n2_values = n2_values.to(device = device, dtype=torch.float64)
            isat_values = isat_values.to(device = device, dtype=torch.float64)
            alpha_values = alpha_values.to(device = device, dtype=torch.float64)
            values = torch.cat((n2_values,isat_values, alpha_values), dim=1)

            outputs = net(images)
            predictions.append(outputs.numpy())
            true_values.append(torch.cat((n2_values, isat_values, alpha_values), dim=1).numpy())
            
            mse += mse_loss(outputs, values).item()
            mae += mae_loss(outputs, values).item()

            mae_n2 += torch.mean(torch.abs(outputs[:, 0] - n2_values)).item()
            mae_isat += torch.mean(torch.abs(outputs[:, 1] - isat_values)).item()
            mae_alpha += torch.mean(torch.abs(outputs[:, 2] - alpha_values)).item()
        predictions = np.concatenate(predictions, axis=0)
        true_values = np.concatenate(true_values, axis=0)
    
    average_mse = mse / len(test_loader)
    average_mae = mae / len(test_loader)
    mae_n2 /= len(test_loader)
    mae_isat /= len(test_loader)
    mae_alpha /= len(test_loader)


    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average MAE: {average_mae:.4f}")   
    print(f"MAE n2: {mae_n2}, MAE isat: {mae_isat}, MAE alpha: {mae_alpha}")
    plot_prediction(true_values, predictions, path)

def count_parameters_pandas(
        model: torch.nn.Module
        ) -> int:
    """
    Counts and prints the number of trainable parameters in a given model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: The total number of trainable parameters in the model.

    Description:
        This function iterates over the parameters of the provided model, counts the total number of trainable parameters, and prints the total count. It also returns this count.
    """
    data = []
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        data.append([name, params])
        total_params += params

    print(f"Total Trainable Params: {total_params}")
    return total_params