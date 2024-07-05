#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
from engine.utils import set_seed
from torch.utils.data import DataLoader
set_seed(10)

def exam(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device 
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
    test_model(loader, model, device)


def test_model(
        totalloader: DataLoader, 
        net: torch.nn.Module,
        device: int
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
    
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mae_loss = torch.nn.L1Loss(reduction='sum')
    
    mse_n2 = 0
    mse_isat = 0
    mse_alpha = 0
    mae_n2 = 0
    mae_isat = 0
    mae_alpha = 0
    total = 0

    with torch.no_grad():
        for images, n2_values, isat_values, alpha_values in totalloader:
            images = images.to(device = device, dtype=torch.float32)
            n2_values = n2_values.to(device = device, dtype=torch.float32)
            isat_values = isat_values.to(device = device, dtype=torch.float32)
            alpha_values = alpha_values.to(device = device, dtype=torch.float32)

            outputs_n2, outputs_isat, outputs_alpha = net(images)
            
            mse_n2 += mse_loss(outputs_n2, n2_values).item()
            mse_isat += mse_loss(outputs_isat, isat_values).item()
            mse_alpha += mse_loss(outputs_alpha, alpha_values).item()
            mae_n2 += mae_loss(outputs_n2, n2_values).item()
            mae_isat += mae_loss(outputs_isat, isat_values).item()
            mae_alpha += mae_loss(outputs_alpha, alpha_values).item()
            total += n2_values.size(0)
    
    average_mse_n2 = mse_n2 / total
    average_mse_isat = mse_isat / total
    average_mse_alpha = mse_alpha / total
    average_mae_n2 = mae_n2 / total
    average_mae_isat = mae_isat / total
    average_mae_alpha = mae_alpha / total

    print(f"Average MSE for 'n2': {average_mse_n2:.4f}")
    print(f"Average MSE for 'isat': {average_mse_isat:.4f}")
    print(f"Average MSE for 'alpha': {average_mse_alpha:.4f}")
    print(f"Average MAE for 'n2': {average_mae_n2:.4f}")
    print(f"Average MAE for 'isat': {average_mae_isat:.4f}")
    print(f"Average MAE for 'alpha': {average_mae_alpha:.4f}")

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