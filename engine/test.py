#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch

def test_model(totalloader, net, device):
    """
    Tests the regression performance of a trained neural network model on a given dataset.

    Parameters:
    - totalloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
    - net (torch.nn.Module): The trained neural network model to be evaluated.

    The function iterates over the test dataset to evaluate the model's performance on regression tasks,
    computing the Mean Squared Error (MSE) and Mean Absolute Error (MAE) for predictions.
    """
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mae_loss = torch.nn.L1Loss(reduction='sum')
    
    mse_n2 = 0
    mse_isat = 0
    mae_n2 = 0
    mae_isat = 0
    total = 0

    with torch.no_grad():
        for images, n2_values, isat_values in totalloader:
            images = images.to(device)
            n2_values = n2_values.to(device)
            isat_values = isat_values.to(device)

            outputs_n2, outputs_isat = net(images)
            
            mse_n2 += mse_loss(outputs_n2, n2_values).item()
            mse_isat += mse_loss(outputs_isat, isat_values).item()
            mae_n2 += mae_loss(outputs_n2, n2_values).item()
            mae_isat += mae_loss(outputs_isat, isat_values).item()
            total += n2_values.size(0)
    
    # Calculating the average losses
    average_mse_n2 = mse_n2 / total
    average_mse_isat = mse_isat / total
    average_mae_n2 = mae_n2 / total
    average_mae_isat = mae_isat / total

    print(f"Average MSE for 'n2': {average_mse_n2:.2f}")
    print(f"Average MSE for 'isat': {average_mse_isat:.2f}")
    print(f"Average MAE for 'n2': {average_mae_n2:.2f}")
    print(f"Average MAE for 'isat': {average_mae_isat:.2f}")

def count_parameters_pandas(model):
    """
    Counts the total number of trainable parameters in a neural network model and prints a summary.

    Parameters:
    - model (torch.nn.Module): The neural network model whose parameters are to be counted.

    This function iterates through all trainable parameters of the given model, summarizing the count
    of parameters per module and the total count of trainable parameters across the model. The summary
    is printed in a tabular format using pandas DataFrame for clear visualization.

    Returns:
    - total_params (int): The total number of trainable parameters in the model.

    Example usage:
    - total_trainable_params = count_parameters_pandas(my_model)
      Prints a table summarizing parameters per module and the total trainable parameters.
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