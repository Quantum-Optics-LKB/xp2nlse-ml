#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from torch.utils.data import DataLoader
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from engine.utils import plot_prediction, set_seed


set_seed(10)

def exam(
        model_settings: tuple,
        test_set: NetworkDataset,
        dataset: EngineDataset,
        ) -> None:
    model, _, _, _, device, new_path, _ = model_settings
    test_loader = DataLoader(test_set, batch_size=dataset.batch_size, shuffle=True)
    
    print("---- MODEL ANALYSIS ----")
    count_parameters_pandas(model)

    print("---- MODEL TESTING ----")
    test_model(test_loader, model, device, new_path)


def test_model(
        model_settings: tuple,
        test_loader: DataLoader,
        dataset: EngineDataset,
        ) -> None:
   
    model, _, _, _, device, new_path, _ = model_settings

    mse_loss = torch.nn.MSELoss(reduction='mean')
    mae_loss = torch.nn.L1Loss(reduction='mean')
    
    mse = 0
    mae = 0
    mae_n2 = 0.0
    mae_isat = 0.0
    mae_alpha = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, n2_labels, isat_labels, alpha_labels in test_loader:
            images = images.to(device = device, dtype=torch.float64)
            
            n2_labels = n2_labels.to(device = device, dtype=torch.float64)
            isat_labels = isat_labels.to(device = device, dtype=torch.float64)
            alpha_labels = alpha_labels.to(device = device, dtype=torch.float64)
            labels = torch.cat((n2_labels,isat_labels, alpha_labels), dim=1)

            outputs = model(images)
            predictions.append(outputs.numpy())
            true_labels.append(labels.numpy())
            
            mse += mse_loss(outputs, labels).item()
            mae += mae_loss(outputs, labels).item()

            mae_n2 += torch.mean(torch.abs(outputs[:, 0] - n2_labels)).item()
            mae_isat += torch.mean(torch.abs(outputs[:, 1] - isat_labels)).item()
            mae_alpha += torch.mean(torch.abs(outputs[:, 2] - alpha_labels)).item()
            
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
    
    average_mse = mse / len(test_loader)
    average_mae = mae / len(test_loader)
    mae_n2 /= len(test_loader)
    mae_isat /= len(test_loader)
    mae_alpha /= len(test_loader)


    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average MAE: {average_mae:.4f}")   
    print(f"MAE n2: {mae_n2}, MAE isat: {mae_isat}, MAE alpha: {mae_alpha}")
    plot_prediction(true_labels, predictions, new_path)

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