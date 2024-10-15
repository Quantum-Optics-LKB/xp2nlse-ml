#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from io import TextIOWrapper
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
        file: TextIOWrapper,
        ) -> None:
    model, _, _, _, _, _, _ = model_settings
    test_loader = DataLoader(test_set, batch_size=dataset.batch_size, shuffle=True)
    
    print("---- MODEL ANALYSIS ----")
    count_parameters_pandas(model, file)

    print("---- MODEL TESTING ----")
    test_model(model_settings, test_loader, dataset, file)


def test_model(
        model_settings: tuple,
        test_loader: DataLoader,
        dataset: EngineDataset,
        file: TextIOWrapper,
        ) -> None:
   
    model, _, _, _, device, new_path, _ = model_settings

    mse_loss = torch.nn.MSELoss(reduction='mean')
    mae_loss = torch.nn.L1Loss(reduction='mean')
    
    mse = 0
    mae = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, n2_labels, isat_labels, alpha_labels in test_loader:
            images = images.to(device = device)
            
            n2_labels = n2_labels.to(device = device)
            isat_labels = isat_labels.to(device = device)
            alpha_labels = alpha_labels.to(device = device)
            labels = torch.cat((n2_labels,isat_labels, alpha_labels), dim=1)

            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            
            mse += mse_loss(outputs, labels).item()
            mae += mae_loss(outputs, labels).item()
            
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
    
    average_mse = mse / len(test_loader)
    average_mae = mae / len(test_loader)

    file.write(f"Average MSE: {average_mse:.4f}\n")
    file.write(f"Average MAE: {average_mae:.4f}\n")
    print(f"Average MSE: {average_mse:.4f}")
    print(f"Average MAE: {average_mae:.4f}")   
    plot_prediction(true_labels, predictions, new_path)

def count_parameters_pandas(
        model: torch.nn.Module,
        file: TextIOWrapper,
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

    file.write(f"Total Trainable Params: {total_params}\n")
    print(f"Total Trainable Params: {total_params}")
    return total_params