#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
from torch.utils.data import DataLoader
from engine.seed_settings import set_seed
set_seed(10)

def exam(model, loader,device ):
    print("---- MODEL ANALYSIS ----")
    count_parameters_pandas(model)

    print("---- MODEL TESTING ----")
    test_model(loader, model, device)


def test_model(
        totalloader: DataLoader, 
        net: torch.nn.Module,
        device: int):
    
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mae_loss = torch.nn.L1Loss(reduction='sum')
    
    mse_n2 = 0
    mse_isat = 0
    mae_n2 = 0
    mae_isat = 0
    total = 0

    with torch.no_grad():
        for images, n2_values, isat_values in totalloader:
            images = images.to(device = device, dtype=torch.float32)
            n2_values = n2_values.to(device = device, dtype=torch.float32)
            isat_values = isat_values.to(device = device, dtype=torch.float32)

            outputs_n2, outputs_isat = net(images)
            
            mse_n2 += mse_loss(outputs_n2, n2_values).item()
            mse_isat += mse_loss(outputs_isat, isat_values).item()
            mae_n2 += mae_loss(outputs_n2, n2_values).item()
            mae_isat += mae_loss(outputs_isat, isat_values).item()
            total += n2_values.size(0)
    
    average_mse_n2 = mse_n2 / total
    average_mse_isat = mse_isat / total
    average_mae_n2 = mae_n2 / total
    average_mae_isat = mae_isat / total

    print(f"Average MSE for 'n2': {average_mse_n2:.4f}")
    print(f"Average MSE for 'isat': {average_mse_isat:.4f}")
    print(f"Average MAE for 'n2': {average_mae_n2:.4f}")
    print(f"Average MAE for 'isat': {average_mae_isat:.4f}")

def count_parameters_pandas(
        model: torch.nn.Module
        ) -> int:
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