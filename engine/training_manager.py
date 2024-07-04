#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from engine.test import exam
from engine.utils import set_seed
from torch.utils.data import DataLoader
from engine.model import Inception_ResNetv2
from engine.field_dataset import FieldDataset
from engine.utils import data_split, plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine.training import load_checkpoint, network_training
set_seed(10)

def create_loaders(
        sets: np.ndarray, 
        batch_size: int
        ) -> DataLoader:
    
    set, n2label, isatlabel, alphalabel= sets 
    fieldset = FieldDataset(set, n2label, isatlabel, alphalabel)
    fieldloader = DataLoader(fieldset, batch_size=batch_size, shuffle=True)

    return fieldloader

def network_init(
        learning_rate: float, 
        channels: int, 
        model: torch.nn.Module
        ) -> tuple:
    """
    Initializes the neural network model, criterion, optimizer, and scheduler.

    Parameters:
    - learning_rate (float): The initial learning rate for the optimizer.
    - channels (int): The number of channels in the input data.
    - model (torch.nn.Module class): The neural network model class to be initialized.

    Returns:
    - cnn (torch.nn.Module): The initialized neural network model.
    - optimizer (torch.optim.Optimizer): The Adam optimizer initialized with the model parameters.
    - criterion (torch.nn.Module): The CrossEntropyLoss criterion for the model's output.
    - scheduler (torch.optim.lr_scheduler): A ReduceLROnPlateau learning rate scheduler.
    """
    cnn = model(channels)
    weight_decay = 1e-5
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    return cnn, optimizer, criterion, scheduler

def prepare_training(
        nlse_settings: tuple, 
        labels: tuple, 
        E: np.ndarray,
        path: str, 
        learning_rate: float, 
        batch_size: int, 
        num_epochs: int, 
        accumulation_steps: int,
        device_number: torch.device
        ) -> tuple:
    
    device = torch.device(f"cuda:{device_number}")
    _, input_power, _, _, _, _, _, _ = nlse_settings

    number_of_n2, n2_values, number_of_isat, isat_values, number_of_alpha, alpha_values = labels
    
    n2_values_normalized = n2_values/np.min(n2_values)
    isat_values_normalized = isat_values/np.max(isat_values)
    alpha_values_normalized = alpha_values/np.max(alpha_values)

    new_path = f"{path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}"

    os.makedirs(new_path, exist_ok=True)

    assert E.shape[1] == 3
    assert E.shape[0] == n2_values.shape[0], f"field[0] is {E.shape[0]}, n2_values[0] is {n2_values.shape[0]}"
    assert E.shape[0] == isat_values.shape[0], f"field[0] is {E.shape[0]}, isat_values[0] is {isat_values.shape[0]}"
    assert E.shape[0] == alpha_values.shape[0], f"field[0] is {E.shape[0]}, alpha_values[0] is {alpha_values.shape[0]}"

    print("---- MODEL INITIALIZING ----")
    cnn, optimizer, criterion, scheduler = network_init(learning_rate, E.shape[1], Inception_ResNetv2)
    cnn = cnn.to(device)
    
    print("---- DATA TREATMENT ----")
    train, validation, test = data_split(E, n2_values_normalized, isat_values_normalized, alpha_values_normalized, 0.8, 0.1, 0.1)


    trainloader = create_loaders(train, batch_size)
    validationloader = create_loaders(validation, batch_size)
    testloader = create_loaders(test, batch_size )

    model_settings = cnn, optimizer, criterion, scheduler, num_epochs, accumulation_steps, device

    return trainloader, validationloader, testloader, model_settings, new_path

def manage_training(
        trainloader: DataLoader,
        validationloader: DataLoader,
        testloader: DataLoader,
        model_settings: tuple, 
        nlse_settings: tuple, 
        new_path: str, 
        resolution_training: int,
        labels: tuple
        ) -> None:

    number_of_n2, _, number_of_isat, _, number_of_alpha, _ = labels
    n2, in_power, alpha, isat, _, _, _, _ = nlse_settings
    cnn, optimizer, criterion, scheduler, num_epochs, accumulation_steps, device = model_settings

    orig_stdout = sys.stdout
    f = open(f'{new_path}/testing.txt', 'a')
    sys.stdout = f

    try:
        checkpoint = load_checkpoint(new_path)
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        val_loss_list = checkpoint['val_loss_list']
    except FileNotFoundError:
        start_epoch = 0
        loss_list = []
        val_loss_list = []
    
    print("---- MODEL TRAINING ----")
    loss_list, val_loss_list, cnn = network_training(cnn, optimizer, criterion, scheduler,start_epoch, num_epochs, trainloader, validationloader, accumulation_steps, device, new_path, loss_list, val_loss_list)
    
    print("---- MODEL SAVING ----")
    directory_path = f'{new_path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}/'
    directory_path += f'n2_net_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}.pth'
    torch.save(cnn.state_dict(), f'{new_path}/n2_net_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}.pth')

    file_name = f"{new_path}/params.txt"
    classes = {
        'n2': tuple(map(str, n2)),
        'isat' : tuple(map(str, isat)),
        'alpha' : tuple(map(str, alpha))
    }
    with open(file_name, "a") as file:
        file.write(f"resolution: {resolution_training}\n")
        file.write(f"accumulator: {accumulation_steps}\n")
        file.write(f"num_of_n2: {number_of_n2}\n")
        file.write(f"in_power: {in_power}\n")
        file.write(f"num_of_isat: {number_of_isat}\n")
        file.write(f"num_of_alpha: {number_of_alpha}\n")
        file.write(f"num_epochs: {num_epochs}\n")
        file.write(f"file: {file}\n")
        file.write(f"model: {Inception_ResNetv2}\n")
        file.write(f"classes: {classes}\n")

    plot_loss(loss_list,val_loss_list, new_path, resolution_training,number_of_n2,number_of_isat, number_of_alpha)

    exam(cnn, testloader, device)

    sys.stdout = orig_stdout
    f.close()   