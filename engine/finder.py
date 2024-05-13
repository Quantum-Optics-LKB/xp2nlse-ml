#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import sys
import torch
import numpy as np
from engine.loss_plot import plotter
from engine.test import count_parameters_pandas, test_model
from engine.data_prep_for_training import data_split, data_treatment
from engine.training import network_training
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine.model import Inception_ResNetv2

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

def lauch_training(
        numbers: tuple, 
        values: tuple, 
        E: np.ndarray,
        path: str, 
        resolution: int, 
        learning_rate: float, 
        batch_size: int, 
        num_epochs: int, 
        accumulation_steps: int,
        device_number: torch.device
        ) -> None:
    """
    Prepares the dataset and launches the training process for a neural network model.

    Parameters:
    - numbers (tuple): A tuple containing the numbers of n2, power, and isat instances.
    - values (tuple): A tuple containing n2 values, power values, and isat values arrays.
    - path (str): The path where training outputs will be saved.
    - resolution (int): The resolution of the input data.
    - learning_rate (float): The learning rate for the optimizer.
    - batch_size (int): The size of batches for training.
    - num_epochs (int): The number of training epochs.
    - accumulation_steps (int): The number of steps to accumulate gradients before an optimizer step.

    This function iterates over different power values and model configurations to train multiple models. 
    For each model and power configuration, it initializes the model, splits the dataset, prepares DataLoader
    objects, and then trains the model using the `network_training` function. Results, including losses and 
    the trained model, are saved to the specified path.
    """
    device = torch.device(f"cuda:{device_number}")
    n2, powers, alpha, isat = numbers

    number_of_power = len(powers)
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    n2_values, isat_values = values

    n2_values_normalized = (n2_values - np.min(n2_values))/(np.max(n2_values)- np.min(n2_values))
    isat_values_normalized = (isat_values - np.min(isat_values))/(np.max(isat_values) - np.min(isat_values))
    n2 = np.linspace(np.max(n2_values), np.min(n2_values), number_of_n2)
    isat = np.linspace(np.min(isat_values), np.max(isat_values), number_of_isat)

    new_path = f"{path}/training_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}"

    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    else:
        exit()
    
    orig_stdout = sys.stdout
    f = open(f'{new_path}/testing.txt', 'a')
    sys.stdout = f

    print("---- DATA LOADING ----")
    assert E.shape[1] == 2*number_of_power
    assert E.shape[0] == n2_values.shape[0], f"field[0] is {E.shape[0]}, n2_values[0] is {n2_values.shape[0]}"
    assert E.shape[0] == isat_values.shape[0], f"field[0] is {E.shape[0]}, isat_values[0] is {isat_values.shape[0]}"

    print("---- MODEL INITIALIZING ----")
    cnn, optimizer, criterion, scheduler = network_init(learning_rate, E.shape[1], Inception_ResNetv2)
    cnn = cnn.to(device)
    
    print("---- DATA TREATMENT ----")
    train_set, validation_set, test_set = data_split(E,n2_values_normalized,isat_values_normalized, 0.8, 0.1, 0.1)

    train, train_n2_label,train_isat_label = train_set
    validation, validation_n2_label, validation_isat_label = validation_set
    test, test_n2_label, test_isat_label = test_set

    training_train = True
    training_valid = False
    training_test = False

    trainloader = data_treatment(train, train_n2_label,train_isat_label, batch_size, device, training_train)
    validationloader = data_treatment(validation, validation_n2_label, validation_isat_label, batch_size, device, training_valid)
    testloader = data_treatment(test, test_n2_label, test_isat_label, batch_size, device, training_test )

    print("---- MODEL TRAINING ----")
    loss_list, val_loss_list, cnn = network_training(cnn, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, accumulation_steps, device)
    
    print("---- MODEL SAVING ----")
    torch.save(cnn.state_dict(), f'{new_path}/n2_net_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}.pth')

    file_name = f"{new_path}/params.txt"
    classes = {
        'n2': tuple(map(str, n2)),
        'isat' : tuple(map(str, isat))
    }
    with open(file_name, "a") as file:
        file.write(f"resolution: {resolution}\n")
        file.write(f"batch_size: {batch_size}\n")
        file.write(f"accumulator: {accumulation_steps}\n")
        file.write(f"num_of_n2: {number_of_n2}\n")
        file.write(f"num_of_power: {number_of_power}\n")
        file.write(f"num_of_isat: {number_of_isat}\n")
        file.write(f"num_epochs: {num_epochs}\n")
        file.write(f"learning rate: {learning_rate}\n")
        file.write(f"file: {file}\n")
        file.write(f"training_train: {training_train}\n")
        file.write(f"training_valid: {training_valid}\n")
        file.write(f"training_test: {training_test}\n")
        file.write(f"model: {Inception_ResNetv2}\n")
        file.write(f"classes: {classes}\n")

    plotter(loss_list,val_loss_list, new_path, resolution,number_of_n2,number_of_isat)

    print("---- MODEL ANALYSIS ----")
    count_parameters_pandas(cnn)

    print("---- MODEL TESTING ----")
    test_model(testloader, cnn, device)

    sys.stdout = orig_stdout
    f.close()