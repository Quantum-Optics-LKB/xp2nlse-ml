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

def network_training(net, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, device, accumulation_steps, backend):
    """
    Trains a neural network using given data loaders, optimizer, loss functions, and a learning rate scheduler.
    Supports gradient accumulation and device-specific data handling for efficient training.

    Parameters:
    - net (torch.nn.Module): The neural network model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer to use for adjusting model parameters.
    - criterion (list[torch.nn.Module]): A list of loss functions. The first is used for n2 label predictions,
      and the second for power label predictions.
    - scheduler (torch.optim.lr_scheduler): A learning rate scheduler to adjust the learning rate based on
      validation loss.
    - num_epochs (int): The number of epochs to train the model.
    - trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - validationloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - device (torch.device): The device (CPU/GPU) on which the training is executed.
    - accumulation_steps (int): The number of steps over which gradients are accumulated. Zero means that
      accumulation is not used.
    - backend (str): Indicates whether training is performed on CPU or GPU. Expected values are "CPU" or "GPU".

    Returns:
    tuple: Contains three elements:
        - loss_list (np.ndarray): Array of training loss values for each epoch.
        - val_loss_list (np.ndarray): Array of validation loss values for each epoch.
        - net (torch.nn.Module): The trained neural network model.

    This function orchestrates the training and validation process for a neural network model, implementing
    gradient accumulation if required, and performing device-specific data handling for optimized training.
    It tracks and prints the loss per epoch for both training and validation phases and returns the trained model
    along with the loss metrics.

    Example Usage:
        trained_model, training_losses, validation_losses = network_training(
            net=my_model, optimizer=my_optimizer, criterion=[loss_n2, loss_powers],
            scheduler=my_scheduler, num_epochs=20, trainloader=train_loader,
            validationloader=val_loader, device=torch.device('cuda'), accumulation_steps=4, backend="GPU"
        )
    """
    loss_list = np.zeros(num_epochs)
    val_loss_list = np.zeros(num_epochs)

    for epoch in range(num_epochs):  
        running_loss = 0.0
        net.train()

        for i, (images, powers, powers_labels, labels) in enumerate(trainloader, 0):
            # Process original images
            if backend == "GPU":
                images = images
                powers_labels = powers_labels
                powers_values = torch.from_numpy(powers.cpu().numpy()[:,np.newaxis]).float().to(device)
                n2_labels = labels
            else:
                images = images.to(device) 
                powers_labels = powers_labels.to(device) 
                powers_values = torch.from_numpy(powers.numpy()[:,np.newaxis]).float().to(device)
                n2_labels = labels.to(device)


            # # Forward pass with original images
            outputs_n2, outputs_power = net(images, powers_values)

            loss_n2 = criterion[0](outputs_n2, n2_labels)
            loss_powers = criterion[1](outputs_power, powers_labels)
            loss_n2.backward(retain_graph=True)
            loss_powers.backward(retain_graph=True)

            if accumulation_steps == 0:
                optimizer.step()
            else:
                if (i + 1) % accumulation_steps == 0:
                    for param in net.parameters():
                        param.grad /= accumulation_steps

                    optimizer.step()

                    for param in net.parameters():
                        param.grad.zero_()
            
            running_loss += loss_n2.item() + loss_powers.item()
        
        # Validation loop
        val_running_loss = 0.0
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradients tracked
            for i, (images, powers, powers_labels, labels) in enumerate(validationloader, 0):
                if backend == "GPU":
                    images = images
                    powers_labels = powers_labels
                    powers_values = torch.from_numpy(powers.cpu().numpy()[:,np.newaxis]).float().to(device)
                    n2_labels = labels
                else:
                    images = images.to(device) 
                    powers_labels = powers_labels.to(device) 
                    powers_values = torch.from_numpy(powers.numpy()[:,np.newaxis]).float().to(device)
                    n2_labels = labels.to(device)

                # Forward pass with original images
                outputs_n2, outputs_power = net(images,  powers_values)

                loss_n2 = criterion[0](outputs_n2, n2_labels)
                loss_powers = criterion[1](outputs_power, powers_labels)
                
                val_running_loss += loss_n2.item() + loss_powers.item()

        # Step the scheduler with the validation loss
        scheduler.step(val_running_loss / len(validationloader))

        # Print the epoch's result
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader)}, Validation Loss: {val_running_loss / len(validationloader)}')
        # print(f'Epoch {epoch+1}, Train accuracy: {np.exp(-running_loss / len(trainloader)) *100} %, Validation accuracy: {np.exp(-val_running_loss / len(validationloader))*100} %')

        loss_list[epoch] = running_loss / len(trainloader)
        val_loss_list[epoch] = val_running_loss / len(validationloader)
    
    return loss_list, val_loss_list, net