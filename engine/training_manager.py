#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import torch
import numpy as np
import torch.nn as nn
from engine.test import exam
import torch.nn.functional as F
from engine.model import network
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine.utils import data_split, plot_loss, set_seed
from engine.training import load_checkpoint, network_training
set_seed(10)

class MultivariateNLLLoss(nn.Module):
    """
    Computes the negative log-likelihood (NLL) for multivariate Gaussian distributions.

    This loss function takes into account the predicted means, covariance parameters,
    and the true target values to calculate the NLL for a multivariate Gaussian distribution.

    Methods:
    --------
    forward(mean_predictions, cov_params, true_values):
        Computes the loss given the predicted means, covariance parameters, and true values.

    construct_covariance_matrix(cov_params):
        Constructs the covariance matrix from the predicted covariance parameters using Cholesky decomposition.
    """
    def __init__(self):
        super(MultivariateNLLLoss, self).__init__()

    def forward(self, mean_predictions, cov_params, true_values):
        """
        Computes the negative log-likelihood (NLL) for multivariate Gaussian predictions.

        Parameters:
        -----------
        mean_predictions : torch.Tensor
            Predicted mean values, shape [batch_size, 3].
        cov_params : torch.Tensor
            Covariance parameters for constructing the covariance matrix, shape [batch_size, 6].
        true_values : torch.Tensor
            True target values, shape [batch_size, 3].

        Returns:
        --------
        torch.Tensor
            The mean NLL loss over the batch.
        """
        # Construct the covariance matrix from the predicted parameters
        cov_matrix = self.construct_covariance_matrix(cov_params)
        cov_matrix = cov_matrix 
        
        # Compute the inverse and determinant of the covariance matrix for each batch
        cov_inv = torch.inverse(cov_matrix)
        cov_det = torch.det(cov_matrix)
        
        # Calculate the Mahalanobis distance
        diff = (true_values - mean_predictions).unsqueeze(-1)  # Shape [batch_size, 3, 1]
        mahalanobis_term = torch.matmul(torch.matmul(diff.transpose(1, 2), cov_inv), diff)  # Shape [batch_size, 1, 1]
        
        # Log-likelihood (negative log of multivariate Gaussian)
        nll = 0.5 * (mahalanobis_term.squeeze() + torch.log(cov_det) + 3 * torch.log(torch.tensor(2 * torch.pi)))
        
        return torch.mean(nll)  # Return the average loss over the batch

    @staticmethod
    def construct_covariance_matrix(cov_params):
        """
        Constructs a covariance matrix using Cholesky decomposition from the predicted parameters.

        Parameters:
        -----------
        cov_params : torch.Tensor
            Predicted covariance parameters, shape [batch_size, 6].
            - First three values represent the log variances (for n2, Isat, and alpha).
            - Next three values represent the covariances between the parameters.

        Returns:
        --------
        torch.Tensor
            The batch covariance matrix of shape [batch_size, 3, 3], guaranteed to be positive semi-definite.
        """
        batch_size = cov_params.size(0)

        # Extract the predicted log variances and covariances
        var_n2 = F.softplus(cov_params[:, 0]) + 1e-4  # Ensure variance is positive
        var_isat = F.softplus(cov_params[:, 1]) + 1e-4
        var_alpha = F.softplus(cov_params[:, 2]) + 1e-4
        
        # Cholesky decomposition requires constructing a lower triangular matrix L
        L = torch.zeros(batch_size, 3, 3).to(cov_params.device)

        # Fill the lower triangular matrix
        L[:, 0, 0] = torch.sqrt(var_n2)  # sqrt of variance (Cholesky decomposition)
        L[:, 1, 0] = cov_params[:, 3]  # Covariance between n2 and Isat
        L[:, 1, 1] = torch.sqrt(var_isat)  # sqrt of variance
        L[:, 2, 0] = cov_params[:, 4]  # Covariance between n2 and alpha
        L[:, 2, 1] = cov_params[:, 5]  # Covariance between Isat and alpha
        L[:, 2, 2] = torch.sqrt(var_alpha)  # sqrt of variance

        # The covariance matrix is L * L^T (since it must be positive semi-definite)
        cov_matrix = torch.matmul(L, L.transpose(1, 2))
        
        return cov_matrix
    
def prepare_training(
        dataset: EngineDataset,
        ) -> tuple:
    """
    Prepares the dataset for training and initializes the model.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset object containing fields and labels to be processed.

    Returns:
    --------
    tuple
        A tuple containing the training set, validation set, test set, 
        and model settings (model, optimizer, criterion, scheduler, device, and path).
    """
    # Determine the device for computation (e.g., GPU or CPU)
    device = torch.device(dataset.device_number)

    # Create a directory for saving training outputs
    new_path = f"{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}"
    os.makedirs(new_path, exist_ok=True)
    
    print("---- DATA TREATMENT ----")

    # Split the dataset indices into training, validation, and test subsets
    indices = np.arange(len(dataset.n2_labels))
    train_index, validation_index = data_split(indices, 0.8, 0.1, 0.1)

    # Partition the dataset fields into training, validation, and test sets
    training_field = dataset.field[:train_index,:,:,:]
    validation_field = dataset.field[train_index:validation_index,:,:,:]
    test_field = dataset.field[validation_index:,:,:,:]

    # Normalize the labels (n2, alpha, isat) to a [0, 1] range
    dataset.n2_min = np.min(dataset.n2_labels)
    dataset.n2_max = np.max(dataset.n2_labels)

    dataset.alpha_min = np.min(dataset.alpha_labels)
    dataset.alpha_max = np.max(dataset.alpha_labels)

    dataset.isat_min = np.min(dataset.isat_labels)
    dataset.isat_max = np.max(dataset.isat_labels)

    dataset.n2_labels -= dataset.n2_min
    dataset.n2_labels /= dataset.n2_max - dataset.n2_min
    dataset.n2_labels = np.abs(dataset.n2_labels)

    dataset.isat_labels -= dataset.isat_min
    dataset.isat_labels /= dataset.isat_max - dataset.isat_min
    dataset.isat_labels = np.abs(dataset.isat_labels)

    dataset.alpha_labels -= dataset.alpha_min
    dataset.alpha_labels /= dataset.alpha_max - dataset.alpha_min
    dataset.alpha_labels = np.abs(dataset.alpha_labels)

    # Split the standardized labels into training, validation, and test subsets
    training_n2_labels = dataset.n2_labels[:train_index]
    training_isat_labels = dataset.isat_labels[:train_index]
    training_alpha_labels = dataset.alpha_labels[:train_index]

    validation_n2_labels = dataset.n2_labels[train_index:validation_index]
    validation_isat_labels = dataset.isat_labels[train_index:validation_index]
    validation_alpha_labels = dataset.alpha_labels[train_index:validation_index]

    test_n2_labels = dataset.n2_labels[validation_index:]
    test_isat_labels = dataset.isat_labels[validation_index:]
    test_alpha_labels = dataset.alpha_labels[validation_index:]

    # Create `NetworkDataset` objects for training, validation, and test sets
    training_set = NetworkDataset(set=training_field, 
                                  n2_labels=training_n2_labels, 
                                  isat_labels=training_isat_labels, 
                                  alpha_labels=training_alpha_labels)

    validation_set = NetworkDataset(set=validation_field, 
                                    n2_labels=validation_n2_labels, 
                                    isat_labels=validation_isat_labels, 
                                    alpha_labels=validation_alpha_labels)

    test_set = NetworkDataset(set=test_field, 
                              n2_labels=test_n2_labels, 
                              isat_labels=test_isat_labels, 
                              alpha_labels=test_alpha_labels)

    print("---- MODEL INITIALIZING ----")
    # Initialize the machine learning model
    model = network()

    # Set weight decay for regularization
    weight_decay =  1e-5

    # Define the loss function
    criterion = MultivariateNLLLoss()

    # Configure the optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=dataset.learning_rate, weight_decay=weight_decay)

    # Set up a learning rate scheduler to reduce the learning rate on plateau
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, )

    # Transfer the model to the specified device (e.g., GPU or CPU)
    model = model.to(device)
    
    # Bundle the model settings into a tuple
    model_settings = model, optimizer, criterion, scheduler, device, new_path

    # Return the training set, validation set, test set, and model settings
    return training_set, validation_set, test_set, model_settings

def manage_training(
        dataset: EngineDataset,
        training_set: NetworkDataset,
        validation_set: NetworkDataset, 
        test_set: NetworkDataset, 
        model_settings: tuple
        ) -> None:
    """
    Manages the training, saving, and evaluation of a machine learning model.

    Parameters:
    -----------
    dataset : EngineDataset
        The dataset object containing simulation and training data.
    training_set : NetworkDataset
        The training dataset for model training.
    validation_set : NetworkDataset
        The validation dataset for model evaluation during training.
    test_set : NetworkDataset
        The test dataset for final model evaluation.
    model_settings : tuple
        A tuple containing model, optimizer, criterion, scheduler, device, and training path.

    Returns:
    --------
    None
        This function saves the model, logs details, and evaluates performance.
    """

    # Unpack model settings
    model, optimizer, criterion, scheduler, device, new_path = model_settings

    # Open a log file to record test results
    f = open(f'{new_path}/testing.txt', 'a')

    # Attempt to load a previous checkpoint if it exists
    try:
        checkpoint = load_checkpoint(new_path)
        model.load_state_dict(checkpoint['state_dict']) # Load model weights
        optimizer.load_state_dict(checkpoint['optimizer']) # Load optimizer state
        scheduler.load_state_dict(checkpoint['scheduler']) # Load scheduler state
        start_epoch = checkpoint['epoch'] # Resume from last epoch
        loss_list = checkpoint['loss_list'] # Retrieve training loss history
        val_loss_list = checkpoint['val_loss_list'] # Retrieve validation loss history
        loss_threshold = checkpoint["loss_threshold"] # Retrieve loss threshold
        new_learning_rate = checkpoint["learning_rate"] # Retrieve learning rate
        dataset.accumulator = checkpoint["accumulator"] # Retrieve accumulator value

        # Reinitialize optimizer and scheduler
        weight_decay =  1e-5
        optimizer = torch.optim.AdamW(model.parameters(), lr=new_learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    except FileNotFoundError:

        # Initialize new training if no checkpoint is found
        start_epoch = 0
        loss_list = []
        val_loss_list = []
        loss_threshold = 0.05
    
    # Begin training
    print("---- MODEL TRAINING ----")
    model_settings = model, optimizer, criterion, scheduler, device, new_path, start_epoch, loss_threshold
    
    # Train the model and update loss histories
    loss_list, val_loss_list, model = network_training(
        model_settings, dataset, training_set, validation_set, loss_list, val_loss_list, f
        )
    
    # Save the trained model's state
    print("---- MODEL SAVING ----")
    directory_path = f'{new_path}/'
    directory_path += f'n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    torch.save(model.state_dict(), directory_path)

    # Log dataset parameters
    file_name = f"{new_path}/params.txt"
    with open(file_name, "a") as file:
        file.write(f"n2: {dataset.n2_values} \n")
        file.write(f"alpha: {dataset.alpha_values}\n")
        file.write(f"Isat: {dataset.isat_values}\n")
        file.write(f"num_of_n2: {dataset.number_of_n2}\n")
        file.write(f"num_of_isat: {dataset.number_of_isat}\n")
        file.write(f"num_of_alpha: {dataset.number_of_alpha}\n")
        file.write(f"in_power: {dataset.input_power}\n")
        file.write(f"delta_z: {dataset.delta_z}\n")
        file.write(f"cell_length: {dataset.length}\n")
        file.write(f"waist_input_beam: {dataset.waist} m\n")
        file.write(f"non_locality_length: {dataset.non_locality} m\n")
        file.write(f"num_epochs: {dataset.num_epochs}\n")
        file.write(f"resolution training: {dataset.resolution_training}\n")
        file.write(f"resolution simulation: {dataset.resolution_simulation}\n")
        file.write(f"window training: {dataset.window_training}\n")
        file.write(f"window simulation: {dataset.window_simulation}\n")
        file.write(f"accumulator: {dataset.accumulator}\n")
        file.write(f"batch size: {dataset.batch_size}\n")
        file.write(f"learning_rate: {dataset.learning_rate}\n")
    
    # Log standardization parameters
    file_name = f"{new_path}/standardize.txt"
    with open(file_name, "a") as file:
        file.write(f"{dataset.n2_max}\n")
        file.write(f"{dataset.n2_min}\n")
        file.write(f"{dataset.isat_max}\n")
        file.write(f"{dataset.isat_min}\n")
        file.write(f"{dataset.alpha_max}\n")
        file.write(f"{dataset.alpha_min}\n")
    
    # Plot training and validation loss
    plot_loss(loss_list,val_loss_list, new_path, dataset.resolution_training, dataset.number_of_n2, dataset.number_of_isat, dataset.number_of_alpha)

    # Evaluate the model on the test set
    exam(model_settings, test_set, dataset, f)

    # Close the log file
    f.close()   