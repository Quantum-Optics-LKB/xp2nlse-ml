#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from engine.test import exam
from engine.utils import set_seed
from engine.model import network
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from engine.utils import data_split, plot_loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from engine.training import load_checkpoint, network_training
set_seed(10)

class MultivariateNLLLoss(nn.Module):
    def __init__(self):
        super(MultivariateNLLLoss, self).__init__()

    def forward(self, mean_predictions, cov_params, true_values):
        """
        Forward pass for the Multivariate Negative Log Likelihood (MNLL) loss function.

        Args:
        mean_predictions: Predicted means of shape [batch_size, 3].
        cov_params: Covariance matrix predictions, lower triangular elements of shape [batch_size, 6].
        true_values: Ground truth values for n2, Isat, alpha of shape [batch_size, 3].

        Returns:
        Loss: The negative log likelihood loss.
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
        Construct a covariance matrix from the predicted lower triangular parameters.

        Args:
        cov_params: tensor of shape [batch_size, 6], where 3 are the log variances, and 3 are covariances.

        Returns:
        cov_matrix: the batch covariance matrix of shape [batch_size, 3, 3].
        """
        batch_size = cov_params.size(0)

        # Extract the predicted log variances and covariances
        var_n2 = F.softplus(cov_params[:, 0])  # Ensure variance is positive with softplus
        var_isat = F.softplus(cov_params[:, 1])
        var_alpha = F.softplus(cov_params[:, 2])
        
        cov_n2_isat = cov_params[:, 3]
        cov_n2_alpha = cov_params[:, 4]
        cov_isat_alpha = cov_params[:, 5]

        # Construct the covariance matrix as lower triangular
        cov_matrix = torch.zeros(batch_size, 3, 3).to(cov_params.device)
        
        # Fill diagonal (variances)
        cov_matrix[:, 0, 0] = var_n2
        cov_matrix[:, 1, 1] = var_isat
        cov_matrix[:, 2, 2] = var_alpha
        
        
        # Fill lower triangular (covariances)
        cov_matrix[:, 1, 0] = cov_n2_isat
        cov_matrix[:, 2, 0] = cov_n2_alpha
        cov_matrix[:, 2, 1] = cov_isat_alpha
        
        # Copy to the upper triangular (since covariance matrices are symmetric)
        cov_matrix[:, 0, 1] = cov_matrix[:, 1, 0]
        cov_matrix[:, 0, 2] = cov_matrix[:, 2, 0]
        cov_matrix[:, 1, 2] = cov_matrix[:, 2, 1]
        
        return cov_matrix
    
def prepare_training(
        dataset: EngineDataset,
        ) -> tuple:
    
    device = torch.device(dataset.device_number)
    new_path = f"{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}"
    os.makedirs(new_path, exist_ok=True)
    
    print("---- DATA TREATMENT ----")
    indices = np.arange(len(dataset.n2_labels))
    train_index, validation_index = data_split(indices, 0.8, 0.1, 0.1)

    training_field = dataset.field[:train_index,:,:,:]
    validation_field = dataset.field[train_index:validation_index,:,:,:]
    test_field = dataset.field[:validation_index,:,:,:]

    dataset.n2_min_standard = np.min(dataset.n2_labels)
    dataset.n2_max_standard = np.max(dataset.n2_labels)

    dataset.alpha_min_standard = np.min(dataset.alpha_labels)
    dataset.alpha_max_standard = np.max(dataset.alpha_labels)

    dataset.isat_min_standard = np.min(dataset.isat_labels)
    dataset.isat_max_standard = np.max(dataset.isat_labels)

    dataset.n2_labels -= dataset.n2_min_standard
    dataset.n2_labels /= dataset.n2_max_standard - dataset.n2_min_standard

    dataset.isat_labels -= dataset.isat_min_standard
    dataset.isat_labels /= dataset.isat_max_standard - dataset.isat_min_standard

    dataset.alpha_labels -= dataset.alpha_min_standard
    dataset.alpha_labels /= dataset.alpha_max_standard - dataset.alpha_min_standard

    training_n2_labels = dataset.n2_labels[:train_index]
    training_isat_labels = dataset.isat_labels[:train_index]
    training_alpha_labels = dataset.alpha_labels[:train_index]

    validation_n2_labels = dataset.n2_labels[train_index:validation_index]
    validation_isat_labels = dataset.isat_labels[train_index:validation_index]
    validation_alpha_labels = dataset.alpha_labels[train_index:validation_index]

    test_n2_labels = dataset.n2_labels[validation_index:]
    test_isat_labels = dataset.isat_labels[validation_index:]
    test_alpha_labels = dataset.alpha_labels[validation_index:]

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
    model = network()
    weight_decay =  1e-4
    criterion = MultivariateNLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=dataset.learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=50, steps_per_epoch=len(training_set))
    model = model.to(device)
    
    model_settings = model, optimizer, criterion, scheduler, device, new_path

    return training_set, validation_set, test_set, model_settings

def manage_training(
        dataset: EngineDataset,
        training_set: NetworkDataset,
        validation_set: NetworkDataset, 
        test_set: NetworkDataset, 
        model_settings: tuple
        ) -> None:

    model, optimizer, criterion, scheduler, device, new_path = model_settings

    f = open(f'{new_path}/testing.txt', 'a')

    try:
        checkpoint = load_checkpoint(new_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        val_loss_list = checkpoint['val_loss_list']
        weights = checkpoint['weights']
    except FileNotFoundError:
        start_epoch = 0
        loss_list = []
        val_loss_list = []

        n2_ratio = dataset.n2_min_standard / dataset.n2_max_standard
        isat_ratio = dataset.isat_max_standard / dataset.isat_min_standard
        alpha_ratio = dataset.alpha_max_standard / dataset.alpha_min_standard
        weights = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=False, device=device)
        weights /= weights.max() 
    
    print("---- MODEL TRAINING ----")
    model_settings = model, optimizer, criterion, scheduler, device, new_path, start_epoch, weights
    loss_list, val_loss_list, model = network_training(model_settings, dataset, training_set, validation_set, loss_list, val_loss_list, f)
    
    print("---- MODEL SAVING ----")
    directory_path = f'{new_path}/'
    directory_path += f'n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    torch.save(model.state_dict(), directory_path)

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
    
    file_name = f"{new_path}/standardize.txt"
    with open(file_name, "a") as file:
        file.write(f"{dataset.n2_max_standard}\n")
        file.write(f"{dataset.n2_min_standard}\n")
        file.write(f"{dataset.isat_max_standard}\n")
        file.write(f"{dataset.isat_min_standard}\n")
        file.write(f"{dataset.alpha_max_standard}\n")
        file.write(f"{dataset.alpha_min_standard}\n")
        
    plot_loss(loss_list,val_loss_list, new_path, dataset.resolution_training, dataset.number_of_n2, dataset.number_of_isat, dataset.number_of_alpha)

    exam(model_settings, test_set, dataset, f)

    f.close()   