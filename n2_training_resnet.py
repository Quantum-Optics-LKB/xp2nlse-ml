#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch
import numpy as np
from field_dataset import FieldDataset
from torch.utils.data import DataLoader
import torch.optim 

def data_split(E, n2_labels, puiss_labels, puiss_values, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
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

def data_treatment(myset, n2label,puissvalue, puisslabel, batch_size, device, training):
    fieldset = FieldDataset(myset,puissvalue, puisslabel, n2label, training, device)
    fieldloader = DataLoader(fieldset, batch_size=batch_size, shuffle=True)

    return fieldloader

def network_training(net, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, device, accumulation_steps, backend):
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