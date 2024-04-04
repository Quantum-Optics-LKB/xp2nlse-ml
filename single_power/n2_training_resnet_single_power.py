#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch
import numpy as np
import torch.optim

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
            outputs_n2 = net(images, powers_values)

            loss_n2 = criterion(outputs_n2, n2_labels)
            loss_n2.backward()

            if accumulation_steps == 0:
                optimizer.step()
            else:
                if (i + 1) % accumulation_steps == 0:
                    for param in net.parameters():
                        param.grad /= accumulation_steps

                    optimizer.step()

                    for param in net.parameters():
                        param.grad.zero_()
            
            running_loss += loss_n2.item()
        
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
                outputs_n2 = net(images,  powers_values)

                loss_n2 = criterion(outputs_n2, n2_labels)
                
                val_running_loss += loss_n2.item()

        # Step the scheduler with the validation loss
        scheduler.step(val_running_loss / len(validationloader))

        # Print the epoch's result
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader)}, Validation Loss: {val_running_loss / len(validationloader)}')
        # print(f'Epoch {epoch+1}, Train accuracy: {np.exp(-running_loss / len(trainloader)) *100} %, Validation accuracy: {np.exp(-val_running_loss / len(validationloader))*100} %')

        loss_list[epoch] = running_loss / len(trainloader)
        val_loss_list[epoch] = val_running_loss / len(validationloader)
    
    return loss_list, val_loss_list, net