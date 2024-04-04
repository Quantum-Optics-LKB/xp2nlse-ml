#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from datetime import datetime
import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from loss_plot_2D import plotter
from n2_test_resnet import count_parameters_pandas, test_model_classification
from n2_training_resnet import data_split, data_treatment, network_training
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def network_init(learning_rate, channels, class_n2, class_power,batch):
    
    cnn = Inception_ResNetv2(in_channels=channels, batch_size=batch,class_n2=class_n2, class_power=class_power)
    weight_decay = 1e-5
    criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]#[nn.CrossEntropyLoss(), nn.MSELoss()]
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return cnn, optimizer, criterion, scheduler

path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
resolution = 512
number_of_n2 = 10
number_of_puiss = 10
num_epochs = 60
learning_rate = 0.001
batch_size = 10
accumulation_steps = 10
n2_label_clean = np.tile(np.arange(0, number_of_n2), number_of_puiss)
n2_label_noisy = np.repeat(n2_label_clean, 23)


backend = "GPU"
if backend == "GPU":
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap"]


puiss_label_clean = np.repeat(np.arange(0, number_of_puiss), number_of_n2)
puiss_label_noisy = np.repeat(puiss_label_clean, 23)
puiss_value_clean = np.repeat(np.linspace(0.02, .5, number_of_puiss), number_of_n2)
puiss_value_noisy = (np.repeat(puiss_value_clean, 23) - 0.02 ) / (0.5 - 0.02)

count = 0
for model_index in tqdm(range(4, 6), position=4,desc="Iteration", leave=False):
    if model_index == 2:
        from model_resnetv2 import Inception_ResNetv2
    elif model_index == 3:
        from model_resnetv3 import Inception_ResNetv2
    elif model_index == 4:
        from model_resnetv4 import Inception_ResNetv2
    elif model_index == 5:
        from model_resnetv5 import Inception_ResNetv2

    for data_types_index in range(5):
        for noisy in ["noise","no_noise"]:
            count += 1
            
            model_version =  str(Inception_ResNetv2).split('.')[0][8:]
            stamp = f"{noisy}_{data_types[data_types_index]}_{model_version}"
            
            if not os.path.isdir(f"{path}/{stamp}_training"):
                os.makedirs(f"{path}/{stamp}_training")
            new_path = f"{path}/{stamp}_training"

            orig_stdout = sys.stdout
            f = open(f'{new_path}/testing.txt', 'w')
            sys.stdout = f

            print("---- DATA LOADING ----")

            if data_types_index == 0:
                if noisy == "noise":
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended_noise.npy' # 10 x 1
                else:
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended.npy' # 10 x 1
            elif data_types_index == 1:
                if noisy == "noise":
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended_noise.npy' # 10 x 1
                else:
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended.npy' # 10 x 1
            elif data_types_index == 2:
                if noisy == "noise":
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[2]}_out_extended_noise.npy' # 10 x 1
                else:
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[2]}_out_extended.npy' # 10 x 1
            elif data_types_index == 3:
                if noisy == "noise":
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended_noise.npy' # 10 x 1
                else:
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[1]}_out_extended.npy' # 10 x 1
            
            elif data_types_index == 4:
                if noisy == "noise":
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[2]}_out_extended_noise.npy' # 10 x 1
                else:
                    file = f'{path}/Es_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_{data_types[2]}_out_extended.npy' # 10 x 1
            
            
            if data_types_index == 0:
                E_noisy = np.load(file, 'r')[:,[0],:,:]
                assert E_noisy.shape[1] == 1
            elif data_types_index == 1 or data_types_index == 2:
                E_noisy = np.load(file, 'r')
                assert E_noisy.shape[1] == 2
            elif data_types_index == 3 or data_types_index == 4:
                E_noisy = np.load(file, 'r')[:,[1],:,:]
                assert E_noisy.shape[1] == 1

            assert E_noisy.shape[0] == n2_label_noisy.shape[0]
            assert E_noisy.shape[0] == puiss_label_noisy.shape[0]
            assert E_noisy.shape[0] == puiss_value_noisy.shape[0]

            
            print("---- MODEL INITIALIZING ----")
            cnn, optimizer, criterion, scheduler = network_init(learning_rate, E_noisy.shape[1], number_of_n2,number_of_puiss, batch_size)
            cnn = cnn.to(device)

            print("---- DATA TREATMENT ----")
            train_set, validation_set, test_set = data_split(E_noisy,n2_label_noisy, puiss_label_noisy, puiss_value_noisy, 0.8, 0.1, 0.1)

            train, train_n2_label, train_puiss_label,train_puiss_value = train_set
            validation, validation_n2_label, validation_puiss_label,validation_puiss_value = validation_set
            test, test_n2_label, test_puiss_label,test_puiss_value = test_set

            training_train = True
            training_valid = False
            training_test = False

            trainloader = data_treatment(train, train_n2_label, train_puiss_value,train_puiss_label, batch_size, device, training_train)
            validationloader = data_treatment(validation, validation_n2_label, train_puiss_value, validation_puiss_label, batch_size, device, training_valid)
            testloader = data_treatment(test, test_n2_label, test_puiss_value, test_puiss_label, batch_size, device,training_test )

            print("---- MODEL TRAINING ----")
            loss_list, val_loss_list, cnn = network_training(cnn, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, device,accumulation_steps, backend)

            print("---- MODEL SAVING ----")
            torch.save(cnn.state_dict(), f'{new_path}/n2_net_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_2D.pth')

            file_name = f"{new_path}/params.txt"
            classes = {
                'n2': tuple(map(str, np.linspace(-1e-9, -1e-10, number_of_n2))),
                'power' : tuple(map(str, np.linspace(0.02, .5, number_of_puiss)))
            }
            with open(file_name, "a") as file:
                file.write(f"resolution: {resolution}\n")
                file.write(f"batch_size: {batch_size}\n")
                file.write(f"accumulator: {accumulation_steps}\n")
                file.write(f"num_of_n2: {number_of_n2}\n")
                file.write(f"num_of_power: {1}\n")
                file.write(f"num_epochs: {num_epochs}\n")
                file.write(f"learning rate: {learning_rate}\n")
                file.write(f"file: {file}\n")
                file.write(f"training_train: {training_train}\n")
                file.write(f"training_valid: {training_valid}\n")
                file.write(f"model: {Inception_ResNetv2}\n")
                file.write(f"classes: {classes}\n")
            plotter(loss_list,val_loss_list, new_path, resolution,number_of_n2,number_of_puiss)

            print("---- MODEL ANALYSIS ----")
            count_parameters_pandas(cnn)

            print("---- MODEL TESTING ----")
            test_model_classification(testloader, cnn, classes, device, backend)

            sys.stdout = orig_stdout
            f.close()               