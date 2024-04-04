#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from data_creator_2D_cupy import normalize_data

def network_init(learning_rate, channels, class_n2, class_power,batch, index_power):
    
    # weights = torch.from_numpy(create_1d_gradient_filter(class_power, index_power))
    cnn = Inception_ResNetv2(in_channels=channels, batch_size=batch,class_n2=class_n2, class_power=class_power)
    weight_decay = 1e-5
    criterion = nn.CrossEntropyLoss()#, nn.CrossEntropyLoss(weight=weights)]#[nn.CrossEntropyLoss(), nn.MSELoss()]
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return cnn, optimizer, criterion, scheduler


path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
resolution = 512
number_of_n2 = 10
number_of_puiss = 10
batch_size = 20
learning_rate = 0.001
puiss_label_clean = np.arange(0, number_of_puiss)
puiss_value_clean = np.linspace(0.02, .5001, number_of_puiss)
puiss_value_normed = (puiss_value_clean - np.min(puiss_value_clean) ) / (np.max(puiss_value_clean) - np.min(puiss_value_clean))

backend = "GPU"
if backend == "GPU":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("---- DATA LOADING ----")

E = np.load("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/field.npy")[:,0,:,:]
E_data = np.zeros((E.shape[0], 2, E.shape[1], E.shape[2]))

E_data[:,0,:, :] = np.abs(E)**2
out_amp = np.abs(E)**2
E[out_amp < 50000] = 0
E_data[:,1,:, :] = np.angle(E)

cut = (E.shape[2] - E.shape[1])//2
E_reshape = E[:,:,cut:E.shape[2] - cut] 
E_resized =  zoom(E_reshape, (1, 512/E_reshape.shape[1],512/E_reshape.shape[2]), order=3)

E_amp_pha_unwrap = np.zeros((E.shape[0], 2, 512, 512))
E_amp_pha_unwrap[:,0,:,:] = np.abs(E_resized)**2
E_amp_pha_unwrap[:,1,:,:] = unwrap_phase(np.angle(E_resized))
E_amp_pha_unwrap = normalize_data(E_amp_pha_unwrap)

E_amp_pha = np.zeros((E.shape[0], 2, 512, 512))
E_amp_pha[:,0,:,:] = np.abs(E_resized)**2
E_amp_pha[:,1,:,:] = np.angle(E_resized)
E_amp_pha = normalize_data(E_amp_pha)

# # Normalize the data
# E_norm = np.roll(normalize_data(E_data), (-15,10), axis=(2,3))
# E_norm[:,:,256,256] = 1


for i in range(0, 10):
    plt.figure(figsize=(10,10 ))
    plt.imshow(E_data[i,0,:,:], cmap='viridis')
    plt.savefig(f"{path}/old_data_{i}_density.png")
    plt.close()

    plt.figure(figsize=(10,10 ))
    plt.imshow(E_data[i,1,:,:], cmap='viridis')
    plt.savefig(f"{path}/old_data_{i}_phase.png")
    plt.close()

# field_data = [E_amp_pha[:,[0],:,:], E_amp_pha, E_amp_pha_unwrap, E_amp_pha[:,[1],:,:],E_amp_pha_unwrap[:,[1],:,:]]
# classes = {
#         'n2': tuple(map(str, np.linspace(-1e-9, -1e-10, number_of_n2))),
#         'power' : tuple(map(str, np.linspace(0.02, .5001, number_of_puiss)))
#     }

# data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap"]
# noise_types = ["noise","no_noise"]

# results_index_accuracy = np.zeros((2, 5))
# results_index_std = np.zeros((2, 5))

# results_n2_accuracy = np.zeros((2, 5))
# results_n2_std = np.zeros((2, 5))
# for noise_index in range(2):

#     for data_types_index in range(5):
#         print(data_types_index)

#         E = field_data[data_types_index]

#         for model_index in [3]:#range(2, 6):
                
#                 if model_index == 2:
#                     from model_resnetv2_1powers import Inception_ResNetv2
#                 elif model_index == 3:
#                     from model_resnetv3_1powers import Inception_ResNetv2
#                 elif model_index == 4:
#                     from model_resnetv4_1powers import Inception_ResNetv2
#                 elif model_index == 5:
#                     from model_resnetv5_1powers import Inception_ResNetv2
                
#                 model_version =  str(Inception_ResNetv2).split('.')[0][8:]
                
#                 result = np.zeros(10)
#                 result_index = np.zeros(10)
#                 for puiss_index in range(10):
                    
#                     stamp = f"{noise_types[noise_index]}_power{str(puiss_value_clean[puiss_index])[:4]}_{data_types[data_types_index]}_{model_version}"
                    
#                     new_path = f"{path}/{stamp}_training"

                    
#                     cnn = Inception_ResNetv2(in_channels=E.shape[1], batch_size=batch_size,class_n2=number_of_n2, class_power=number_of_puiss)
#                     cnn = cnn.to(device)

#                     cnn.load_state_dict(torch.load(f'{new_path}/n2_net_w{resolution}_n2{number_of_n2}_puiss{1}_2D.pth'))

#                     with torch.no_grad():
#                         images = torch.from_numpy(E[[puiss_index],:,:,:]).float().to(device)
#                         labels_power = torch.from_numpy(puiss_label_clean[[puiss_index],]).long().to(device)
#                         powers_values = torch.from_numpy(puiss_value_normed[[puiss_index],np.newaxis]).float().to(device)
                            
                    
#                         outputs_n2 = cnn(images, powers_values)
#                         _, predicted_n2 = torch.max(outputs_n2, 1)

#                         result[puiss_index] = classes['n2'][predicted_n2]
#                         result_index[puiss_index] = predicted_n2
                    
                    
#                 results_index_accuracy[noise_index, data_types_index] = np.mean(result)
#                 results_n2_std[noise_index, data_types_index] = np.std(result)
#                 results_index_accuracy[noise_index, data_types_index] = np.mean(result_index)
#                 results_index_std[noise_index, data_types_index] = np.std(result_index)



# print(results_index_accuracy)
# print(results_index_std)
# print(results_n2_accuracy)
# print(results_n2_std)               