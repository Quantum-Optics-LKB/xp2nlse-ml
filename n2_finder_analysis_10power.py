#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

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
number_of_puiss = 1
num_epochs = 60
learning_rate = 0.001
batch_size = 20
accumulation_steps = 5
n2_label_clean = np.tile(np.arange(0, number_of_n2), number_of_puiss)
n2_label_noisy = np.repeat(n2_label_clean, 23)


backend = "GPU"
if backend == "GPU":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap"]


data_type_results = {}
for data_types_index in range(5):

    model_result = {}
    for model_index in range(2, 6):
        
        model_version =  f"model_resnetv{model_index}"

        noise_dict = {}
        for noisy in ["noise","no_noise"]:

            noise_list = {}    
            stamp = f"{noisy}_{data_types[data_types_index]}_{model_version}"
            new_path = f"{path}/{stamp}_training"

            f = open(f'{new_path}/testing.txt', 'r')

            count = 0
            for line in f.readlines():
                if "TESTING" in line:
                    break
                count += 1
            f.close()

            f = open(f'{new_path}/testing.txt', 'r')
            lines = f.readlines()
            noise_list["accuracy"] = float(lines[count+2].split(" ")[-1].split("%")[0])
            noise_list["index_error"] = float(lines[count+13].split(" ")[-1].split("\n")[0])
            f.close
            noise_dict[noisy] = noise_list 
        model_result[model_version] = noise_dict
    data_type_results[data_types[data_types_index]] = model_result

df = pd.DataFrame.from_dict(data_type_results, orient="columns")
df.to_json(f'{path}/model_analysis_10power.json') 