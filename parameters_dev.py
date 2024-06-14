#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.finder import lauch_training
from engine.use import get_parameters
import cupy as cp
from engine.generate_augment import data_creation, data_augmentation, generate_labels

saving_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
input_alpha_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/alpha.npy"
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/experiment.npy"

device = 0
resolution_in = 2048
window_in = 70e-3
smallest_out_res = 3008
out_pixel_size = 3.76e-6
window_out = out_pixel_size * smallest_out_res
resolution_training = 256

number_of_n2 = 50
number_of_isat = 50
n2 = -np.logspace(-10, -8, number_of_n2) #m/W^2
isat = np.logspace(4, 6, number_of_isat) #W/m^2

delta_z=1e-5
length=20e-2

in_power = 1.05 #W
alpha = 22 #m^-1
waist = 2.3e-3 #m
nl_length = 0#100e-6 #m

expansion=False
generate=False
training=False
learning_rate=0.01
batch_size=50
accumulator=1
num_epochs=50

use=True
plot_generate_compare=True

cameras = resolution_in, window_in, window_out, resolution_training
nlse_settings = n2, in_power, alpha, isat, waist, nl_length, delta_z, length

if expansion or generate or training:
    if generate:
        with cp.cuda.Device(device):
            E = data_creation(nlse_settings, cameras ,saving_path)
    else:
        file = f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}.npy'
        E = np.load(file)

    
    labels = generate_labels(n2, isat)
    
    E, labels = data_augmentation(E, in_power, expansion, saving_path, labels)

    if training:
        print("---- TRAINING ----")
        lauch_training(nlse_settings, labels,E, saving_path, resolution_training, learning_rate, batch_size, num_epochs, accumulator, device)

if use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(exp_image_path, saving_path, resolution_training, nlse_settings, device, cameras, plot_generate_compare)