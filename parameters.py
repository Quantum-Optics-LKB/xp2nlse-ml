#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import gc
import numpy as np
from engine.finder import launch_training, prep_training
from engine.use import get_parameters
import cupy as cp
from engine.generate_augment import data_creation, data_augmentation, generate_labels

saving_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
device = 1

###Data generation Parameters:
delta_z=1e-4
resolution_input_beam = 2048
window_input = 50e-3
output_camera_resolution = 3008
output_pixel_size = 3.76e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2
resolution_training = 256
generate=False
expansion=False


###Parameter spaces:
number_of_n2 = 20
number_of_isat = 20
n2 = -5*np.logspace(-11, -9, number_of_n2) #m/W^2
isat = np.logspace(4, 5, number_of_isat) #W/m^2

###Laser Parameters:
input_power = 1.05 #W
alpha = 22 #m^-1
waist_input_beam = 2.3e-3 #m
non_locality_length = 0 #m

###Training Parameters:
training=True
learning_rate=0.01
batch_size=33
accumulator=3
num_epochs=100

###Find your parameters (n2 and Isat):
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp/experiment.npy"
use=True
plot_generate_compare=True

cameras = resolution_input_beam, window_input, window_out, resolution_training
nlse_settings = n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length

if expansion or generate or training:
    if generate:
        with cp.cuda.Device(device):
            E = data_creation(nlse_settings, cameras ,saving_path)
    else:
        if expansion:
            file = f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.npy'
            E = np.load(file)
        else:
            E = np.zeros((number_of_n2*number_of_isat, 3, resolution_training, resolution_training), dtype=np.float16)

    labels = generate_labels(n2, isat)
    E, labels = data_augmentation(E, input_power, expansion, saving_path, labels)

    if training:
        print("---- TRAINING ----")
        trainloader, validationloader, testloader, model_settings, new_path = prep_training(nlse_settings, labels, E, saving_path, learning_rate, batch_size, num_epochs, accumulator, device)
        del E
        gc.collect()
        launch_training(trainloader, validationloader, testloader, model_settings, nlse_settings, new_path, resolution_training, labels)

if use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(exp_image_path, saving_path, resolution_training, nlse_settings, device, cameras, plot_generate_compare)