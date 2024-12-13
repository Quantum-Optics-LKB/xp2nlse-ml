#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.utils import set_seed
from engine.parameter_manager import manager
set_seed(10)

saving_path="data"

###Data generation Parameters:
resolution_simulation = 1024
window_simulation = 20e-3 #m
output_camera_resolution = 2000
output_pixel_size = 3.45e-6 #m
window_training = output_pixel_size * output_camera_resolution #m
length=20e-2 #m
generate = False
create_visual = False

###Parameter spaces:
number_of_n2 = 50
number_of_isat = 50
number_of_alpha = 50

n2_values = -np.linspace(1e-9, 1e-10, number_of_n2)
isat_values = np.linspace(5e4, 1e6, number_of_isat)
alpha_values = np.linspace(21, 30, number_of_alpha)

###Laser Parameters:
input_power = 2.1 #W
waist = 1.7e-3 #m
non_locality = 0 #m

###Training Parameters:
training = False
learning_rate=1e-4
batch_size=128
accumulator=32
num_epochs=200

###Find your parameters (n2, Isat and alpha):
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/Stage_louis/paper/data/E_detuning9_power2.1_1000.npy"
use = True
plot_generate_compare = True

manager(generate=generate, training=training, create_visual=create_visual, use=use, 
        plot_generate_compare=plot_generate_compare, resolution_simulation=resolution_simulation,
          window_simulation=window_simulation, window_training=window_training,
          n2_values=n2_values, input_power=input_power, alpha_values=alpha_values, isat_values=isat_values, 
          waist=waist, non_locality=non_locality, length=length, saving_path=saving_path, 
          exp_image_path=exp_image_path, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, 
          accumulator=accumulator)