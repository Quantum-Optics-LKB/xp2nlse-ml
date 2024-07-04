#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.parameter_manager import manager
saving_path="data"

###Data generation Parameters:
output_camera_resolution = 2056
output_pixel_size = 3.45e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2 #m
generate = True
create_visual = False

###Parameter spaces:
number_of_n2 = 2
number_of_isat = 1
number_of_alpha = 1
n2 = -np.logspace(-9, -8, number_of_n2) #m^2/W [-1e-9 -> -1e-8]
isat = 6*np.logspace(3, 5, number_of_isat) #W/m^2 [6e3 -> 6e5]
alpha = np.linspace(25, 45, number_of_alpha) #m^-1 [25 -> 45]

###Laser Parameters:
input_power = 0.570 #W
waist_input_beam = 3.564e-3 #m

###Training Parameters:
training=True

###Find your parameters (n2 and Isat):
exp_image_path="/your/experiment/path/experiment.npy"
use=True
plot_generate_compare=True

manager(generate, training, create_visual, use, plot_generate_compare,
         window_out, n2, number_of_n2, alpha, number_of_alpha, isat, number_of_isat, 
         input_power, waist_input_beam, cell_length, 
         saving_path, exp_image_path, device=1)