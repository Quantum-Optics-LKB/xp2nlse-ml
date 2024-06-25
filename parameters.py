#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.parameter_manager import manager
saving_path="/your/saving/path/"

###Data generation Parameters:
output_camera_resolution = 3008
output_pixel_size = 3.76e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2 #m
generate = False
create_visual = False

###Parameter spaces:
number_of_n2 = 30
number_of_isat = 30
n2 = -5*np.logspace(-10, -9, number_of_n2) #m/W^2 [-5e-10 -> -5e-9]
isat = np.logspace(4, 5, number_of_isat) #W/m^2 [1e4 -> 1e5]

###Laser Parameters:
input_power = 1.05 #W
alpha = 22 #m^-1
waist_input_beam = 2.3e-3 #m

###Training Parameters:
training=True

###Find your parameters (n2 and Isat):
exp_image_path="/your/experiment/path/experiment.npy"
use=True
plot_generate_compare=True

manager(generate, training, create_visual, use, plot_generate_compare,
         window_out, n2, number_of_n2, alpha, isat, number_of_isat, 
         input_power, waist_input_beam, cell_length, 
         saving_path, exp_image_path)