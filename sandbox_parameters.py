#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from engine.nlse_sandbox import sandbox

saving_path="data"

###Data generation Parameters:
resolution_input_beam = 1024
window_input = 20e-3 #m
output_camera_resolution = 2000
output_pixel_size = 3.45e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2 #m

###Parameter spaces:
n2 = -1e-9 #switch this to an actual range using numpy to launch the real simulation 
isat = 1e6 #switch this to an actual range using numpy to launch the real simulation
alpha = 130 #switch this to an actual range using numpy to launch the real simulation

###Laser Parameters:
input_power = 2.1 #W
waist_input_beam = 1.7e-3#m
non_locality_length = 0 #m

###Find your parameters (n2 and Isat):
exp_image_path="data/field.npy"

sandbox(resolution_input_beam, window_input, window_out, 
        n2, input_power, alpha, isat, waist_input_beam, 
        non_locality_length, cell_length, 
        exp_image_path, saving_path)