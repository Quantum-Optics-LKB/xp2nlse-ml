#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import pandas as pd
import torch
import numpy as np
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from engine.nlse_generator import normalize_data
from engine.model import Inception_ResNetv2
from tqdm import tqdm

def reshape_resize(E, resolution_out):
    if E.shape[2] != E.shape[1]:
        cut = (E.shape[2] - E.shape[1])//2
        E_reshape = E[:,0,:,cut:E.shape[2] - cut]
    else:
        E_reshape = E
    
    if resolution_out != E_reshape.shape[1]:
        E_resized = zoom(E_reshape, (1, resolution_out/E_reshape.shape[1],resolution_out/E_reshape.shape[2]), order=3)
    else:
        E_resized = E_reshape
    
    return E_resized

def formatting(E_resized, resolution_out):
    E_formatted = np.zeros((E_resized.shape[0], 2, resolution_out, resolution_out))
    E_formatted[:,0,:,:] = np.abs(E_resized)**2
    E_formatted[:,1,:,:] = np.angle(E_resized)
    E_formatted = normalize_data(E_formatted)

    return E_formatted

def get_parameters(exp_path, saving_path, resolution_out, numbers, device_number):
    number_of_n2, number_of_power, number_of_isat = numbers

    n2 = np.linspace(-1e-11, -1e-10, number_of_n2)
    isat = np.linspace(1e4, 1e6, number_of_isat)

    device = torch.device(f"cuda:{device_number}")
    
    E_experiment = np.load(exp_path)
    E_resized = reshape_resize(E_experiment, resolution_out)
    E = formatting(E_resized, resolution_out)
            
    result_index_n2 = np.zeros(number_of_power)
    result_index_isat = np.zeros(number_of_power)
    
    for power_index in tqdm(range(number_of_power), position=4,desc="Iteration", leave=False):

        cnn = Inception_ResNetv2(in_channels=E.shape[1], class_n2=number_of_n2, class_isat=number_of_isat)
        cnn = cnn.to(device)
        cnn.load_state_dict(torch.load(f'{saving_path}/n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{1}.pth'))

        with torch.no_grad():
            images = torch.from_numpy(E[[power_index],:,:,:]).float().to(device)
                
            outputs_n2, outputs_isat = cnn(images)
            _, predicted_n2 = torch.max(outputs_n2, 1)
            _, predicted_isat = torch.max(outputs_isat, 1)

            result_index_n2[power_index] = predicted_n2
            result_index_isat[power_index] = predicted_isat
    
    mean_index_n2 = int(np.mean(result_index_n2))
    std_index_n2 = np.std(result_index_n2)

    mean_index_isat = int(np.mean(result_index_isat))
    std_index_isat = np.std(result_index_isat)

    print(f"n2 is {n2[mean_index_n2]} ± {std_index_n2/number_of_n2}")
    print(f"Isat is {isat[mean_index_isat]} ± {std_index_isat/number_of_isat}")
