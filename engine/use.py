#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch
import numpy as np
from scipy.ndimage import zoom
from engine.nlse_generator import normalize_data
from engine.model import Inception_ResNetv2
from skimage.restoration import unwrap_phase


def formatting(E_resized, resolution_out, number_of_power):
    E_formatted = np.zeros((1, 2*number_of_power, resolution_out, resolution_out))

    even_indices = np.arange(0, number_of_power*2, 2)
    data_even = np.abs(E_resized)**2

    odd_indices = np.arange(1, number_of_power*2, 2)
    data_odd = unwrap_phase(np.angle(E_resized), rng=0)
    
    E_formatted[0, even_indices, :, :] = data_even

    E_formatted[0, odd_indices, :, :] = data_odd

    E_formatted = normalize_data(E_formatted)
    return E_formatted

def get_parameters(exp_path, saving_path, resolution_out, numbers, device_number):
    n2, in_power, alpha, isat, waist, nl_length = numbers
    
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    min_n2 = n2.min()
    max_n2 = n2.max()
    min_isat = isat.min()
    max_isat = isat.max()

    device = torch.device(f"cuda:{device_number}")
    
    E = zoom(np.load(exp_path), (1, 1, 256/3008, 256/3008))

    cnn = Inception_ResNetv2(in_channels=E.shape[1])
    cnn.to(device)
    cnn.load_state_dict(torch.load(f'{saving_path}/training_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}/n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}.pth'))
    with torch.no_grad():
        images = torch.from_numpy(E).float().to(device)
        outputs_n2, outputs_isat = cnn(images)

    print(f"n2 = {outputs_n2[0,0]*(max_n2 - min_n2) + min_n2:.2e} m^2/W")
    print(f"Isat = {outputs_isat[0,0]*(max_isat - min_isat) + min_isat:.2e} W/m^2")