#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import gc
import cupy as cp
import numpy as np
from tqdm import tqdm
from NLSE import NLSE
from engine.utils import experiment_noise, set_seed
from cupyx.scipy.ndimage import zoom
from scipy.constants import c, epsilon_0
from engine.engine_dataset import EngineDataset
set_seed(10)

def simulation(
    dataset: EngineDataset, 
    ) -> np.ndarray:

    crop = int(0.5*(dataset.window_simulation - dataset.window_training)*dataset.resolution_simulation/dataset.window_simulation)
    alpha = dataset.alpha_values[:, np.newaxis, np.newaxis]

    X = np.linspace(-dataset.window_simulation/ 2, dataset.window_simulation / 2, num=dataset.resolution_simulation, endpoint=False, dtype=np.float64)
    Y = np.linspace(-dataset.window_simulation/ 2, dataset.window_simulation / 2, num=dataset.resolution_simulation, endpoint=False, dtype=np.float64)
    XX, YY = np.meshgrid(X, Y)
    
    beam = np.ones((dataset.number_of_alpha, dataset.resolution_simulation, dataset.resolution_simulation), dtype=np.complex64)*np.exp(-(XX**2 + YY**2) / dataset.waist**2)
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = experiment_noise(beam, poisson_noise_lam, normal_noise_sigma)

    for n2_index, n2_value in tqdm(enumerate(dataset.n2_values),desc=f"NLSE", total=len(dataset.n2_values), unit="n2"):
      for isat_index, isat_value in enumerate(dataset.isat_values):

        simu = NLSE(power=dataset.input_power, alpha=alpha, window=dataset.window_simulation, n2=n2_value, 
                      V=None, L=dataset.length, NX=dataset.resolution_simulation, NY=dataset.resolution_simulation, 
                      Isat=isat_value, nl_length=dataset.non_locality)
        
        if dataset.non_locality != 0:
          simu.nl_profile =  simu.nl_profile[np.newaxis, np.newaxis, :,:]
        simu.delta_z = dataset.delta_z
        A = simu.out_field(beam, z=dataset.length, verbose=False, plot=False, normalize=True, precision="single")

        if crop != 0:
          A = A[:,crop:-crop,crop:-crop]

        zoom_factor = dataset.resolution_training / A.shape[-1]
        A = zoom(cp.asarray(A), (1, zoom_factor, zoom_factor),order=5).get()

        density = np.abs(A)**2 * c * epsilon_0 / 2
        phase = np.angle(A)
        
        start_index = dataset.number_of_alpha * dataset.number_of_isat * n2_index + dataset.number_of_alpha * isat_index
        end_index = dataset.number_of_alpha * dataset.number_of_isat * (n2_index) + dataset.number_of_alpha * (isat_index + 1)

        dataset.field[start_index:end_index,0,:,:] = density
        dataset.field[start_index:end_index,1,:,:] = phase

    dataset.field[:,0,:,:] -= np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
    dataset.field[:,0,:,:] /= np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
    
    if dataset.saving_path != "":
      path = f'{dataset.saving_path}/Es_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}'
      np.save(path, dataset.field)
