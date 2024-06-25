#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol


import torch
import numpy as np
from torch.utils.data import Dataset
from engine.seed_settings import set_seed
set_seed(10)

class FieldDataset(Dataset):
    def __init__(
            self, 
            data: np.ndarray, 
            n2_values: np.ndarray, 
            isat_values: np.ndarray):
        
        self.n2_values = torch.from_numpy(n2_values).to(torch.float16).unsqueeze(1)
        self.isat_values = torch.from_numpy(isat_values).to(torch.float16).unsqueeze(1)
        self.data = torch.from_numpy(data).to(torch.float16)

    def __len__(
            self) -> int:
        return len(self.data)

    def __getitem__(
            self, 
            idx: int) -> tuple:
        data_item = self.data[idx,:,:,:]
        n2_label = self.n2_values[idx]
        isat_label = self.isat_values[idx]

        return  data_item, n2_label, isat_label