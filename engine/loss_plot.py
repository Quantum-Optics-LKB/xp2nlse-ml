#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
import matplotlib.pyplot as plt

def plotter(
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    path: str, 
    resolution: int, 
    number_of_n2: int, 
    number_of_isat: int,
    number_of_alpha: int,
    ) -> None:
    
    plt.figure(figsize=(10, 6))

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 12

    plt.plot(np.log(y_train), label="Training Loss", marker='^', linestyle='-', color='blue', mfc='lightblue', mec='indigo', markersize=10, mew=2)

    plt.plot(np.log(y_val), label="Validation Loss", marker='^', linestyle='-', color='orange', mfc='#FFEDA0', mec='darkorange', markersize=10, mew=2)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title("Training and Validation Log Losses")
    plt.legend()
    plt.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}.png")
    plt.close()