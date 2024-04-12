#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import matplotlib.pyplot as plt
import numpy as np

def plotter(
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    path: str, 
    resolution: int, 
    number_of_n2: int, 
    number_of_isat: int
    ) -> None:
    """
    Generates and saves a plot comparing training and validation losses over epochs with a dark theme. 
    The plot includes customized color cycles for better visual distinction and supports a dark background 
    for aesthetic or presentation purposes.

    Parameters:
    - y_train (np.ndarray): A sequence of training loss values over epochs. Each entry corresponds 
      to the loss at a particular epoch.
    - y_val (np.ndarray): A sequence of validation loss values over epochs, similarly indexed.
    - path (str): The directory path where the plot image will be saved. The file name is constructed 
      using other parameters to reflect the plot's context.
    - resolution (int): The resolution parameter, typically representing the resolution of the input data 
      or model output, used in naming the saved plot file.
    - number_of_n2 (int): Specifies the number of nonlinear refractive index (n2) values considered, 
      used in the file name to provide context about the simulation or model configuration.
    - number_of_isat (int): Specifies the number of isat considered, also used 
      in the file naming to give further context.
    """
    
    plt.figure(figsize=(10, 6))

    # Set font to DejaVu Serif
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 12

    plt.plot(y_train, label="Training Loss", marker='^', linestyle='-', color='blue', mfc='lightblue', mec='indigo', markersize=10, mew=2)

    plt.plot(y_val, label="Validation Loss", marker='^', linestyle='-', color='orange', mfc='#FFEDA0', mec='darkorange', markersize=10, mew=2)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_isat{number_of_isat}.png")
    plt.close()
    