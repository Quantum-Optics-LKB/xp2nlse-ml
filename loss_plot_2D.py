#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

def plotter(
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    path: str, 
    resolution: int, 
    number_of_n2: int, 
    number_of_puiss: int
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
    - number_of_puiss (int): Specifies the number of power levels (puissances) considered, also used 
      in the file naming to give further context.

    This function sets up a matplotlib plot with a dark theme, custom color cycles for lines, and then plots 
    the training and validation losses. It labels the axes, adds a title, a legend, and finally saves the 
    plot to the specified path with a name that incorporates the resolution, n2, and power level details.

    Note:
    - The function does not return any value.
    - It is assumed that the necessary directories already exist at the specified path.
    - The color and style settings are configured for a dark theme; these can be adjusted as per requirements.
    """
    # for dark theme
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = "#00000080"
    plt.rcParams['axes.facecolor'] = "#00000080"
    plt.rcParams['savefig.facecolor'] = "#00000080"
    # plt.rcParams['savefig.transparent'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    # for plots
    tab_colors = ['tab:blue', 'tab:orange', 'forestgreen', 'tab:red', 'tab:purple', 'tab:brown',
                'tab:pink', 'tab:gray', 'tab:olive', 'teal']
    fills = ['lightsteelblue', 'navajowhite', 'darkseagreen', 'lightcoral', 'violet', 'indianred',
            'lavenderblush', 'lightgray', 'darkkhaki', 'darkturquoise']
    edges = tab_colors
    custom_cycler = (cycler(color=tab_colors)) + \
        (cycler(markeredgecolor=edges))+(cycler(markerfacecolor=fills))
    plt.rc('axes', prop_cycle=custom_cycler)
    plt.plot(y_train, label="Training Loss")
    plt.plot(y_val, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend()
    plt.savefig(f"{path}/losses_w{resolution}_n2{number_of_n2}_puiss{number_of_puiss}_2D.png")
    plt.close()
    