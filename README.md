# Nonlinear Schrödinger Equation Parameter Estimation with Neural Networks

## Source
The code used for the model is adapted from the github repository:
https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch
which is itself an unofficial adaptation from [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, Christian Szegedy, 2016](https://link-url-here.org).

## Overview

This repository contains the implementation of a Convolutional Neural Network (CNN) model designed for estimating parameters in the context of the Nonlinear Schrödinger Equation (NLSE). The project aims to facilitate advanced research and applications by providing a reliable and efficient tool for parameter analysis in complex quantum optical systems.

## Workflow

- **Create your setup**: Create the architecture of your set up
- **Record input beam**: Recover the input beam from your setup. This will allow the data generation to be as close as possible to your data
- **Generate training data**: Provide your input image to the `generate_data_for_training.py` code. It will automatically generate the data using 
[NLSE](https://github.com/Quantum-Optics-LKB/NLSE). You can provide your range in which $n_2$, $I_{sat}$ lie within as well as the power at which the laser is.
- **Train the model**: You can train the model and using the generated data.
- **Use the model at will**: Just provide your output image 

## Getting Started

### Prerequisites


### Installation

Step-by-step instructions on setting up the project environment and installing any necessary dependencies.

```bash
git clone https://github.com/Quantum-Optics-LKB/nlse_parameter_nn.git
cd nlse_parameter_nn