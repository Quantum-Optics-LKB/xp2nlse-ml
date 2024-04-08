# Nonlinear Schrödinger Equation Parameter Estimation with Neural Networks

## Source

The code for this model is adapted from an unofficial PyTorch implementation of Inception-v4 and Inception-ResNet-v2, available at [this repository](https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch). This adaptation is inspired by the paper ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" by Christian Szegedy, et al., 2016](https://doi.org/10.48550/arXiv.1602.07261).

## Overview

This repository introduces a Convolutional Neural Network (CNN) model dedicated to the estimation of parameters within the Nonlinear Schrödinger Equation (NLSE) representing the propagation of a laser beam inside a hot Rubidium vapor cell.

## Workflow

1. **Create Your Setup**
2. **Record Input Beam**
3. **Generate Training Data**
4. **Train the Model**
5. **Deploy the Model**

## Getting Started

To utilize this repository for Nonlinear Schrödinger Equation (NLSE) parameter estimation with Neural Networks, follow these steps to set up and run the model.

### Prerequisites:

#### Installation

First, clone the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/Quantum-Optics-LKB/nlse_parameter_nn.git
cd nlse_parameter_nn
```

#### External Dependencies

- **NumPy**
- **Matplotlib**
- **SciPy**.
- **CuPy**
- **Pillow (PIL)**
- **Pandas**
- **PyTorch**
- **Albumentations**
- **Skimage**


### Usage Command

```plaintext
usage: parameters.py [-h] [--saving_path SAVING_PATH] [--image_path IMAGE_PATH] [--resolution_in RESOLUTION_IN] [--resolution_out RESOLUTION_OUT] [--number_of_n2 NUMBER_OF_N2]
                     [--number_of_power NUMBER_OF_POWER] [--number_of_isat NUMBER_OF_ISAT] [--is_from_image] [--visualize] [--expension] [--generate] [--single_power] [--multiple_power] [--delta_z DELTA_Z]
                     [--trans TRANS] [--length LENGTH] [--factor_window FACTOR_WINDOW] [--training] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--accumulator ACCUMULATOR]
                     [--num_epochs NUM_EPOCHS]
```

### Options

- `-h, --help`: Show this help message and exit.
- `--saving_path SAVING_PATH`: Directory path for saving output files.
- `--image_path IMAGE_PATH`: Path to the input image file. Default is `<saving_path>/exp_data/input_beam.tiff`.
- `--resolution_in RESOLUTION_IN`: Input resolution.
- `--resolution_out RESOLUTION_OUT`: Output resolution.
- `--number_of_n2 NUMBER_OF_N2`: Number of different n2 values.
- `--number_of_power NUMBER_OF_POWER`: Number of different power levels.
- `--number_of_isat NUMBER_OF_ISAT`: Number of different Isat values.
- `--is_from_image`: Specifies if the input is from an image.
- `--visualize`: Enable visualization of the data generation process.
- `--expension`: Enable expansion in the processing steps.
- `--generate`: Enable data generation.
- `--single_power`: Train the model with a single power setting.
- `--multiple_power`: Train the model with multiple power settings.
- `--delta_z DELTA_Z`: Step of the NLSE propagation.
- `--trans TRANS`: Transmission through the optical cell.
- `--length LENGTH`: Length of the optical cell.
- `--factor_window FACTOR_WINDOW`: Factor window applied to the waist.
- `--training`: Enable training mode.
- `--learning_rate LEARNING_RATE`: Learning rate for the training.
- `--batch_size BATCH_SIZE`: Batch size for training.
- `--accumulator ACCUMULATOR`: Number of steps for gradient accumulation.
- `--num_epochs NUM_EPOCHS`: Number of epochs for training.

### Example Command

To start the model with specific options enabled, you could use a command like this:

```bash
./parameters.py --saving_path "/path/to/save" --image_path "/path/to/input_image.tiff" --resolution_in 1024 --resolution_out 512 --number_of_n2 20 --number_of_power 20 --number_of_isat 20 --generate --training --single_power --learning_rate 0.001 --batch_size 16 --num_epochs 100
```