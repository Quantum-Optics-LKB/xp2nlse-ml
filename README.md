# Nonlinear Schrödinger Equation Parameter Estimation with Neural Networks

## Source

The code for this model is adapted from an unofficial PyTorch implementation of Inception-v4 and Inception-ResNet-v2, available at [this repository](https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch). This adaptation is inspired by the paper ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" by Christian Szegedy, et al., 2016](https://doi.org/10.48550/arXiv.1602.07261).

## Overview

This repository introduces a Convolutional Neural Network (CNN) model dedicated to the estimation of parameters within the Nonlinear Schrödinger Equation (NLSE) representing the propagation of a laser beam inside a hot Rubidium vapor cell.

## Workflow

1. **Create Your Setup**: Design your experimental or simulation setup.
2. **Record Input Beam**: Capture the beam profile used as input for generating training data.
3. **Generate Training Data**: Use `parameters.py` with the `--generate` option to create training datasets.
4. **Train the Model**: Train the CNN using the generated data.
5. **Deploy the Model**: Apply the trained model to new data to estimate parameters.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed. This project requires the following external libraries:

- **NumPy**
- **Matplotlib**
- **SciPy**
- **CuPy**
- **Pillow**
- **Pandas**
- **PyTorch**
- **Albumentations**
- **Skimage**

These dependencies can be installed using mamba:

### Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Quantum-Optics-LKB/nlse_parameter_nn.git
cd nlse_parameter_nn
```

### Usage

The `parameters.py` script controls the data generation, training, and parameter estimation processes:

#### Usage
```plaintext
parameters.py [-h] [--device DEVICE] [--saving_path SAVING_PATH] [--input_image_path INPUT_IMAGE_PATH]
                     [--exp_image_path EXP_IMAGE_PATH] [--output_image_path OUTPUT_IMAGE_PATH]
                     [--resolution_in RESOLUTION_IN] [--resolution_out RESOLUTION_OUT]
                     [--number_of_n2 NUMBER_OF_N2] [--number_of_power NUMBER_OF_POWER] [--number_of_isat NUMBER_OF_ISAT]
                     [--visualize] [--expansion] [--generate] [--expanded]
                     [--delta_z DELTA_Z] [--trans TRANS] [--length LENGTH] [--factor_window FACTOR_WINDOW]
                     [--training] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                     [--accumulator ACCUMULATOR] [--num_epochs NUM_EPOCHS] [--use]
```

#### Options
```plaintext
-h, --help            show this help message and exit
  --device DEVICE       Which GPU you are using
  --saving_path SAVING_PATH
                        Directory path for saving output files.
  --input_image_path INPUT_IMAGE_PATH
                        Path to the input image file. Default is <saving_path>/exp_data/input_beam.tiff
  --exp_image_path EXP_IMAGE_PATH
                        Path to the experiment image file. Default is <saving_path>/exp_data/field_9.npy
  --output_image_path OUTPUT_IMAGE_PATH
                        Path to the input image file. Default is <saving_path>/exp_data/input_beam.tiff
  --resolution_in RESOLUTION_IN
                        Input resolution.
  --resolution_out RESOLUTION_OUT
                        Output resolution.
  --number_of_n2 NUMBER_OF_N2
                        Number of different n2
  --number_of_power NUMBER_OF_POWER
                        Number of different power
  --number_of_isat NUMBER_OF_ISAT
                        Number of different Isat
  --expansion           Enable expansion.
  --generate            Enable generation.
  --expanded            Add if your data was expanded in a previous run
  --delta_z DELTA_Z     Step of the propagation of NLSE
  --trans TRANS         Transmission through the cell
  --length LENGTH       Length of the cell
  --factor_window FACTOR_WINDOW
                        Factor window that is multiplied by the waist
  --training            Enable training.
  --learning_rate LEARNING_RATE
                        Learning rate
  --batch_size BATCH_SIZE
                        Batch size
  --accumulator ACCUMULATOR
                        Number of accumulation steps to allow for gradient accumulation
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --use                 Find your parameters
```

### Example Command

Run the following command to start the model with specific options:

```bash
python parameters.py --saving_path "/path/to/save" --input_image_path "/path/to/input_image.tiff" --resolution_in 1024 --resolution_out 512 --number_of_n2 20 --number_of_power 20 --number_of_isat 20 --generate --expansion --training --learning_rate 0.001 --batch_size 16 --num_epochs 100
```

This command sets up the environment to generate data, expand the dataset if needed, train the model with specific settings, and then use the model to find parameters from new experimental images.