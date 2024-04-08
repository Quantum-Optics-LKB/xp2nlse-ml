# Nonlinear Schrödinger Equation Parameter Estimation with Neural Networks

## Source

The code for this model is adapted from an unofficial PyTorch implementation of Inception-v4 and Inception-ResNet-v2, available at [this repository](https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch). This adaptation is inspired by the paper "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" by Christian Szegedy, et al., 2016.

## Overview

This repository introduces a Convolutional Neural Network (CNN) model dedicated to the estimation of parameters within the Nonlinear Schrödinger Equation (NLSE) framework. Aimed at advancing research in quantum optical systems, this tool provides an efficient means for parameter analysis, thereby facilitating the exploration of complex dynamics.

## Workflow

1. **Create Your Setup**: Design the architecture of your experimental or simulated setup.
2. **Record Input Beam**: Capture the input beam profile from your setup to closely align the data generation with your experimental conditions.
3. **Generate Training Data**: Use `generate_data_for_training.py` to produce training data. This script leverages [NLSE](https://github.com/Quantum-Optics-LKB/NLSE) for data synthesis, allowing you to specify parameters such as $n_2$, $I_{sat}$, and laser power.
4. **Train the Model**: With the generated data, proceed to train the CNN model for parameter estimation.
5. **Deploy the Model**: Once trained, the model is ready to estimate parameters from new output images.

## Getting Started

### Prerequisites

[List any prerequisites or dependencies needed before installing the project.]

### Installation

Follow these steps to set up the project environment on your local machine:

```bash
git clone https://github.com/Quantum-Optics-LKB/nlse_parameter_nn.git
cd nlse_parameter_nn
```

[Further steps regarding environment setup, library installations, and any post-installation configuration.]

---

### Additional Enhancements

- **Prerequisites**: Detail the software and hardware prerequisites for someone looking to use your project. This could include programming languages, library versions, and hardware specifications.
- **Detailed Installation Guide**: Beyond just cloning the repository, include instructions on setting up a virtual environment, installing dependencies, and any environment variables that need to be set.
- **Usage Examples**: Provide a few examples of how the model can be used, including commands to run scripts or code snippets.
- **Contribution Guidelines**: Encourage community involvement by adding a section for contributions. This can include how to submit issues, propose pullups, and contribute to the code.
- **License**: Specify the license under which your project is released, making it clear how others can use or contribute to your project.