# Architecture and implementation

This document describes the current code structure and data flow of XP2NLSE-ML.

## Core objects

### EngineDataset

`engine/engine_dataset.py` stores:

- the physical parameter grids;
- flattened labels for every Cartesian-product sample;
- simulation geometry and numerical parameters;
- training hyperparameters;
- the in-memory field tensor.

The field tensor shape is:

```text
N_samples x 2 x H x W
```

where the two channels are density and phase.

## Simulation pipeline

`engine/generate.py::simulation` is the synthetic-data generator.

For every `alpha` and `n2` combination, it:

1. prepares all `Isat` values in one leading batch dimension;
2. creates a Gaussian complex beam;
3. configures the external `NLSE` solver;
4. propagates to the specified cell length;
5. crops the simulation window to the training window;
6. resamples to `resolution_training`;
7. converts the propagated complex field to physical density and phase;
8. writes the result into the preallocated dataset tensor.

Current density conversion:

```text
density = abs(A)^2 * c * epsilon_0 / 2
```

Current phase conversion:

```text
phase = angle(A)
```

## Persistence

`engine/utils.py` defines `save_field_raw` and `load_field_raw`.

Raw storage consists of:

- `.raw`: contiguous binary array bytes;
- `.json`: dtype, shape, and order metadata.

The writer uses chunked writes and atomic `os.replace` finalization.

The generator also currently writes a compressed NumPy `.npz` copy. The manager reload path, however, uses the raw/JSON representation.

## Preprocessing and splitting

`engine/parameter_manager.py` performs image normalization before training:

- density: per-image min shift followed by per-image max division;
- phase: affine mapping from `[-pi, pi]` to `[0, 1]`.

`shuffle_dataset` applies one shared random permutation to fields and all three labels.

`prepare_training` then:

- derives 80/10/10 split boundaries;
- computes target min/max values;
- min-max normalizes each target independently;
- constructs `NetworkDataset` instances.

## Network architecture

`engine/model.py` contains three logical components.

### SubModel

`SubModel` wraps Torchvision ConvNeXt-Tiny:

- `weights=None`;
- first convolution replaced to accept 2 channels;
- classifier replaced with `Identity`.

The network later indexes `features[:, :, 0, 0]`, yielding a 768-dimensional representation.

### Shared representation

The shared MLP is:

```text
768
 -> Linear(2048) -> BatchNorm -> ReLU -> Dropout
 -> Linear(1024) -> BatchNorm -> ReLU -> Dropout
 -> Linear(512)  -> BatchNorm -> ReLU -> Dropout
```

### Output heads

`Isat` and `alpha` each have a one-unit sigmoid head.

`n2` uses `N2CondNet`, which embeds the predicted `Isat` and `alpha` into 512-dimensional representations and concatenates them with the shared 512-dimensional feature vector.

A separate linear layer emits six covariance parameters.

The model returns:

```python
mean_predictions, cov_predictions
```

where `mean_predictions.shape == (B, 3)`.

## Loss

`MultivariateNLLLoss` in `engine/training_manager.py` constructs a lower-triangular Cholesky factor.

Diagonal terms come from softplus-transformed variance parameters with a floor. Off-diagonal terms are tanh-bounded and scaled.

Instead of explicitly inverting covariance matrices, the loss uses:

```python
torch.linalg.solve_triangular(...)
```

The Gaussian NLL is augmented with a large Smooth-L1 term on normalized `n2`.

## Training controller

`prepare_training` initializes:

- the network;
- AdamW;
- `MultivariateNLLLoss`;
- `ReduceLROnPlateau`;
- the target device.

`manage_training` handles:

- checkpoint resume;
- training invocation;
- model-state saving;
- parameter logging;
- target normalization logging;
- loss plotting;
- test evaluation.

The checkpoint stores model, optimizer, scheduler, epoch, loss histories, threshold, learning rate, and accumulator state.

## Training loop

`engine/training.py::network_training` is intended to:

- build training/validation DataLoaders;
- sample augmentation parameters;
- perform gradient accumulation;
- clip gradient norm;
- evaluate validation loss, MAE, and R2;
- adapt the optimizer/accumulator when the n2 MAE threshold is reached;
- checkpoint the best validation loss;
- early stop after patience is exhausted.

Important: the current file contains an unconditional `break` at the start of the epoch body, which prevents this intended loop from executing normally.

## Augmentations

`engine/utils.py` defines:

- density elastic deformation;
- density/phase affine shear and translation;
- random phase shift;
- circular masking.

These are constructed in the current training loop. The present `network_training` implementation does not currently apply the constructed augmentation pipelines to `images` before the forward pass.

## Evaluation

`engine/test.py` computes:

- average normalized MSE;
- average normalized MAE;
- per-target normalized MAE;
- expected-vs-predicted plots.

`plot_prediction` also writes `predictions.csv` and calls a linear-summary plotting helper.

There is currently a duplicate/invalid trailing CSV call in `test_model` using undefined names.

## Experimental inference

`engine/use.py::get_parameters`:

1. reconstructs the expected training-output directory;
2. loads target min/max values from `standardize.txt`;
3. loads the model state dictionary;
4. reads an experimental complex `.npy`;
5. resamples it to training resolution;
6. builds normalized density and phase;
7. predicts normalized parameters;
8. de-normalizes to physical units.

Optional comparison mode performs a new NLSE simulation using the inferred physical values.

## Interpretability

The effective, later implementation in `engine/interpretability.py` provides:

- uniform-ish sampling through normalized 3D label space;
- gradient saliency for all outputs and both channels;
- raw saliency magnitude statistics;
- first/middle/last Conv2d Grad-CAM;
- density-only and phase-only channel ablation for CAM generation.

The module currently contains two full implementations in one file; the later definitions shadow the earlier ones.

## Reproducibility

`set_seed(10)` is called across many modules and configures Python, NumPy, and PyTorch random generators plus deterministic cuDNN settings.

This improves repeatability, although GPU kernels and external NLSE/CuPy behavior may still depend on environment and hardware.
