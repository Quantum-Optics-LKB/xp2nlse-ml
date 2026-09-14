# Usage guide

## Recommended workflow

### 1. Prepare an experimental complex field

Both sandbox comparison and final inference expect a NumPy `.npy` containing a complex 2D array.

Typical shape:

```text
(camera_height, camera_width)
```

The code resizes it internally to `resolution_training x resolution_training`.

### 2. Tune simulation geometry in the sandbox

Edit `sandbox_parameters.py`.

Key values:

```python
saving_path = "data"

resolution_simulation = 1024
window_simulation = 20e-3
output_camera_resolution = 2000
output_pixel_size = 3.45e-6
window_training = output_pixel_size * output_camera_resolution
length = 20e-2

n2 = -1e-9
isat = 1e6
alpha = 130

input_power = 2.1
waist = 1.7e-3

exp_image_path = "data/field.npy"
```

Run:

```bash
python sandbox_parameters.py
```

The output is `<saving_path>/sandbox.png`.

### 3. Define the training grid

In `parameters.py`, define arrays:

```python
number_of_n2 = 50
number_of_isat = 50
number_of_alpha = 50

n2_values = -np.linspace(1e-9, 1e-10, number_of_n2)
isat_values = np.linspace(5e4, 1e6, number_of_isat)
alpha_values = np.linspace(21, 30, number_of_alpha)
```

Total sample count is:

```text
number_of_n2 * number_of_isat * number_of_alpha
```

Memory can become large quickly because the field tensor is dense `float32` with two image channels.

Approximate in-memory field size:

```text
N_samples * 2 * resolution_training^2 * 4 bytes
```

### 4. Call the manager explicitly

The current `manager` signature includes interpretability parameters without usable scalar defaults. The safest pattern is therefore to pass them explicitly.

Example:

```python
from engine.parameter_manager import manager

manager(
    generate=True,
    training=True,
    create_visual=False,
    use=False,
    plot_generate_compare=False,

    window_training=window_training,
    n2_values=n2_values,
    alpha_values=alpha_values,
    isat_values=isat_values,
    input_power=input_power,
    waist=waist,
    length=length,
    saving_path=saving_path,
    exp_image_path=exp_image_path,
    resolution_simulation=resolution_simulation,
    window_simulation=window_simulation,

    interpretability=False,
    interp_methods=["saliency", "gradcam"],
    interp_num_samples=24,
    interp_output_subdir="interpretability_dataset",
    interp_overlay_alpha=0.5,
    interp_dpi=250,

    device_number=0,
    resolution_training=224,
    non_locality=0,
    delta_z=1e-4,
    learning_rate=1e-4,
    batch_size=128,
    num_epochs=200,
    accumulator=32,
)
```

## Pipeline switches

### generate

When `True`, the manager runs NLSE simulation and writes dataset artifacts.

When `False` but training/visualization/interpretability needs the field, the manager reconstructs the raw dataset filename and loads `.raw + .json`.

### training

When `True`, data are normalized, shuffled, split, and passed to the training pipeline.

Current repository caveat: `network_training()` contains a debug `break` and should be fixed before expecting a normal new training run.

### create_visual

Writes density and phase grids for each alpha value.

Both PNG and SVG files are currently emitted.

For large parameter grids these figures can be extremely large and expensive to create.

### use

Loads trained weights and an experimental complex field, then prints physical estimates:

```text
n2 = ... m^2/W
Isat = ... W/m^2
alpha = ... m^-1
```

### plot_generate_compare

After inference, re-runs the NLSE with the inferred values and writes a simulated-vs-experimental comparison plot.

### interpretability

Requires:

- simulation field loaded in `dataset.field`;
- a matching trained `.pth` file;
- matching parameter-grid metadata.

Supported method strings:

```python
["saliency", "gradcam"]
```

Current effective interpretability code writes SVG output because the PNG filename is overwritten before saving.

## Device selection

`device_number` is used in both:

```python
with cp.cuda.Device(device_number):
```

and:

```python
torch.device(dataset.device_number)
```

Depending on the installed PyTorch version, passing an integer directly to `torch.device` may not be portable. If you encounter device construction errors, use an explicit CUDA device string in the implementation, e.g. `cuda:0`.

## File naming

### Generated dataset

Base:

```text
<saving_path>/Es_w<resolution_training>_n2<Nn2>_isat<NIsat>_alpha<Nalpha>_power<P>
```

Generation currently produces:

```text
<base>.npz
<base>.raw
<base>.json
```

### Training directory

```text
<saving_path>/training_n2<Nn2>_isat<NIsat>_alpha<Nalpha>_power<P>/
```

### Model weights

```text
n2_net_w<resolution_training>_n2<Nn2>_isat<NIsat>_alpha<Nalpha>_power<P>.pth
```

### Target scaling

`standardize.txt` contains six lines in this order:

```text
n2_max
n2_min
isat_max
isat_min
alpha_max
alpha_min
```

### Parameter record

`params.txt` contains physical ranges, simulation geometry, training resolution, epochs, batch size, accumulator, and learning rate.

## Input normalization

Simulation fields are normalized in `parameter_manager.py`.

Density, per image:

```python
density -= density.min()
density /= density.max()
```

Phase:

```python
phase01 = (phase + np.pi) / (2 * np.pi)
```

Experimental inference applies analogous preprocessing.

## Target normalization

Each target is normalized independently:

```text
y01 = (y - ymin) / (ymax - ymin)
```

Predictions use sigmoid outputs, and inference maps them back to physical units using `standardize.txt`.

## Checkpoint resume

If `checkpoint.pth.tar` exists, training management reloads:

- model weights;
- optimizer state;
- scheduler state;
- epoch;
- training and validation loss histories;
- current threshold;
- learning rate;
- gradient-accumulation multiplier.

It then reconstructs AdamW and ReduceLROnPlateau.

## Output metrics

Testing is performed on normalized targets.

Current outputs include:

- per-target MAE;
- overall average MSE;
- overall average MAE;
- true-vs-predicted scatter plots;
- `predictions.csv`.

Interpret these metrics as normalized-domain metrics unless you explicitly convert values back to physical units.

## Interpretability details

### Saliency

For each selected sample and target, the module differentiates the scalar model output with respect to the two-channel input.

Per-channel maps are normalized separately for visualization.

The effective implementation also prints:

- channel gradient sum;
- mean;
- 99th percentile;
- phase/density ratios.

### Grad-CAM

Three convolution depths are automatically selected:

- first Conv2d;
- middle Conv2d;
- last Conv2d.

For each requested output, the effective implementation computes:

- density-only CAM with phase set to zero;
- phase-only CAM with density set to zero.

This is a channel-ablation interpretation, not a standard two-channel Grad-CAM on the untouched input.

## Generated model weights and Git

Do not add large `.pth` checkpoints to ordinary Git history.

The repository now ignores `engine/model_weights/`, but training weights are generated in experiment directories under `saving_path`. Keep these as external experiment artifacts or use Git LFS if you intentionally decide to version them.

## Current issues to fix before a production run

The following are visible in the current code:

- remove the unconditional `break` in `network_training`;
- remove the undefined trailing `save_predictions_to_csv` call in `test_model`;
- deduplicate `interpretability.py`;
- replace type-object defaults in `manager` with actual values;
- decide on one canonical dataset format instead of writing both NPZ and raw/JSON;
- decide whether training augmentations should actually be applied to each batch;
- make device construction explicit and consistent between CuPy and PyTorch.

These changes are code maintenance tasks; they do not change the conceptual workflow documented above.
