#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

"""
engine/interpretability.py

Consistent interpretability module (dataset-driven, no experimental data).

Requirements implemented (per your spec):

(1) BIG SALIENCY FIGURE (one per sample):
    Row 0: input density + input phase (robust display scaling)
    Row 1: |∂ŷ_n2/∂density| + |∂ŷ_n2/∂phase|
    Row 2: |∂ŷ_Isat/∂density| + |∂ŷ_Isat/∂phase|
    Row 3: |∂ŷ_alpha/∂density| + |∂ŷ_alpha/∂phase|
    - independent colorbar per subplot
    - prints which triplet is being processed

(2) DEPTH GRAD-CAM FIGURE (one per sample AND per output parameter):
    You asked for a single figure per output parameter where the context is repeated per depth.
    Layout (7 rows x 3 cols) — columns correspond to depths: start / middle / end Conv2d
      Row 0 (2 cols): input density + input phase (robust scaling) [shown once, spanning width]
      Rows 1-2: depth 1 context (density repeated in row 1 across 3 cols, phase repeated row 2)
      Rows 3-4: depth 2 context (density repeated row 3, phase repeated row 4)
      Rows 5-6: depth 3 context (density repeated row 5, phase repeated row 6)
    And **CAM overlays** are rendered ON TOP of the density/phase in each depth block:
      - Density rows show density with CAM overlay
      - Phase rows show phase with CAM overlay
    - each subplot has its own CAM colorbar (separate from base image scaling)
    - robust scaling for the base image (density/phase) so it doesn't look "almost zero"
    - prints which output parameter is being processed (n2 / Isat / alpha) and which sample

Notes:
- Labels are normalized ONLY inside prepare_training(dataset), same as your training.
- Sampling is uniform in normalized label space [0,1]^3 by nearest-to-grid selection.
- Model is loaded ONLY from the .pth you save in manage_training (no checkpoint dependency).
- dataset.field MUST be loaded before calling run_interpretability_from_dataset(dataset, cfg).

Usage from parameter_manager.py after dataset.field is loaded:
    from engine.interpretability import run_interpretability_from_dataset, InterpretabilityConfig
    if interpretability:
        cfg = InterpretabilityConfig(num_samples=24, sample_split="test", out_dir="", dpi=250, verbose=True)
        run_interpretability_from_dataset(dataset, cfg)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Optional, List, Any

from mpl_toolkits.axes_grid1 import make_axes_locatable

from engine.training_manager import prepare_training
from engine.model import network
from engine.utils import set_seed

set_seed(10)


# =========================
# Config
# =========================

@dataclass
class InterpretabilityConfig:
    num_samples: int = 24
    out_dir: str = ""                       # default: <training_dir>/interpretability_dataset
    sample_split: str = "test"              # "train" | "val" | "test"
    methods: Tuple[str, ...] = ("saliency", "gradcam")
    param_names: Tuple[str, ...] = ("n2", "Isat", "alpha")
    grid_side: Optional[int] = None         # None -> inferred from num_samples
    dpi: int = 250
    verbose: bool = True

    # CAM overlay visual parameters
    cam_alpha: float = 0.50                 # opacity of CAM overlay
    cam_cmap: str = "jet"                   # CAM colormap
    density_cmap: str = "gray"
    phase_cmap: str = "twilight"


# =========================
# Paths (use ONLY what you save)
# =========================

def _training_dir(dataset) -> str:
    return (
        f"{dataset.saving_path}/training_n2{dataset.number_of_n2}"
        f"_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}"
        f"_power{dataset.input_power:.2f}"
    )

def _weights_path(dataset) -> str:
    return (
        f"{_training_dir(dataset)}/"
        f"n2_net_w{dataset.resolution_training}"
        f"_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}"
        f"_power{dataset.input_power:.2f}.pth"
    )


# =========================
# Small utils
# =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def add_colorbar(fig, ax, im, size="4%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    fig.colorbar(im, cax=cax)

def nan_safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)

def robust_limits(a: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    """
    Robust display scaling for base images (density/phase).
    """
    a = nan_safe(a)
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin) < 1e-12:
        vmin = float(np.min(a))
        vmax = float(np.max(a))
    if (vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax

def norm01_np(a: np.ndarray) -> np.ndarray:
    a = nan_safe(a)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def get_mu_tensor(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("mu", "mean", "pred", "y", "output"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")


# =========================
# Model loading
# =========================

def load_model_from_pth(dataset) -> nn.Module:
    device = torch.device(dataset.device_number)
    model = network().to(device)

    wpath = _weights_path(dataset)
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Model weights not found: {wpath}")

    state = torch.load(wpath, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================
# Saliency
# =========================

def compute_saliency(model: nn.Module, x: torch.Tensor, param_index: int) -> torch.Tensor:
    """
    x: (1,2,H,W)
    returns (2,H,W) saliency in [0,1] per-channel for visualization
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    mu = get_mu_tensor(model(x))
    if mu.ndim != 2 or mu.shape[1] < 3:
        raise ValueError(f"Expected mu shape (B,3). Got {tuple(mu.shape)}")

    target = mu[0, param_index]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    target.backward(retain_graph=False)

    sal = x.grad.detach().abs()[0]  # (2,H,W)

    raw_sum = sal.view(2, -1).sum(dim=1)          # [2]
    raw_mean = sal.view(2, -1).mean(dim=1)        # [2]
    raw_p99 = sal.view(2, -1).quantile(0.99, dim=1)

    out = []
    for c in range(sal.shape[0]):
        s = sal[c]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        out.append(s)
    return torch.stack(out, dim=0)


# =========================
# Grad-CAM (start/mid/end conv2d)
# =========================

def list_conv2d_layers(model: nn.Module) -> List[tuple]:
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((name, m))
    return convs

def pick_start_mid_end_convs(model: nn.Module) -> List[tuple]:
    convs = list_conv2d_layers(model)
    if len(convs) < 3:
        raise RuntimeError(f"Need >=3 Conv2d layers for start/mid/end. Found {len(convs)}")
    return [convs[0], convs[len(convs) // 2], convs[-1]]

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self, x: torch.Tensor, param_index: int) -> torch.Tensor:
        """
        returns CAM (H,W) in [0,1]
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        mu = get_mu_tensor(self.model(x))
        target = mu[0, param_index]
        target.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients
        if A is None or dA is None:
            raise RuntimeError("Grad-CAM missing activations/gradients. Check chosen layer.")

        weights = dA.mean(dim=(2, 3), keepdim=True)         # (1,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)        # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def build_gradcam_engines(model: nn.Module) -> List[tuple]:
    layers = pick_start_mid_end_convs(model)
    return [(lname, GradCAM(model, layer)) for lname, layer in layers]


# =========================
# Sampling uniformly in label space [0,1]^3 (labels normalized in prepare_training)
# =========================

def _labels01_from_networkdataset(subset) -> np.ndarray:
    n = len(subset)
    y = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        item = subset[i]
        y[i, 0] = float(item[1].item())
        y[i, 1] = float(item[2].item())
        y[i, 2] = float(item[3].item())
    return y

def sample_uniform_triplets_indices(labels01: np.ndarray, num_samples: int, grid_side: Optional[int] = None) -> List[int]:
    N = labels01.shape[0]
    if N == 0:
        return []
    K = int(min(num_samples, N))
    if grid_side is None:
        grid_side = max(2, int(round(K ** (1.0 / 3.0))))

    lin = np.linspace(0.0, 1.0, grid_side)
    targets = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)

    if targets.shape[0] > K:
        pick = np.linspace(0, targets.shape[0] - 1, K).round().astype(int)
        targets = targets[pick]

    chosen: List[int] = []
    used = np.zeros(N, dtype=bool)

    for t in targets:
        diff = labels01 - t[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        dist2[used] = np.inf
        j = int(np.argmin(dist2))
        if np.isfinite(dist2[j]):
            chosen.append(j)
            used[j] = True
        if len(chosen) >= K:
            break

    if len(chosen) < K:
        rem = np.where(~used)[0]
        np.random.shuffle(rem)
        chosen.extend(rem[: (K - len(chosen))].tolist())

    return chosen[:K]

def denormalize_triplet(dataset, y01: np.ndarray) -> np.ndarray:
    n2 = y01[0] * (dataset.n2_max - dataset.n2_min) + dataset.n2_min
    isat = y01[1] * (dataset.isat_max - dataset.isat_min) + dataset.isat_min
    alpha = y01[2] * (dataset.alpha_max - dataset.alpha_min) + dataset.alpha_min
    return np.array([n2, isat, alpha], dtype=np.float64)


# =========================
# Plotting
# =========================

def plot_saliency_all_outputs_big(
    x_np: np.ndarray,                 # (2,H,W)
    sal_n2: np.ndarray,               # (2,H,W) in [0,1]
    sal_isat: np.ndarray,             # (2,H,W) in [0,1]
    sal_alpha: np.ndarray,            # (2,H,W) in [0,1]
    triplet_str: str,
    save_path: str,
    cfg: InterpretabilityConfig,
) -> None:
    density = nan_safe(x_np[0])
    phase = nan_safe(x_np[1])

    vmin_d, vmax_d = robust_limits(density, 1, 99)
    vmin_p, vmax_p = robust_limits(phase,   1, 99)

    fig, axs = plt.subplots(4, 2, figsize=(12, 16), layout="tight")
    fig.suptitle(f"Input + Saliency (all outputs)\n{triplet_str}", fontsize=12)

    # Row 0: input
    im00 = axs[0, 0].imshow(density, cmap=cfg.density_cmap, vmin=vmin_d, vmax=vmax_d)
    axs[0, 0].set_title("Input density")
    axs[0, 0].axis("off")
    add_colorbar(fig, axs[0, 0], im00)

    im01 = axs[0, 1].imshow(phase, cmap=cfg.phase_cmap, vmin=vmin_p, vmax=vmax_p)
    axs[0, 1].set_title("Input phase")
    axs[0, 1].axis("off")
    add_colorbar(fig, axs[0, 1], im01)

    def _row(r: int, sal: np.ndarray, name: str):
        sd = nan_safe(sal[0])
        sp = nan_safe(sal[1])

        imd = axs[r, 0].imshow(sd, cmap="hot", vmin=0.0, vmax=1.0)
        axs[r, 0].set_title(f"{name}: |dŷ/d(density)|")
        axs[r, 0].axis("off")
        add_colorbar(fig, axs[r, 0], imd)

        imp = axs[r, 1].imshow(sp, cmap="hot", vmin=0.0, vmax=1.0)
        axs[r, 1].set_title(f"{name}: |dŷ/d(phase)|")
        axs[r, 1].axis("off")
        add_colorbar(fig, axs[r, 1], imp)

    _row(1, sal_n2, "Output n2")
    _row(2, sal_isat, "Output Isat")
    _row(3, sal_alpha, "Output alpha")

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=cfg.dpi)
    plt.close(fig)


def _overlay_cam(ax, base_img, base_cmap, base_vmin, base_vmax, cam01, cfg: InterpretabilityConfig):
    """
    Draw base image with robust scaling + CAM overlay.
    CAM is always [0,1], rendered with cfg.cam_cmap and alpha cfg.cam_alpha.
    Returns (base_im, cam_im) so you can attach colorbars.
    """
    base_im = ax.imshow(base_img, cmap=base_cmap, vmin=base_vmin, vmax=base_vmax)
    cam_im = ax.imshow(cam01, cmap=cfg.cam_cmap, vmin=0.0, vmax=1.0, alpha=cfg.cam_alpha)
    ax.axis("off")
    return base_im, cam_im


def plot_gradcam_depths_per_output(
    x_np: np.ndarray,                   # (2,H,W)
    cams_by_depth: List[tuple],         # [(layer_name, cam(H,W)), ...] len=3
    output_name: str,
    triplet_str: str,
    save_path: str,
    cfg: InterpretabilityConfig,
) -> None:
    """
    DEPTH GRAD-CAM FIGURE (7 rows x 3 cols), matching your requested repetition pattern:

    Row 0: input density + input phase (shown once spanning width)
    Rows 1-2: depth 1 (density repeated across 3 cols, phase repeated across 3 cols) WITH CAM overlay
    Rows 3-4: depth 2 ...
    Rows 5-6: depth 3 ...

    Columns correspond to depths: start / mid / end Conv2d.
    """
    density = nan_safe(x_np[0])
    phase = nan_safe(x_np[1])

    # Robust limits for base images
    vmin_d, vmax_d = robust_limits(density, 1, 99)
    vmin_p, vmax_p = robust_limits(phase,   1, 99)

    # CAMs normalized
    layer_names = []
    cam_list = []
    for lname, cam in cams_by_depth:
        layer_names.append(lname)
        cam_list.append(norm01_np(cam))
    if len(cam_list) != 3:
        raise ValueError("Expected exactly 3 CAMs (start/mid/end).")

    # Use a gridspec so row0 can be (density, phase) spanning 3 cols
    fig = plt.figure(figsize=(20, 24), layout="tight")
    gs = fig.add_gridspec(nrows=7, ncols=3)

    fig.suptitle(f"Depth Grad-CAM (start/mid/end) for output={output_name}\n{triplet_str}", fontsize=12)

    # --- Row 0: two big panels spanning width: density (cols 0-1), phase (col 2)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])

    im_d = ax0.imshow(density, cmap=cfg.density_cmap, vmin=vmin_d, vmax=vmax_d)
    ax0.set_title("Input density (robust scaling)")
    ax0.axis("off")
    add_colorbar(fig, ax0, im_d)

    im_p = ax1.imshow(phase, cmap=cfg.phase_cmap, vmin=vmin_p, vmax=vmax_p)
    ax1.set_title("Input phase (robust scaling)")
    ax1.axis("off")
    add_colorbar(fig, ax1, im_p)

    # --- Rows 1..6: repeated context per depth with CAM overlay
    # Depth blocks: (row_density, row_phase) = (1,2), (3,4), (5,6)
    depth_blocks = [(1, 2), (3, 4), (5, 6)]

    for depth_j, (r_d, r_p) in enumerate(depth_blocks):
        cam01 = cam_list[depth_j]
        lname = layer_names[depth_j]

        # Density row repeated across 3 columns (but each column is still distinct axis)
        for c in range(3):
            ax = fig.add_subplot(gs[r_d, c])
            base_im, cam_im = _overlay_cam(
                ax=ax,
                base_img=density,
                base_cmap=cfg.density_cmap,
                base_vmin=vmin_d,
                base_vmax=vmax_d,
                cam01=cam01,
                cfg=cfg,
            )
            ax.set_title(f"Depth {depth_j+1} (Conv2d={lname}) | Density + CAM")
            # Separate CAM colorbar (your request was “different color bar than the map”)
            add_colorbar(fig, ax, cam_im)

        # Phase row repeated across 3 columns
        for c in range(3):
            ax = fig.add_subplot(gs[r_p, c])
            base_im, cam_im = _overlay_cam(
                ax=ax,
                base_img=phase,
                base_cmap=cfg.phase_cmap,
                base_vmin=vmin_p,
                base_vmax=vmax_p,
                cam01=cam01,
                cfg=cfg,
            )
            ax.set_title(f"Depth {depth_j+1} (Conv2d={lname}) | Phase + CAM")
            add_colorbar(fig, ax, cam_im)

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=cfg.dpi)
    plt.close(fig)


# =========================
# Main entry
# =========================

def run_interpretability_from_dataset(dataset, cfg: InterpretabilityConfig) -> None:
    """
    dataset.field must already be loaded. This function does not load simulation data from disk.
    It normalizes labels by calling prepare_training(dataset) (same as training).
    """
    # Guard: avoid silent "all zeros" from EngineDataset initialization
    if not hasattr(dataset, "field") or dataset.field is None:
        raise RuntimeError("dataset.field is None. Load the field (Es_w...) before interpretability.")
    if float(np.max(dataset.field)) == 0.0 and float(np.min(dataset.field)) == 0.0:
        raise RuntimeError(
            "dataset.field appears to be all zeros. You likely didn't load the field from disk before interpretability."
        )

    if cfg.out_dir.strip() == "":
        cfg.out_dir = os.path.join(_training_dir(dataset), "interpretability_dataset")

    per_sample_dir = os.path.join(cfg.out_dir, "per_sample")
    ensure_dir(per_sample_dir)

    # Prepare training split objects (this normalizes labels; it does not modify field)
    train_set, val_set, test_set, _ = prepare_training(dataset)

    split = cfg.sample_split.lower()
    if split == "train":
        base_set = train_set
    elif split in ("val", "valid", "validation"):
        base_set = val_set
    else:
        base_set = test_set

    labels01 = _labels01_from_networkdataset(base_set)
    chosen_idx = sample_uniform_triplets_indices(labels01, cfg.num_samples, cfg.grid_side)
    if len(chosen_idx) == 0:
        raise RuntimeError("No samples selected (empty split?).")

    # Load model (pth only)
    device = torch.device(dataset.device_number)
    model = load_model_from_pth(dataset)

    methods = tuple(m.lower() for m in cfg.methods)
    do_sal = "saliency" in methods
    do_cam = "gradcam" in methods

    # Build CAM engines once
    gradcam_engines = build_gradcam_engines(model) if do_cam else []

    # For each sample, produce outputs
    for s, idx in enumerate(chosen_idx):
        item = base_set[idx]
        x = item[0].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,2,H,W)
        x_np = x.detach().cpu().numpy()[0]  # (2,H,W)

        y01 = labels01[idx]
        yphys = denormalize_triplet(dataset, y01)

        triplet_str = (
            f"sample {s:04d} | split_idx={idx:06d} | "
            f"y01=(n2={y01[0]:.3f}, Isat={y01[1]:.3f}, alpha={y01[2]:.3f}) | "
            f"yphys=(n2={yphys[0]:.3e}, Isat={yphys[1]:.3e}, alpha={yphys[2]:.3f})"
        )

        if cfg.verbose:
            d = nan_safe(x_np[0])
            p = nan_safe(x_np[1])
            print(f"\n=== SAMPLE/TRIPLET === {triplet_str}")
            print(
                f"[triplet {s:04d}] input stats | "
                f"density min/max={float(d.min()):.3e}/{float(d.max()):.3e} | "
                f"phase   min/max={float(p.min()):.3e}/{float(p.max()):.3e}"
            )

        # (1) Big saliency figure
        if do_sal:
            if cfg.verbose:
                print(f"[triplet {s:04d}] computing saliency: outputs n2, Isat, alpha")

            sal_n2 = compute_saliency(model, x, 0).detach().cpu().numpy()
            sal_isat = compute_saliency(model, x, 1).detach().cpu().numpy()
            sal_alpha = compute_saliency(model, x, 2).detach().cpu().numpy()

            out_sal = os.path.join(per_sample_dir, f"triplet{s:04d}_saliency_all_outputs.png")
            out_sal = os.path.join(per_sample_dir, f"triplet{s:04d}_saliency_all_outputs.svg")
            if cfg.verbose:
                print(f"[triplet {s:04d}] saving saliency figure -> {out_sal}")

            plot_saliency_all_outputs_big(
                x_np=x_np,
                sal_n2=sal_n2,
                sal_isat=sal_isat,
                sal_alpha=sal_alpha,
                triplet_str=triplet_str,
                save_path=out_sal,
                cfg=cfg,
            )

        # (2) Depth grad-cam figure for EACH chosen output parameter (n2, Isat, alpha)
        if do_cam:
            for k, output_name in enumerate(cfg.param_names):
                if cfg.verbose:
                    print(f"[triplet {s:04d}] computing depth Grad-CAM for output={output_name}")

                cams_by_depth = []
                for lname, engine in gradcam_engines:
                    cam = engine.compute(x, k).detach().cpu().numpy()
                    cams_by_depth.append((lname, cam))

                out_cam = os.path.join(per_sample_dir, f"triplet{s:04d}_gradcam_depths_{output_name}.png")
                out_cam = os.path.join(per_sample_dir, f"triplet{s:04d}_gradcam_depths_{output_name}.svg")
                if cfg.verbose:
                    print(f"[triplet {s:04d}] saving Grad-CAM figure ({output_name}) -> {out_cam}")

                plot_gradcam_depths_per_output(
                    x_np=x_np,
                    cams_by_depth=cams_by_depth,
                    output_name=output_name,
                    triplet_str=triplet_str,
                    save_path=out_cam,
                    cfg=cfg,
                )

    print(f"\n[OK] Interpretability saved to: {cfg.out_dir}")
    print(f"     split={cfg.sample_split} | N={len(chosen_idx)} | methods={cfg.methods}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

"""
engine/interpretability.py

Implements:

(1) BIG SALIENCY FIGURE (one per sample):
    Row 0: input density + input phase (robust scaling)
    Row 1: per-channel saliency for n2 (density, phase)
    Row 2: per-channel saliency for Isat
    Row 3: per-channel saliency for alpha
    - independent colorbar per subplot
    - prints which triplet is being processed
    - prints channel saliency strength stats (sum/mean/p99 + phase/density ratio)

(2) DEPTH GRAD-CAM FIGURE (one per sample AND per output parameter):
    Layout 7x3: start / mid / end Conv2d as columns
    Row 0: input density + input phase (robust scaling) shown once
    Rows 1-2: depth 1 density+CAM / phase+CAM
    Rows 3-4: depth 2 density+CAM / phase+CAM
    Rows 5-6: depth 3 density+CAM / phase+CAM
    - each subplot has its own CAM colorbar
    - IMPORTANT: channel-dependent inference:
        density rows use CAM from density-only inference (phase=0)
        phase rows use CAM from phase-only inference (density=0)
    - prints which output parameter is processed and which sample
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Optional, List, Any, Dict

from mpl_toolkits.axes_grid1 import make_axes_locatable

from engine.training_manager import prepare_training
from engine.model import network
from engine.utils import set_seed

set_seed(10)


# =========================
# Config
# =========================

@dataclass
class InterpretabilityConfig:
    num_samples: int = 24
    out_dir: str = ""                       # default: <training_dir>/interpretability_dataset
    sample_split: str = "test"              # "train" | "val" | "test"
    methods: Tuple[str, ...] = ("saliency", "gradcam")
    param_names: Tuple[str, ...] = ("n2", "Isat", "alpha")
    grid_side: Optional[int] = None         # None -> inferred from num_samples
    dpi: int = 250
    verbose: bool = True

    # CAM overlay visual parameters
    cam_alpha: float = 0.50
    cam_cmap: str = "jet"
    density_cmap: str = "gray"
    phase_cmap: str = "twilight"


# =========================
# Paths
# =========================

def _training_dir(dataset) -> str:
    return (
        f"{dataset.saving_path}/training_n2{dataset.number_of_n2}"
        f"_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}"
        f"_power{dataset.input_power:.2f}"
    )

def _weights_path(dataset) -> str:
    return (
        f"{_training_dir(dataset)}/"
        f"n2_net_w{dataset.resolution_training}"
        f"_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}"
        f"_power{dataset.input_power:.2f}.pth"
    )


# =========================
# Small utils
# =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def add_colorbar(fig, ax, im, size="4%", pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    fig.colorbar(im, cax=cax)

def nan_safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)

def robust_limits(a: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    a = nan_safe(a)
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin) < 1e-12:
        vmin = float(np.min(a))
        vmax = float(np.max(a))
    if (vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax

def norm01_np(a: np.ndarray) -> np.ndarray:
    a = nan_safe(a)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def get_mu_tensor(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    if isinstance(model_out, torch.Tensor):
        return model_out
    if isinstance(model_out, dict):
        for k in ("mu", "mean", "pred", "y", "output"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
    raise TypeError(f"Unsupported model output type: {type(model_out)}")

def channel_ablate(x: torch.Tensor, keep: str) -> torch.Tensor:
    """
    x: (1,2,H,W)
    keep: "density" or "phase"
    returns a new tensor where the other channel is zeroed.
    """
    y = x.clone()
    if keep == "density":
        y[:, 1] = 0.0
    elif keep == "phase":
        y[:, 0] = 0.0
    else:
        raise ValueError("keep must be 'density' or 'phase'")
    return y


# =========================
# Model loading
# =========================

def load_model_from_pth(dataset) -> nn.Module:
    device = torch.device(dataset.device_number)
    model = network().to(device)

    wpath = _weights_path(dataset)
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Model weights not found: {wpath}")

    state = torch.load(wpath, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# =========================
# Saliency (improved + channel stats)
# =========================

def compute_saliency(model: nn.Module, x: torch.Tensor, param_index: int, verbose: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    x: (1,2,H,W)
    returns:
      sal_viz: (2,H,W) in [0,1] per-channel for visualization
      stats: raw channel saliency magnitudes BEFORE normalization
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    mu = get_mu_tensor(model(x))
    if mu.ndim != 2 or mu.shape[1] < 3:
        raise ValueError(f"Expected mu shape (B,3). Got {tuple(mu.shape)}")

    target = mu[0, param_index]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    target.backward(retain_graph=False)

    sal = x.grad.detach().abs()[0]  # (2,H,W)

    flat = sal.view(2, -1)
    raw_sum  = flat.sum(dim=1)
    raw_mean = flat.mean(dim=1)
    raw_p99  = torch.quantile(flat, 0.99, dim=1)

    # ratio phase/density
    ratio_sum  = (raw_sum[1]  / (raw_sum[0]  + 1e-12))
    ratio_mean = (raw_mean[1] / (raw_mean[0] + 1e-12))
    ratio_p99  = (raw_p99[1]  / (raw_p99[0]  + 1e-12))

    stats = {
        "raw_sum": raw_sum,
        "raw_mean": raw_mean,
        "raw_p99": raw_p99,
        "ratio_sum_phase_over_density": ratio_sum,
        "ratio_mean_phase_over_density": ratio_mean,
        "ratio_p99_phase_over_density": ratio_p99,
    }

    if verbose:
        print(
            f"[saliency] param_index={param_index} | "
            f"sum(d,p)=({raw_sum[0].item():.3e},{raw_sum[1].item():.3e}) ratio={ratio_sum.item():.2f} | "
            f"mean(d,p)=({raw_mean[0].item():.3e},{raw_mean[1].item():.3e}) ratio={ratio_mean.item():.2f} | "
            f"p99(d,p)=({raw_p99[0].item():.3e},{raw_p99[1].item():.3e}) ratio={ratio_p99.item():.2f}"
        )

    # per-channel normalization for visualization (keeps maps readable)
    out = []
    for c in range(sal.shape[0]):
        s = sal[c]
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        out.append(s)
    sal_viz = torch.stack(out, dim=0)

    return sal_viz, stats


# =========================
# Grad-CAM (start/mid/end conv2d) with channel-dependent inference
# =========================

def list_conv2d_layers(model: nn.Module) -> List[tuple]:
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            convs.append((name, m))
    return convs

def pick_start_mid_end_convs(model: nn.Module) -> List[tuple]:
    convs = list_conv2d_layers(model)
    if len(convs) < 3:
        raise RuntimeError(f"Need >=3 Conv2d layers for start/mid/end. Found {len(convs)}")
    return [convs[0], convs[len(convs) // 2], convs[-1]]

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute(self, x: torch.Tensor, param_index: int) -> torch.Tensor:
        """
        returns CAM (H,W) in [0,1]
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        mu = get_mu_tensor(self.model(x))
        target = mu[0, param_index]
        target.backward(retain_graph=False)

        A = self.activations
        dA = self.gradients
        if A is None or dA is None:
            raise RuntimeError("Grad-CAM missing activations/gradients. Check chosen layer.")

        weights = dA.mean(dim=(2, 3), keepdim=True)         # (1,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)        # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def build_gradcam_engines(model: nn.Module) -> List[tuple]:
    layers = pick_start_mid_end_convs(model)
    return [(lname, GradCAM(model, layer)) for lname, layer in layers]


# =========================
# Sampling uniformly in label space [0,1]^3
# =========================

def _labels01_from_networkdataset(subset) -> np.ndarray:
    n = len(subset)
    y = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        item = subset[i]
        y[i, 0] = float(item[1].item())
        y[i, 1] = float(item[2].item())
        y[i, 2] = float(item[3].item())
    return y

def sample_uniform_triplets_indices(labels01: np.ndarray, num_samples: int, grid_side: Optional[int] = None) -> List[int]:
    N = labels01.shape[0]
    if N == 0:
        return []
    K = int(min(num_samples, N))
    if grid_side is None:
        grid_side = max(2, int(round(K ** (1.0 / 3.0))))

    lin = np.linspace(0.0, 1.0, grid_side)
    targets = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)

    if targets.shape[0] > K:
        pick = np.linspace(0, targets.shape[0] - 1, K).round().astype(int)
        targets = targets[pick]

    chosen: List[int] = []
    used = np.zeros(N, dtype=bool)

    for t in targets:
        diff = labels01 - t[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        dist2[used] = np.inf
        j = int(np.argmin(dist2))
        if np.isfinite(dist2[j]):
            chosen.append(j)
            used[j] = True
        if len(chosen) >= K:
            break

    if len(chosen) < K:
        rem = np.where(~used)[0]
        np.random.shuffle(rem)
        chosen.extend(rem[: (K - len(chosen))].tolist())

    return chosen[:K]

def denormalize_triplet(dataset, y01: np.ndarray) -> np.ndarray:
    n2 = y01[0] * (dataset.n2_max - dataset.n2_min) + dataset.n2_min
    isat = y01[1] * (dataset.isat_max - dataset.isat_min) + dataset.isat_min
    alpha = y01[2] * (dataset.alpha_max - dataset.alpha_min) + dataset.alpha_min
    return np.array([n2, isat, alpha], dtype=np.float64)


# =========================
# Plotting
# =========================

def plot_saliency_all_outputs_big(
    x_np: np.ndarray,                 # (2,H,W)
    sal_n2: np.ndarray,               # (2,H,W) in [0,1]
    sal_isat: np.ndarray,             # (2,H,W) in [0,1]
    sal_alpha: np.ndarray,            # (2,H,W) in [0,1]
    triplet_str: str,
    save_path: str,
    cfg: InterpretabilityConfig,
) -> None:
    density = nan_safe(x_np[0])
    phase = nan_safe(x_np[1])

    vmin_d, vmax_d = robust_limits(density, 1, 99)
    vmin_p, vmax_p = robust_limits(phase,   1, 99)

    fig, axs = plt.subplots(4, 2, figsize=(12, 16), layout="tight")
    fig.suptitle(f"Input + Saliency (all outputs)\n{triplet_str}", fontsize=12)

    # Row 0: input
    im00 = axs[0, 0].imshow(density, cmap=cfg.density_cmap, vmin=vmin_d, vmax=vmax_d)
    axs[0, 0].set_title("Input density")
    axs[0, 0].axis("off")
    add_colorbar(fig, axs[0, 0], im00)

    im01 = axs[0, 1].imshow(phase, cmap=cfg.phase_cmap, vmin=vmin_p, vmax=vmax_p)
    axs[0, 1].set_title("Input phase")
    axs[0, 1].axis("off")
    add_colorbar(fig, axs[0, 1], im01)

    def _row(r: int, sal: np.ndarray, name: str):
        sd = nan_safe(sal[0])
        sp = nan_safe(sal[1])

        imd = axs[r, 0].imshow(sd, cmap="hot", vmin=0.0, vmax=1.0)
        axs[r, 0].set_title(f"{name}: |∂ŷ/∂density|")
        axs[r, 0].axis("off")
        add_colorbar(fig, axs[r, 0], imd)

        imp = axs[r, 1].imshow(sp, cmap="hot", vmin=0.0, vmax=1.0)
        axs[r, 1].set_title(f"{name}: |∂ŷ/∂phase|")
        axs[r, 1].axis("off")
        add_colorbar(fig, axs[r, 1], imp)

    _row(1, sal_n2, "Output n2")
    _row(2, sal_isat, "Output Isat")
    _row(3, sal_alpha, "Output alpha")

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=cfg.dpi)
    plt.close(fig)

def _overlay_cam(ax, base_img, base_cmap, base_vmin, base_vmax, cam01, cfg: InterpretabilityConfig):
    base_im = ax.imshow(base_img, cmap=base_cmap, vmin=base_vmin, vmax=base_vmax)
    cam_im = ax.imshow(cam01, cmap=cfg.cam_cmap, vmin=0.0, vmax=1.0, alpha=cfg.cam_alpha)
    ax.axis("off")
    return base_im, cam_im

def plot_gradcam_depths_per_output_channel_inference(
    x_np: np.ndarray,                         # (2,H,W)
    cams_den_by_depth: List[tuple],           # [(layer_name, cam(H,W)), ...] len=3 (density-only inference)
    cams_ph_by_depth: List[tuple],            # [(layer_name, cam(H,W)), ...] len=3 (phase-only inference)
    output_name: str,
    triplet_str: str,
    save_path: str,
    cfg: InterpretabilityConfig,
) -> None:
    """
    7x3 figure, columns correspond to start/mid/end Conv2d.

    Density rows use CAM computed with phase channel zeroed (density-only inference).
    Phase rows use CAM computed with density channel zeroed (phase-only inference).
    """
    density = nan_safe(x_np[0])
    phase = nan_safe(x_np[1])

    vmin_d, vmax_d = robust_limits(density, 1, 99)
    vmin_p, vmax_p = robust_limits(phase,   1, 99)

    # Normalize CAMs and extract layer names
    layer_names = [lname for lname, _ in cams_den_by_depth]
    cam_den_list = [norm01_np(cam) for _, cam in cams_den_by_depth]
    cam_ph_list  = [norm01_np(cam) for _, cam in cams_ph_by_depth]

    if len(cam_den_list) != 3 or len(cam_ph_list) != 3:
        raise ValueError("Expected exactly 3 CAMs per channel (start/mid/end).")

    fig = plt.figure(figsize=(20, 24), layout="tight")
    gs = fig.add_gridspec(nrows=7, ncols=3)
    fig.suptitle(
        f"Depth Grad-CAM (channel-dependent inference) for output={output_name}\n{triplet_str}",
        fontsize=12,
    )

    # Row 0: inputs (density spans cols 0-1, phase col 2)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])

    im_d = ax0.imshow(density, cmap=cfg.density_cmap, vmin=vmin_d, vmax=vmax_d)
    ax0.set_title("Input density (robust scaling)")
    ax0.axis("off")
    add_colorbar(fig, ax0, im_d)

    im_p = ax1.imshow(phase, cmap=cfg.phase_cmap, vmin=vmin_p, vmax=vmax_p)
    ax1.set_title("Input phase (robust scaling)")
    ax1.axis("off")
    add_colorbar(fig, ax1, im_p)

    depth_blocks = [(1, 2), (3, 4), (5, 6)]

    for depth_j, (r_d, r_p) in enumerate(depth_blocks):
        lname = layer_names[depth_j]

        cam_den = cam_den_list[depth_j]
        cam_ph  = cam_ph_list[depth_j]

        # Density row: CAM from density-only inference
        for c in range(3):
            ax = fig.add_subplot(gs[r_d, c])
            _, cam_im = _overlay_cam(
                ax=ax,
                base_img=density,
                base_cmap=cfg.density_cmap,
                base_vmin=vmin_d,
                base_vmax=vmax_d,
                cam01=cam_den,
                cfg=cfg,
            )
            ax.set_title(f"Depth {depth_j+1} (Conv2d={lname}) | Density + CAM (phase=0)")
            add_colorbar(fig, ax, cam_im)

        # Phase row: CAM from phase-only inference
        for c in range(3):
            ax = fig.add_subplot(gs[r_p, c])
            _, cam_im = _overlay_cam(
                ax=ax,
                base_img=phase,
                base_cmap=cfg.phase_cmap,
                base_vmin=vmin_p,
                base_vmax=vmax_p,
                cam01=cam_ph,
                cfg=cfg,
            )
            ax.set_title(f"Depth {depth_j+1} (Conv2d={lname}) | Phase + CAM (density=0)")
            add_colorbar(fig, ax, cam_im)

    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=cfg.dpi)
    plt.close(fig)


# =========================
# Main entry
# =========================

def run_interpretability_from_dataset(dataset, cfg: InterpretabilityConfig) -> None:
    """
    dataset.field must already be loaded. This function does not load simulation data from disk.
    It normalizes labels by calling prepare_training(dataset) (same as training).
    """
    if not hasattr(dataset, "field") or dataset.field is None:
        raise RuntimeError("dataset.field is None. Load the field (Es_w...) before interpretability.")
    if float(np.max(dataset.field)) == 0.0 and float(np.min(dataset.field)) == 0.0:
        raise RuntimeError("dataset.field appears to be all zeros. You likely didn't load the field before interpretability.")

    if cfg.out_dir.strip() == "":
        cfg.out_dir = os.path.join(_training_dir(dataset), "interpretability_dataset")

    per_sample_dir = os.path.join(cfg.out_dir, "per_sample")
    ensure_dir(per_sample_dir)

    train_set, val_set, test_set, _ = prepare_training(dataset)

    split = cfg.sample_split.lower()
    if split == "train":
        base_set = train_set
    elif split in ("val", "valid", "validation"):
        base_set = val_set
    else:
        base_set = test_set

    labels01 = _labels01_from_networkdataset(base_set)
    chosen_idx = sample_uniform_triplets_indices(labels01, cfg.num_samples, cfg.grid_side)
    if len(chosen_idx) == 0:
        raise RuntimeError("No samples selected (empty split?).")

    device = torch.device(dataset.device_number)
    model = load_model_from_pth(dataset)

    methods = tuple(m.lower() for m in cfg.methods)
    do_sal = "saliency" in methods
    do_cam = "gradcam" in methods

    gradcam_engines = build_gradcam_engines(model) if do_cam else []

    for s, idx in enumerate(chosen_idx):
        item = base_set[idx]
        x = item[0].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,2,H,W)
        x_np = x.detach().cpu().numpy()[0]  # (2,H,W)

        y01 = labels01[idx]
        yphys = denormalize_triplet(dataset, y01)

        triplet_str = (
            f"sample {s:04d} | split_idx={idx:06d} | "
            f"y01=(n2={y01[0]:.3f}, Isat={y01[1]:.3f}, alpha={y01[2]:.3f}) | "
            f"yphys=(n2={yphys[0]:.3e}, Isat={yphys[1]:.3e}, alpha={yphys[2]:.3f})"
        )

        if cfg.verbose:
            d = nan_safe(x_np[0])
            p = nan_safe(x_np[1])
            print(f"\n=== SAMPLE/TRIPLET === {triplet_str}")
            print(
                f"[triplet {s:04d}] input stats | "
                f"density min/max={float(d.min()):.3e}/{float(d.max()):.3e} | "
                f"phase   min/max={float(p.min()):.3e}/{float(p.max()):.3e}"
            )

        # (1) Big saliency figure
        if do_sal:
            if cfg.verbose:
                print(f"[triplet {s:04d}] computing improved saliency + channel stats: outputs n2, Isat, alpha")

            sal_n2, stats_n2 = compute_saliency(model, x, 0, verbose=cfg.verbose)
            sal_isat, stats_isat = compute_saliency(model, x, 1, verbose=cfg.verbose)
            sal_alpha, stats_alpha = compute_saliency(model, x, 2, verbose=cfg.verbose)

            out_sal = os.path.join(per_sample_dir, f"triplet{s:04d}_saliency_all_outputs.png")
            out_sal = os.path.join(per_sample_dir, f"triplet{s:04d}_saliency_all_outputs.svg")
            if cfg.verbose:
                print(f"[triplet {s:04d}] saving saliency figure -> {out_sal}")

            plot_saliency_all_outputs_big(
                x_np=x_np,
                sal_n2=sal_n2.detach().cpu().numpy(),
                sal_isat=sal_isat.detach().cpu().numpy(),
                sal_alpha=sal_alpha.detach().cpu().numpy(),
                triplet_str=triplet_str,
                save_path=out_sal,
                cfg=cfg,
            )

        # (2) Depth grad-cam figure for EACH output parameter with channel-dependent inference
        if do_cam:
            x_den = channel_ablate(x, "density")
            x_ph  = channel_ablate(x, "phase")

            for k, output_name in enumerate(cfg.param_names):
                if cfg.verbose:
                    print(f"[triplet {s:04d}] computing channel-dependent depth Grad-CAM for output={output_name}")

                cams_den_by_depth = []
                cams_ph_by_depth  = []

                for lname, engine in gradcam_engines:
                    cam_den = engine.compute(x_den, k).detach().cpu().numpy()
                    cam_ph  = engine.compute(x_ph,  k).detach().cpu().numpy()
                    cams_den_by_depth.append((lname, cam_den))
                    cams_ph_by_depth.append((lname, cam_ph))

                out_cam = os.path.join(per_sample_dir, f"triplet{s:04d}_gradcam_depths_{output_name}.png")
                out_cam = os.path.join(per_sample_dir, f"triplet{s:04d}_gradcam_depths_{output_name}.svg")
                if cfg.verbose:
                    print(f"[triplet {s:04d}] saving Grad-CAM figure ({output_name}) -> {out_cam}")

                plot_gradcam_depths_per_output_channel_inference(
                    x_np=x_np,
                    cams_den_by_depth=cams_den_by_depth,
                    cams_ph_by_depth=cams_ph_by_depth,
                    output_name=output_name,
                    triplet_str=triplet_str,
                    save_path=out_cam,
                    cfg=cfg,
                )

    print(f"\n[OK] Interpretability saved to: {cfg.out_dir}")
    print(f"     split={cfg.sample_split} | N={len(chosen_idx)} | methods={cfg.methods}")