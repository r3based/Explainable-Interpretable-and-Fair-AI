"""Unified heatmap visualisation for LIME, SHAP, and counterfactual outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.cifar10 import CIFAR10_CLASSES, denormalize


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) normalised tensor to a uint8 (H, W, 3) array."""
    img = denormalize(tensor.cpu()).permute(1, 2, 0).numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
) -> np.ndarray:
    """Blend a heatmap onto an RGB image.

    Parameters
    ----------
    image_np: (H, W, 3) uint8 array.
    heatmap:  (H, W) float array (arbitrary range).
    symmetric: centre the colourmap at zero (good for signed attributions).

    Returns
    -------
    (H, W, 3) uint8 blended image.
    """
    h = heatmap.copy().astype(np.float32)
    if symmetric:
        abs_max = np.abs(h).max() + 1e-8
        h = (h / abs_max + 1.0) / 2.0
    else:
        h_min, h_max = h.min(), h.max() + 1e-8
        h = (h - h_min) / (h_max - h_min)

    coloured = matplotlib.colormaps[cmap](h)[..., :3]
    base     = image_np.astype(np.float32) / 255.0
    blended  = (1 - alpha) * base + alpha * coloured
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def plot_lime_result(
    image_tensor: torch.Tensor,
    lime_result,
    class_idx: int,
    save_path: str | Path | None = None,
    top_k: int = 10,
) -> plt.Figure:
    """Three-panel figure: original | positive regions | full heatmap."""
    image_np   = tensor_to_numpy(image_tensor)
    heatmap    = lime_result.heatmap
    segments   = lime_result.segments
    coeffs     = lime_result.coefficients
    class_name = CIFAR10_CLASSES[class_idx]

    top_seg_ids = np.argsort(coeffs)[-top_k:]
    pos_mask    = np.isin(segments, top_seg_ids).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"LIME  —  class: {class_name}", fontsize=13)

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    highlighted = image_np.copy().astype(np.float32)
    highlighted[pos_mask == 0] *= 0.3
    axes[1].imshow(highlighted.clip(0, 255).astype(np.uint8))
    axes[1].set_title(f"Top-{top_k} positive superpixels")
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap(image_np, heatmap, alpha=0.6))
    axes[2].set_title("Signed attribution heatmap")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gallery(
    results: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grid: one row per image; columns = [original, LIME, (SHAP), (Counterfactual)].

    Each dict needs 'image_tensor', 'lime_heatmap', 'class_idx';
    optionally 'shap_heatmap', 'cf_image_tensor'.
    """
    n        = len(results)
    has_shap = any("shap_heatmap"    in r for r in results)
    has_cf   = any("cf_image_tensor" in r for r in results)
    n_cols   = 2 + int(has_shap) + int(has_cf)

    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "LIME"]
    if has_shap:
        col_titles.append("SHAP")
    if has_cf:
        col_titles.append("Counterfactual")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row, res in enumerate(results):
        img_np   = tensor_to_numpy(res["image_tensor"])
        cls_name = CIFAR10_CLASSES[res["class_idx"]]

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(cls_name, fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_heatmap(img_np, res["lime_heatmap"], alpha=0.6))
        axes[row, 1].axis("off")

        col_offset = 2
        if has_shap and "shap_heatmap" in res:
            axes[row, col_offset].imshow(overlay_heatmap(img_np, res["shap_heatmap"], alpha=0.6))
            axes[row, col_offset].axis("off")
            col_offset += 1

        if has_cf and "cf_image_tensor" in res:
            axes[row, col_offset].imshow(tensor_to_numpy(res["cf_image_tensor"]))
            axes[row, col_offset].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
