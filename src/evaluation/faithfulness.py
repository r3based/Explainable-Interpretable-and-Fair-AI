"""Baseline faithfulness metrics for image explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _integrate_curve(scores: list[float], fractions: list[float]) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(scores, fractions))
    return float(np.trapz(scores, fractions))


def to_common_heatmap(result_or_heatmap: Any) -> np.ndarray:
    """Convert explainer outputs into a common 2D heatmap representation."""
    if hasattr(result_or_heatmap, "heatmap"):
        array = np.asarray(result_or_heatmap.heatmap)
    elif hasattr(result_or_heatmap, "perturbation"):
        perturbation = result_or_heatmap.perturbation
        if isinstance(perturbation, torch.Tensor):
            array = perturbation.detach().cpu().numpy()
        else:
            array = np.asarray(perturbation)
    elif isinstance(result_or_heatmap, torch.Tensor):
        array = result_or_heatmap.detach().cpu().numpy()
    else:
        array = np.asarray(result_or_heatmap)

    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        return array.mean(axis=0).astype(np.float32)
    if array.ndim == 3 and array.shape[-1] in {1, 3}:
        return array.mean(axis=-1).astype(np.float32)
    if array.ndim == 2:
        return array.astype(np.float32)

    raise ValueError(f"Cannot convert input with shape {array.shape} to a 2D heatmap.")


def normalize_heatmap(heatmap: Any, use_abs: bool = True, eps: float = 1e-8) -> np.ndarray:
    """Normalize any supported heatmap-like input to [0, 1]."""
    array = to_common_heatmap(heatmap)
    if use_abs:
        array = np.abs(array)
    array = array.astype(np.float32)
    array -= array.min()
    denom = array.max()
    if denom <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array / denom).astype(np.float32)


def _prepare_image(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image).float()
    else:
        tensor = image.detach().clone().float()

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise ValueError("Expected image with shape (C, H, W) or (1, C, H, W).")
    return tensor


def _build_baseline(image: torch.Tensor, baseline: str = "zero") -> torch.Tensor:
    if baseline == "zero":
        return torch.zeros_like(image)
    if baseline == "mean":
        mean = image.mean(dim=(2, 3), keepdim=True)
        return mean.expand_as(image).clone()
    raise ValueError("baseline must be one of: zero, mean")


def _ordered_pixel_indices(heatmap: Any) -> np.ndarray:
    importance = normalize_heatmap(heatmap).reshape(-1)
    return np.argsort(importance)[::-1]


def _apply_topk_pixels(
    source: torch.Tensor,
    target: torch.Tensor,
    pixel_order: np.ndarray,
    count: int,
) -> torch.Tensor:
    updated = target.clone()
    if count <= 0:
        return updated

    flat_source = source.reshape(source.shape[0], source.shape[1], -1)
    flat_target = updated.reshape(updated.shape[0], updated.shape[1], -1)
    selected = pixel_order[:count].tolist()
    flat_target[:, :, selected] = flat_source[:, :, selected]
    return updated


def _score_curve_batch(
    images: list[torch.Tensor],
    predict_fn,
    class_idx: int,
    device: str | torch.device | None = None,
) -> np.ndarray:
    batch = torch.cat(images, dim=0)
    if device is not None:
        batch = batch.to(device)
    with torch.no_grad():
        probs = predict_fn(batch)
    return probs[:, int(class_idx)].detach().cpu().numpy().astype(np.float32)


def deletion_curve(
    image: torch.Tensor | np.ndarray,
    heatmap: Any,
    predict_fn,
    class_idx: int,
    steps: int = 20,
    baseline: str = "zero",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Progressively remove important pixels and record the target score."""
    image_tensor = _prepare_image(image)
    base = _build_baseline(image_tensor, baseline=baseline)
    pixel_order = _ordered_pixel_indices(heatmap)
    n_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
    fractions = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    masked_images: list[torch.Tensor] = []
    counts: list[int] = []
    for fraction in fractions:
        count = min(n_pixels, int(round(float(fraction) * n_pixels)))
        counts.append(count)
        masked = image_tensor.clone()
        masked = _apply_topk_pixels(source=base, target=masked, pixel_order=pixel_order, count=count)
        masked_images.append(masked)

    scores = _score_curve_batch(masked_images, predict_fn=predict_fn, class_idx=class_idx, device=device)
    return {
        "fractions": fractions.tolist(),
        "scores": scores.tolist(),
        "pixel_counts": counts,
        "class_idx": int(class_idx),
        "baseline": baseline,
    }


def insertion_curve(
    image: torch.Tensor | np.ndarray,
    heatmap: Any,
    predict_fn,
    class_idx: int,
    steps: int = 20,
    baseline: str = "zero",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Progressively reveal important pixels and record the target score."""
    image_tensor = _prepare_image(image)
    base = _build_baseline(image_tensor, baseline=baseline)
    pixel_order = _ordered_pixel_indices(heatmap)
    n_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
    fractions = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    revealed_images: list[torch.Tensor] = []
    counts: list[int] = []
    for fraction in fractions:
        count = min(n_pixels, int(round(float(fraction) * n_pixels)))
        counts.append(count)
        revealed = _apply_topk_pixels(source=image_tensor, target=base, pixel_order=pixel_order, count=count)
        revealed_images.append(revealed)

    scores = _score_curve_batch(revealed_images, predict_fn=predict_fn, class_idx=class_idx, device=device)
    return {
        "fractions": fractions.tolist(),
        "scores": scores.tolist(),
        "pixel_counts": counts,
        "class_idx": int(class_idx),
        "baseline": baseline,
    }


def deletion_auc(
    image: torch.Tensor | np.ndarray,
    heatmap: Any,
    predict_fn,
    class_idx: int,
    steps: int = 20,
    baseline: str = "zero",
    device: str | torch.device | None = None,
) -> float:
    curve = deletion_curve(
        image=image,
        heatmap=heatmap,
        predict_fn=predict_fn,
        class_idx=class_idx,
        steps=steps,
        baseline=baseline,
        device=device,
    )
    return _integrate_curve(curve["scores"], curve["fractions"])


def insertion_auc(
    image: torch.Tensor | np.ndarray,
    heatmap: Any,
    predict_fn,
    class_idx: int,
    steps: int = 20,
    baseline: str = "zero",
    device: str | torch.device | None = None,
) -> float:
    curve = insertion_curve(
        image=image,
        heatmap=heatmap,
        predict_fn=predict_fn,
        class_idx=class_idx,
        steps=steps,
        baseline=baseline,
        device=device,
    )
    return _integrate_curve(curve["scores"], curve["fractions"])
