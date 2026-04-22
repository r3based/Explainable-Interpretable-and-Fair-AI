"""Baseline stability metrics for local explanations."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
import torch

from .faithfulness import normalize_heatmap


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float = 0.05,
    seed: int = 42,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device=image.device).manual_seed(int(seed))
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * float(sigma)
    return (image + noise).clamp(clamp_min, clamp_max)


def compare_heatmaps(
    heatmap_a: Any,
    heatmap_b: Any,
    topk_fraction: float = 0.05,
) -> dict[str, float]:
    a = normalize_heatmap(heatmap_a).reshape(-1)
    b = normalize_heatmap(heatmap_b).reshape(-1)

    std_a = float(a.std())
    std_b = float(b.std())
    if std_a < 1e-8 and std_b < 1e-8:
        correlation = 1.0 if np.allclose(a, b) else 0.0
    elif std_a < 1e-8 or std_b < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    k = max(1, int(round(len(a) * float(topk_fraction))))
    topk_a = set(np.argsort(a)[-k:].tolist())
    topk_b = set(np.argsort(b)[-k:].tolist())
    union = len(topk_a | topk_b)
    iou = 1.0 if union == 0 else float(len(topk_a & topk_b) / union)

    return {
        "correlation": correlation,
        "topk_iou": iou,
    }


def stability_under_noise(
    image: torch.Tensor,
    explain_fn: Callable[[torch.Tensor], Any],
    noise_std: float = 0.05,
    repeats: int = 3,
    seed: int = 42,
    topk_fraction: float = 0.05,
) -> dict[str, Any]:
    base_result = explain_fn(image)
    base_heatmap = normalize_heatmap(base_result)

    records: list[dict[str, float | int]] = []
    for repeat_idx in range(int(repeats)):
        noisy = add_gaussian_noise(image=image, sigma=noise_std, seed=seed + repeat_idx)
        noisy_result = explain_fn(noisy)
        comparison = compare_heatmaps(base_heatmap, noisy_result, topk_fraction=topk_fraction)
        records.append(
            {
                "perturbation_seed": int(seed + repeat_idx),
                "correlation": comparison["correlation"],
                "topk_iou": comparison["topk_iou"],
            }
        )

    aggregate = aggregate_stability_scores(records)
    aggregate.update(
        {
            "noise_std": float(noise_std),
            "repeats": int(repeats),
            "mode": "noise",
            "raw": records,
        }
    )
    return aggregate


def stability_under_seed_variation(
    image: torch.Tensor,
    explain_fn: Callable[[torch.Tensor, int], Any],
    seeds: list[int] | tuple[int, ...],
    topk_fraction: float = 0.05,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for seed-variation stability.")

    outputs: list[tuple[int, Any]] = []
    for seed in seeds:
        outputs.append((int(seed), explain_fn(image, int(seed))))

    records: list[dict[str, float | int]] = []
    for (seed_a, output_a), (seed_b, output_b) in combinations(outputs, 2):
        comparison = compare_heatmaps(output_a, output_b, topk_fraction=topk_fraction)
        records.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "correlation": comparison["correlation"],
                "topk_iou": comparison["topk_iou"],
            }
        )

    aggregate = aggregate_stability_scores(records)
    aggregate.update(
        {
            "seeds": [int(seed) for seed in seeds],
            "mode": "seed_variation",
            "raw": records,
        }
    )
    return aggregate


def aggregate_stability_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "correlation_mean": float("nan"),
            "correlation_std": float("nan"),
            "topk_iou_mean": float("nan"),
            "topk_iou_std": float("nan"),
        }

    correlations = np.array([float(record["correlation"]) for record in records], dtype=np.float32)
    ious = np.array([float(record["topk_iou"]) for record in records], dtype=np.float32)
    return {
        "correlation_mean": float(correlations.mean()),
        "correlation_std": float(correlations.std(ddof=0)),
        "topk_iou_mean": float(ious.mean()),
        "topk_iou_std": float(ious.std(ddof=0)),
    }
