from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .counterfactual_generator import CounterfactualResult


def _to_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 4:
        image = image[0]
    return image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0)


def _normalize_perturbation(delta: torch.Tensor) -> torch.Tensor:
    x = delta.abs()
    maxv = x.max().item()
    if maxv < 1e-12:
        return x
    return x / maxv


def save_counterfactual_panel(result: CounterfactualResult, save_path: str | Path, title: str | None = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = _to_hwc(result.original_image)
    cf = _to_hwc(result.counterfactual_image)
    perturb = _normalize_perturbation(_to_hwc(result.perturbation))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original.numpy())
    axes[0].set_title(f"Original\nclass={result.original_class}")
    axes[1].imshow(perturb.numpy())
    axes[1].set_title(f"|Perturbation|\nL2={result.perturbation_l2:.4f}")
    axes[2].imshow(cf.numpy())
    axes[2].set_title(f"Counterfactual\nclass={result.final_class}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title or f"success={result.success}, steps={result.steps_run}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
