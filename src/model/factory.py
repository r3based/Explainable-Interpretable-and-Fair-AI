"""Model loading helpers for baseline, fine-tuned, and robust variants."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .vit import ViTWrapper
from .vit_cifar import ViTCIFAR10Classifier

ModelKind = Literal["anchor", "finetuned", "robust"]


def load_project_model(
    model_kind: ModelKind = "anchor",
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    unfreeze_last_blocks: int = 0,
    require_checkpoint: bool = False,
):
    """Load a model variant with the common project prediction API."""
    if model_kind == "anchor":
        return ViTWrapper(device=device)

    if model_kind not in {"finetuned", "robust"}:
        raise ValueError("model_kind must be one of: anchor, finetuned, robust")

    if checkpoint:
        return ViTCIFAR10Classifier.load_from_checkpoint(checkpoint, device=device)

    if require_checkpoint:
        raise ValueError(f"--checkpoint is required for model_kind={model_kind}")

    return ViTCIFAR10Classifier(
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        unfreeze_last_blocks=unfreeze_last_blocks,
        device=device,
    )
