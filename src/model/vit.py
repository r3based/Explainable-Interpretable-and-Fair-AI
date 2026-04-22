"""ViT-B/16 inference wrapper.

Loads a timm ViT-B/16 pretrained on ImageNet-21k → fine-tuned on ImageNet-1k,
exposes a clean predict interface, and maps ImageNet logits to CIFAR-10 class
indices via a fixed label remapping so that evaluation stays in CIFAR-10 space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Callable


# ImageNet-1k class indices that best correspond to each CIFAR-10 class.
# These are the canonical "anchor" labels used throughout the project.
CIFAR10_TO_IMAGENET: dict[int, int] = {
    0: 404,   # airplane  → airliner
    1: 407,   # automobile → beach wagon
    2: 88,    # bird       → macaw
    3: 281,   # cat        → tabby cat
    4: 352,   # deer       → impala
    5: 207,   # dog        → golden retriever
    6: 32,    # frog       → tree frog
    7: 603,   # horse      → horse cart
    8: 628,   # ship       → liner (ocean liner)
    9: 867,   # truck      → trailer truck
}

IMAGENET_TO_CIFAR10: dict[int, int] = {v: k for k, v in CIFAR10_TO_IMAGENET.items()}


class ViTWrapper(nn.Module):
    """Thin wrapper around timm ViT-B/16 for the project pipeline.

    forward() returns raw ImageNet-1k logits (1000 dims).
    predict() returns (cifar10_class_idx, confidence) for a single image tensor.
    predict_proba() returns a (N, 10) probability tensor in CIFAR-10 class order,
    keeping only the 10 anchor columns and renormalising.
    """

    MODEL_NAME = "vit_base_patch16_224"

    def __init__(self, device: str | torch.device | None = None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(self.MODEL_NAME, pretrained=True)
        self.model.eval().to(self.device)
        # Freeze all parameters – we never train the backbone
        for p in self.model.parameters():
            p.requires_grad_(False)

        anchor_indices = [CIFAR10_TO_IMAGENET[i] for i in range(10)]
        self.register_buffer(
            "_anchor_idx",
            torch.tensor(anchor_indices, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Core forward – full 1000-class logits
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    # ------------------------------------------------------------------
    # CIFAR-10-space helpers
    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return (N, 10) softmax probabilities over CIFAR-10 anchor classes."""
        with torch.no_grad():
            logits = self(x)                          # (N, 1000)
        anchor_logits = logits[:, self._anchor_idx]   # (N, 10)
        return anchor_logits.softmax(dim=-1)

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        """Return (cifar10_class_idx, confidence) for a single image (1, C, H, W)."""
        probs = self.predict_proba(x)                 # (1, 10)
        idx   = int(probs.argmax(dim=-1).item())
        conf  = float(probs[0, idx].item())
        return idx, conf

    # ------------------------------------------------------------------
    # Convenience: callable for LIME / SHAP black-box interface
    # ------------------------------------------------------------------
    def as_black_box(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a function (N, C, H, W) → (N, 10) probs (no grad)."""
        def _call(x: torch.Tensor) -> torch.Tensor:
            return self.predict_proba(x)
        return _call
