"""SHAP image explanations for the CIFAR-10 ViT pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import shap
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    shap = None


@dataclass
class SHAPResult:
    heatmap: np.ndarray
    raw_values: np.ndarray | None
    class_idx: int
    runtime_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class _TargetProbabilityModel(nn.Module):
    """Wrap the shared ViT model so SHAP explains one CIFAR-10 target at a time."""

    def __init__(self, model: nn.Module, class_idx: int):
        super().__init__()
        self.model = model
        self.class_idx = int(class_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.model.cifar10_probabilities(x)
        return probs[:, self.class_idx : self.class_idx + 1]


class SHAPImageExplainer:
    """Expected-gradients SHAP on top of the existing ViT wrapper."""

    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        nsamples: int = 200,
        batch_size: int = 8,
        local_smoothing: float = 0.0,
        seed: int = 42,
    ) -> None:
        if shap is None:
            raise ImportError("SHAP is not installed. Add 'shap' to the environment before using SHAPImageExplainer.")
        if background_data.dim() != 4:
            raise ValueError("background_data must have shape (N, C, H, W).")

        self.model = model
        self.background_data = background_data.detach().cpu()
        self.nsamples = int(nsamples)
        self.batch_size = int(batch_size)
        self.local_smoothing = float(local_smoothing)
        self.seed = int(seed)

    def explain(
        self,
        image: torch.Tensor,
        predict_fn=None,
        class_idx: int | None = None,
        device: str | torch.device | None = None,
    ) -> SHAPResult:
        """Explain a single image in the existing normalized ViT tensor format."""
        image_tensor = self._ensure_batched_image(image)
        target_device = self._resolve_device(device)
        image_tensor = image_tensor.to(target_device)
        background = self.background_data.to(target_device)
        class_idx_was_explicit = class_idx is not None

        if class_idx is None:
            class_idx = self._infer_class_idx(image_tensor, predict_fn)

        if not 0 <= int(class_idx) < 10:
            raise ValueError("class_idx must be in the CIFAR-10 range [0, 9].")

        target_model = _TargetProbabilityModel(self.model, class_idx=int(class_idx))
        t0 = time.perf_counter()
        explainer = shap.GradientExplainer(
            target_model,
            background,
            batch_size=self.batch_size,
            local_smoothing=self.local_smoothing,
        )
        shap_values = explainer.shap_values(
            image_tensor,
            nsamples=self.nsamples,
            rseed=self.seed,
        )
        runtime = time.perf_counter() - t0

        raw_values = self._coerce_shap_values(shap_values)
        per_image_values = raw_values[0]
        heatmap = per_image_values.mean(axis=0).astype(np.float32)

        return SHAPResult(
            heatmap=heatmap,
            raw_values=per_image_values.astype(np.float32),
            class_idx=int(class_idx),
            runtime_seconds=runtime,
            extra={
                "background_size": int(background.shape[0]),
                "nsamples": self.nsamples,
                "batch_size": self.batch_size,
                "local_smoothing": self.local_smoothing,
                "seed": self.seed,
                "device": str(target_device),
                "target_selection": "explicit" if class_idx_was_explicit else "predicted",
            },
        )

    def _ensure_batched_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4 and image.shape[0] == 1:
            return image
        raise ValueError("image must have shape (C, H, W) or (1, C, H, W).")

    def _resolve_device(self, device: str | torch.device | None) -> torch.device:
        target_device = torch.device(device or getattr(self.model, "device", "cpu"))
        model_device = torch.device(getattr(self.model, "device", target_device))
        if model_device != target_device:
            self.model.to(target_device)
            if hasattr(self.model, "device"):
                self.model.device = target_device
        return target_device

    def _infer_class_idx(self, image: torch.Tensor, predict_fn=None) -> int:
        with torch.no_grad():
            if predict_fn is not None:
                probs = predict_fn(image)
                return int(probs.argmax(dim=-1).item())

            if hasattr(self.model, "predict"):
                predicted_idx, _ = self.model.predict(image)
                return int(predicted_idx)

            probs = self.model.cifar10_probabilities(image)
            return int(probs.argmax(dim=-1).item())

    def _coerce_shap_values(self, shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            if len(shap_values) != 1:
                raise ValueError("Expected a single SHAP output for the target-specific model.")
            shap_values = shap_values[0]

        values = np.asarray(shap_values)
        if values.ndim == 5 and values.shape[-1] == 1:
            values = np.squeeze(values, axis=-1)
        if values.ndim == 3:
            values = values[np.newaxis, ...]
        if values.ndim != 4:
            raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
        return values.astype(np.float32)
