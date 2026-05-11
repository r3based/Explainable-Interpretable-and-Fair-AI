"""Fine-tunable ViT classifier for CIFAR-10.

This module keeps the same prediction interface as ``ViTWrapper`` while using
a real 10-class classification head instead of ImageNet anchor remapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import timm


@dataclass
class ViTCIFARConfig:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 10
    freeze_backbone: bool = False
    unfreeze_last_blocks: int = 0


class ViTCIFAR10Classifier(nn.Module):
    """Pretrained ViT with a CIFAR-10 head and project-compatible helpers."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 10,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = ViTCIFARConfig(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.configure_trainable_layers(
            freeze_backbone=freeze_backbone,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        self.to(self.device)

    def configure_trainable_layers(self, freeze_backbone: bool, unfreeze_last_blocks: int = 0) -> None:
        """Choose which parts are trainable for head-only or partial fine-tuning."""
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

        if not freeze_backbone:
            return

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        head = getattr(self.model, "head", None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(True)

        blocks = getattr(self.model, "blocks", None)
        if blocks is not None and unfreeze_last_blocks > 0:
            for block in list(blocks)[-int(unfreeze_last_blocks):]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

        norm = getattr(self.model, "norm", None)
        if norm is not None and unfreeze_last_blocks > 0:
            for parameter in norm.parameters():
                parameter.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    def cifar10_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def cifar10_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return self.cifar10_logits(x).softmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.cifar10_probabilities(x)

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        probs = self.predict_proba(x)
        idx = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, idx].item())
        return idx, conf

    def as_black_box(self) -> Callable[[torch.Tensor], torch.Tensor]:
        def _call(x: torch.Tensor) -> torch.Tensor:
            return self.predict_proba(x)

        return _call

    def as_cifar10_classifier(self) -> nn.Module:
        return self

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "extra": extra or {},
            },
            target,
        )
        return target

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> "ViTCIFAR10Classifier":
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = payload.get("config", {})
        model = cls(
            model_name=config.get("model_name", "vit_base_patch16_224"),
            pretrained=False,
            num_classes=int(config.get("num_classes", 10)),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            unfreeze_last_blocks=int(config.get("unfreeze_last_blocks", 0)),
            device=device,
        )
        state_dict = payload.get("state_dict", payload)
        model.model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model
