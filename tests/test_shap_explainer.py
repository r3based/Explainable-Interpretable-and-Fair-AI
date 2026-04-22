from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

pytest.importorskip("shap")

from src.explainers.shap_explainer import SHAPImageExplainer, SHAPResult


class DummyViTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.proj = nn.Conv2d(3, 10, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.fill_(0.1)
            self.proj.bias.copy_(torch.linspace(-0.2, 0.2, steps=10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).mean(dim=(2, 3))

    def cifar10_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).softmax(dim=-1)

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        probs = self.cifar10_probabilities(x)
        idx = int(probs.argmax(dim=-1).item())
        return idx, float(probs[0, idx].item())


def test_shap_explainer_returns_expected_result_shape() -> None:
    model = DummyViTWrapper().eval()
    background = torch.zeros(2, 3, 4, 4)
    image = torch.ones(3, 4, 4) * 0.25

    explainer = SHAPImageExplainer(
        model=model,
        background_data=background,
        nsamples=8,
        batch_size=2,
        seed=7,
    )
    result = explainer.explain(image=image, class_idx=2, device="cpu")

    assert isinstance(result, SHAPResult)
    assert result.heatmap.shape == (4, 4)
    assert result.raw_values is not None
    assert result.raw_values.shape == (3, 4, 4)
    assert result.class_idx == 2
    assert math.isfinite(result.runtime_seconds)
    assert result.extra["background_size"] == 2
