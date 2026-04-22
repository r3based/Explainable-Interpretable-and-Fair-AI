from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from counterfactuals import CounterfactualConfig, GradientCounterfactualGenerator
from utils import set_seed


class ToyBrightnessClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_intensity = x.mean(dim=(1, 2, 3))
        logits = torch.stack([0.5 - mean_intensity, mean_intensity - 0.5], dim=1)
        return logits


def test_untargeted_counterfactual_success() -> None:
    set_seed(7)
    model = ToyBrightnessClassifier().eval()
    image = torch.full((1, 3, 16, 16), 0.2)
    generator = GradientCounterfactualGenerator(model)
    config = CounterfactualConfig(steps=120, step_size=0.08, lambda_l2=1e-3, lambda_tv=0.0)
    result = generator.generate(image, config=config)

    assert result.original_class == 0
    assert result.final_class == 1
    assert result.success is True
    assert result.perturbation_l2 > 0.0
