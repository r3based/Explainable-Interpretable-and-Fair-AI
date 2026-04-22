from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from counterfactuals import CounterfactualConfig, GradientCounterfactualGenerator, save_counterfactual_panel
from utils import set_seed


class ToyBrightnessClassifier(torch.nn.Module):
    """A tiny deterministic image classifier used for sanity checks.

    class 0: dark image
    class 1: bright image
    """

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_intensity = x.mean(dim=(1, 2, 3))
        logits = torch.stack([0.5 - mean_intensity, mean_intensity - 0.5], dim=1)
        return logits


def build_demo_image(size: int = 32, value: float = 0.2) -> torch.Tensor:
    return torch.full((1, 3, size, size), fill_value=value, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "counterfactual_demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--lambda-l2", type=float, default=1e-2)
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--target-class", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ToyBrightnessClassifier().eval()
    image = build_demo_image()

    config = CounterfactualConfig(
        steps=args.steps,
        step_size=args.step_size,
        lambda_l2=args.lambda_l2,
        lambda_tv=args.lambda_tv,
        target_mode="untargeted" if args.target_class is None else "targeted",
    )

    generator = GradientCounterfactualGenerator(model)
    result = generator.generate(image, target_class=args.target_class, config=config)

    save_counterfactual_panel(result, output_dir / "counterfactual_panel.png", title="Toy counterfactual sanity check")

    summary = {
        "original_class": result.original_class,
        "final_class": result.final_class,
        "success": result.success,
        "steps_run": result.steps_run,
        "original_confidence": result.original_confidence,
        "final_confidence": result.final_confidence,
        "perturbation_l2": result.perturbation_l2,
        "perturbation_linf": result.perturbation_linf,
        "config": result.config,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
