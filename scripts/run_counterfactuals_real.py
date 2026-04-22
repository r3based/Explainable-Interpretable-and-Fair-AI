from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from counterfactuals import (
    CounterfactualConfig,
    GradientCounterfactualGenerator,
    save_counterfactual_panel,
)
from data.cifar10 import get_cifar10, denormalize
from model.resnet import load_model
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--save-dir", type=str, default="artifacts/counterfactuals_real")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--lambda-l2", type=float, default=0.01)
    parser.add_argument("--lambda-tv", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_to_saveable_image(x: torch.Tensor) -> torch.Tensor:
    """
    x shape: [1, C, H, W] or [C, H, W]
    Returns [1, C, H, W] in [0,1]
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    dataset = get_cifar10(root=args.root, train=args.train)
    model = load_model(device)
    generator = GradientCounterfactualGenerator(model)

    config = CounterfactualConfig(
        steps=args.steps,
        step_size=args.step_size,
        lambda_l2=args.lambda_l2,
        lambda_tv=args.lambda_tv,
    )

    all_results = []

    for idx in range(args.start_index, args.start_index + args.num_samples):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)

        result = generator.generate(image, config=config)

        # если image нормализован под ViT/модель, для картинки лучше денормализовать
        try:
            original_vis = denormalize(result.original_image.clone())
            counterfactual_vis = denormalize(result.counterfactual_image.clone())
        except Exception:
            original_vis = tensor_to_saveable_image(result.original_image.clone())
            counterfactual_vis = tensor_to_saveable_image(result.counterfactual_image.clone())

        sample_dir = save_root / f"sample_{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        panel_path = sample_dir / "counterfactual_panel.png"
        save_counterfactual_panel(
            original_image=original_vis,
            counterfactual_image=counterfactual_vis,
            perturbation=result.perturbation,
            save_path=str(panel_path),
        )

        summary = {
            "index": idx,
            "true_label": int(label),
            "original_class": int(result.original_class),
            "final_class": int(result.final_class),
            "success": bool(result.success),
            "steps_run": int(result.steps_run),
            "original_confidence": float(result.original_confidence),
            "final_confidence": float(result.final_confidence),
            "perturbation_l2": float(result.perturbation_l2),
            "perturbation_linf": float(result.perturbation_linf),
            "panel_path": str(panel_path),
            "config": result.config,
        }

        with open(sample_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results.append(summary)
        print(
            f"[{idx}] success={summary['success']} "
            f"{summary['original_class']} -> {summary['final_class']} "
            f"L2={summary['perturbation_l2']:.4f} "
            f"Linf={summary['perturbation_linf']:.4f}"
        )

    success_rate = sum(int(r["success"]) for r in all_results) / max(len(all_results), 1)
    avg_l2 = sum(r["perturbation_l2"] for r in all_results) / max(len(all_results), 1)
    avg_linf = sum(r["perturbation_linf"] for r in all_results) / max(len(all_results), 1)
    avg_steps = sum(r["steps_run"] for r in all_results) / max(len(all_results), 1)

    aggregate = {
        "num_samples": len(all_results),
        "success_rate": success_rate,
        "avg_l2": avg_l2,
        "avg_linf": avg_linf,
        "avg_steps_run": avg_steps,
        "results": all_results,
    }

    with open(save_root / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print("\nAggregate:")
    print(json.dumps({
        "num_samples": len(all_results),
        "success_rate": success_rate,
        "avg_l2": avg_l2,
        "avg_linf": avg_linf,
        "avg_steps_run": avg_steps,
    }, indent=2))


if __name__ == "__main__":
    main()