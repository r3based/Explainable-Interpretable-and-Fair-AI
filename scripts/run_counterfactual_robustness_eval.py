from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import bootstrap  # noqa: F401

from src.counterfactuals import CounterfactualConfig, generate_counterfactual_for_normalized_input
from src.data import CIFAR10_CLASSES, VIT_MEAN, VIT_STD, get_cifar10
from src.model import load_project_model
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model vulnerability with counterfactual perturbations.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/counterfactual_robustness")
    parser.add_argument("--model-kind", type=str, default="anchor", choices=["anchor", "finetuned", "robust"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--lambda-l2", type=float, default=0.01)
    parser.add_argument("--lambda-tv", type=float, default=0.0001)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def select_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    size = min(int(subset_size), int(dataset_len))
    return [int(index) for index in rng.choice(dataset_len, size=size, replace=False).tolist()]


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    n = max(len(records), 1)
    successes = np.array([bool(record["success"]) for record in records], dtype=np.float32)
    l2 = np.array([float(record["perturbation_l2"]) for record in records], dtype=np.float32)
    linf = np.array([float(record["perturbation_linf"]) for record in records], dtype=np.float32)
    steps = np.array([float(record["steps_run"]) for record in records], dtype=np.float32)
    confidence_drop = np.array(
        [float(record["original_confidence"]) - float(record["final_confidence"]) for record in records],
        dtype=np.float32,
    )
    return {
        "num_images": len(records),
        "success_rate": float(successes.mean()) if len(records) else None,
        "failure_rate": float(1.0 - successes.mean()) if len(records) else None,
        "l2_mean": float(l2.mean()) if len(records) else None,
        "l2_std": float(l2.std(ddof=0)) if len(records) else None,
        "linf_mean": float(linf.mean()) if len(records) else None,
        "linf_std": float(linf.std(ddof=0)) if len(records) else None,
        "steps_mean": float(steps.mean()) if len(records) else None,
        "confidence_drop_mean": float(confidence_drop.mean()) if len(records) else None,
        "successful_images": int(successes.sum()) if len(records) else 0,
        "failed_images": int(n - successes.sum()) if len(records) else 0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw" / args.model_kind
    raw_dir.mkdir(parents=True, exist_ok=True)

    model = load_project_model(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        device=device,
        model_name=args.model_name,
        require_checkpoint=args.model_kind != "anchor" and not args.allow_random_init,
    )
    classifier = model.as_cifar10_classifier()
    classifier.eval()

    dataset = get_cifar10(root=args.data_dir, train=False)
    selected_indices = select_subset_indices(len(dataset), args.subset_size, args.seed)
    config = CounterfactualConfig(
        steps=args.steps,
        step_size=args.step_size,
        lambda_l2=args.lambda_l2,
        lambda_tv=args.lambda_tv,
        confidence_margin=args.confidence_margin,
        target_mode="untargeted",
    )
    mean = torch.tensor(VIT_MEAN, dtype=torch.float32)
    std = torch.tensor(VIT_STD, dtype=torch.float32)

    records: list[dict[str, object]] = []
    for dataset_index in selected_indices:
        image_tensor, true_label = dataset[dataset_index]
        image_tensor = image_tensor.to(device)
        pred_class, pred_conf = model.predict(image_tensor.unsqueeze(0))
        result = generate_counterfactual_for_normalized_input(
            model=classifier,
            normalized_image=image_tensor.unsqueeze(0),
            mean=mean,
            std=std,
            config=config,
        )
        record = {
            "dataset_index": int(dataset_index),
            "true_class": CIFAR10_CLASSES[int(true_label)],
            "pred_class": CIFAR10_CLASSES[int(pred_class)],
            "pred_confidence": float(pred_conf),
            "original_class": CIFAR10_CLASSES[int(result.original_class)],
            "final_class": CIFAR10_CLASSES[int(result.final_class)],
            "success": bool(result.success),
            "steps_run": int(result.steps_run),
            "original_confidence": float(result.original_confidence),
            "final_confidence": float(result.final_confidence),
            "perturbation_l2": float(result.perturbation_l2),
            "perturbation_linf": float(result.perturbation_linf),
        }
        records.append(record)
        with open(raw_dir / f"image_{int(dataset_index):05d}.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(
            f"idx={dataset_index} success={record['success']} "
            f"{record['original_class']}->{record['final_class']} "
            f"L2={record['perturbation_l2']:.3f} Linf={record['perturbation_linf']:.3f}"
        )

    payload = {
        "args": vars(args),
        "device": str(device),
        "selected_indices": selected_indices,
        "summary": summarize(records),
        "per_image": records,
    }
    output_path = output_dir / f"{args.model_kind}_counterfactual_robustness.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
