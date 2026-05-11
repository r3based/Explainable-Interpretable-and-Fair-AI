"""LIME explanations for ViT-B/16 on CIFAR-10.

Usage:
    python scripts/run_lime.py [--n-images 10] [--n-samples 1000] [--n-segments 50]
                               [--output-dir artifacts/lime] [--data-dir data] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import bootstrap  # noqa: F401

from src.data.cifar10 import CIFAR10_CLASSES, get_cifar10
from src.explainers.lime_explainer import LIMEImageExplainer
from src.model import load_project_model
from src.utils import set_seed
from src.visualization.heatmap import plot_lime_result, tensor_to_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIME explanations for ViT on CIFAR-10")
    p.add_argument("--n-images",   type=int, default=10,            help="Number of images to explain")
    p.add_argument("--n-samples",  type=int, default=1000,          help="LIME neighbourhood samples")
    p.add_argument("--n-segments", type=int, default=50,            help="SLIC superpixel count")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output-dir", type=str, default="artifacts/lime")
    p.add_argument("--data-dir",   type=str, default="data")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--model-kind", type=str, default="anchor", choices=["anchor", "finetuned", "robust"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    p.add_argument("--allow-random-init", action="store_true")
    return p.parse_args()


def _pick_indices(dataset, n_images: int, seed: int) -> list[int]:
    """One image per class, up to n_images total."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(dataset[idx][1]) for idx in range(len(dataset))]
    for idx, label in enumerate(targets):
        by_class[int(label)].append(int(idx))

    selected: list[int] = []
    n_per_class = max(1, n_images // 10)
    for cls_indices in by_class.values():
        chosen = rng.choice(cls_indices, size=min(n_per_class, len(cls_indices)), replace=False)
        selected.extend(chosen.tolist())
    return selected[:n_images]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", args.model_kind)
    model = load_project_model(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        require_checkpoint=args.model_kind != "anchor" and not args.allow_random_init,
    )
    device    = model.device
    predict_fn = model.as_black_box()
    log.info("Device: %s", device)

    log.info("Loading CIFAR-10 test split …")
    dataset = get_cifar10(root=args.data_dir, train=False)
    indices = _pick_indices(dataset, args.n_images, args.seed)

    explainer = LIMEImageExplainer(
        n_samples=args.n_samples,
        n_segments=args.n_segments,
        seed=args.seed,
    )

    summary: list[dict] = []

    for i, ds_idx in enumerate(indices):
        img_tensor, true_label = dataset[ds_idx]
        img_tensor = img_tensor.unsqueeze(0)

        class_idx, confidence = model.predict(img_tensor)
        class_name = CIFAR10_CLASSES[class_idx]
        true_name  = CIFAR10_CLASSES[true_label]

        log.info("[%02d/%02d] true=%-12s pred=%-12s conf=%.3f",
                 i + 1, len(indices), true_name, class_name, confidence)

        img_np = tensor_to_numpy(img_tensor.squeeze(0))
        result = explainer.explain(
            image=img_np,
            predict_fn=predict_fn,
            class_idx=class_idx,
            device=device,
        )
        log.info("  LIME done in %.1fs  (segments=%d)",
                 result.runtime_seconds, result.extra["n_segments_actual"])

        fig_path = out_dir / f"lime_{i:02d}_{class_name}.png"
        fig = plot_lime_result(
            image_tensor=img_tensor.squeeze(0),
            lime_result=result,
            class_idx=class_idx,
            save_path=fig_path,
        )
        plt.close(fig)
        np.save(out_dir / f"lime_{i:02d}_{class_name}_heatmap.npy", result.heatmap)

        summary.append({
            "index":      int(ds_idx),
            "true_class": true_name,
            "pred_class": class_name,
            "confidence": round(float(confidence), 4),
            "runtime_s":  round(float(result.runtime_seconds), 2),
            "n_segments": int(result.extra["n_segments_actual"]),
            "figure":     str(fig_path),
        })

    with open(out_dir / "lime_summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "model": {"kind": args.model_kind, "checkpoint": args.checkpoint, "model_name": args.model_name},
                "results": summary,
            },
            f,
            indent=2,
        )
    log.info("Done — %d explanations → %s", len(summary), out_dir)


if __name__ == "__main__":
    main()
