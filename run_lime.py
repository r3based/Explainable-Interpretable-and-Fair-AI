"""End-to-end LIME pipeline: load ViT → run on CIFAR-10 → explain with LIME → save outputs.

Usage (from repo root with venv active):
    python run_lime.py [--n-images 10] [--n-samples 500] [--n-segments 50] [--seed 42]
                       [--output-dir outputs/heatmaps] [--data-dir data]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.cifar10 import get_cifar10, denormalize, CIFAR10_CLASSES
from src.model.vit import ViTWrapper
from src.explainers.lime_explainer import LIMEImageExplainer
from src.visualization.heatmap import plot_lime_result, tensor_to_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIME explanations for ViT on CIFAR-10")
    p.add_argument("--n-images",   type=int, default=10,   help="Number of images to explain")
    p.add_argument("--n-samples",  type=int, default=1000, help="LIME neighbourhood samples")
    p.add_argument("--n-segments", type=int, default=50,   help="SLIC superpixel count")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/heatmaps")
    p.add_argument("--data-dir",   type=str, default="data")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    log.info("Loading ViT-B/16 …")
    model  = ViTWrapper()
    device = model.device
    log.info("Using device: %s", device)

    predict_fn = model.as_black_box()

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    log.info("Loading CIFAR-10 test split …")
    dataset = get_cifar10(root=args.data_dir, train=False)

    # Pick a fixed set of images (one per class for visual diversity)
    rng     = np.random.default_rng(args.seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(10)}
    for idx, (_, label) in enumerate(dataset):
        by_class[label].append(idx)

    selected_indices: list[int] = []
    n_per_class = max(1, args.n_images // 10)
    for cls_indices in by_class.values():
        chosen = rng.choice(cls_indices, size=min(n_per_class, len(cls_indices)), replace=False)
        selected_indices.extend(chosen.tolist())
    selected_indices = selected_indices[: args.n_images]

    # ------------------------------------------------------------------ #
    # LIME explainer
    # ------------------------------------------------------------------ #
    explainer = LIMEImageExplainer(
        n_samples=args.n_samples,
        n_segments=args.n_segments,
        seed=args.seed,
    )

    summary: list[dict] = []

    for i, ds_idx in enumerate(selected_indices):
        img_tensor, true_label = dataset[ds_idx]
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)

        # Model prediction
        class_idx, confidence = model.predict(img_tensor)
        class_name = CIFAR10_CLASSES[class_idx]
        true_name  = CIFAR10_CLASSES[true_label]

        log.info(
            "[%02d/%02d] true=%-12s pred=%-12s conf=%.3f",
            i + 1, len(selected_indices), true_name, class_name, confidence,
        )

        # Convert tensor to uint8 numpy for LIME
        img_np = tensor_to_numpy(img_tensor.squeeze(0))  # (224, 224, 3)

        # Run LIME
        result = explainer.explain(
            image=img_np,
            predict_fn=predict_fn,
            class_idx=class_idx,
            device=device,
        )

        log.info("  LIME done in %.1fs  (segments=%d)", result.runtime_seconds, result.extra["n_segments_actual"])

        # Save figure
        fig_path = out_dir / f"lime_{i:02d}_{class_name}.png"
        fig = plot_lime_result(
            image_tensor=img_tensor.squeeze(0),
            lime_result=result,
            class_idx=class_idx,
            save_path=fig_path,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

        # Save raw heatmap array
        np.save(out_dir / f"lime_{i:02d}_{class_name}_heatmap.npy", result.heatmap)

        summary.append({
            "index": int(ds_idx),
            "true_class": true_name,
            "pred_class": class_name,
            "confidence": round(confidence, 4),
            "runtime_s": round(result.runtime_seconds, 2),
            "n_segments": result.extra["n_segments_actual"],
            "figure": str(fig_path),
        })

    # ------------------------------------------------------------------ #
    # Save run summary
    # ------------------------------------------------------------------ #
    summary_path = out_dir / "lime_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"args": vars(args), "results": summary}, f, indent=2)
    log.info("Summary saved → %s", summary_path)
    log.info("Done. %d explanations written to %s", len(summary), out_dir)


if __name__ == "__main__":
    main()
