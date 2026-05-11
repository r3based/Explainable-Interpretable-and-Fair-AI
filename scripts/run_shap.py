from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bootstrap  # noqa: F401

from src.data import CIFAR10_CLASSES, DEFAULT_REFERENCE_DIR, get_cifar10, load_or_build_reference_set, summarize_reference_set
from src.explainers.shap_explainer import SHAPImageExplainer
from src.model import load_project_model
from src.utils import set_seed
from src.visualization.heatmap import overlay_heatmap, tensor_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP explanations for the ViT CIFAR-10 pipeline.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/shap")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=5)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--background-strategy", type=str, default="stratified", choices=["random", "stratified"])
    parser.add_argument("--reference-manifest", type=str, default=None)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-reference-set", action="store_true")
    parser.add_argument("--model-kind", type=str, default="anchor", choices=["anchor", "finetuned", "robust"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def _select_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    size = min(int(subset_size), int(dataset_len))
    rng = np.random.default_rng(seed)
    return [int(index) for index in rng.choice(dataset_len, size=size, replace=False).tolist()]


def _save_overlay(image_tensor, heatmap: np.ndarray, class_name: str, save_path: Path) -> None:
    image_np = tensor_to_numpy(image_tensor)
    overlay = overlay_heatmap(image_np, heatmap=heatmap, alpha=0.6)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[1].imshow(overlay)
    axes[1].set_title(f"SHAP ({class_name})")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(
        f"Running SHAP: model={args.model_kind}, subset_size={args.subset_size}, "
        f"background_size={args.background_size}, nsamples={args.nsamples}, device={args.device}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.reference_manifest) if args.reference_manifest else (
        DEFAULT_REFERENCE_DIR / f"cifar10_train_{args.background_strategy}_{args.background_size}_seed{args.seed}.json"
    )

    model = load_project_model(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        require_checkpoint=args.model_kind != "anchor" and not args.allow_random_init,
    )
    predict_fn = model.as_black_box()

    dataset = get_cifar10(root=args.data_dir, train=False)
    background_dataset = get_cifar10(root=args.data_dir, train=True)
    background_tensor, manifest, manifest_path = load_or_build_reference_set(
        dataset=background_dataset,
        manifest_path=manifest_path,
        strategy=args.background_strategy,
        size=args.background_size,
        seed=args.seed,
        reuse_existing=args.reuse_reference_set,
    )
    print(f"Reference set ready: {manifest_path}", flush=True)

    explainer = SHAPImageExplainer(
        model=model,
        background_data=background_tensor,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    selected_indices = _select_subset_indices(len(dataset), args.subset_size, args.seed)
    summary: list[dict[str, object]] = []

    for run_idx, dataset_index in enumerate(selected_indices):
        print(
            f"SHAP image {run_idx + 1}/{len(selected_indices)}: dataset_index={dataset_index}",
            flush=True,
        )
        image_tensor, true_label = dataset[dataset_index]
        prediction_idx, confidence = model.predict(image_tensor.unsqueeze(0))
        result = explainer.explain(
            image=image_tensor,
            predict_fn=predict_fn,
            class_idx=None,
            device=args.device,
        )

        class_name = CIFAR10_CLASSES[result.class_idx]
        heatmap_path = output_dir / f"shap_{run_idx:02d}_{class_name}_heatmap.npy"
        raw_values_path = output_dir / f"shap_{run_idx:02d}_{class_name}_raw.npy"
        figure_path = output_dir / f"shap_{run_idx:02d}_{class_name}.png"
        np.save(heatmap_path, result.heatmap)
        if result.raw_values is not None:
            np.save(raw_values_path, result.raw_values)
        _save_overlay(image_tensor=image_tensor, heatmap=result.heatmap, class_name=class_name, save_path=figure_path)
        print(
            f"SHAP image {run_idx + 1}/{len(selected_indices)} done: "
            f"class={class_name}, runtime={result.runtime_seconds:.2f}s, figure={figure_path}",
            flush=True,
        )

        summary.append(
            {
                "index": int(dataset_index),
                "true_class": CIFAR10_CLASSES[int(true_label)],
                "pred_class": CIFAR10_CLASSES[int(prediction_idx)],
                "confidence": float(confidence),
                "explained_class": class_name,
                "runtime_seconds": float(result.runtime_seconds),
                "heatmap_path": str(heatmap_path),
                "raw_values_path": str(raw_values_path) if result.raw_values is not None else None,
                "figure_path": str(figure_path),
                "extra": result.extra,
            }
        )

    payload = {
        "config": vars(args),
        "model": {"kind": args.model_kind, "checkpoint": args.checkpoint, "model_name": args.model_name},
        "reference_manifest_path": str(manifest_path),
        "reference_manifest": summarize_reference_set(manifest),
        "selected_indices": selected_indices,
        "results": summary,
    }
    with open(output_dir / "shap_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved SHAP summary to {output_dir / 'shap_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
