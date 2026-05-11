from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import bootstrap  # noqa: F401

from src.counterfactuals import CounterfactualConfig, generate_counterfactual_for_normalized_input
from src.data import DEFAULT_REFERENCE_DIR, CIFAR10_CLASSES, get_cifar10, load_or_build_reference_set
from src.data.cifar10 import VIT_MEAN, VIT_STD
from src.evaluation.faithfulness import deletion_curve, insertion_curve
from src.explainers.lime_explainer import LIMEImageExplainer
from src.explainers.shap_explainer import SHAPImageExplainer
from src.model import load_project_model
from src.utils import set_seed
from src.visualization.heatmap import tensor_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline faithfulness metrics for explanation methods.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/faithfulness")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--background-strategy", type=str, default="stratified", choices=["random", "stratified"])
    parser.add_argument("--reference-manifest", type=str, default=None)
    parser.add_argument("--reuse-reference-set", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["lime", "shap", "counterfactual"], choices=["lime", "shap", "counterfactual"])
    parser.add_argument("--curve-steps", type=int, default=20)
    parser.add_argument("--baseline", type=str, default="zero", choices=["zero", "mean"])
    parser.add_argument("--lime-samples", type=int, default=200)
    parser.add_argument("--lime-segments", type=int, default=50)
    parser.add_argument("--shap-nsamples", type=int, default=64)
    parser.add_argument("--shap-batch-size", type=int, default=8)
    parser.add_argument("--cf-steps", type=int, default=80)
    parser.add_argument("--cf-step-size", type=float, default=0.03)
    parser.add_argument("--cf-lambda-l2", type=float, default=1e-2)
    parser.add_argument("--cf-lambda-tv", type=float, default=1e-4)
    parser.add_argument("--model-kind", type=str, default="anchor", choices=["anchor", "finetuned", "robust"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def _select_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    size = min(int(subset_size), int(dataset_len))
    return [int(index) for index in rng.choice(dataset_len, size=size, replace=False).tolist()]


def _curve_auc(curve: dict[str, object]) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(curve["scores"], curve["fractions"]))
    return float(np.trapz(curve["scores"], curve["fractions"]))


def _counterfactual_heatmap(model, image_tensor: torch.Tensor, args: argparse.Namespace) -> object:
    config = CounterfactualConfig(
        steps=args.cf_steps,
        step_size=args.cf_step_size,
        lambda_l2=args.cf_lambda_l2,
        lambda_tv=args.cf_lambda_tv,
        target_mode="untargeted",
    )
    mean = torch.tensor(VIT_MEAN, dtype=image_tensor.dtype)
    std  = torch.tensor(VIT_STD,  dtype=image_tensor.dtype)
    return generate_counterfactual_for_normalized_input(
        model=model.as_cifar10_classifier(),
        normalized_image=image_tensor.unsqueeze(0),
        mean=mean,
        std=std,
        config=config,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(
        f"Running faithfulness eval: model={args.model_kind}, methods={args.methods}, "
        f"subset_size={args.subset_size}, device={args.device}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    raw_curve_dir = output_dir / "raw_curves"
    raw_curve_dir.mkdir(parents=True, exist_ok=True)

    model = load_project_model(
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        require_checkpoint=args.model_kind != "anchor" and not args.allow_random_init,
    )
    predict_fn = model.as_black_box()
    dataset = get_cifar10(root=args.data_dir, train=False)
    selected_indices = _select_subset_indices(len(dataset), args.subset_size, args.seed)

    lime_explainer = None
    shap_explainer = None
    reference_manifest_path = None

    if "lime" in args.methods:
        lime_explainer = LIMEImageExplainer(
            n_samples=args.lime_samples,
            n_segments=args.lime_segments,
            seed=args.seed,
        )

    if "shap" in args.methods:
        reference_manifest_path = Path(args.reference_manifest) if args.reference_manifest else (
            DEFAULT_REFERENCE_DIR / f"cifar10_train_{args.background_strategy}_{args.background_size}_seed{args.seed}.json"
        )
        background_dataset = get_cifar10(root=args.data_dir, train=True)
        background_tensor, _, reference_manifest_path = load_or_build_reference_set(
            dataset=background_dataset,
            manifest_path=reference_manifest_path,
            strategy=args.background_strategy,
            size=args.background_size,
            seed=args.seed,
            reuse_existing=args.reuse_reference_set,
        )
        shap_explainer = SHAPImageExplainer(
            model=model,
            background_data=background_tensor,
            nsamples=args.shap_nsamples,
            batch_size=args.shap_batch_size,
            seed=args.seed,
        )

    per_method: dict[str, list[dict[str, object]]] = {method: [] for method in args.methods}

    for image_pos, dataset_index in enumerate(selected_indices, start=1):
        print(f"Faithfulness image {image_pos}/{len(selected_indices)}: dataset_index={dataset_index}", flush=True)
        image_tensor, true_label = dataset[dataset_index]
        class_idx, confidence = model.predict(image_tensor.unsqueeze(0))

        method_outputs: dict[str, object] = {}
        if lime_explainer is not None:
            print(f"  generating LIME for image {image_pos}", flush=True)
            method_outputs["lime"] = lime_explainer.explain(
                image=tensor_to_numpy(image_tensor),
                predict_fn=predict_fn,
                class_idx=class_idx,
                device=args.device,
            )
        if shap_explainer is not None:
            print(f"  generating SHAP for image {image_pos}", flush=True)
            method_outputs["shap"] = shap_explainer.explain(
                image=image_tensor,
                predict_fn=predict_fn,
                class_idx=class_idx,
                device=args.device,
            )
        if "counterfactual" in args.methods:
            print(f"  generating counterfactual for image {image_pos}", flush=True)
            method_outputs["counterfactual"] = _counterfactual_heatmap(model=model, image_tensor=image_tensor, args=args)

        for method_name, method_output in method_outputs.items():
            print(f"  scoring {method_name}: deletion/insertion curves", flush=True)
            deletion = deletion_curve(
                image=image_tensor,
                heatmap=method_output,
                predict_fn=predict_fn,
                class_idx=class_idx,
                steps=args.curve_steps,
                baseline=args.baseline,
                device=args.device,
            )
            insertion = insertion_curve(
                image=image_tensor,
                heatmap=method_output,
                predict_fn=predict_fn,
                class_idx=class_idx,
                steps=args.curve_steps,
                baseline=args.baseline,
                device=args.device,
            )

            record = {
                "dataset_index": int(dataset_index),
                "true_class": CIFAR10_CLASSES[int(true_label)],
                "pred_class": CIFAR10_CLASSES[int(class_idx)],
                "confidence": float(confidence),
                "deletion_auc": _curve_auc(deletion),
                "insertion_auc": _curve_auc(insertion),
                "deletion_curve": deletion,
                "insertion_curve": insertion,
            }
            per_method[method_name].append(record)

            method_curve_dir = raw_curve_dir / method_name
            method_curve_dir.mkdir(parents=True, exist_ok=True)
            with open(method_curve_dir / f"image_{int(dataset_index):05d}.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            print(
                f"  {method_name} done: deletion_auc={record['deletion_auc']:.4f}, "
                f"insertion_auc={record['insertion_auc']:.4f}",
                flush=True,
            )

    methods_summary: dict[str, dict[str, object]] = {}
    for method_name, records in per_method.items():
        deletion_values = np.array([record["deletion_auc"] for record in records], dtype=np.float32)
        insertion_values = np.array([record["insertion_auc"] for record in records], dtype=np.float32)
        methods_summary[method_name] = {
            "num_images": len(records),
            "deletion_auc_mean": float(deletion_values.mean()) if len(records) else None,
            "deletion_auc_std": float(deletion_values.std(ddof=0)) if len(records) else None,
            "insertion_auc_mean": float(insertion_values.mean()) if len(records) else None,
            "insertion_auc_std": float(insertion_values.std(ddof=0)) if len(records) else None,
            "per_image": records,
        }

    payload = {
        "config": vars(args),
        "model": {"kind": args.model_kind, "checkpoint": args.checkpoint, "model_name": args.model_name},
        "selected_indices": selected_indices,
        "reference_manifest_path": str(reference_manifest_path) if reference_manifest_path is not None else None,
        "methods": methods_summary,
    }
    with open(output_dir / "faithfulness_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved faithfulness metrics to {output_dir / 'faithfulness_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
