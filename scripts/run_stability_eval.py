from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import bootstrap  # noqa: F401

from src.counterfactuals import CounterfactualConfig, generate_counterfactual_for_normalized_input
from src.data import DEFAULT_REFERENCE_DIR, get_cifar10, load_or_build_reference_set
from src.data.cifar10 import VIT_MEAN, VIT_STD
from src.evaluation.stability import aggregate_stability_scores, stability_under_noise, stability_under_seed_variation
from src.explainers.lime_explainer import LIMEImageExplainer
from src.explainers.shap_explainer import SHAPImageExplainer
from src.model import load_project_model
from src.utils import set_seed
from src.visualization.heatmap import tensor_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline stability metrics for explanation methods.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/stability")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--background-strategy", type=str, default="stratified", choices=["random", "stratified"])
    parser.add_argument("--reference-manifest", type=str, default=None)
    parser.add_argument("--reuse-reference-set", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["lime", "shap", "counterfactual"], choices=["lime", "shap", "counterfactual"])
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--seed-variants", type=int, default=3)
    parser.add_argument("--topk-fraction", type=float, default=0.05)
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
    return parser.parse_args()


def _select_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    size = min(int(subset_size), int(dataset_len))
    return [int(index) for index in rng.choice(dataset_len, size=size, replace=False).tolist()]


def _counterfactual_result(model, image_tensor: torch.Tensor, args: argparse.Namespace) -> object:
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    reference_manifest_path = None
    background_tensor = None
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

    seed_list = [args.seed + offset for offset in range(args.seed_variants)]
    per_method: dict[str, list[dict[str, object]]] = {method: [] for method in args.methods}

    for dataset_index in selected_indices:
        image_tensor, _ = dataset[dataset_index]
        class_idx, _ = model.predict(image_tensor.unsqueeze(0))
        image_np = tensor_to_numpy(image_tensor)

        def explain_lime(current_image: torch.Tensor, run_seed: int) -> object:
            explainer = LIMEImageExplainer(
                n_samples=args.lime_samples,
                n_segments=args.lime_segments,
                seed=run_seed,
            )
            return explainer.explain(
                image=tensor_to_numpy(current_image),
                predict_fn=predict_fn,
                class_idx=class_idx,
                device=args.device,
            )

        def explain_shap(current_image: torch.Tensor, run_seed: int) -> object:
            explainer = SHAPImageExplainer(
                model=model,
                background_data=background_tensor,
                nsamples=args.shap_nsamples,
                batch_size=args.shap_batch_size,
                seed=run_seed,
            )
            return explainer.explain(
                image=current_image,
                predict_fn=predict_fn,
                class_idx=class_idx,
                device=args.device,
            )

        def explain_counterfactual(current_image: torch.Tensor, run_seed: int) -> object:
            set_seed(run_seed)
            return _counterfactual_result(model=model, image_tensor=current_image, args=args)

        explainers = {}
        if "lime" in args.methods:
            explainers["lime"] = explain_lime
        if "shap" in args.methods:
            explainers["shap"] = explain_shap
        if "counterfactual" in args.methods:
            explainers["counterfactual"] = explain_counterfactual

        for method_name, explain_fn in explainers.items():
            noise_result = stability_under_noise(
                image=image_tensor,
                explain_fn=lambda img, fn=explain_fn: fn(img, args.seed),
                noise_std=args.noise_std,
                repeats=args.noise_repeats,
                seed=args.seed,
                topk_fraction=args.topk_fraction,
            )
            seed_result = stability_under_seed_variation(
                image=image_tensor,
                explain_fn=explain_fn,
                seeds=seed_list,
                topk_fraction=args.topk_fraction,
            )
            combined = aggregate_stability_scores(noise_result["raw"] + seed_result["raw"])
            record = {
                "dataset_index": int(dataset_index),
                "predicted_class": int(class_idx),
                "noise": noise_result,
                "seed_variation": seed_result,
                "combined": combined,
            }
            per_method[method_name].append(record)

            method_dir = output_dir / "raw" / method_name
            method_dir.mkdir(parents=True, exist_ok=True)
            with open(method_dir / f"image_{int(dataset_index):05d}.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)

    methods_summary: dict[str, dict[str, object]] = {}
    for method_name, records in per_method.items():
        all_raw = []
        for record in records:
            all_raw.extend(record["noise"]["raw"])
            all_raw.extend(record["seed_variation"]["raw"])
        aggregate = aggregate_stability_scores(all_raw)
        methods_summary[method_name] = {
            "num_images": len(records),
            "correlation_mean": aggregate["correlation_mean"],
            "correlation_std": aggregate["correlation_std"],
            "topk_iou_mean": aggregate["topk_iou_mean"],
            "topk_iou_std": aggregate["topk_iou_std"],
            "per_image": records,
        }

    payload = {
        "config": vars(args),
        "model": {"kind": args.model_kind, "checkpoint": args.checkpoint, "model_name": args.model_name},
        "selected_indices": selected_indices,
        "reference_manifest_path": str(reference_manifest_path) if reference_manifest_path is not None else None,
        "methods": methods_summary,
    }
    with open(output_dir / "stability_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
