from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.counterfactuals import CounterfactualConfig, generate_counterfactual_for_normalized_input
from src.data import DEFAULT_REFERENCE_DIR, get_cifar10, load_or_build_reference_set
from src.data.cifar10 import VIT_MEAN, VIT_STD
from src.evaluation.runtime import benchmark_all_methods
from src.explainers.lime_explainer import LIMEImageExplainer
from src.explainers.shap_explainer import SHAPImageExplainer
from src.model.vit import ViTWrapper
from src.utils import set_seed
from src.visualization.heatmap import tensor_to_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark explanation runtime across methods.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/runtime")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--background-strategy", type=str, default="stratified", choices=["random", "stratified"])
    parser.add_argument("--reference-manifest", type=str, default=None)
    parser.add_argument("--reuse-reference-set", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["lime", "shap", "counterfactual"], choices=["lime", "shap", "counterfactual"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lime-samples", type=int, default=200)
    parser.add_argument("--lime-segments", type=int, default=50)
    parser.add_argument("--shap-nsamples", type=int, default=64)
    parser.add_argument("--shap-batch-size", type=int, default=8)
    parser.add_argument("--cf-steps", type=int, default=80)
    parser.add_argument("--cf-step-size", type=float, default=0.03)
    parser.add_argument("--cf-lambda-l2", type=float, default=1e-2)
    parser.add_argument("--cf-lambda-tv", type=float, default=1e-4)
    return parser.parse_args()


def _select_subset_indices(dataset_len: int, subset_size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    size = min(int(subset_size), int(dataset_len))
    return [int(index) for index in rng.choice(dataset_len, size=size, replace=False).tolist()]


def _counterfactual_callable(model: ViTWrapper, image_tensor: torch.Tensor, args: argparse.Namespace):
    config = CounterfactualConfig(
        steps=args.cf_steps,
        step_size=args.cf_step_size,
        lambda_l2=args.cf_lambda_l2,
        lambda_tv=args.cf_lambda_tv,
        target_mode="untargeted",
    )
    mean = torch.tensor(VIT_MEAN, dtype=image_tensor.dtype)
    std  = torch.tensor(VIT_STD,  dtype=image_tensor.dtype)
    return lambda: generate_counterfactual_for_normalized_input(
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

    model = ViTWrapper(device=args.device)
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

    method_callables: dict[str, list] = {method: [] for method in args.methods}
    for dataset_index in selected_indices:
        image_tensor, _ = dataset[dataset_index]
        class_idx, _ = model.predict(image_tensor.unsqueeze(0))

        if lime_explainer is not None:
            method_callables["lime"].append(
                lambda img=image_tensor, cls=class_idx: lime_explainer.explain(
                    image=tensor_to_numpy(img),
                    predict_fn=predict_fn,
                    class_idx=cls,
                    device=args.device,
                )
            )
        if shap_explainer is not None:
            method_callables["shap"].append(
                lambda img=image_tensor, cls=class_idx: shap_explainer.explain(
                    image=img,
                    predict_fn=predict_fn,
                    class_idx=cls,
                    device=args.device,
                )
            )
        if "counterfactual" in args.methods:
            method_callables["counterfactual"].append(_counterfactual_callable(model=model, image_tensor=image_tensor, args=args))

    benchmark_rows = benchmark_all_methods(
        method_callables=method_callables,
        repeats=args.repeats,
        warmup=args.warmup,
        device=args.device,
        measure_peak_memory=True,
    )

    payload = {
        "config": vars(args),
        "selected_indices": selected_indices,
        "reference_manifest_path": str(reference_manifest_path) if reference_manifest_path is not None else None,
        "methods": {row["method"]: row for row in benchmark_rows},
    }
    with open(output_dir / "runtime_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
