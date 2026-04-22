"""Reproducible SHAP reference-set selection for CIFAR-10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .cifar10 import CIFAR10_CLASSES


DEFAULT_REFERENCE_DIR = Path("artifacts/reference_sets")


def _infer_split(dataset: Dataset) -> str:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    train = getattr(base, "train", None)
    if train is True:
        return "train"
    if train is False:
        return "test"
    return "unknown"


def _extract_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if isinstance(targets, torch.Tensor):
            return [int(x) for x in targets.tolist()]
        return [int(x) for x in targets]

    targets: list[int] = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        targets.append(int(label))
    return targets


def _dataset_records(dataset: Dataset) -> list[dict[str, int]]:
    if isinstance(dataset, Subset):
        base_targets = _extract_targets(dataset.dataset)
        records = []
        for local_index, absolute_index in enumerate(dataset.indices):
            records.append(
                {
                    "local_index": int(local_index),
                    "absolute_index": int(absolute_index),
                    "label": int(base_targets[int(absolute_index)]),
                }
            )
        return records

    targets = _extract_targets(dataset)
    return [
        {"local_index": idx, "absolute_index": idx, "label": int(label)}
        for idx, label in enumerate(targets)
    ]


def _stratified_records(
    records: list[dict[str, int]],
    size: int,
    seed: int,
) -> list[dict[str, int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict[str, int]]] = {class_idx: [] for class_idx in range(len(CIFAR10_CLASSES))}
    for record in records:
        by_class.setdefault(record["label"], []).append(record)

    for class_idx, class_records in by_class.items():
        if not class_records:
            continue
        perm = rng.permutation(len(class_records))
        by_class[class_idx] = [class_records[i] for i in perm]

    selected: list[dict[str, int]] = []
    while len(selected) < size:
        progress = False
        for class_idx in range(len(CIFAR10_CLASSES)):
            class_records = by_class.get(class_idx, [])
            if class_records:
                selected.append(class_records.pop())
                progress = True
                if len(selected) == size:
                    break
        if not progress:
            break

    if len(selected) != size:
        raise ValueError(f"Could not build a stratified reference set of size {size}.")
    return selected


def materialize_reference_tensor(dataset: Dataset, indices: list[int]) -> torch.Tensor:
    """Load a stacked tensor from absolute dataset indices."""
    tensors = [dataset[int(index)][0] for index in indices]
    return torch.stack(tensors)


def build_reference_set(
    dataset: Dataset,
    strategy: str = "stratified",
    size: int = 32,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a reproducible background tensor and manifest metadata."""
    if size <= 0:
        raise ValueError("Reference set size must be positive.")

    records = _dataset_records(dataset)
    if size > len(records):
        raise ValueError(f"Requested reference size {size} exceeds dataset size {len(records)}.")

    strategy = strategy.lower()
    if strategy not in {"random", "stratified"}:
        raise ValueError("strategy must be one of: random, stratified")

    if strategy == "random":
        rng = np.random.default_rng(seed)
        chosen_positions = rng.choice(len(records), size=size, replace=False)
        selected = [records[int(pos)] for pos in chosen_positions.tolist()]
    else:
        selected = _stratified_records(records=records, size=size, seed=seed)

    local_indices = [record["local_index"] for record in selected]
    tensors = torch.stack([dataset[idx][0] for idx in local_indices])

    absolute_indices = [record["absolute_index"] for record in selected]
    label_counts: dict[str, int] = {}
    for record in selected:
        label_name = CIFAR10_CLASSES[int(record["label"])]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

    manifest = {
        "seed": int(seed),
        "dataset_split": _infer_split(dataset),
        "strategy": strategy,
        "background_size": int(size),
        "chosen_indices": [int(index) for index in absolute_indices],
        "class_distribution": label_counts,
    }
    return tensors, manifest


def save_reference_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


def load_reference_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_reference_set(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_split": manifest.get("dataset_split", "unknown"),
        "strategy": manifest.get("strategy"),
        "background_size": manifest.get("background_size"),
        "seed": manifest.get("seed"),
        "class_distribution": manifest.get("class_distribution", {}),
    }


def load_or_build_reference_set(
    dataset: Dataset,
    manifest_path: str | Path,
    strategy: str = "stratified",
    size: int = 32,
    seed: int = 42,
    reuse_existing: bool = True,
) -> tuple[torch.Tensor, dict[str, Any], Path]:
    """Reuse an existing manifest when possible, otherwise create one."""
    path = Path(manifest_path)
    if reuse_existing and path.exists():
        manifest = load_reference_manifest(path)
        tensor = materialize_reference_tensor(dataset, manifest["chosen_indices"])
        return tensor, manifest, path

    tensor, manifest = build_reference_set(dataset=dataset, strategy=strategy, size=size, seed=seed)
    save_reference_manifest(manifest, path)
    return tensor, manifest, path
