from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.reference_set import build_reference_set
from src.evaluation.faithfulness import deletion_auc, deletion_curve, insertion_auc, insertion_curve
from src.evaluation.runtime import measure_runtime
from src.evaluation.stability import compare_heatmaps


class TinyDataset(Dataset):
    def __init__(self):
        self.train = True
        self.targets = [0, 1, 0, 1, 0, 1]
        self.images = [
            torch.full((3, 4, 4), fill_value=float(index) / 10.0)
            for index in range(len(self.targets))
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


def _dummy_predict_fn(batch: torch.Tensor) -> torch.Tensor:
    score = batch.mean(dim=(1, 2, 3))
    logits = torch.stack([score, -score], dim=1)
    return logits.softmax(dim=-1)


def test_reference_set_builder_is_reproducible() -> None:
    dataset = TinyDataset()
    background_a, manifest_a = build_reference_set(dataset, strategy="stratified", size=4, seed=11)
    background_b, manifest_b = build_reference_set(dataset, strategy="stratified", size=4, seed=11)

    assert manifest_a == manifest_b
    assert torch.equal(background_a, background_b)
    assert manifest_a["background_size"] == 4


def test_faithfulness_curves_and_aucs_are_finite() -> None:
    image = torch.linspace(0.0, 1.0, steps=3 * 4 * 4).reshape(3, 4, 4)
    heatmap = np.arange(16, dtype=np.float32).reshape(4, 4)

    deletion = deletion_curve(image, heatmap, _dummy_predict_fn, class_idx=0, steps=4, baseline="zero")
    insertion = insertion_curve(image, heatmap, _dummy_predict_fn, class_idx=0, steps=4, baseline="zero")

    assert len(deletion["fractions"]) == 5
    assert len(insertion["scores"]) == 5

    deletion_score = deletion_auc(image, heatmap, _dummy_predict_fn, class_idx=0, steps=4)
    insertion_score = insertion_auc(image, heatmap, _dummy_predict_fn, class_idx=0, steps=4)

    assert math.isfinite(deletion_score)
    assert math.isfinite(insertion_score)


def test_stability_comparison_returns_finite_values() -> None:
    heatmap_a = np.arange(16, dtype=np.float32).reshape(4, 4)
    heatmap_b = np.flipud(heatmap_a)

    comparison = compare_heatmaps(heatmap_a, heatmap_b, topk_fraction=0.25)

    assert math.isfinite(comparison["correlation"])
    assert math.isfinite(comparison["topk_iou"])


def test_runtime_helper_returns_summary_for_dummy_callable() -> None:
    result = measure_runtime(lambda: sum(range(50)), repeats=2, warmup=1, device="cpu")

    assert len(result["per_run_seconds"]) == 2
    assert result["runtime_mean_sec"] >= 0.0
    assert result["runtime_median_sec"] >= 0.0
