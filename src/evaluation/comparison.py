"""Build cross-method comparison tables and strict summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


QUALITY_COLUMNS = [
    "method",
    "num_images",
    "deletion_auc_mean",
    "deletion_auc_std",
    "insertion_auc_mean",
    "insertion_auc_std",
    "stability_corr_mean",
    "stability_iou_mean",
]

COST_COLUMNS = [
    "method",
    "device",
    "runtime_mean_sec",
    "runtime_std_sec",
    "runtime_median_sec",
    "peak_memory_mb",
]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def write_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, divider]
    for row in rows:
        values = [str(row.get(field, "")) for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_quality_table(
    faithfulness_metrics: dict[str, Any],
    stability_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    faith_methods = faithfulness_metrics.get("methods", {})
    stability_methods = stability_metrics.get("methods", {})
    rows: list[dict[str, Any]] = []

    for method in sorted(set(faith_methods) | set(stability_methods)):
        faith = faith_methods.get(method, {})
        stability = stability_methods.get(method, {})
        rows.append(
            {
                "method": method,
                "num_images": faith.get("num_images", stability.get("num_images", 0)),
                "deletion_auc_mean": faith.get("deletion_auc_mean"),
                "deletion_auc_std": faith.get("deletion_auc_std"),
                "insertion_auc_mean": faith.get("insertion_auc_mean"),
                "insertion_auc_std": faith.get("insertion_auc_std"),
                "stability_corr_mean": stability.get("correlation_mean"),
                "stability_iou_mean": stability.get("topk_iou_mean"),
            }
        )
    return rows


def build_cost_table(runtime_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_methods = runtime_metrics.get("methods", {})
    rows: list[dict[str, Any]] = []
    for method in sorted(runtime_methods):
        stats = runtime_methods[method]
        rows.append(
            {
                "method": method,
                "device": stats.get("device"),
                "runtime_mean_sec": stats.get("runtime_mean_sec"),
                "runtime_std_sec": stats.get("runtime_std_sec"),
                "runtime_median_sec": stats.get("runtime_median_sec"),
                "peak_memory_mb": stats.get("peak_memory_mb"),
            }
        )
    return rows


def _dominates_faithfulness(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    candidate_deletion = candidate.get("deletion_auc_mean")
    candidate_insertion = candidate.get("insertion_auc_mean")
    other_deletion = other.get("deletion_auc_mean")
    other_insertion = other.get("insertion_auc_mean")
    if None in {candidate_deletion, candidate_insertion, other_deletion, other_insertion}:
        return False

    no_worse = candidate_deletion <= other_deletion and candidate_insertion >= other_insertion
    strictly_better = candidate_deletion < other_deletion or candidate_insertion > other_insertion
    return bool(no_worse and strictly_better)


def _dominates_stability(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
    candidate_corr = candidate.get("stability_corr_mean")
    candidate_iou = candidate.get("stability_iou_mean")
    other_corr = other.get("stability_corr_mean")
    other_iou = other.get("stability_iou_mean")
    if None in {candidate_corr, candidate_iou, other_corr, other_iou}:
        return False

    no_worse = candidate_corr >= other_corr and candidate_iou >= other_iou
    strictly_better = candidate_corr > other_corr or candidate_iou > other_iou
    return bool(no_worse and strictly_better)


def _faithfulness_winner(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if all(row["method"] == other["method"] or _dominates_faithfulness(row, other) for other in rows):
            return row["method"]
    return None


def _stability_winner(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if all(row["method"] == other["method"] or _dominates_stability(row, other) for other in rows):
            return row["method"]
    return None


def _best_method(rows: list[dict[str, Any]], key: str, reverse: bool) -> str | None:
    valid_rows = [row for row in rows if row.get(key) is not None]
    if not valid_rows:
        return None
    best = sorted(valid_rows, key=lambda row: row[key], reverse=reverse)[0]
    return str(best["method"])


def _pareto_dominated_shap(quality_rows: list[dict[str, Any]], cost_rows: list[dict[str, Any]]) -> bool | None:
    quality_by_method = {row["method"]: row for row in quality_rows}
    cost_by_method = {row["method"]: row for row in cost_rows}
    if "shap" not in quality_by_method or "shap" not in cost_by_method:
        return None

    shap_row = quality_by_method["shap"] | cost_by_method["shap"]
    for method, quality_row in quality_by_method.items():
        if method == "shap" or method not in cost_by_method:
            continue

        other_row = quality_row | cost_by_method[method]
        no_worse = (
            other_row.get("deletion_auc_mean") <= shap_row.get("deletion_auc_mean")
            and other_row.get("insertion_auc_mean") >= shap_row.get("insertion_auc_mean")
            and other_row.get("stability_corr_mean") >= shap_row.get("stability_corr_mean")
            and other_row.get("stability_iou_mean") >= shap_row.get("stability_iou_mean")
            and other_row.get("runtime_mean_sec") <= shap_row.get("runtime_mean_sec")
        )
        strictly_better = (
            other_row.get("deletion_auc_mean") < shap_row.get("deletion_auc_mean")
            or other_row.get("insertion_auc_mean") > shap_row.get("insertion_auc_mean")
            or other_row.get("stability_corr_mean") > shap_row.get("stability_corr_mean")
            or other_row.get("stability_iou_mean") > shap_row.get("stability_iou_mean")
            or other_row.get("runtime_mean_sec") < shap_row.get("runtime_mean_sec")
        )
        if no_worse and strictly_better:
            return True
    return False


def summarize_comparison(
    quality_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    most_expensive = _best_method(cost_rows, key="runtime_mean_sec", reverse=True)
    best_deletion = _best_method(quality_rows, key="deletion_auc_mean", reverse=False)
    best_insertion = _best_method(quality_rows, key="insertion_auc_mean", reverse=True)
    best_corr = _best_method(quality_rows, key="stability_corr_mean", reverse=True)
    best_iou = _best_method(quality_rows, key="stability_iou_mean", reverse=True)
    most_faithful = _faithfulness_winner(quality_rows)
    most_stable = _stability_winner(quality_rows)
    shap_dominated = _pareto_dominated_shap(quality_rows, cost_rows)

    if most_faithful is None:
        faithfulness_note = (
            "No single method dominated both faithfulness metrics; "
            f"best deletion AUC: {best_deletion}, best insertion AUC: {best_insertion}."
        )
    else:
        faithfulness_note = f"{most_faithful} dominated the baseline deletion and insertion metrics."

    if most_stable is None:
        stability_note = (
            "No single method dominated both stability metrics; "
            f"best correlation: {best_corr}, best top-k IoU: {best_iou}."
        )
    else:
        stability_note = f"{most_stable} dominated the baseline stability metrics."

    if shap_dominated is None:
        shap_note = "SHAP was not present in both the quality and runtime summaries."
    elif shap_dominated:
        shap_note = "SHAP was Pareto-dominated by at least one other method under the baseline quality/cost metrics."
    else:
        shap_note = "SHAP was not Pareto-dominated under the baseline quality/cost metrics."

    return {
        "most_faithful_method": most_faithful,
        "most_stable_method": most_stable,
        "most_computationally_expensive_method": most_expensive,
        "best_deletion_auc_method": best_deletion,
        "best_insertion_auc_method": best_insertion,
        "best_stability_corr_method": best_corr,
        "best_stability_iou_method": best_iou,
        "shap_reasonable_tradeoff": None if shap_dominated is None else bool(not shap_dominated),
        "summary_text": "\n".join(
            [
                faithfulness_note,
                stability_note,
                f"Most computationally expensive method by mean runtime: {most_expensive}.",
                shap_note,
            ]
        ),
    }
