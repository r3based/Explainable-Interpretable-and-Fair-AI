from __future__ import annotations

import argparse
from pathlib import Path

import bootstrap  # noqa: F401

from src.evaluation.comparison import (
    COST_COLUMNS,
    QUALITY_COLUMNS,
    build_cost_table,
    build_quality_table,
    load_json,
    markdown_table,
    summarize_comparison,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison tables and summaries from saved evaluation metrics.")
    parser.add_argument(
        "--faithfulness-path",
        type=str,
        default="artifacts/eval/faithfulness/faithfulness_metrics.json",
    )
    parser.add_argument(
        "--stability-path",
        type=str,
        default="artifacts/eval/stability/stability_metrics.json",
    )
    parser.add_argument(
        "--runtime-path",
        type=str,
        default="artifacts/eval/runtime/runtime_metrics.json",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/reports")
    parser.add_argument("--classifier-paths", nargs="*", default=[], help="Optional classifier metric JSON files.")
    parser.add_argument(
        "--counterfactual-robustness-paths",
        nargs="*",
        default=[],
        help="Optional counterfactual robustness JSON files.",
    )
    return parser.parse_args()


CLASSIFIER_COLUMNS = [
    "model",
    "checkpoint",
    "clean_accuracy",
    "clean_loss",
    "attack",
    "robust_accuracy",
    "attack_success_rate",
]

COUNTERFACTUAL_COLUMNS = [
    "model",
    "checkpoint",
    "num_images",
    "success_rate",
    "failure_rate",
    "l2_mean",
    "linf_mean",
    "steps_mean",
]


def _classifier_rows(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = load_json(path)
        args = payload.get("args", {})
        clean = payload.get("clean", {})
        attacks = payload.get("attacks", {})
        if not attacks:
            rows.append(
                {
                    "model": args.get("model_kind"),
                    "checkpoint": args.get("checkpoint"),
                    "clean_accuracy": clean.get("accuracy"),
                    "clean_loss": clean.get("loss"),
                    "attack": None,
                    "robust_accuracy": None,
                    "attack_success_rate": None,
                }
            )
        for attack_name, metrics in attacks.items():
            rows.append(
                {
                    "model": args.get("model_kind"),
                    "checkpoint": args.get("checkpoint"),
                    "clean_accuracy": clean.get("accuracy"),
                    "clean_loss": clean.get("loss"),
                    "attack": attack_name,
                    "robust_accuracy": metrics.get("robust_accuracy"),
                    "attack_success_rate": metrics.get("attack_success_rate"),
                }
            )
    return rows


def _counterfactual_rows(paths: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        payload = load_json(path)
        args = payload.get("args", {})
        summary = payload.get("summary", {})
        rows.append(
            {
                "model": args.get("model_kind"),
                "checkpoint": args.get("checkpoint"),
                "num_images": summary.get("num_images"),
                "success_rate": summary.get("success_rate"),
                "failure_rate": summary.get("failure_rate"),
                "l2_mean": summary.get("l2_mean"),
                "linf_mean": summary.get("linf_mean"),
                "steps_mean": summary.get("steps_mean"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faithfulness = load_json(args.faithfulness_path)
    stability = load_json(args.stability_path)
    runtime = load_json(args.runtime_path)

    quality_rows = build_quality_table(faithfulness_metrics=faithfulness, stability_metrics=stability)
    cost_rows = build_cost_table(runtime_metrics=runtime)
    summary = summarize_comparison(quality_rows=quality_rows, cost_rows=cost_rows)

    write_csv(quality_rows, output_dir / "quality_table.csv", QUALITY_COLUMNS)
    write_csv(cost_rows, output_dir / "cost_table.csv", COST_COLUMNS)
    write_json(summary, output_dir / "summary.json")
    classifier_rows = _classifier_rows(args.classifier_paths)
    counterfactual_rows = _counterfactual_rows(args.counterfactual_robustness_paths)
    if classifier_rows:
        write_csv(classifier_rows, output_dir / "classifier_table.csv", CLASSIFIER_COLUMNS)
    if counterfactual_rows:
        write_csv(counterfactual_rows, output_dir / "counterfactual_robustness_table.csv", COUNTERFACTUAL_COLUMNS)

    report_md = "\n\n".join(
        [
            "# Evaluation Report",
            "## Table A — Explanation Quality",
            markdown_table(quality_rows, QUALITY_COLUMNS),
            "## Table B — Computational Cost",
            markdown_table(cost_rows, COST_COLUMNS),
            "## Summary",
            summary["summary_text"],
        ]
    )
    if classifier_rows:
        report_md += "\n\n" + "\n\n".join(
            ["## Table C - Classifier Robustness", markdown_table(classifier_rows, CLASSIFIER_COLUMNS)]
        )
    if counterfactual_rows:
        report_md += "\n\n" + "\n\n".join(
            [
                "## Table D - Counterfactual Robustness",
                markdown_table(counterfactual_rows, COUNTERFACTUAL_COLUMNS),
            ]
        )

    with open(output_dir / "report.md", "w", encoding="utf-8") as handle:
        handle.write(report_md)


if __name__ == "__main__":
    main()
