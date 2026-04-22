from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
        default=str(ROOT / "artifacts" / "eval" / "faithfulness" / "faithfulness_metrics.json"),
    )
    parser.add_argument(
        "--stability-path",
        type=str,
        default=str(ROOT / "artifacts" / "eval" / "stability" / "stability_metrics.json"),
    )
    parser.add_argument(
        "--runtime-path",
        type=str,
        default=str(ROOT / "artifacts" / "eval" / "runtime" / "runtime_metrics.json"),
    )
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "reports"))
    return parser.parse_args()


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
    with open(output_dir / "report.md", "w", encoding="utf-8") as handle:
        handle.write(report_md)


if __name__ == "__main__":
    main()
