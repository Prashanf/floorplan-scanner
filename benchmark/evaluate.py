"""Benchmark gate evaluation against tape/laser ground truth.

Usage: python benchmark/evaluate.py --results-dir benchmark/results/ --ground-truth-dir benchmark/ground_truth/
"""

from __future__ import annotations


def evaluate(results_dir: str, ground_truth_dir: str) -> str:
    """Match pipeline outputs to ground-truth YAML by room id, compute per-wall,
    ceiling, opening and floor-area errors, score the gates (openings, ceiling,
    repeatability, photo stitch footprint) and interval calibration, print a
    markdown table and write results_dir/gates_report.md. Returns the markdown.
    """
    raise NotImplementedError("Not yet implemented")


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError("Not yet implemented")


if __name__ == "__main__":
    main()
