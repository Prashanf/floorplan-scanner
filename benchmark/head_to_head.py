"""Head-to-head comparison against a consumer scanning app export.

Usage: python benchmark/head_to_head.py --ours benchmark/results/room-1.json \
    --theirs benchmark/competitor/room-1_polycam.json --ground-truth benchmark/ground_truth/room-1.yaml
"""

from __future__ import annotations


def compare(ours_path: str, theirs_path: str, ground_truth_path: str) -> str:
    """Compare shared dimensions (walls, openings), print a table of ground truth,
    our value/error, their value/error and winner, plus a 'Beat or tied on X/Y' line.
    Returns the markdown.
    """
    raise NotImplementedError("Not yet implemented")


def main() -> None:
    """CLI entry point."""
    raise NotImplementedError("Not yet implemented")


if __name__ == "__main__":
    main()
