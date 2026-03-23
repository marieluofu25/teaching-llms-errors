#!/usr/bin/env python3
"""Plot coverage/concentration from evaluate_pattern_sets JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="PNG path")
    args = p.parse_args()

    import matplotlib.pyplot as plt

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)
    hr = m["residual_metrics"]["high_residual"]
    lr = m["residual_metrics"]["low_residual"]
    labels = ["High residual\ncoverage", "High residual\nconcentration", "Low residual\ncoverage", "Low residual\nconcentration"]
    vals = [
        hr["coverage_of_errors"],
        hr["concentration_in_group"],
        lr["coverage_of_errors"],
        lr["concentration_in_group"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, vals, color=["#2c7fb8", "#2c7fb8", "#bdc9e1", "#bdc9e1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title(f"Residual proxies ({m.get('label', 'run')})")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
