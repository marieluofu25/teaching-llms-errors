#!/usr/bin/env python3
"""Merge Slurm shard CSVs from run_gemma_mmlu_inference → one sorted predictions table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Merge MMLU prediction shard CSVs")
    p.add_argument("--shards", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    parts = [pd.read_csv(sp) for sp in args.shards]
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["row_id"], keep="last").sort_values("row_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} n={len(out)}")


if __name__ == "__main__":
    main()
