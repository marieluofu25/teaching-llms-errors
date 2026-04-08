#!/usr/bin/env python3
"""Concatenate ``activations`` arrays from shard ``.npz`` files (same ``d_in``, row-major)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Merge activation .npz shards on axis 0")
    p.add_argument("--shards", nargs="+", type=Path, required=True, help="Ordered shard paths")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--meta-template",
        type=Path,
        default=None,
        help="Optional .meta.json from first shard; merged meta written next to --output",
    )
    args = p.parse_args()

    arrs = []
    for sp in args.shards:
        data = np.load(sp, allow_pickle=True)
        if "activations" not in data:
            raise KeyError(f"Missing activations in {sp}")
        arrs.append(np.asarray(data["activations"], dtype=np.float32))
    d = arrs[0].shape[1]
    for i, a in enumerate(arrs):
        if a.shape[1] != d:
            raise ValueError(f"{args.shards[i]} has d_in {a.shape[1]} != {d}")
    merged = np.concatenate(arrs, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, activations=merged)
    print(f"Wrote {args.output} shape={merged.shape}")

    if args.meta_template and args.meta_template.is_file():
        meta = json.loads(args.meta_template.read_text(encoding="utf-8"))
        meta["n_rows"] = int(merged.shape[0])
        meta["d_in"] = int(merged.shape[1])
        meta["hidden_dim"] = int(merged.shape[1])
        meta["merged_from"] = [str(s) for s in args.shards]
        meta["output_npz"] = str(args.output.resolve())
        out_meta = args.output.with_suffix(".meta.json")
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
