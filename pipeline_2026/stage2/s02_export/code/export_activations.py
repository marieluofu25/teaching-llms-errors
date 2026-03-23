#!/usr/bin/env python3
"""
Export row-aligned neural activations for each row of a residual CSV.

Uses a local Hugging Face causal LM: hidden states at a chosen layer, pooled to one
vector per row. This produces a *real* .npz (not random dummy data) suitable for
run_sae_diffing and build_html_report.

Note: This is a *surrogate* representation (your chosen open model), not internal
activations from the closed API model that produced the original MMLU predictions.
Same questions / row order as the CSV are preserved.

If you use ``--max-rows`` manually, every downstream step (``run_sae_diffing``, ``build_html_report``)
must use a residual CSV with exactly that many rows—``mmlu-real`` does this by writing a subset CSV
when ``EXPORT_MAX_ROWS`` is set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.io_schema import validate_activations_shape


def _text_column(df: pd.DataFrame) -> str:
    if "questions" in df.columns:
        return "questions"
    if "question" in df.columns:
        return "question"
    raise ValueError("Need column 'questions' (MMLU) or 'question' (MathCAMPs).")


def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """hidden: (batch, seq, dim), mask: (batch, seq)"""
    if pooling == "last":
        # index of last real token per sequence
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        b = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[b, lengths]
    if pooling == "mean":
        m = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return summed / denom
    raise ValueError("pooling must be 'last' or 'mean'")


def export_activations(
    texts: list[str],
    *,
    model_name: str,
    layer_index: int,
    pooling: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    use_fp16: bool,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_fp16 and device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    n = len(texts)
    dim: int | None = None
    out_list: list[np.ndarray] = []

    for start in tqdm(range(0, n, batch_size), desc="export_activations"):
        batch_texts = texts[start : start + batch_size]
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hs_tuple = outputs.hidden_states
        # index: 0 = embeddings; -1 = last layer before head (typical)
        idx = layer_index if layer_index >= 0 else len(hs_tuple) + layer_index
        if not (0 <= idx < len(hs_tuple)):
            raise IndexError(
                f"layer_index {layer_index} resolves to {idx}; "
                f"model has {len(hs_tuple)} hidden state tensors."
            )
        h = hs_tuple[idx]
        pooled = _pool_hidden(h, enc["attention_mask"], pooling)
        vecs = pooled.float().cpu().numpy()
        if dim is None:
            dim = vecs.shape[1]
        out_list.append(vecs)

    stacked = np.concatenate(out_list, axis=0)
    assert stacked.shape == (n, dim or 0)
    return stacked.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export HF model hidden states row-aligned with residual CSV → .npz",
    )
    p.add_argument("--residual-csv", type=Path, required=True)
    p.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model id (default: gpt2, hidden dim 768)",
    )
    p.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Hidden state index: -1 = last layer before LM head",
    )
    p.add_argument(
        "--pooling",
        choices=("last", "mean"),
        default="last",
        help="How to pool sequence to one vector per row",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only first N rows (sanity check); default = all rows",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu | mps (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 on GPU (faster, less RAM)",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    args = p.parse_args()

    df = pd.read_csv(args.residual_csv)
    tcol = _text_column(df)
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()
    texts = df[tcol].fillna("").astype(str).tolist()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_fp16 = args.fp16 and device.type == "cuda"

    acts = export_activations(
        texts,
        model_name=args.model,
        layer_index=args.layer_index,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        use_fp16=use_fp16,
    )
    validate_activations_shape(len(df), acts.shape[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, activations=acts)

    meta = {
        "residual_csv": str(args.residual_csv.resolve()),
        "n_rows": int(acts.shape[0]),
        "hidden_dim": int(acts.shape[1]),
        "representation_type": "hidden_pooled",
        "d_in": int(acts.shape[1]),
        "model": args.model,
        "layer_index": args.layer_index,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": "float16" if use_fp16 else "float32",
        "output_npz": str(args.output.resolve()),
        "note": (
            "Activations are from the specified HF model, not from closed API models. "
            "Row order matches residual CSV."
        ),
    }
    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.output} shape={acts.shape} and {meta_path}")


if __name__ == "__main__":
    main()
