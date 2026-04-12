#!/usr/bin/env python3
"""MMLU inference with a Hugging Face causal LM (default: Gemma-2-9B) — unified-model Phase 1.

Loads the full legacy pickle, runs batched generation, writes a predictions CSV with
``ai_preds`` / ``ai_correct`` aligned to ``data_y``. Gold may be **letters A–D** or
**numeric indices 1–4** (MMLU convention: 1=A, …, 4=D). Supports resume and row
ranges for Slurm array tasks (use **one output file per array task**, then merge).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.repo_paths import teaching_llms_errors_repo_root
from stage2.s01_residual.code.mmlu_gold_utils import mmlu_gold_letter as _mmlu_gold_letter
from stage2.s01_residual.code.run_residual_control import load_mmlu_dataframe


def _parse_choice(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return None


def _build_prompt(question: str) -> str:
    q = (question or "").strip()
    inst = (
        "Answer with only one letter: A, B, C, or D. "
        "Do not write anything else after the letter.\n\n"
    )
    return f"{inst}{q}\n\nAnswer:"


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def main() -> None:
    p = argparse.ArgumentParser(description="Gemma/HF MMLU batched inference")
    p.add_argument(
        "--mmlu-pickle",
        type=Path,
        default=teaching_llms_errors_repo_root()
        / "pipeline/stage1_do_errors_exist/datasets/mmlu_df_gpt35.pkl",
    )
    p.add_argument("--model", type=str, default="google/gemma-2-9b")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--max-input-length", type=int, default=2048)
    p.add_argument("--max-input-chars", type=int, default=12000)
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip row_ids already present in output-csv (single-writer only)",
    )
    p.add_argument("--row-start", type=int, default=None)
    p.add_argument("--row-end", type=int, default=None)
    p.add_argument("--ai-model-label", type=str, default="google/gemma-2-9b")
    args = p.parse_args()

    if not args.mmlu_pickle.is_file():
        raise FileNotFoundError(args.mmlu_pickle)

    full_df = load_mmlu_dataframe(args.mmlu_pickle)
    full_df = full_df.reset_index(drop=True)
    full_df["row_id"] = range(len(full_df))

    if args.row_start is not None or args.row_end is not None:
        rs = int(args.row_start or 0)
        re_ = int(args.row_end) if args.row_end is not None else len(full_df)
        work = full_df.iloc[rs:re_].copy()
    else:
        work = full_df.copy()

    done_ids: set[int] = set()
    if args.resume and args.output_csv.is_file():
        existing = pd.read_csv(args.output_csv, usecols=["row_id"])
        done_ids = set(int(x) for x in existing["row_id"].tolist())

    pending = work[~work["row_id"].isin(done_ids)].copy()
    if pending.empty:
        print("All rows already in output; nothing to do.")
        return

    if not args.resume and args.output_csv.is_file():
        args.output_csv.unlink()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw: dict = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if args.device != "cpu" and torch.cuda.is_available():
        load_kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    if args.device == "cpu" or not torch.cuda.is_available():
        dev = torch.device(args.device or "cpu")
        model = model.to(dev)
    model.eval()
    mdev = _model_device(model)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    first_write = not (args.resume and args.output_csv.is_file())

    mbs = max(1, args.micro_batch_size)
    out_buffer: list[dict] = []
    n_correct = 0
    n_parsed_empty = 0
    n_gold_unrecognized = 0

    def flush() -> None:
        nonlocal first_write
        if not out_buffer:
            return
        odf = pd.DataFrame(out_buffer)
        odf.to_csv(
            args.output_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False
        out_buffer.clear()

    n = len(pending)
    for start in tqdm(range(0, n, mbs), desc="gemma_mmlu_inference"):
        sub = pending.iloc[start : start + mbs]
        prompts = [
            _build_prompt(str(t)[: args.max_input_chars]) for t in sub["questions"]
        ]
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        enc = {k: v.to(mdev) for k, v in enc.items()}
        attn = enc["attention_mask"]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for j in range(len(sub)):
            row = sub.iloc[j]
            n_in = int(attn[j].sum())
            new_tokens = gen[j, n_in:]
            decoded = tok.decode(new_tokens, skip_special_tokens=True)
            letter = _parse_choice(decoded) or ""
            if not letter:
                n_parsed_empty += 1
            gold = _mmlu_gold_letter(row["data_y"])
            if gold is None:
                n_gold_unrecognized += 1
            correct = bool(letter and gold and letter == gold)
            if correct:
                n_correct += 1
            out_buffer.append(
                {
                    "row_id": int(row["row_id"]),
                    "questions": row["questions"],
                    "data_y": row["data_y"],
                    "ai_preds": letter,
                    "ai_correct": correct,
                    "subject": row.get("subject", ""),
                    "subcat": row.get("subcat", ""),
                    "cat": row.get("cat", ""),
                    "ai_model": args.ai_model_label,
                    "model_raw_tail": decoded[:500].replace("\n", " "),
                }
            )
        flush()

    flush()

    acc = float(n_correct / n) if n else None
    meta = {
        "model": args.model,
        "n_rows_written": int(n),
        "n_correct": int(n_correct),
        "accuracy": acc,
        "n_parsed_empty": int(n_parsed_empty),
        "n_gold_unrecognized": int(n_gold_unrecognized),
        "output_csv": str(args.output_csv),
        "row_start": args.row_start,
        "row_end": args.row_end,
        "resume": args.resume,
    }
    meta_path = args.output_csv.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.output_csv} (+ {meta_path})")
    if acc is not None and acc == 0.0 and n > 0:
        print(
            "WARNING: accuracy is 0.0 — verify data_y (1–4 vs A–D), parsing, and model output.",
            flush=True,
        )
    if n > 0 and n_gold_unrecognized > max(10, n // 100):
        print(
            f"WARNING: n_gold_unrecognized={n_gold_unrecognized}/{n} — many rows lack a valid gold letter.",
            flush=True,
        )


if __name__ == "__main__":
    main()
