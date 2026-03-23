#!/usr/bin/env python3
"""
Map row-aligned pooled hidden states (N, d_in) through a pretrained sparse autoencoder.

Produces a new .npz with the same key ``activations`` but shape (N, d_sae) for use with
``run_sae_diffing`` and ``build_html_report``. Sidecar ``*.meta.json`` records
``representation_type: sae_latent`` and SAELens release / hook id.

Requires: pip install 'sae-lens>=6.0,<7' (see requirements.txt and docs/sae_checkpoints.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lib.io_schema import validate_activations_shape

try:
    from sae_lens import SAE
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "sae-lens is required for encode_sae_latents. Install with:\n"
        "  pip install 'sae-lens>=6.0,<7'\n"
        "See docs/sae_checkpoints.md for checkpoint / layer alignment."
    ) from exc


def _device_from_arg(device: str | None) -> torch.device:
    if device:
        d = torch.device(device)
        if d.type == "mps":
            # SAELens / some ops may be unstable on MPS; allow but warn.
            return d
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_batches(
    sae: SAE,
    acts: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """acts: (N, d_in) float32 → (N, d_sae) float32."""
    n, d_in = acts.shape
    out_chunks: list[np.ndarray] = []
    sae.eval()
    sae.to(device)
    for start in tqdm(range(0, n, batch_size), desc="encode_sae_latents"):
        end = min(start + batch_size, n)
        x = torch.from_numpy(acts[start:end].astype(np.float32, copy=False)).to(
            device=device, dtype=torch.float32
        )
        with torch.no_grad():
            z = sae.encode(x)
        if isinstance(z, torch.Tensor):
            z_np = z.float().detach().cpu().numpy()
        else:
            z_np = np.asarray(z, dtype=np.float32)
        out_chunks.append(z_np.astype(np.float32, copy=False))
    stacked = np.concatenate(out_chunks, axis=0)
    if stacked.shape != (n, sae.cfg.d_sae):
        raise RuntimeError(
            f"Encode output shape {stacked.shape} != expected ({n}, {sae.cfg.d_sae})"
        )
    return stacked


def main() -> None:
    p = argparse.ArgumentParser(
        description="Encode pooled hidden .npz through a SAELens pretrained SAE → (N, d_sae)",
    )
    p.add_argument("--input-npz", type=Path, required=True, help="Hidden export, key activations (N, d_in)")
    p.add_argument(
        "--input-meta",
        type=Path,
        default=None,
        help="Sidecar JSON from export_activations (default: input-npz with .meta.json)",
    )
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--sae-release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--sae-id", type=str, default="blocks.11.hook_resid_pre")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu | mps (default: auto)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--skip-d-in-check",
        action="store_true",
        help="Allow d_in mismatch vs SAE (not recommended; breaks interpretability)",
    )
    args = p.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    if "activations" not in data:
        raise KeyError("Expected 'activations' in input npz")
    acts = np.asarray(data["activations"], dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError("activations must be 2D (N, d_in)")

    meta_path = args.input_meta
    if meta_path is None:
        meta_path = args.input_npz.with_suffix(".meta.json")
    in_meta: dict = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            in_meta = json.load(f)
        if "n_rows" in in_meta:
            validate_activations_shape(int(in_meta["n_rows"]), acts.shape[0])

    device = _device_from_arg(args.device)
    print(f"Loading SAE release={args.sae_release!r} sae_id={args.sae_id!r} device={device}…")
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=str(device),
    )

    d_in_npz = int(acts.shape[1])
    d_in_sae = int(sae.cfg.d_in)
    if d_in_npz != d_in_sae and not args.skip_d_in_check:
        raise ValueError(
            f"Input activations width {d_in_npz} != SAE d_in {d_in_sae}. "
            f"Use a matching HF export (see docs/sae_checkpoints.md) or --skip-d-in-check."
        )

    latents = encode_batches(sae, acts, batch_size=args.batch_size, device=device)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, activations=latents)

    base_model = in_meta.get("model", "unknown")
    layer_index = in_meta.get("layer_index")
    out_meta = {
        "representation_type": "sae_latent",
        "n_rows": int(latents.shape[0]),
        "d_in": d_in_sae,
        "d_sae": int(sae.cfg.d_sae),
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "sae_lens_note": (
            "Columns are SAE feature indices (sparse dictionary), not raw hidden dimensions."
        ),
        "source_hidden_npz": str(args.input_npz.resolve()),
        "source_hidden_meta": str(meta_path.resolve()) if meta_path.is_file() else None,
        "residual_csv": in_meta.get("residual_csv"),
        "base_model": base_model,
        "hidden_export_layer_index": layer_index,
        "model": (
            f"{base_model} + SAELens SAE {args.sae_release}/{args.sae_id} "
            f"(d_in={d_in_sae}, d_sae={int(sae.cfg.d_sae)})"
        ),
        "output_npz": str(args.output_npz.resolve()),
        "batch_size": args.batch_size,
        "device": str(device),
        "methodology": (
            "SAE embeddings for group-wise comparison; see Jiang et al., arXiv:2512.10092."
        ),
    }
    out_meta_path = args.output_npz.with_suffix(".meta.json")
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    print(f"Wrote {args.output_npz} shape={latents.shape} and {out_meta_path}")


if __name__ == "__main__":
    main()
