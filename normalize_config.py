#!/usr/bin/env python
"""
Normalize a HuggingFace config.json so mlc-llm v0.19.0 (our pinned compiler) can read
models saved by NEWER transformers.

Why: recent transformers changed the config schema. It now nests the RoPE base inside a
`rope_parameters` (or `rope_scaling`) dict instead of a top-level `rope_theta`, and writes
`dtype` instead of `torch_dtype`. mlc-llm v0.19.0 still expects the OLD top-level keys, so
convert_weight fails on a freshly fine-tuned/merged model with:

    TypeError: QWen2Config.__init__() missing 1 required positional argument: 'rope_theta'

This decouples your training transformers version from the (pinned, old) build compiler.
Run AFTER download, BEFORE convert_weight. Idempotent; safe on already-old configs.

Usage:  python normalize_config.py <hf_model_dir | path/to/config.json>
"""
import json
import sys
from pathlib import Path


def normalize(path: str):
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        sys.exit(f"config.json not found: {p}")

    cfg = json.loads(p.read_text(encoding="utf-8"))
    changed = []

    # rope_theta: newer transformers nests it inside rope_parameters / rope_scaling.
    if "rope_theta" not in cfg:
        for key in ("rope_parameters", "rope_scaling"):
            rp = cfg.get(key)
            if isinstance(rp, dict) and rp.get("rope_theta") is not None:
                cfg["rope_theta"] = rp["rope_theta"]
                changed.append(f"rope_theta <- {key}.rope_theta ({rp['rope_theta']})")
                break

    # torch_dtype: newer transformers writes `dtype` instead.
    if "torch_dtype" not in cfg and "dtype" in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
        changed.append(f"torch_dtype <- dtype ({cfg['dtype']})")

    if changed:
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"normalized {p}:")
        for c in changed:
            print(f"  + {c}")
    else:
        print(f"no changes needed: {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python normalize_config.py <hf_model_dir | path/to/config.json>")
    normalize(sys.argv[1])
