#!/usr/bin/env python3
"""Writes a model's vision tower as a raceline vision file (#28).

  .venv/bin/python tools/convert_vision.py models/Qwen3.5-4B models/qwen3.5-4b.vision.safetensors

The Qwen3.5 / 3.6 / 3.8 checkpoints carry a vision tower (model.visual.*)
that the .rlm conversion leaves out. This file holds it next to the model:
safetensors, the weights in float16 under their names below model.visual.,
and in the metadata what raceline.vision needs besides (the tower's config,
the image processor's settings, the image token ids and the checkpoint it
comes from, which raceline compares with the model's). Only the shards that
hold the tower are read, so it also works from a download of just those
(tools/fetch_model.py --include) plus the JSON files.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", help="the original checkpoint directory")
    ap.add_argument("out", help="the vision file to write (.vision.safetensors)")
    ap.add_argument("--source", help="name of the checkpoint (default: the directory's name, as in the .rlm)")
    args = ap.parse_args()

    import mlx.core as mx

    d = args.checkpoint
    cfg = json.load(open(os.path.join(d, "config.json")))
    if "vision_config" not in cfg:
        sys.exit(f"{d} has no vision tower")
    proc = json.load(open(os.path.join(d, "preprocessor_config.json")))
    index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    names = sorted(n for n in index if n.startswith("model.visual."))
    weights = {}
    for shard in sorted({index[n] for n in names}):
        for name, arr in mx.load(os.path.join(d, shard)).items():
            if name.startswith("model.visual."):
                weights[name[len("model.visual."):]] = arr.astype(mx.float16)
    missing = [n for n in names if n[len("model.visual."):] not in weights]
    if missing:
        sys.exit(f"missing tensors: {missing[:5]}")
    from raceline.vision import FORMAT

    tok = json.load(open(os.path.join(d, "tokenizer_config.json")))
    added = {v["content"]: int(k) for k, v in (tok.get("added_tokens_decoder") or {}).items()}
    tokens = {k: added.get(t) for k, t in (("image_pad", "<|image_pad|>"), ("vision_start", "<|vision_start|>"),
                                         ("vision_end", "<|vision_end|>"))}
    if cfg.get("image_token_id") is not None and tokens["image_pad"] != cfg["image_token_id"]:
        sys.exit(f"the tokenizer's <|image_pad|> is {tokens['image_pad']}, the config's {cfg['image_token_id']}")
    size = proc.get("size") or {}
    processor = {"patch_size": proc["patch_size"], "temporal_patch_size": proc["temporal_patch_size"],
                 "merge_size": proc["merge_size"], "image_mean": proc["image_mean"], "image_std": proc["image_std"],
                 "min_pixels": size.get("shortest_edge", proc.get("min_pixels")),
                 "max_pixels": size.get("longest_edge", proc.get("max_pixels"))}
    meta = {"format": FORMAT, "source": args.source or os.path.basename(os.path.normpath(d)),
            "config": json.dumps(cfg["vision_config"]), "processor": json.dumps(processor),
            "tokens": json.dumps(tokens)}
    mx.save_safetensors(args.out, weights, metadata=meta)
    n = sum(int(np.prod(w.shape)) for w in weights.values())
    print(f"{args.out}: {len(weights)} tensors, {n / 1e6:.0f}M parameters, {os.path.getsize(args.out) / 1e9:.2f} GB; "
          f"source {meta['source']}, tokens {tokens}")


if __name__ == "__main__":
    main()
