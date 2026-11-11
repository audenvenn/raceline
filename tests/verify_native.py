#!/usr/bin/env python3
"""Check the engine on a native-format model against MLX running the same weights.

The .rlm file is decoded back to float matrices (codebook value * row scale *
column scale), loaded into the original model in MLX, and evaluated with
float32 activations. That checks the file writer, the kernel's reading of the
codes and scales, and the engine's handling of column scales in one go.

  .venv/bin/python tests/verify_native.py models/Qwen3.5-4B models/qwen3.5-4b.fp4.rlm build/ref/tokens_96.i32
"""

import argparse
import os
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from rlmfile import Native  # noqa: E402

PARTS = {  # file tensor -> {part: module path below the layer}
    "lin.in_proj": {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "b": "linear_attn.in_proj_b",
                    "a": "linear_attn.in_proj_a"},
    "lin.out_proj": {"out": "linear_attn.out_proj"},
    "attn.qkv": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj"},
    "attn.o_proj": {"out": "self_attn.o_proj"},
    "mlp.gate_up": {"gate": "mlp.gate_proj", "up": "mlp.up_proj"},
    "mlp.down": {"out": "mlp.down_proj"},
}


def load_decoded(float_model, rlm, check=False):
    """The original model in MLX with every quantized matrix replaced by what the file holds."""
    nat = Native(rlm)
    hdr = nat.hdr
    model, _ = load(float_model)
    lm = model.language_model
    tm = lm.model
    for i, layer in enumerate(tm.layers):
        for name, parts in PARTS.items():
            t = hdr["tensors"].get(f"layers.{i}.{name}")
            if t is None:
                continue
            W = nat.matrix(f"layers.{i}.{name}")
            for part, path in parts.items():
                mod = layer
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                p = t["parts"][part]
                mod.weight = mx.array(W[p["offset"]:p["offset"] + p["n"]].astype(np.float16))
    emb = nat.matrix("embed")
    tm.embed_tokens.weight = mx.array(emb.astype(np.float16))
    if not lm.args.tie_word_embeddings:
        lm.lm_head.weight = mx.array(nat.matrix("lm_head").astype(np.float16))
    elif check:  # one matrix in two layouts: they must decode to the same values
        print("tied head and lookup table identical:", bool(np.array_equal(nat.matrix("lm_head"), emb)))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("float_model")
    ap.add_argument("rlm")
    ap.add_argument("tokens")
    ap.add_argument("--redline", default="build/cmake/redline-engine")
    args = ap.parse_args()

    model = load_decoded(args.float_model, args.rlm, check=True)
    tm = model.language_model.model

    tokens = np.fromfile(args.tokens, dtype=np.int32)
    x = mx.array(tokens.tolist())[None]
    ref = np.array(model(x, input_embeddings=tm.embed_tokens(x).astype(mx.float32))[0, -1].astype(mx.float32))
    del model
    mx.clear_cache()

    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        r = subprocess.run([args.redline, "logits", args.rlm, args.tokens, tmp.name], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"engine failed:\n{r.stdout}\n{r.stderr}")
        eng = np.fromfile(tmp.name, dtype=np.float32)

    top_r, top_e = np.argsort(-ref)[:5], np.argsort(-eng)[:5]
    rel = np.linalg.norm(eng - ref) / np.linalg.norm(ref)
    same = top_r[0] == top_e[0]
    print(f"logits: relative error {rel:.2e}, max|d| {np.abs(eng - ref).max():.3f}, argmax {int(top_e[0])} "
          f"({'same' if same else 'DIFFERENT'} as MLX {int(top_r[0])}), top-5 overlap {len(set(top_r) & set(top_e))}/5")
    ok = same and rel < 2e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
