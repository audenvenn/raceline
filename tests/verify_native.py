#!/usr/bin/env python3
"""Check the engine on a native-format model against MLX running the same weights.

The .rlm file is decoded back to float matrices (codebook value * row scale *
column scale), loaded into the original model in MLX, and evaluated with
float32 activations. That checks the file writer, the kernel's reading of the
codes and scales, and the engine's handling of column scales in one go.

  .venv/bin/python tests/verify_native.py models/Qwen3.5-4B models/qwen3.5-4b.fp4.rlm build/ref/tokens_96.i32
"""

import argparse
import json
import struct
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np
from mlx_lm import load

LEVELS = {
    "fp4": np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32),
    "int15": np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], dtype=np.float32),
}
PARTS = {  # file tensor -> {part: module path below the layer}
    "lin.in_proj": {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "b": "linear_attn.in_proj_b",
                    "a": "linear_attn.in_proj_a"},
    "lin.out_proj": {"out": "linear_attn.out_proj"},
    "attn.qkv": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj"},
    "attn.o_proj": {"out": "self_attn.o_proj"},
    "mlp.gate_up": {"gate": "mlp.gate_proj", "up": "mlp.up_proj"},
    "mlp.down": {"out": "mlp.down_proj"},
}


def read_native(path):
    f = open(path, "rb")
    assert f.read(4) == b"RLM1"
    _, n = struct.unpack("<IQ", f.read(12))
    hdr = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def matrix(t):
        k, book = t["k"], hdr["quant"]["book"]
        if t["kind"] == "q4n":
            blocks = t["n_pad"] // 256
            b = np.asarray(mm[t["w"]:t["w"] + blocks * k * 128]).reshape(blocks, k, 128)
            codes = np.empty((blocks, k, 256), dtype=np.uint8)
            codes[:, :, 0::2], codes[:, :, 1::2] = b & 15, b >> 4
            codes = codes.transpose(0, 2, 1).reshape(t["n_pad"], k)[:t["n"]]
        else:
            b = np.asarray(mm[t["w"]:t["w"] + t["n"] * k // 2]).reshape(t["n"], k // 2)
            codes = np.empty((t["n"], k), dtype=np.uint8)
            codes[:, 0::2], codes[:, 1::2] = b & 15, b >> 4
        s = np.frombuffer(mm, dtype=np.float32, count=t["n"], offset=t["s"])
        c = np.frombuffer(mm, dtype=np.float32, count=k, offset=t["c"])
        return LEVELS[book][codes] * s[:, None] * c[None, :]

    return hdr, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("float_model")
    ap.add_argument("rlm")
    ap.add_argument("tokens")
    ap.add_argument("--redline", default="build/cmake/redline-engine")
    args = ap.parse_args()

    hdr, matrix = read_native(args.rlm)
    model, _ = load(args.float_model)
    lm = model.language_model
    tm = lm.model
    for i, layer in enumerate(tm.layers):
        for name, parts in PARTS.items():
            t = hdr["tensors"].get(f"layers.{i}.{name}")
            if t is None:
                continue
            W = matrix(t)
            for part, path in parts.items():
                mod = layer
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                p = t["parts"][part]
                mod.weight = mx.array(W[p["offset"]:p["offset"] + p["n"]].astype(np.float16))
    emb = matrix(hdr["tensors"]["embed"])
    tm.embed_tokens.weight = mx.array(emb.astype(np.float16))
    if not lm.args.tie_word_embeddings:
        lm.lm_head.weight = mx.array(matrix(hdr["tensors"]["lm_head"]).astype(np.float16))
    else:  # one matrix in two layouts: they must decode to the same values
        head = matrix(hdr["tensors"]["lm_head"])
        print("tied head and lookup table identical:", bool(np.array_equal(head, emb)))

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
