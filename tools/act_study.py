#!/usr/bin/env python3
"""What would 8-bit activations cost? (#6)

The GPU accelerators multiply 8-bit operands about 1.75 times as fast as half
precision against 4-bit weights (bench/hw/nax_probe.mm: int8 x int8 29 TFLOPS,
half x fp4 17 on the M6). This measures the accuracy side before any kernel is
written: the model a .rlm file holds, decoded to float in MLX
(tests/verify_native.py), with the input of every quantized matrix rounded as
an int8 prompt pass would round it, against the unquantized model.

Rounding: the engine feeds a matrix X * c (its column scales folded in), so the
simulation rounds that per row, to int8 with a scale of max|row| / 127, and
divides c out again: X -> Q(X * c) / c.

Variants: every matrix; only the matrices whose input is the (rotated)
residual stream after a norm (attention q/k/v, the DeltaNet input projections,
the MLP gate/up); and those plus the attention and DeltaNet output
projections (leaving the MLP down projection, whose input is the MLP's
spikiest activation, in half precision).

  .venv/bin/python tools/act_study.py models/Qwen3.5-4B models/qwen3.5-4b.rot.rlm
"""

import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from quant_study import DATA, Reference, build_splits  # noqa: E402
from rlmfile import Native  # noqa: E402
from verify_native import PARTS, load_decoded  # noqa: E402

INPUT_OF_STREAM = ("lin.in_proj", "attn.qkv", "mlp.gate_up")   # read the normed residual stream
OUTPUT_PROJ = ("lin.out_proj", "attn.o_proj")


class RoundedInput(nn.Module):
    """A linear layer whose input is rounded as an int8 pass with per-row scales would round it."""

    def __init__(self, inner, c, bits=8):
        super().__init__()
        self.inner = inner
        self.c = mx.array(c.astype(np.float32))
        self.levels = float(2 ** (bits - 1) - 1)

    def __call__(self, x):
        xc = x.astype(mx.float32) * self.c
        s = mx.max(mx.abs(xc), axis=-1, keepdims=True) / self.levels
        s = mx.where(s > 0, s, 1.0)
        return self.inner((mx.round(xc / s) * s / self.c).astype(x.dtype))


def patch(model, nat, names, bits=8):
    """Wraps the modules of the file tensors `names` (layers.*.<name>) in RoundedInput; -> undo list."""
    undo = []
    for i, layer in enumerate(model.language_model.model.layers):
        for name in names:
            tensor = f"layers.{i}.{name}"
            if tensor not in nat.hdr["tensors"]:
                continue
            c = nat.column_scales(tensor)
            for path in PARTS[name].values():
                parent = layer
                *head, attr = path.split(".")
                for a in head:
                    parent = getattr(parent, a)
                inner = getattr(parent, attr)
                setattr(parent, attr, RoundedInput(inner, c, bits))
                undo.append((parent, attr, inner))
    return undo


def unpatch(undo):
    for parent, attr, inner in undo:
        setattr(parent, attr, inner)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref", help="the original checkpoint directory")
    ap.add_argument("rlm", help="the converted model file")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()

    nat = Native(args.rlm)
    _, eval_seqs = build_splits(_tokenizer(args.ref))
    path = f"{DATA}/ref_logp_{os.path.basename(os.path.normpath(args.ref))}.f32"
    if not os.path.exists(path):      # the float model's log-probabilities, once (tools/quant_study.py's cache)
        from mlx_lm import load
        float_model, _ = load(args.ref)
        Reference(float_model, eval_seqs, path)
        del float_model
        mx.clear_cache()
    model = load_decoded(args.ref, args.rlm)
    ref = Reference(model, eval_seqs, path)      # reads the cache
    every = tuple(PARTS)
    variants = [("4-bit weights, half-precision inputs (the file as it runs today)", ()),
                ("int8 inputs to every matrix", every),
                ("int8 inputs where the input is the normed residual stream", INPUT_OF_STREAM),
                ("... and to the attention / DeltaNet output projections", INPUT_OF_STREAM + OUTPUT_PROJ)]
    results = {}
    for label, names in variants:
        undo = patch(model, nat, names)
        r = ref.score(model)
        unpatch(undo)
        results[label] = r
        print(f"{label:72s} KL {r['kl']:.4f}  ppl {r['ppl']:.3f}  top-1 {100 * r['top1']:.1f}%", flush=True)
    if args.json:
        json.dump(dict(model=args.rlm, positions=len(eval_seqs) * 96, results=results), open(args.json, "w"), indent=1)


def _tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


if __name__ == "__main__":
    main()
