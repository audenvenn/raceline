#!/usr/bin/env python3
"""Layer-by-layer check of the engine against MLX on the same quantized model.

The reference is the MLX model evaluated in float32 (MLX's default for these
checkpoints is bfloat16, which is itself a noisy approximation of the same
quantized model; its distance to the float32 reference is printed for scale).

  .venv/bin/python tests/verify_against_mlx.py models/Qwen3.5-4B-MLX-4bit models/qwen3.5-4b.rlm \
      build/ref/tokens_96.i32
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def run_mlx(model, tokens, dtype):
    """Returns ([h after each layer] as float32 [T,H], logits of the last token)."""
    model.set_dtype(dtype)
    lm = model.language_model
    tm = lm.model
    cache = lm.make_cache()
    h = tm.embed_tokens(mx.array(tokens)[None])
    fa_mask = create_attention_mask(h, cache[tm.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[tm.ssm_idx])
    hs = []
    for layer, c in zip(tm.layers, cache):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        mx.eval(h)
        hs.append(np.array(h[0].astype(mx.float32)))
    out = tm.norm(h[:, -1:, :])
    logits = tm.embed_tokens.as_linear(out) if lm.args.tie_word_embeddings else lm.lm_head(out)
    return hs, np.array(logits[0, 0].astype(mx.float32))


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def kl(p_logits, q_logits):
    """KL(p || q) in nats, from logits."""
    p = p_logits.astype(np.float64) - p_logits.max()
    q = q_logits.astype(np.float64) - q_logits.max()
    lp = p - np.log(np.exp(p).sum())
    lq = q - np.log(np.exp(q).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlx_model")
    ap.add_argument("rlm_model")
    ap.add_argument("tokens")
    ap.add_argument("--redline", default="build/cmake/redline-engine")
    ap.add_argument("--tol", type=float, default=2e-2, help="max relative error allowed on any layer")
    args = ap.parse_args()

    tokens = np.fromfile(args.tokens, dtype=np.int32)
    model, _ = load(args.mlx_model)
    ref_h, ref_logits = run_mlx(model, tokens.tolist(), mx.float32)
    bf_h, bf_logits = run_mlx(model, tokens.tolist(), mx.bfloat16)
    hidden = ref_h[0].shape[1]
    kinds = ["linear" if l.is_linear else "attn" for l in model.language_model.model.layers]
    # Release MLX's copy of the weights before the engine maps its own: a 27B
    # model does not fit twice in 24 GB.
    del model
    gc.collect()
    mx.clear_cache()

    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([args.redline, "dump", args.rlm_model, args.tokens, d], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"engine failed:\n{r.stdout}\n{r.stderr}")
        eng_h = [np.fromfile(os.path.join(d, f"layer_{i:02d}.f32"), dtype=np.float32).reshape(-1, hidden)
                 for i in range(len(ref_h))]
        eng_logits = np.fromfile(os.path.join(d, "logits.f32"), dtype=np.float32)

    print(f"{len(tokens)} tokens, {len(ref_h)} layers; reference = MLX float32")
    print(f"{'layer':>5s} {'kind':>6s} {'engine rel.err':>15s} {'MLX-bf16 rel.err':>17s}")
    worst = 0.0
    for i, (e, r_, b) in enumerate(zip(eng_h, ref_h, bf_h)):
        er, br = rel(e, r_), rel(b, r_)
        worst = max(worst, er)
        kind = kinds[i]
        if i < 4 or i % 4 == 3 or er > args.tol:
            print(f"{i:5d} {kind:>6s} {er:15.2e} {br:17.2e}")

    print("\nlast-token logits vs float32 reference:")
    for name, lg in (("engine", eng_logits), ("MLX bf16", bf_logits)):
        top_ref, top = np.argsort(-ref_logits)[:5], np.argsort(-lg)[:5]
        print(f"  {name:9s} argmax {int(top[0])} ({'same' if top[0] == top_ref[0] else 'DIFFERENT'} as reference "
              f"{int(top_ref[0])}), top-5 overlap {len(set(top) & set(top_ref))}/5, "
              f"max|dlogit| {np.abs(lg - ref_logits).max():.3f}, KL {kl(ref_logits, lg):.2e} nats")

    ok = worst <= args.tol and int(np.argmax(eng_logits)) == int(np.argmax(ref_logits))
    print(f"\n{'PASS' if ok else 'FAIL'}: worst layer relative error {worst:.2e} (tolerance {args.tol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
