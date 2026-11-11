#!/usr/bin/env python3
"""Distance of real runtimes from the float model, position by position.

tools/quantize.py reports the KL of a conversion as it simulates it. This
measures what the runtimes actually compute: the float model in MLX gives the
reference distribution at the held-out positions tools/quantize.py uses, then
the raceline engine on a .rlm file and mlx-lm on an MLX model give theirs.
Each step runs in its own process and leaves its log-probabilities in a
directory, so the float pass (slow for a 27B: it streams the checkpoint a
layer at a time) is done once.

  bench/kl_compare.py float  models/Qwen3.8-27B            out/
  bench/kl_compare.py engine models/qwen3.8-27b.rlm         out/
  bench/kl_compare.py mlx    models/Qwen3.8-27B-4bit        out/
  bench/kl_compare.py report out/
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)


def sequences(model_dir_or_tokenizer):
    from quantize import default_texts, token_sequences
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(model_dir_or_tokenizer)
    _, ev_files = default_texts()
    return np.asarray(token_sequences(tok, ev_files, 12))


def save(out, name, logits):
    """Log-probabilities [positions][vocab] as float16."""
    import mlx.core as mx

    z = mx.array(logits)
    lp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    np.save(os.path.join(out, f"{name}.npy"), np.array(lp.astype(mx.float16)))


def run_float(src_dir, out):
    import mlx.core as mx
    from mlx_lm.models import qwen3_5
    from quantize import EVAL_POS, T, Source, build_layer, run_layer

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    args = qwen3_5.TextModelArgs.from_dict(cfg.get("text_config", cfg))
    src = Source(src_dir)
    ids = sequences(os.path.join(src_dir, "tokenizer.json"))
    np.save(os.path.join(out, "tokens.npy"), ids)
    h = mx.array(src.get("embed_tokens.weight")[ids])
    for i in range(args.num_hidden_layers):
        h = run_layer(build_layer(args, i, src.layer_weights(i)), h)
        print(f"\r  float model: layer {i + 1}/{args.num_hidden_layers}", end="", flush=True)
    print()
    x = mx.fast.rms_norm(h[:, T - EVAL_POS - 1:T - 1], mx.array(src.get("norm.weight")), args.rms_norm_eps)
    x = x.reshape(-1, args.hidden_size)
    head = src.get("lm_head.weight" if "lm_head.weight" in src.st or src.prefix + "lm_head.weight" in src.st
                   else "embed_tokens.weight")
    logits = np.concatenate([np.array(x @ mx.array(head[r:r + 16384]).T) for r in range(0, head.shape[0], 16384)], axis=1)
    save(out, "float", logits[:, :args.vocab_size])


def run_engine(rlm, out):
    import ctypes

    import mlx.core as mx
    from quantize import EVAL_POS, T
    from raceline.engine import _library
    from rlmfile import Native

    ids = np.load(os.path.join(out, "tokens.npy"))
    nat = Native(rlm)
    cfg = nat.hdr["config"]
    H, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
    lib = _library()
    e = lib.rl_open(os.path.abspath(rlm).encode(), T, T)
    if lib.rl_has_mtp(e):
        lib.rl_set_mtp(e, 0)
    rows = []
    for seq in ids:
        lib.rl_reset(e)
        lib.rl_forward(e, (ctypes.c_int32 * T)(*seq.tolist()), T)
        hid = np.ctypeslib.as_array(lib.rl_hidden(e), shape=(T, H))[T - EVAL_POS - 1:T - 1].copy()
        rows.append(hid)
    lib.rl_close(e)
    # The engine's own final norm (with its weight, folded or not) and the file's head, as the engine applies them.
    x = mx.fast.rms_norm(mx.array(np.concatenate(rows)), mx.array(nat.floats("final_norm")), eps)
    n = nat.hdr["tensors"]["lm_head"]["n"]
    logits = np.concatenate([np.array(x @ mx.array(nat.block_rows("lm_head", r, min(n, r + 16384))).T)
                             for r in range(0, n, 16384)], axis=1)
    save(out, "engine_" + os.path.basename(rlm).replace(".rlm", ""), logits[:, :cfg["vocab_size"]])


def run_mlx(model_dir, out):
    import mlx.core as mx
    from mlx_lm import load
    from quantize import EVAL_POS, T

    ids = np.load(os.path.join(out, "tokens.npy"))
    model, _ = load(model_dir)
    rows = []
    for seq in ids:
        logits = model(mx.array(seq)[None])[0, T - EVAL_POS - 1:T - 1].astype(mx.float32)
        mx.eval(logits)
        rows.append(np.array(logits))
    save(out, "mlx_" + os.path.basename(os.path.normpath(model_dir)), np.concatenate(rows))


def report(out):
    ref = np.load(os.path.join(out, "float.npy")).astype(np.float32)
    p = np.exp(ref)
    print(f"{ref.shape[0]} positions\n")
    print(f"{'runtime':40s} {'mean KL':>8s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'top 5% share':>13s} {'top-1':>7s}")
    for path in sorted(glob.glob(os.path.join(out, "*.npy"))):
        name = os.path.basename(path)[:-4]
        if name in ("float", "tokens"):
            continue
        lq = np.load(path).astype(np.float32)
        kl = (p * (ref - lq)).sum(axis=1)
        top = np.sort(kl)[::-1]
        share = top[:max(1, len(top) // 20)].sum() / top.sum()
        agree = (lq.argmax(1) == ref.argmax(1)).mean()
        print(f"{name:40s} {kl.mean():8.4f} {np.median(kl):8.4f} {np.percentile(kl, 90):8.4f} "
              f"{np.percentile(kl, 99):8.4f} {100 * share:12.1f}% {100 * agree:6.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["float", "engine", "mlx", "report"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"float": run_float, "engine": run_engine, "mlx": run_mlx}.get(a.what, lambda m, o: report(o))(a.model, a.out)


if __name__ == "__main__":
    main()
