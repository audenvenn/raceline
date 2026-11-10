#!/usr/bin/env python3
"""Baseline: mlx-lm prefill and decode speed on a local MLX model.

Prompts are real source code (Python standard library) cut to an exact token
count, so prefill is measured on realistic text and decode continues it without
hitting EOS early.

  .venv/bin/python bench/baselines/mlx_bench.py models/Qwen3.5-4B-MLX-4bit
"""

import argparse
import glob
import json
import os
import sysconfig
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def corpus_tokens(tokenizer, n):
    """First n tokens of a deterministic slab of stdlib source."""
    stdlib = sysconfig.get_paths()["stdlib"]
    ids = []
    for path in sorted(glob.glob(os.path.join(stdlib, "*.py"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            ids.extend(tokenizer.encode(f.read(), add_special_tokens=False))
        if len(ids) >= n:
            break
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[128, 1024, 4096, 16384])
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    print(f"model: {args.model}  load {load_s:.2f}s  mlx {mx.__version__}")
    print(f"{'prompt':>8s} {'prefill tok/s':>14s} {'ttft s':>8s} {'decode tok/s':>13s} {'peak GB':>8s}")

    sampler = make_sampler(temp=0.0)
    all_ids = corpus_tokens(tokenizer, max(args.prompt_tokens))
    results = []
    for n in args.prompt_tokens:
        best = None
        for _ in range(args.reps):
            mx.clear_cache()
            mx.reset_peak_memory()
            last = None
            t0 = time.perf_counter()
            ttft = None
            for resp in stream_generate(model, tokenizer, all_ids[:n], max_tokens=args.gen_tokens, sampler=sampler):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                last = resp
            r = dict(prompt_tokens=last.prompt_tokens, prefill_tps=last.prompt_tps, ttft_s=ttft,
                     gen_tokens=last.generation_tokens, decode_tps=last.generation_tps,
                     peak_gb=last.peak_memory)
            if best is None or r["decode_tps"] > best["decode_tps"]:
                keep_prefill = max(r["prefill_tps"], best["prefill_tps"]) if best else r["prefill_tps"]
                best = r
                best["prefill_tps"] = keep_prefill
            else:
                best["prefill_tps"] = max(best["prefill_tps"], r["prefill_tps"])
            best["ttft_s"] = min(best["ttft_s"], ttft)
        results.append(best)
        print(f"{best['prompt_tokens']:8d} {best['prefill_tps']:14.1f} {best['ttft_s']:8.2f} "
              f"{best['decode_tps']:13.1f} {best['peak_gb']:8.2f}", flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(model=args.model, engine=f"mlx-lm (mlx {mx.__version__})", results=results), f, indent=2)


if __name__ == "__main__":
    main()
