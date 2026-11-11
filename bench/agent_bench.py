#!/usr/bin/env python3
"""Prefill and decode speed of the runtime's backends, as an application sees them.

Each measurement runs in its own process (so peak memory is that run's own)
through the same `Backend` interface the agent uses: a prompt of real source
code cut to an exact token count, then greedy decoding that continues it.

  bench/agent_bench.py models/qwen3.5-4b.fp4.rlm models/Qwen3.5-4B-MLX-4bit --prompt 1024 8192
  REDLINE_MTP=0 bench/agent_bench.py models/qwen3.5-4b.fp4.rlm        # without speculative decoding

Compare engines back to back, with nothing else using the GPU.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import sysconfig
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def corpus_tokens(tokenizer, n):
    """First n tokens of a deterministic slab of Python standard library source."""
    ids = []
    for path in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))):
        with open(path, encoding="utf-8", errors="ignore") as f:
            ids.extend(tokenizer.encode(f.read(), add_special_tokens=False))
        if len(ids) >= n:
            break
    return ids[:n]


def one(model, prompt_tokens, gen_tokens):
    sys.path.insert(0, ROOT)
    from redline.backends import Sampling, load_backend

    t0 = time.perf_counter()
    b = load_backend(model)
    load_s = time.perf_counter() - t0
    ids = corpus_tokens(b.tokenizer, prompt_tokens)
    greedy = Sampling(temperature=0.0)
    b.prefill(ids[:64])                                  # compile kernels, page the weights in
    for _ in b.generate(greedy, 8, ids[:64]):
        pass
    best = {}
    for _ in range(2):
        b.reset()
        if hasattr(b, "counters"):
            b.counters.update(drafts=0, accepted=0)
        t0 = time.perf_counter()
        b.prefill(ids)
        prefill = len(ids) / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        out = list(b.generate(greedy, gen_tokens, ids))
        decode = len(out) / (time.perf_counter() - t0)
        if decode > best.get("decode_tps", 0):
            best = dict(prefill_tps=prefill, decode_tps=decode, tokens=out[:16])
            c = getattr(b, "counters", None)
            if c and c.get("drafts"):
                best["drafts_accepted"] = c["accepted"] / c["drafts"]
    print(json.dumps(dict(model=model, backend=b.name, mtp=bool(getattr(b, "mtp", False)), load_s=load_s,
                          prompt_tokens=len(ids), gen_tokens=gen_tokens, **best)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--prompt", type=int, nargs="+", default=[1024])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()
    if args.one:
        return one(args.models[0], args.prompt[0], args.gen)

    results = []
    print(f"{'model':34s} {'backend':8s} {'prompt':>7s} {'load':>6s} {'prefill':>12s} {'decode':>12s} {'drafts ok':>10s} {'peak mem':>9s}")
    for n in args.prompt:
        for model in args.models:
            r = subprocess.run(["/usr/bin/time", "-l", sys.executable, os.path.abspath(__file__), model, "--one",
                                "--prompt", str(n), "--gen", str(args.gen)], capture_output=True, text=True)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
            if line is None:
                print(f"{model}: failed\n{r.stderr[-2000:]}")
                continue
            res = json.loads(line)
            peak = re.search(r"(\d+)\s+peak memory footprint", r.stderr)
            res["peak_gb"] = int(peak.group(1)) / 1e9 if peak else None
            results.append(res)
            acc = f"{100 * res['drafts_accepted']:.0f}%" if "drafts_accepted" in res else "-"
            mem = f"{res['peak_gb']:.1f} GB" if res["peak_gb"] else "-"
            print(f"{os.path.basename(os.path.normpath(model)):34s} {res['backend']:8s} {res['prompt_tokens']:7d} "
                  f"{res['load_s']:5.1f}s {res['prefill_tps']:6.0f} tok/s {res['decode_tps']:6.1f} tok/s {acc:>10s} {mem:>9s}",
                  flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
