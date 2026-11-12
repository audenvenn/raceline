#!/usr/bin/env python3
"""Prefill and decode speed of the runtime's backends, as an application sees them.

Each measurement runs in its own process (so peak memory is that run's own)
through the same `Backend` interface the agent uses: a prompt of real source
code cut to an exact token count, then greedy decoding that continues it.

How fast drafts go depends on the text: the Python standard library
(`--corpus stdlib`, the default) is code the model has seen in training and
partly remembers; `--corpus unseen` is this project's own code from after the
models were trained (frozen copies in bench/corpus/). `--samples N` continues
N different stretches of the text and reports the total.

  bench/agent_bench.py models/qwen3.5-4b.fp4.rlm models/Qwen3.5-4B-MLX-4bit --prompt 1024 8192
  bench/agent_bench.py models/qwen3.5-4b.fp4.rlm --corpus unseen --samples 4
  RACELINE_MTP=0 bench/agent_bench.py models/qwen3.5-4b.fp4.rlm        # without speculative decoding

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


def corpus_tokens(tokenizer, n, corpus="stdlib", skip=0):
    """n tokens of source code, from token `skip` on: Python standard library
    files (sorted by name) or the frozen files in bench/corpus/."""
    if corpus == "stdlib":
        paths = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, "bench", "corpus", "*.txt")))
    ids = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            ids.extend(tokenizer.encode(f.read(), add_special_tokens=False))
        if len(ids) >= skip + n:
            break
    if len(ids) < skip + n:
        raise SystemExit(f"the {corpus} corpus has {len(ids)} tokens, {skip + n} needed")
    return ids[skip:skip + n]


def warn_if_busy():
    """Speed numbers taken next to other work are worthless: say so up front."""
    load = os.getloadavg()[0]
    if load > 3:
        print(f"warning: the load average is {load:.1f}; other processes are using the machine and the "
              "numbers will be low (`ps -Ao pid,%cpu,command -r | head`)", file=sys.stderr, flush=True)


def one(model, prompt_tokens, gen_tokens, corpus="stdlib", samples=1):
    sys.path.insert(0, ROOT)
    from raceline.backends import Sampling, load_backend

    t0 = time.perf_counter()
    b = load_backend(model)
    load_s = time.perf_counter() - t0
    greedy = Sampling(temperature=0.0)
    warm = corpus_tokens(b.tokenizer, 64, corpus)
    b.prefill(warm)                                      # compile kernels, page the weights in
    for _ in b.generate(greedy, 8, warm):
        pass
    total = dict(prefill_s=0.0, decode_s=0.0, generated=0, drafts=0, accepted=0)
    first = None
    for k in range(samples):           # stretches far enough apart that they share no text
        ids = corpus_tokens(b.tokenizer, prompt_tokens, corpus, skip=k * (prompt_tokens + gen_tokens))
        best = None
        for _ in range(2):
            b.reset()
            if hasattr(b, "counters"):
                b.counters.update(drafts=0, accepted=0)
            t0 = time.perf_counter()
            b.prefill(ids)
            t_prefill = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = list(b.generate(greedy, gen_tokens, ids))
            t_decode = time.perf_counter() - t0
            if best is None or t_decode < best[1]:
                c = dict(getattr(b, "counters", {}) or {})
                best = (t_prefill, t_decode, len(out), c.get("drafts", 0), c.get("accepted", 0), out[:16])
        first = first or best[5]
        for key, v in zip(("prefill_s", "decode_s", "generated", "drafts", "accepted"), best[:5]):
            total[key] += v
    res = dict(prefill_tps=samples * prompt_tokens / total["prefill_s"], decode_tps=total["generated"] / total["decode_s"],
               tokens=first)
    if total["drafts"]:
        res["drafts_accepted"] = total["accepted"] / total["drafts"]
    print(json.dumps(dict(model=model, backend=b.name, mtp=bool(getattr(b, "mtp", False)), load_s=load_s,
                          prompt_tokens=prompt_tokens, gen_tokens=gen_tokens, corpus=corpus, samples=samples, **res)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--prompt", type=int, nargs="+", default=[1024])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--corpus", choices=["stdlib", "unseen"], default="stdlib")
    ap.add_argument("--samples", type=int, default=1, help="stretches of the text to continue (reported as one total)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()
    if args.one:
        return one(args.models[0], args.prompt[0], args.gen, args.corpus, args.samples)

    warn_if_busy()
    results = []
    print(f"{'model':34s} {'backend':8s} {'prompt':>7s} {'load':>6s} {'prefill':>12s} {'decode':>12s} {'drafts ok':>10s} {'peak mem':>9s}")
    for n in args.prompt:
        for model in args.models:
            r = subprocess.run(["/usr/bin/time", "-l", sys.executable, os.path.abspath(__file__), model, "--one",
                                "--prompt", str(n), "--gen", str(args.gen), "--corpus", args.corpus,
                                "--samples", str(args.samples)], capture_output=True, text=True)
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
