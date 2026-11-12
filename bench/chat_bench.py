#!/usr/bin/env python3
"""Decode speed on chat answers, by sampling mode, through the runtime.

Three ordinary requests, thinking off, up to 300 tokens each; greedy and two
sampled settings with a fixed seed. Prose is harder to draft than source code,
and sampling harder than greedy decoding, so this is the conservative
companion of bench/agent_bench.py. One backend per run:

  bench/chat_bench.py models/qwen3.5-4b.rlm
  bench/chat_bench.py models/Qwen3.5-4B-MLX-4bit
  RACELINE_MTP=0 bench/chat_bench.py models/qwen3.5-4b.rlm      # the engine without drafts
  bench/chat_bench.py models/qwen3.5-4b.rlm --json out.json      # also write the results
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import Sampling, load_backend  # noqa: E402
from raceline.chat import Runtime  # noqa: E402

PROMPTS = [
    "Write a Python function that merges two sorted lists, then explain how it works in a short paragraph.",
    "Explain the difference between a process and a thread to a new programmer, in about 150 words.",
    "Give me a step-by-step plan for migrating a small web app from SQLite to PostgreSQL.",
]
MODES = [
    ("greedy", Sampling(temperature=0.0)),
    ("temperature 0.7, top_p 0.8, top_k 20", Sampling(temperature=0.7, top_p=0.8, top_k=20, seed=1)),
    ("temperature 1.0, top_p 0.95, top_k 20", Sampling(temperature=1.0, top_p=0.95, top_k=20, seed=1)),
]


def warn_if_busy():
    """Speed numbers taken next to other work are worthless: say so up front."""
    load = os.getloadavg()[0]
    if load > 3:
        print(f"warning: the load average is {load:.1f}; other processes are using the machine and the "
              "numbers will be low (`ps -Ao pid,%cpu,command -r | head`)", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()
    warn_if_busy()
    backend = load_backend(args.model)
    results = []
    rt = Runtime(backend, Sampling(temperature=0.0))
    for _ in rt.chat([{"role": "user", "content": "Hi"}], thinking=False, max_tokens=8):   # warm up
        pass
    experts = getattr(backend, "expert_cache", lambda: None)   # a MoE model with its experts on disk
    for label, sampling in MODES:
        tokens = seconds = drafts = accepted = 0
        e0 = experts()
        for prompt in PROMPTS:
            for kind, value in rt.chat([{"role": "user", "content": prompt}], sampling=sampling, max_tokens=300,
                                       thinking=False):
                if kind == "done":
                    tokens += value["completion_tokens"]
                    seconds += value["decode_seconds"]
                    drafts += value.get("drafts", 0)
                    accepted += value.get("drafts_accepted", 0)
        note = f"   {100 * accepted / drafts:.0f}% of drafts accepted" if drafts else ""
        e1, cache = experts(), {}
        if e0 and e1:
            hits, reads = e1[1] - e0[1], e1[2] - e0[2]
            cache = dict(expert_slots=e1[0], experts_from_cache=hits / max(1, hits + reads), experts_read=reads)
            note += f"   {100 * cache['experts_from_cache']:.1f}% of experts from the cache"
        print(f"{backend.name:8s} {label:40s} {tokens:4d} tokens  {tokens / seconds:6.1f} tok/s{note}", flush=True)
        results.append(dict(model=args.model, backend=backend.name, mtp=bool(getattr(backend, "mtp", False)),
                            sampling=label, tokens=tokens, decode_tps=tokens / seconds,
                            drafts_accepted=accepted / drafts if drafts else None, **cache))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
