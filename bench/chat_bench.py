#!/usr/bin/env python3
"""Decode speed on chat answers, by sampling mode, through the runtime.

Three ordinary requests, thinking off, up to 300 tokens each; greedy and two
sampled settings with a fixed seed. Prose is harder to draft than source code,
and sampling harder than greedy decoding, so this is the conservative
companion of bench/agent_bench.py. One backend per run:

  bench/chat_bench.py models/qwen3.5-4b.rlm
  bench/chat_bench.py models/Qwen3.5-4B-MLX-4bit
  REDLINE_MTP=0 bench/chat_bench.py models/qwen3.5-4b.rlm      # the engine without drafts
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline.backends import Sampling, load_backend  # noqa: E402
from redline.chat import Runtime  # noqa: E402

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


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    backend = load_backend(sys.argv[1])
    rt = Runtime(backend, Sampling(temperature=0.0))
    for _ in rt.chat([{"role": "user", "content": "Hi"}], thinking=False, max_tokens=8):   # warm up
        pass
    for label, sampling in MODES:
        tokens = seconds = drafts = accepted = 0
        for prompt in PROMPTS:
            for kind, value in rt.chat([{"role": "user", "content": prompt}], sampling=sampling, max_tokens=300,
                                       thinking=False):
                if kind == "done":
                    tokens += value["completion_tokens"]
                    seconds += value["decode_seconds"]
                    drafts += value.get("drafts", 0)
                    accepted += value.get("drafts_accepted", 0)
        note = f"   {100 * accepted / drafts:.0f}% of drafts accepted" if drafts else ""
        print(f"{backend.name:8s} {label:40s} {tokens:4d} tokens  {tokens / seconds:6.1f} tok/s{note}", flush=True)


if __name__ == "__main__":
    main()
