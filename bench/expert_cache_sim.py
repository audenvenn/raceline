#!/usr/bin/env python3
"""Expert cache replacement policies replayed on a routing trace (no GPU).

When a mixture-of-experts model's experts live on disk, the engine keeps the
ones in use in a cache of fixed size (src/engine.mm, ExpertCache). This replays
which experts each layer needed and counts the experts each policy has to read:

  lru     least recently used: every pass counts as a use
  cold    passes of more than 16 rows (prompt chunks) do not count as uses:
          their hits keep their place and what they load goes in at the cold
          end (what the engine does)

A prompt chunk needs nearly every expert of every layer, a cyclic scan in which
plain LRU evicts exactly what the next layers need.

  # a synthetic agent session: prompts of 4096, 1200, 800 and 2000 tokens, each followed by decoding
  python bench/expert_cache_sim.py
  # what a real session needed (RACELINE_EXPERT_TRACE=<file> while running the engine with the cache on)
  python bench/expert_cache_sim.py --trace experts.trace --experts 256
"""

import argparse
from collections import OrderedDict

import numpy as np


def synthetic(layers, E, k, zipf, seed=0, plan=((4096, 300), (1200, 300), (800, 300), (2000, 400))):
    """(layer, rows, experts needed) per layer and pass: prompt chunks of up to 1024
    tokens, then single decoded tokens whose experts follow a skewed popularity."""
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(layers):
        p = 1.0 / np.arange(1, E + 1) ** zipf
        pop.append(rng.permutation(p / p.sum()))
    trace = []
    for prompt, gen in plan:
        for start in range(0, prompt, 1024):
            rows = min(1024, prompt - start)
            for layer in range(layers):
                trace.append((layer, rows, set(np.flatnonzero(rng.multinomial(rows * k, pop[layer])).tolist())))
        for _ in range(gen):
            for layer in range(layers):
                trace.append((layer, 1, set(rng.choice(E, k, replace=False, p=pop[layer]).tolist())))
    return trace


def read_trace(path, E):
    """The engine's trace: per request int32 layer, int32 rows, then E bits (little-endian bit order)."""
    raw = np.fromfile(path, dtype=np.uint8)
    size = 8 + (E + 7) // 8
    out = []
    for i in range(0, len(raw) - size + 1, size):
        layer, rows = np.frombuffer(raw[i:i + 8].tobytes(), dtype=np.int32)
        bits = np.unpackbits(raw[i + 8:i + size], bitorder="little")[:E]
        out.append((int(layer), int(rows), set(np.flatnonzero(bits).tolist())))
    return out


def replay(trace, slots, policy, big_rows=16):
    """Experts read during prompt passes and during decoding."""
    order = OrderedDict()                      # cached (layer, expert), coldest first
    reads = {"prompt": 0, "decode": 0}
    for layer, rows, need in trace:
        big = rows > big_rows
        keys = {(layer, e) for e in need}
        for key in keys:
            if key in order:
                if not (big and policy == "cold"):
                    order.move_to_end(key)
                continue
            reads["prompt" if big else "decode"] += 1
            if len(order) >= slots:
                del order[next(x for x in order if x not in keys)]
            order[key] = None
            if big and policy == "cold":
                order.move_to_end(key, last=False)
    return reads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", help="a trace written by the engine (default: a synthetic session)")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--per-token", type=int, default=8, help="experts routed per token (synthetic)")
    ap.add_argument("--zipf", type=float, default=1.0, help="skew of the experts' popularity (synthetic)")
    ap.add_argument("--fractions", default="0.5,0.73,0.9", help="cache sizes, as fractions of all experts")
    args = ap.parse_args()
    trace = (read_trace(args.trace, args.experts) if args.trace
             else synthetic(args.layers, args.experts, args.per_token, args.zipf))
    layers = 1 + max(t[0] for t in trace)
    print(f"{len(trace)} layer passes over {layers} layers")
    for frac in (float(f) for f in args.fractions.split(",")):
        slots = int(frac * layers * args.experts)
        a, b = replay(trace, slots, "lru"), replay(trace, slots, "cold")
        print(f"cache of {frac:.0%} of the experts: experts read for prompts / while decoding: "
              f"lru {a['prompt']} / {a['decode']}, cold {b['prompt']} / {b['decode']}")


if __name__ == "__main__":
    main()
