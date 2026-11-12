#!/usr/bin/env python3
"""A conversation set aside and brought back writes exactly what a fresh run writes (GPU, a minute).

The session keeps conversation A (its recurrent state and attention entries),
serves B, a side request that shares A's first 300 tokens, and comes back to
A: its next tokens and the distribution over them must be bit-identical to a
fresh engine that read all of A at once, with nothing of A computed again.
Then the same after a third conversation that shares nothing has overwritten
the start A shares with B: only that start is computed again. Greedy decoding
with the draft head on, so speculative passes run over the restored entries.

  .venv/bin/python tests/verify_parking.py models/qwen3.5-4b.rlm build/ref/tokens_20k.i32
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import Sampling  # noqa: E402
from raceline.engine import RacelineBackend  # noqa: E402
from raceline.session import Session  # noqa: E402


def run(session, prompt, n=48):
    usage = session.sync(prompt, (300,), 300)
    out = list(session.generate(Sampling(temperature=0.0), n))
    return usage, out, session.backend._logits.copy() if session.backend._logits is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokens", help="int32 token file, at least 3000 tokens")
    args = ap.parse_args()
    toks = np.fromfile(args.tokens, dtype=np.int32).tolist()
    system = toks[:300]
    a = system + toks[1000:2700]          # 2000 tokens, past several 256-position blocks
    b = system + toks[2800:2900]          # a side request after the shared start
    c = toks[5000:5600]                   # shares nothing

    backend = RacelineBackend(args.model, max_ctx=8192)     # one engine: a MoE model is too large for two
    s = Session(backend)
    ok = True

    def check(what, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'}  {what}{('  ' + detail) if detail else ''}", flush=True)

    def fresh(prompt):
        """What the engine writes after reading `prompt` from scratch."""
        _, out, logits = run(Session(backend), prompt)    # a new session resets the engine
        return out, logits

    _, out_a, _ = run(s, a)
    usage_b, out_b, _ = run(s, b)
    check("the long conversation was set aside", s.parked == 1, f"parked {s.parked}")
    check("the side request started from the shared system prompt", usage_b["cached_tokens"] == 300,
          f"cached {usage_b['cached_tokens']}")
    a2 = a + out_a + toks[3000:3040]
    usage2, out_a2, logits2 = run(s, a2)
    check("it came back without recomputing", usage2["cached_tokens"] == len(a) + len(out_a),
          f"cached {usage2['cached_tokens']} of {len(a2)}")
    run(s, b + out_b + toks[2950:2960])    # sets A aside again, from the block its 300 shared tokens end in
    run(s, c)                              # shares nothing: overwrites that start
    a3 = a2 + out_a2 + toks[3100:3110]
    usage3, out_a3, logits3 = run(s, a3)
    check("after its start was overwritten, only that start is computed again",
          usage3["cached_tokens"] == len(a2) + len(out_a2) - 256, f"cached {usage3['cached_tokens']} of {len(a3)}")

    ref2, ref_logits2 = fresh(a2)
    check("it writes what a fresh run writes", out_a2 == ref2,
          "" if out_a2 == ref2 else f"first difference at {next(i for i, (x, y) in enumerate(zip(out_a2, ref2)) if x != y)}")
    check("with bit-identical logits", logits2 is not None and np.array_equal(logits2, ref_logits2))
    ref3, ref_logits3 = fresh(a3)
    check("and so does it after its start was computed again, bit for bit",
          out_a3 == ref3 and logits3 is not None and np.array_equal(logits3, ref_logits3))
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
