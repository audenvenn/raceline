#!/usr/bin/env python3
"""Snapshots taken inside a pass against snapshots of a pass that ends there (#60).

  .venv/bin/python tests/verify_marks.py MODEL.rlm [prompt tokens] [mark]

The engine can keep the recurrent state at a row inside a prompt pass
(Engine::set_mark), so the session need not end a pass at each branch point.
Checks, on build/ref/tokens_20k.i32:
  1. the state saved at the mark is the one a pass ending at the mark leaves
     (byte for byte; of the draft head's parked rows only the one in use);
  2. the prompt's logits are the same either way;
  3. restored, both snapshots continue a different text with the same logits,
     and generate the same tokens with drafts;
  4. through the session: the same prompt with two branch points syncs in
     fewer passes and leaves the same logits and snapshot positions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import Sampling  # noqa: E402
from raceline.engine import RacelineBackend  # noqa: E402
from raceline.session import Session  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    toks = np.fromfile(os.path.join(HERE, "..", "build", "ref", "tokens_20k.i32"), dtype=np.int32).tolist()
    prompt, other = toks[:n], toks[n + 500:n + 600]
    b = RacelineBackend(model, max_ctx=8192)
    lib, e = b._lib, b._e
    ok = True

    def check(cond, what):
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'}  {what}")
        ok &= bool(cond)

    # The bytes of a state that mean something: all of it but the draft head's parked rows past the first.
    size = lib.rl_state_bytes(e)
    unused = (16 - 1) * lib.rl_hidden_size(e) * 4 if lib.rl_has_mtp(e) else 0

    b.reset()
    b.prefill(prompt[:r])
    cut = b.snapshot()
    b.prefill(prompt[r:])
    end_cut = b._logits.copy()

    b.reset()
    snaps = b.prefill(prompt, marks=[r])
    end_mark = b._logits.copy()
    check(len(snaps) == 1 and snaps[0].position == r, f"one snapshot, at position {r}")
    mark = snaps[0]
    same = np.array_equal(cut.state[:size - unused], mark.state[:size - unused])
    check(same, f"the state at the mark equals the cut pass's ({size - unused} bytes compared)")
    check(np.array_equal(end_cut, end_mark), "the prompt's logits are the same with one pass or two")

    outs = {}
    for name, snap in (("cut", cut), ("mark", mark)):
        b.restore(snap)
        b.prefill(other)
        logits = b._logits.copy()
        tokens = list(b.generate(Sampling(temperature=0.0), 40, prompt[:r] + other))
        outs[name] = (logits, tokens)
    check(np.array_equal(outs["cut"][0], outs["mark"][0]), "restored, both continue a new text with the same logits")
    check(outs["cut"][1] == outs["mark"][1], f"... and draft the same {len(outs['cut'][1])} tokens")

    # The session: two branch points in one prompt.
    points, pin = (r, n - 10), r
    runs = {}
    for name, marks in (("cut", False), ("mark", True)):
        b.marks_in_passes = marks
        s = Session(b)
        s.reset()
        p0 = b.counters["passes"]
        s.sync(prompt, points, pin)
        runs[name] = (b.counters["passes"] - p0, b._logits.copy(), [x.position for x in s._snaps])
    b.marks_in_passes = True
    check(np.array_equal(runs["cut"][1], runs["mark"][1]) and runs["cut"][2] == runs["mark"][2],
          f"through the session: the same logits and snapshots at {runs['mark'][2]}")
    check(runs["mark"][0] < runs["cut"][0], f"... in {runs['mark'][0]} passes instead of {runs['cut'][0]}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
