#!/usr/bin/env python3
"""Speculative decoding must never change what the model computes.

Drives the engine through its C API and compares, bit for bit, with a clean
run that feeds the same tokens one at a time:

  1. speculative passes whose drafts are all wrong, partly right or all
     right, for every number of drafts the engine takes: the logits of the
     rows that stand, and the head's next draft, must equal the clean run's;
  2. a draft chain that is abandoned (the token that arrives is not the one
     the head drafted for);
  3. saving the state, decoding on, loading it back and decoding again.

  .venv/bin/python tests/verify_spec.py models/qwen3.5-4b.fp4.rlm build/ref/tokens_96.i32
"""

import argparse
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from redline.engine import _library  # noqa: E402


class Eng:
    def __init__(self, lib, path):
        self.lib, self.e = lib, lib.rl_open(os.path.abspath(path).encode(), 4096, 256)
        self.V, self.stride = lib.rl_vocab(self.e), lib.rl_logits_stride(self.e)

    def close(self):
        self.lib.rl_close(self.e)

    def forward(self, tokens):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        return np.ctypeslib.as_array(self.lib.rl_forward(self.e, arr, len(tokens)), shape=(self.V,)).copy()

    def draft(self, token):
        return np.ctypeslib.as_array(self.lib.rl_draft_logits(self.e, token), shape=(self.V,)).copy()

    def more(self, guess):
        return np.ctypeslib.as_array(self.lib.rl_draft_more(self.e, guess), shape=(self.V,)).copy()

    def forward_draft(self, tokens):
        n = len(tokens)
        ptr = self.lib.rl_forward_draft(self.e, (ctypes.c_int32 * n)(*tokens), n)
        return np.ctypeslib.as_array(ptr, shape=(n, self.stride))[:, :self.V].copy()

    def keep(self, accepted):
        self.lib.rl_keep_drafts(self.e, accepted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rlm")
    ap.add_argument("tokens")
    ap.add_argument("--prompt", type=int, default=24)
    args = ap.parse_args()
    lib = _library()
    toks = np.fromfile(args.tokens, dtype=np.int32).tolist()
    n0, n = args.prompt, len(toks)
    ok = True

    def check(what, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'}  {what}")

    # The clean run: tokens one at a time. ref[i] are the logits after toks[i];
    # ref_draft[i] the head's logits for the token after toks[i], asked before toks[i] is fed.
    e = Eng(lib, args.rlm)
    if not lib.rl_has_mtp(e.e):
        sys.exit(f"{args.rlm} has no MTP head")
    K = lib.rl_max_drafts(e.e)
    ref, ref_draft = {}, {}
    ref[n0 - 1] = e.forward(toks[:n0])
    for i in range(n0, n):
        ref_draft[i] = e.draft(toks[i])
        ref[i] = e.forward([toks[i]])
    e.close()

    # 1. Speculative passes with a prescribed number of right drafts.
    for k in range(1, K + 1):
        e = Eng(lib, args.rlm)
        same = np.array_equal(e.forward(toks[:n0]), ref[n0 - 1])
        i, right, passes, partial = n0, 0, 0, 0
        while i + k < n:
            same = same and np.array_equal(e.draft(toks[i]), ref_draft[i])
            drafts = [toks[i + 1 + j] if j < right else (toks[i + 1 + j] + 1 + j) % e.V for j in range(k)]
            for d in drafts[:-1]:
                e.more(d)
            rows = e.forward_draft([toks[i]] + drafts)
            same = same and all(np.array_equal(rows[r], ref[i + r]) for r in range(right + 1))
            e.keep(right)
            passes, partial = passes + 1, partial + (0 < right < k)
            i += right + 1
            right = (right + 1) % (k + 1)       # 0, 1, ..., k right drafts in turn
        while i < n:                             # the rest plainly
            same = same and np.array_equal(e.forward([toks[i]]), ref[i])
            i += 1
        e.close()
        check(f"{k} draft{'s' if k > 1 else ''} per pass: {passes} passes ({partial} partly accepted), logits and drafts identical", same)

    # 2. An abandoned draft chain.
    e = Eng(lib, args.rlm)
    e.forward(toks[:n0])
    e.draft((toks[n0] + 5) % e.V)
    e.more(17)
    same = np.array_equal(e.forward([toks[n0]]), ref[n0])
    same = same and np.array_equal(e.draft(toks[n0 + 1]), ref_draft[n0 + 1])
    rows = e.forward_draft([toks[n0 + 1], toks[n0 + 2], 3, 4])
    e.keep(1)
    e.draft(9)                                   # abandoned again, right after a partial accept
    same = same and np.array_equal(e.forward([toks[n0 + 3]]), ref[n0 + 3])
    check("abandoned draft chains leave the model where a clean run is", same and np.array_equal(rows[1], ref[n0 + 2]))

    # 3. State save / load around speculative passes.
    size = lib.rl_state_bytes(e.e)
    state = np.empty(size, dtype=np.uint8)
    at = lib.rl_position(e.e)                    # n0 + 4 tokens are in
    lib.rl_save_state(e.e, state.ctypes.data)
    e.draft(toks[at])
    rows = e.forward_draft([toks[at], toks[at + 1], toks[at + 2]])
    e.keep(2)
    e.forward([toks[at + 3]])
    lib.rl_load_state(e.e, state.ctypes.data, at)
    same = np.array_equal(e.draft(toks[at]), ref_draft[at])
    same = same and all(np.array_equal(e.forward([toks[j]]), ref[j]) for j in range(at, at + 5))
    e.close()
    check(f"state saved at {at}, decoded past it, loaded back: identical ({size / 1e6:.0f} MB of state)", same)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
