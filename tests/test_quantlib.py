#!/usr/bin/env python3
"""Checks of the quantizer's numerics (needs MLX, so not part of CI; a few seconds).

  .venv/bin/python tests/test_quantlib.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import quantlib  # noqa: E402


def test_experts_get_their_own_gptq():
    """A stack of experts, each seeing inputs with statistics of its own: GPTQ
    with each expert's own second moment must leave far less output error than
    rounding to nearest, and than GPTQ with the moments of all experts pooled."""
    rng = np.random.default_rng(0)
    E, R, K, n = 8, 256, 128, 400
    W = rng.standard_normal((E * R, K)).astype(np.float32)
    X = []
    for _ in range(E):
        A = rng.standard_normal((K, K)) * 0.3 + np.diag(rng.gamma(0.5, 2.0, K))
        X.append((rng.standard_normal((n, K)) @ A).astype(np.float32))
    sums = np.stack([x.T @ x for x in X])
    counts = np.full(E, float(n))
    pooled = sums.sum(0) / counts.sum()

    def err(codes, s, c):
        Q = quantlib.dequant(codes, s, c, "fp4")
        return sum(float(np.sum(((W[e * R:(e + 1) * R] - Q[e * R:(e + 1) * R]) @ X[e].T) ** 2)) for e in range(E))

    rtn = err(*quantlib.quantize_rtn(W, np.diag(pooled).copy(), "fp4"))
    pool = err(*quantlib.quantize_gptq(W, pooled, "fp4"))
    per = err(*quantlib.quantize_gptq_experts(W, sums, counts, "fp4"))
    assert per < 0.6 * rtn and per < 0.7 * pool, (rtn, pool, per)
    # An expert that saw nothing is quantized with the pooled moments, not with zeros.
    counts0, sums0 = counts.copy(), sums.copy()
    counts0[3], sums0[3] = 0, 0
    codes, s, c = quantlib.quantize_gptq_experts(W, sums0, counts0, "fp4")
    assert np.all(np.isfinite(s)) and np.all(s[3 * R:4 * R] > 0)


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)
            n += 1
    print(f"{n} passed")
