#!/usr/bin/env python3
"""Would rotating the MLP down projection's input cut its 4-bit error? (#15)

The residual-stream rotation reaches every matrix that reads the stream, but
not the down projection, whose input is the MLP's intermediate activation
(QuaRot rotates it online). This quantizes a converted model's down
projections the way tools/quantize.py does (FP4, a scale per row and per
column, GPTQ) twice: as they are, and with their input dimension rotated by
random signs and normalized Hadamard blocks of 256 or 1024 (W' = W Hd, the input
A' = A Hd, so the product is unchanged), the Hessian fitted on the first half
of the dumped tokens, the output error measured on the second.

  .venv/bin/python tools/hadamard_down_study.py models/Qwen3.8-27B models/qwen3.8-27b.rot.rlm DUMP_DIR --layers 4 20 60
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quantlib  # noqa: E402
from quantize import A_GRID, B_GRID, Source, rotation  # noqa: E402
from rlmfile import Native  # noqa: E402


def rms_rows(x):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)


def block_hadamard(n, block=1024, seed=1):
    """Random signs times normalized Sylvester-Hadamard blocks: an orthogonal [n, n] matrix."""
    Hb = np.ones((1, 1))
    while Hb.shape[0] < block:
        Hb = np.block([[Hb, Hb], [Hb, -Hb]])
    Hb /= np.sqrt(block)
    signs = np.where(np.random.default_rng(seed).random(n) < 0.5, -1.0, 1.0)
    M = np.zeros((n, n))
    for i in range(0, n, block):
        M[i:i + block, i:i + block] = Hb
    return (signs[:, None] * M).astype(np.float32)


def gptq_error(W, A_fit, A_eval):
    H = A_fit.T @ A_fit / len(A_fit)
    codes, s, c = quantlib.quantize_gptq(W, H, "fp4", A_GRID, B_GRID, act_order=False)
    W4 = quantlib.dequant(codes, s, c, "fp4")
    return np.linalg.norm(A_eval @ (W - W4).T) / np.linalg.norm(A_eval @ W.T)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("rlm")
    ap.add_argument("dump")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 20, 60])
    args = ap.parse_args()
    src, nat = Source(args.checkpoint), Native(args.rlm)
    hidden = nat.hdr["config"]["hidden_size"]
    Q = rotation(hidden, nat.hdr["quant"]["rotation"]["seed"])
    T = len(np.load(os.path.join(args.dump, "ids.npy")))
    half = T // 2
    for i in args.layers:
        w = src.layer_weights(i)
        X = rms_rows(np.fromfile(os.path.join(args.dump, f"mid_{i:02d}.f32"), dtype=np.float32).reshape(T, hidden))
        g = X @ nat.matrix(f"layers.{i}.mlp.gate_up").T
        n = g.shape[1] // 2
        A = ((g[:, :n] / (1 + np.exp(-g[:, :n]))) * g[:, n:]).astype(np.float32)
        W = (Q.T @ w["mlp.down_proj.weight"]).astype(np.float32)
        file_err = np.linalg.norm(A[half:] @ (W - nat.matrix(f"layers.{i}.mlp.down")).T) / np.linalg.norm(A[half:] @ W.T)
        plain = gptq_error(W, A[:half], A[half:])
        kurt = lambda a: float(np.mean(a ** 4) / np.mean(a ** 2) ** 2)
        line = (f"layer {i}: down's output error, the published file {file_err:.4f}; GPTQ here {plain:.4f} "
                f"(input kurtosis {kurt(A):.0f})")
        for block in (256, 1024):
            Hd = block_hadamard(W.shape[1], block)
            Wr, Ar = W @ Hd, A @ Hd
            rotated = gptq_error(Wr, Ar[:half], Ar[half:])
            line += f"; rotated in blocks of {block}: {rotated:.4f} ({rotated / plain - 1:+.0%}, kurtosis {kurt(Ar):.1f})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
