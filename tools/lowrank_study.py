#!/usr/bin/env python3
"""Would a low-rank correction of the 4-bit weights close the 27B's quality gap? (#15)

For a few layers of a converted model, the quantization error E = W - W4 of a
matrix (W the original weights as the file's model uses them: norm gains
folded in and rotated; W4 decoded from the .rlm file) and the real inputs X of
that matrix, from residual streams the engine dumped (RacelineBackend.dump_layers).
Reports the matrix's output error |X E^T| / |X W^T| as is and after subtracting
a rank-r correction C: the plain SVD of E, and the activation-aware one (the
SVD of E H^1/2, H = X^T X, then H^-1/2; LQER / SVDQuant style), fitted on the
first half of the dumped tokens and measured on the second (another kind of
text), so a correction that only fits its own data shows as such.

  .venv/bin/python tools/lowrank_study.py models/Qwen3.8-27B models/qwen3.8-27b.rot.rlm DUMP_DIR --layers 4 20 40 60

The correction would cost r * (N + K) half-precision values per matrix, read
on every pass: at r = 32 about 3% more bytes for the 27B's MLP matrices.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quantize import Source, rotation  # noqa: E402
from rlmfile import Native  # noqa: E402

RANKS = (16, 32, 64, 128)


def rms_rows(x):
    return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)


def top_r(M, r, seed=0):
    """Rank-r truncated SVD of M by a randomized range finder: (U * s) [n, r], Vt [r, k]."""
    rng = np.random.default_rng(seed)
    Y = M @ rng.standard_normal((M.shape[1], r + 16)).astype(np.float32)
    for _ in range(2):                                   # power iterations: the spectrum of E decays slowly
        Y = M @ (M.T @ Y)
        Y, _ = np.linalg.qr(Y)
    Q, _ = np.linalg.qr(Y)
    U, s, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)
    return (Q @ U[:, :r]) * s[:r], Vt[:r]


def study(name, W, W4, X_fit, X_eval):
    E = W - W4
    ref = np.linalg.norm(X_eval @ W.T)
    base = np.linalg.norm(X_eval @ E.T) / ref
    line = [f"{name:18s} |E|/|W| {np.linalg.norm(E) / np.linalg.norm(W):.3f}  output error {base:.4f}"]
    # Activation-aware: H = X^T X / n from the fitting rows, plus delta * I (fewer rows than columns). With
    # X = U s V^T, H^(+-1/2) = V diag(a^(+-1)) V^T + b^(+-1) (I - V V^T), a = sqrt(s^2/n + delta), b = sqrt(delta):
    # applied through V, no K x K matrix is formed.
    n = len(X_fit)
    _, s, Vt_x = np.linalg.svd(X_fit, full_matrices=False)
    Vx = Vt_x.T                                            # [K, n]
    delta = 0.01 * float((s * s).sum()) / n / X_fit.shape[1]
    a, b = np.sqrt(s * s / n + delta).astype(np.float32), np.float32(np.sqrt(delta))

    def apply(M, inverse):                                 # M @ H^(+-1/2)
        MV = M @ Vx
        ka, kb = (1 / a, 1 / b) if inverse else (a, b)
        return (MV * ka) @ Vx.T + kb * (M - MV @ Vx.T)

    ES = apply(E, False)
    for r in RANKS:
        Us, Vt = top_r(E, r)
        plain = np.linalg.norm(X_eval @ (E - Us @ Vt).T) / ref
        Us, Vt = top_r(ES, r)
        aware = np.linalg.norm(X_eval @ (E - Us @ apply(Vt, True)).T) / ref
        line.append(f"r={r:3d}: plain {plain / base:5.1%} aware {aware / base:5.1%}")
    print("  " + "  ".join(line), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("rlm")
    ap.add_argument("dump", help="directory of layer_NN.f32 / mid_NN.f32 from RacelineBackend.dump_layers")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 20, 40, 60])
    args = ap.parse_args()

    src, nat = Source(args.checkpoint), Native(args.rlm)
    rot = nat.hdr["quant"].get("rotation")
    hidden = nat.hdr["config"]["hidden_size"]
    Q = rotation(hidden, rot["seed"]) if rot else np.eye(hidden, dtype=np.float32)
    T = len(np.load(os.path.join(args.dump, "ids.npy")))
    half = T // 2

    def dumped(kind, i):
        return np.fromfile(os.path.join(args.dump, f"{kind}_{i:02d}.f32"), dtype=np.float32).reshape(T, hidden)

    for i in args.layers:
        w = src.layer_weights(i)
        linear = "linear_attn.in_proj_qkv.weight" in w
        print(f"layer {i} ({'DeltaNet' if linear else 'attention'}); output error relative to the output, "
              "then what is left of it after a rank-r correction (fitted on the first half of the tokens, "
              "measured on the second)", flush=True)
        # The attention block's input projection: the normed stream after layer i-1.
        X = rms_rows(dumped("layer", i - 1))
        if linear:
            parts = ["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
                     "linear_attn.in_proj_a.weight"]
            name = f"layers.{i}.lin.in_proj"
        else:
            parts = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]
            name = f"layers.{i}.attn.qkv"
        W = np.concatenate([w[p] for p in parts]) * w["input_layernorm.weight"][None, :] @ Q
        study(name.split(".", 2)[2], W.astype(np.float32), nat.matrix(name), X[:half], X[half:])
        # The MLP: gate/up read the normed stream between the attention and the MLP, down reads SwiGLU of them.
        X = rms_rows(dumped("mid", i))
        W = np.concatenate([w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]]) * \
            w["post_attention_layernorm.weight"][None, :] @ Q
        W4 = nat.matrix(f"layers.{i}.mlp.gate_up")
        study("mlp.gate_up", W.astype(np.float32), W4, X[:half], X[half:])
        g = X @ W4.T
        n = g.shape[1] // 2
        A = (g[:, :n] / (1 + np.exp(-g[:, :n]))) * g[:, n:]
        W = (Q.T @ w["mlp.down_proj.weight"]).astype(np.float32)
        study("mlp.down", W, nat.matrix(f"layers.{i}.mlp.down"), A[:half], A[half:])


if __name__ == "__main__":
    main()
