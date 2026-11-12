"""Quantizers for weights of the form  W[n][k] ~ s[n] * c[k] * L[q[n][k]].

`L` is a fixed 16-entry codebook the GPU accelerators understand natively, `s`
a scale per output row and `c` a scale per input column. Nothing else may
depend on (n, k): that is what lets the engine hand `q` to the accelerator as a
plain 4-bit operand and apply `s` and `c` outside the matmul.

Codebooks (value of each 4-bit code, in code order)
  fp4    FP4 E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives
  int15  two's-complement nibble, clipped to -7 .. 7
  int16  two's-complement nibble plus one half: +-0.5 .. +-7.5

The scale searches are element-wise and run on the GPU through MLX; so do the
GPTQ blocks. Matrices are processed in row chunks and results are returned as
4-bit codes, so the vocabulary-sized ones stay within memory.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

LEVELS = {
    "fp4": np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32),
    "int15": np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1], dtype=np.float32),
    "int16": np.array([.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -.5],
                      dtype=np.float32),
}
LMAX = {"fp4": 6.0, "int15": 7.0, "int16": 7.5}


def snap(x: np.ndarray, book: str) -> np.ndarray:
    """Nearest codebook value to each element of x (x already divided by its scale)."""
    if book == "int16":
        return np.clip(np.floor(x), -8, 7) + 0.5
    if book == "int15":
        return np.clip(np.rint(x), -7, 7)
    if book == "fp4":
        m = np.abs(x)
        fine = np.rint(m * 2) * 0.5                    # 0 .. 1.5 in steps of 0.5 (1.75 rounds to 2)
        mid = np.rint(m)                               # 2, 3, 4
        coarse = np.minimum(np.rint(m * 0.5) * 2, 6)   # 4, 6
        q = np.where(m < 1.75, fine, np.where(m < 3.5, mid, coarse))
        return np.copysign(q, x)
    raise ValueError(book)


def encode(v: np.ndarray, book: str) -> np.ndarray:
    """Codebook values -> 4-bit codes (uint8)."""
    if book == "int16":
        return (np.rint(v - 0.5).astype(np.int8) & 0x0F).astype(np.uint8)
    if book == "int15":
        return (np.rint(v).astype(np.int8) & 0x0F).astype(np.uint8)
    if book == "fp4":  # sign(1) exponent(2) mantissa(1)
        m = np.abs(v)
        code = np.select([m < 0.25, m < 0.75, m < 1.25, m < 1.75, m < 2.5, m < 3.5, m < 5.0],
                         [0, 1, 2, 3, 4, 5, 6], default=7).astype(np.uint8)
        return code | (np.signbit(v).astype(np.uint8) << 3)
    raise ValueError(book)


def dequant(codes: np.ndarray, s: np.ndarray, c: np.ndarray, book: str) -> np.ndarray:
    """codes [..., K], s [...] and c [K] -> float32 weights."""
    return LEVELS[book][codes] * s[..., None] * c


def _snap_mx(x, book: str):
    if book == "int16":
        return mx.clip(mx.floor(x), -8, 7) + 0.5
    if book == "int15":
        return mx.clip(mx.round(x), -7, 7)
    m = mx.abs(x)
    q = mx.where(m < 1.75, mx.round(m * 2) * 0.5, mx.where(m < 3.5, mx.round(m), mx.minimum(mx.round(m * 0.5) * 2, 6)))
    return mx.where(x < 0, -q, q)


def column_scales(col_rms: np.ndarray, act_rms: np.ndarray | None, alpha: float, beta: float) -> np.ndarray:
    """c[k] = colrms^beta / act_rms^alpha, normalized to geometric mean 1.

    beta evens out columns whose weights are large; alpha gives inputs that are
    large at run time a finer effective step (their error is amplified most).
    """
    c = np.power(col_rms + 1e-12, beta)
    if act_rms is not None and alpha:
        c = c / np.power(act_rms + 1e-12, alpha)
    return (c / np.exp(np.log(c).mean())).astype(np.float32)


RATIOS = np.geomspace(0.22, 1.0, 16).astype(np.float32)
CHUNK = 16384


def row_scales(W: np.ndarray, c: np.ndarray, wgt: np.ndarray, book: str, codes: bool = False):
    """Per row of W / c, the scale with the least weighted squared error.

    `wgt[k]` weights the squared error of column k (input second moment times
    c^2). Returns (s[N], total weighted error) and, if `codes`, the 4-bit codes
    at that scale as a third element.
    """
    N = W.shape[0]
    s = np.empty(N, dtype=np.float32)
    out = np.empty(W.shape, dtype=np.uint8) if codes else None
    wg, cm = mx.array(wgt), mx.array(c)
    total = 0.0
    for i in range(0, N, CHUNK):
        w = mx.array(W[i:i + CHUNK]) / cm
        top = mx.maximum(mx.abs(w).max(axis=1, keepdims=True), 1e-12) / LMAX[book]
        best_e = best_s = None
        for r in RATIOS:
            sc = top * float(r)
            e = (mx.square(_snap_mx(w / sc, book) * sc - w) * wg).sum(axis=1, keepdims=True)
            if best_e is None:
                best_e, best_s = e, sc
            else:
                take = e < best_e
                best_s = mx.where(take, sc, best_s)
                best_e = mx.where(take, e, best_e)
        s[i:i + CHUNK] = np.array(best_s)[:, 0]
        total += best_e.sum().item()
        if codes:
            out[i:i + CHUNK] = encode(np.array(_snap_mx(w / best_s, book)), book)
    return (s, total, out) if codes else (s, total)


def _column_rms(W: np.ndarray) -> np.ndarray:
    acc = np.zeros(W.shape[1], dtype=np.float64)
    for i in range(0, W.shape[0], CHUNK):
        acc += np.square(W[i:i + CHUNK], dtype=np.float32).sum(axis=0, dtype=np.float64)
    return np.sqrt(acc / W.shape[0]).astype(np.float32)


def search_scales(W: np.ndarray, h: np.ndarray | None, book: str, alphas=(0.0,), betas=(0.0,)):
    """-> (normalized input second moments h[K], column scales c[K]).

    The column-scale exponents are searched on a row subsample; rows are
    independent once c is fixed.
    """
    N, K = W.shape
    h = np.ones(K, dtype=np.float32) if h is None else (h / h.mean()).astype(np.float32)
    col_rms, act_rms = _column_rms(W), np.sqrt(h)
    c = column_scales(col_rms, act_rms, alphas[0], betas[0])
    if len(alphas) * len(betas) > 1:
        sub = W if N <= 1024 else np.ascontiguousarray(W[:: N // 1024][:1024])
        best = None
        for a in alphas:
            for b in betas:
                cand = column_scales(col_rms, act_rms, a, b)
                _, err = row_scales(sub, cand, h * cand * cand, book)
                if best is None or err < best:
                    best, c = err, cand
    return h, c


def quantize_rtn(W: np.ndarray, h: np.ndarray | None, book: str, alphas=(0.0,), betas=(0.0,)):
    """Round to nearest with searched row and column scales -> (codes[N][K] uint8, s[N], c[K])."""
    h, c = search_scales(W, h, book, alphas, betas)
    s, _, codes = row_scales(W, c, h * c * c, book, codes=True)
    return codes, s, c


def gptq_factor(Hc: np.ndarray, damp: float) -> np.ndarray:
    """Upper-triangular U with inverse(Hc + damping) = U^T U, in float32.

    Equivalent to cholesky(inv(H), upper) as in the GPTQ reference, but from one
    Cholesky factorization and one triangular inverse instead of a full
    inverse: reverse the order of H, factor it, and reverse back to get an
    upper-triangular R with H = R R^T; then U = inverse(R).
    """
    K = Hc.shape[0]
    Hd = Hc.astype(np.float32).copy()
    Hd[np.diag_indices(K)] += damp * float(np.diag(Hc).mean())
    Lp = mx.linalg.cholesky(mx.array(Hd[::-1, ::-1].copy()), stream=mx.cpu)
    R = Lp[::-1, ::-1]
    U = mx.linalg.tri_inv(R, upper=True, stream=mx.cpu)
    return np.array(U)


_block_steps = {}


def _block_step(book: str):
    """One GPTQ block as a single compiled GPU graph.

    W1 [N, B] are the block's columns (already corrected by earlier blocks), U
    the matching [B, B] upper-triangular piece of the factor, sr [N, 1] the row
    scales. Column j is rounded, and its error is pushed onto the columns to
    its right (row j of U is zero to the left of the diagonal). Returns the
    chosen codebook values and the scaled errors, both [N, B].
    """
    if book not in _block_steps:
        def step(W1, U, sr):
            levs, errs = [], []
            for j in range(W1.shape[1]):
                w = W1[:, j:j + 1]
                lev = _snap_mx(w / sr, book)
                e = (w - lev * sr) / U[j, j]
                W1 = W1 - e * U[j:j + 1, :]
                levs.append(lev)
                errs.append(e)
            return mx.concatenate(levs, axis=1), mx.concatenate(errs, axis=1)
        _block_steps[book] = mx.compile(step)
    return _block_steps[book]


def quantize_gptq(W: np.ndarray, H: np.ndarray, book: str, alphas=(0.0,), betas=(0.0,),
                  damp: float = 0.01, block: int = 128, chunk: int = 32768, act_order: bool = True):
    """GPTQ on the fixed grid s[n] * c[k] * L: columns are rounded one at a
    time and each rounding error is pushed onto the not-yet-rounded columns,
    using the inverse Hessian of the layer's input (H = E[x x^T]).

    Scales come from the round-to-nearest search and stay fixed; only the
    choice of level per weight changes. With `act_order` the columns are
    rounded in order of decreasing input energy, so the inputs that matter
    most are rounded while the most columns are left to absorb their error.
    In a format with per-group scales that order costs a lookup table at run
    time; here the scales do not depend on the order, so it is free.
    Returns (codes, s, c).
    """
    h, c = search_scales(W, np.diag(H).astype(np.float32).copy(), book, alphas, betas)
    s, _ = row_scales(W, c, h * c * c, book)
    return _gptq_codes(W, H, c, s, book, damp, block, chunk, act_order), s, c


def quantize_gptq_experts(W: np.ndarray, sums: np.ndarray, counts: np.ndarray, book: str, alphas=(0.0,),
                          betas=(0.0,), damp: float = 0.01, block: int = 128, act_order: bool = True,
                          prior: float = 64.0):
    """GPTQ for a stack of experts that share their column scales but not their inputs.

    W holds len(counts) blocks of rows, one per expert; sums[e] is the sum of
    x x^T over the inputs expert e saw and counts[e] how many there were. The
    column scales are searched with the inputs of all experts pooled (the
    engine applies one c to a stack); each expert's row scales and its GPTQ
    error propagation use its own second moment. An expert that saw few
    inputs has a poorly determined moment: it is pulled towards the pooled
    one as if `prior` pooled inputs had been added to its own.
    Returns (codes, s, c).
    """
    E, K = len(counts), W.shape[1]
    R = W.shape[0] // E
    assert R * E == W.shape[0] and sums.shape == (E, K, K)
    pooled = sums.sum(axis=0, dtype=np.float64) / max(1.0, float(counts.sum()))
    _, c = search_scales(W, np.diag(pooled).astype(np.float32).copy(), book, alphas, betas)
    codes, s = np.empty(W.shape, dtype=np.uint8), np.empty(W.shape[0], dtype=np.float32)
    for e in range(E):
        H = ((sums[e] + prior * pooled) / (counts[e] + prior)).astype(np.float32)
        h = np.diag(H).copy()
        h = h / h.mean() if h.mean() > 0 else np.ones(K, dtype=np.float32)
        We = W[e * R:(e + 1) * R]
        s[e * R:(e + 1) * R], _ = row_scales(We, c, h * c * c, book)
        codes[e * R:(e + 1) * R] = _gptq_codes(We, H, c, s[e * R:(e + 1) * R], book, damp, block, R, act_order)
    return codes, s, c


def _gptq_codes(W, H, c, s, book, damp, block, chunk, act_order):
    """The GPTQ column loop for fixed scales -> codes[N][K]."""
    N, K = W.shape
    # Work on W / c, whose input is c * x: Hc = diag(c) H diag(c).
    Hc = (H.astype(np.float32) * c[None, :]) * c[:, None]
    dead = np.diag(Hc) <= 0                       # inputs that never fire: their weights do not matter
    Hc[dead, dead] = 1.0
    perm = np.argsort(-np.diag(Hc), kind="stable") if act_order else np.arange(K)
    if act_order:
        Hc = Hc[np.ix_(perm, perm)]
    U = mx.array(gptq_factor(Hc, damp))
    del Hc
    step = _block_step(book)
    cm, live = mx.array(c[perm]), mx.array((~dead[perm]).astype(np.float32))

    codes = np.empty((N, K), dtype=np.uint8)
    for r0 in range(0, N, chunk):
        rows = W[r0:r0 + chunk]
        rest = mx.array(rows[:, perm] if act_order else rows) / cm * live     # columns not yet rounded
        sr = mx.array(s[r0:r0 + chunk, None])
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            lev, err = step(rest[:, :i2 - i1], U[i1:i2, i1:i2], sr)
            if i2 < K:
                rest = rest[:, i2 - i1:] - err @ U[i1:i2, i2:]
                mx.eval(lev, rest)
            else:
                mx.eval(lev)
            codes[r0:r0 + chunk, perm[i1:i2]] = encode(np.array(lev), book)
    return codes
