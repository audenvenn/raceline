#!/usr/bin/env python3
"""How much accuracy does each candidate weight format cost?

The engine's fastest path needs weights the GPU accelerators can take as one
plain 4-bit operand: no per-group scale inside the matmul. That rules out
MLX's "affine, groups of 64" format and leaves formats of the shape

    W[n][k]  ~  s[n] * c[k] * q[n][k]        q a 4-bit integer

(a scale per output row, applied after the matmul, and a scale per input
column, folded into the activations before it). This script measures what such
formats cost against the unquantized model, next to MLX's 4-bit format.

Reference: the original bf16 checkpoint evaluated with float32 activations.
Metric:    KL(reference || candidate) over next-token distributions, plus
           perplexity and top-1 agreement, on text the calibration never saw.

  .venv/bin/python tools/quant_study.py models/Qwen3.5-4B --mlx4 models/Qwen3.5-4B-MLX-4bit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import sysconfig
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantlib  # noqa: E402
from convert import SafeTensors  # noqa: E402
from mlx_lm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402

T = 512          # tokens per sequence
EVAL_POS = 96    # scored positions at the end of each sequence
DATA = "build/quant"


# ------------------------------------------------------------------- data

def build_splits(tokenizer):
    """Disjoint calibration and evaluation sequences: prose, code, Chinese."""
    def chunks(text, n, skip=0):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [ids[i:i + T] for i in range(skip * T, len(ids) - T, T)][:n]

    wiki_eval = open(f"{DATA}/wt2_test.txt", encoding="utf-8").read()[:400_000]
    wiki_cal = open(f"{DATA}/wt2_valid.txt", encoding="utf-8").read()[:400_000]
    zh = open(f"{DATA}/zh_readme.md", encoding="utf-8").read()
    py = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    code_cal = "".join(open(p, encoding="utf-8", errors="ignore").read() for p in py[:6])
    code_eval = "".join(open(p, encoding="utf-8", errors="ignore").read() for p in py[-8:])
    eval_seqs = chunks(wiki_eval, 6) + chunks(code_eval, 4) + chunks(zh[len(zh) // 2:], 2)
    cal_seqs = chunks(wiki_cal, 8) + chunks(code_cal, 6) + chunks(zh[:len(zh) // 2], 2)
    return cal_seqs, eval_seqs


# ------------------------------------------------------------- evaluation

def forward(model, ids):
    """Logits [T, V] with float32 activations whatever dtype the weights have."""
    x = mx.array(ids)[None]
    emb = model.language_model.model.embed_tokens(x).astype(mx.float32)
    return model(x, input_embeddings=emb)[0]


def log_softmax(z):
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


class Reference:
    """Reference log-probabilities for the scored positions, cached on disk."""

    def __init__(self, model, seqs, path):
        self.seqs = seqs
        vocab = model.language_model.args.vocab_size
        shape = (len(seqs), EVAL_POS, vocab)
        if os.path.exists(path) and os.path.getsize(path) == int(np.prod(shape)) * 4:
            self.logp = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
            return
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
        for i, ids in enumerate(seqs):
            lp = log_softmax(forward(model, ids)[T - EVAL_POS - 1:T - 1].astype(mx.float32))
            mm[i] = np.array(lp)
        mm.flush()
        self.logp = np.memmap(path, dtype=np.float32, mode="r", shape=shape)

    def score(self, model):
        kl = ce = agree = 0.0
        n = 0
        for i, ids in enumerate(self.seqs):
            lp = log_softmax(forward(model, ids)[T - EVAL_POS - 1:T - 1].astype(mx.float32))
            ref = mx.array(np.asarray(self.logp[i]))
            tgt = mx.array(ids[T - EVAL_POS:T])
            kl += mx.sum(mx.exp(ref) * (ref - lp)).item()
            ce += -mx.sum(mx.take_along_axis(lp, tgt[:, None], axis=-1)).item()
            agree += mx.sum(mx.argmax(lp, axis=-1) == mx.argmax(ref, axis=-1)).item()
            n += EVAL_POS
        return dict(kl=kl / n, ppl=float(np.exp(ce / n)), top1=agree / n)


# ------------------------------------------------------------ calibration

def collect_input_moments(model, seqs, names):
    """E[x_k^2] for the input of every quantized matrix, keyed by parameter name."""
    sums, counts = {}, {}
    by_id = {id(m): n for n, m in names.items()}
    lin_call, emb_as_linear = nn.Linear.__call__, nn.Embedding.as_linear

    def record(mod, x):
        name = by_id.get(id(mod))
        if name is not None:
            x32 = x.astype(mx.float32).reshape(-1, x.shape[-1])
            sums[name] = sums.get(name, 0) + (x32 * x32).sum(axis=0)
            counts[name] = counts.get(name, 0) + x32.shape[0]

    def lin(self, x):
        record(self, x)
        return lin_call(self, x)

    def emb(self, x):
        record(self, x)
        return emb_as_linear(self, x)

    nn.Linear.__call__, nn.Embedding.as_linear = lin, emb
    try:
        for ids in seqs:
            mx.eval(forward(model, ids), *sums.values())
    finally:
        nn.Linear.__call__, nn.Embedding.as_linear = lin_call, emb_as_linear
    return {n: s / counts[n] for n, s in sums.items()}


# -------------------------------------------------------------- quantizers
# Every quantizer maps a float32 matrix [N, K] to its dequantized float32
# approximation and reports the storage it would need in bits per weight.

def affine_group(W, h, bits=4, group=64):
    """MLX's format: per (row, group) minimum and step. Round to nearest."""
    N, K = W.shape
    g = W.reshape(N, K // group, group)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    step = mx.maximum((hi - lo) / (2 ** bits - 1), 1e-12)
    q = mx.clip(mx.round((g - lo) / step), 0, 2 ** bits - 1)
    return (q * step + lo).reshape(N, K), bits + 32 / group


def _rows_best_clip(Wc, h, c, clips, midrise, row_groups=1):
    """Per row (or per row and K range), pick the clipping ratio with the least
    input-weighted squared error. Wc = W / c; error is measured on W itself."""
    N, K = Wc.shape
    R = row_groups
    Wr = Wc.reshape(N, R, K // R)
    amax = mx.abs(Wr).max(axis=-1, keepdims=True)
    amax = mx.maximum(amax, 1e-12)
    wgt = (h * c * c).reshape(1, R, K // R)  # error in W scales with c
    best_err, best = None, None
    for clip in clips:
        if midrise:  # 16 levels at +-0.5 .. +-7.5 steps, no exact zero
            s = clip * amax / 7.5
            q = mx.clip(mx.floor(Wr / s), -8, 7) + 0.5
        else:        # 15 levels at 0, +-1 .. +-7 steps
            s = clip * amax / 7.0
            q = mx.clip(mx.round(Wr / s), -7, 7)
        Wq = q * s
        err = (mx.square(Wq - Wr) * wgt).sum(axis=-1, keepdims=True)
        if best is None:
            best_err, best = err, Wq
        else:
            take = err < best_err
            best = mx.where(take, Wq, best)
            best_err = mx.where(take, err, best_err)
    return best.reshape(N, K), best_err.sum()


def rank1(W, h, alphas=(0.0,), betas=(0.0,), clips=(1.0,), midrise=False, row_groups=1, chunk=16384):
    """s[n] * c[k] * q with c[k] = colrms(W)^beta / act_rms^alpha, searched per
    matrix to minimize the input-weighted error. row_groups > 1 gives each row
    one scale per K range instead of one overall."""
    N, K = W.shape
    h = h if h is not None else mx.ones((K,))
    h = h / h.mean()
    col = mx.sqrt(mx.square(W).mean(axis=0)) + 1e-12
    act = mx.sqrt(h) + 1e-12
    best_err, best = None, None
    for a in alphas:
        for b in betas:
            c = mx.power(col, b) / mx.power(act, a)
            c = c / mx.exp(mx.log(c).mean())  # geometric mean 1: keeps row scales comparable
            parts, err = [], 0.0
            for i in range(0, N, chunk):  # bounded memory on the vocabulary matrix
                Wq, e = _rows_best_clip(W[i:i + chunk] / c, h, c, clips, midrise, row_groups)
                parts.append(Wq * c)
                err = err + e
            mx.eval(parts, err)
            err = err.item()
            if best is None or err < best_err:
                best_err, best = err, parts
    out = mx.concatenate(best, axis=0) if len(best) > 1 else best[0]
    # 4 bits + one fp16 scale per (row, range) + one fp16 column scale per input
    return out, 4 + 16 * row_groups / K + 16 / N


CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
A_GRID, B_GRID = (0.0, 0.25, 0.5, 0.75), (0.0, 0.5, 1.0)

CANDIDATES = {
    "affine4 g64 (MLX format, own RTN)": lambda W, h: affine_group(W, h, 4, 64),
    "affine8 g64": lambda W, h: affine_group(W, h, 8, 64),
    "row scale, 15 levels": lambda W, h: rank1(W, None),
    "row scale, 15 levels, clip search": lambda W, h: rank1(W, None, clips=CLIPS),
    "row scale, 16 levels, clip search": lambda W, h: rank1(W, None, clips=CLIPS, midrise=True),
    "row+col scale, data-free": lambda W, h: rank1(W, None, betas=B_GRID, clips=CLIPS, midrise=True),
    "row+col scale, calibrated": lambda W, h: rank1(W, h, A_GRID, B_GRID, CLIPS, midrise=True),
    "row+col scale, calibrated, 4 K-ranges": lambda W, h: rank1(W, h, A_GRID, B_GRID, CLIPS, True, 4),
    "row+col scale, calibrated, 8 K-ranges": lambda W, h: rank1(W, h, A_GRID, B_GRID, CLIPS, True, 8),
}


# ------------------------------------------------- library-backed formats
# The formats the engine would actually ship, quantized by tools/quantlib.py
# (numpy), so the study scores the same code the converter uses.

def lib_rtn(book, alphas=(0.0,), betas=(0.0,)):
    def fn(W, h):
        Wn = np.array(W)
        hn = None if h is None else np.array(h)
        codes, sc, c = quantlib.quantize_rtn(Wn, hn, book, alphas, betas)
        return mx.array(quantlib.dequant(codes, sc, c, book)), 4 + 16 / Wn.shape[1] + 16 / Wn.shape[0]
    return fn


def run_gptq(model, cal_seqs, original, book, alphas, betas, workers=6):
    """Layer-by-layer GPTQ. Each layer sees inputs produced by the already
    quantized layers before it, as it will at run time."""
    from concurrent.futures import ThreadPoolExecutor

    lm = model.language_model
    tm = lm.model
    x = mx.array(cal_seqs)
    h = tm.embed_tokens(x).astype(mx.float32)
    fa_mask, ssm_mask = create_attention_mask(h, None), create_ssm_mask(h, None)
    lin_call = nn.Linear.__call__
    bits = n = 0.0

    def quantize_all(mods, hess):
        def one(item):
            name, mod = item
            W = np.array(original(name))
            codes, sc, c = quantlib.quantize_gptq(W, hess[id(mod)], book, alphas, betas)
            return mod, quantlib.dequant(codes, sc, c, book).astype(np.float16), W.shape
        with ThreadPoolExecutor(workers) as ex:
            for mod, Wq, shape in ex.map(one, mods.items()):
                mod.weight = mx.array(Wq)
                nonlocal bits, n
                bits += (4 + 16 / shape[1] + 16 / shape[0]) * shape[0] * shape[1]
                n += shape[0] * shape[1]

    for i, layer in enumerate(tm.layers):
        prefix = f"language_model.model.layers.{i}."
        mods = {prefix + nm + ".weight": m for nm, m in layer.named_modules() if isinstance(m, nn.Linear)}
        ids = {id(m) for m in mods.values()}
        hess = {}

        def hooked(self, a):
            if id(self) in ids:
                a2 = a.astype(mx.float32).reshape(-1, a.shape[-1])
                hess[id(self)] = hess.get(id(self), 0) + a2.T @ a2
            return lin_call(self, a)

        mask = ssm_mask if layer.is_linear else fa_mask
        nn.Linear.__call__ = hooked
        try:
            mx.eval(layer(h, mask=mask, cache=None), *hess.values())
        finally:
            nn.Linear.__call__ = lin_call
        n_tok = h.shape[0] * h.shape[1]
        quantize_all(mods, {k: np.array(v) / n_tok for k, v in hess.items()})
        h = layer(h, mask=mask, cache=None)
        mx.eval(h)
        print(f"\r  gptq layer {i + 1}/{len(tm.layers)}", end="", flush=True)
    print()

    # Output head (the embedding matrix when tied): its input is the final norm.
    out = tm.norm(h).astype(mx.float32).reshape(-1, h.shape[-1])
    H = np.array(out.T @ out) / out.shape[0]
    head = tm.embed_tokens if lm.args.tie_word_embeddings else lm.lm_head
    name = "language_model.model.embed_tokens.weight" if lm.args.tie_word_embeddings else "language_model.lm_head.weight"
    quantize_all({name: head}, {id(head): H})
    if not lm.args.tie_word_embeddings:  # the lookup table is only ever read row by row
        emb = "language_model.model.embed_tokens.weight"
        W = np.array(original(emb))
        codes, sc, c = quantlib.quantize_rtn(W, None, book)
        tm.embed_tokens.weight = mx.array(quantlib.dequant(codes, sc, c, book).astype(np.float16))
    return bits / n


LIB_CANDIDATES = {
    "lib int16 row": lib_rtn("int16"),
    "lib fp4 row": lib_rtn("fp4"),
    "lib int16 row+col calibrated": lib_rtn("int16", A_GRID, B_GRID),
    "lib fp4 row+col calibrated": lib_rtn("fp4", A_GRID, B_GRID),
}
GPTQ_CANDIDATES = {
    "gptq fp4 row": ("fp4", (0.0,), (0.0,)),
    "gptq int16 row": ("int16", (0.0,), (0.0,)),
    "gptq fp4 row+col": ("fp4", A_GRID, B_GRID),
    "gptq int16 row+col": ("int16", A_GRID, B_GRID),
}


# -------------------------------------------------------------------- main

def quantized_modules(model):
    """Parameter name -> module for every matrix the engine stores quantized."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Embedding)) and name.startswith("language_model"):
            out[name + ".weight"] = mod
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="original (bf16) model directory")
    ap.add_argument("--mlx4", help="MLX 4-bit conversion of the same model, scored as shipped")
    ap.add_argument("--only", nargs="*", help="substrings selecting candidates")
    ap.add_argument("--out", default=f"{DATA}/quant_study.json")
    args = ap.parse_args()

    model, tok = load(args.ref)
    cal_seqs, eval_seqs = build_splits(tok._tokenizer)
    print(f"reference {args.ref}: {len(cal_seqs)} calibration and {len(eval_seqs)} evaluation sequences of {T} tokens, "
          f"{len(eval_seqs) * EVAL_POS} scored positions")
    ref = Reference(model, eval_seqs, f"{DATA}/ref_logp_{os.path.basename(os.path.normpath(args.ref))}.f32")
    results = {"reference (bf16 weights)": dict(ref.score(model), bpw=16.0)}

    mods = quantized_modules(model)
    t0 = time.time()
    moments = collect_input_moments(model, cal_seqs, mods)
    print(f"calibration moments for {len(moments)}/{len(mods)} matrices in {time.time() - t0:.0f}s")

    st = SafeTensors(sorted(glob.glob(os.path.join(args.ref, "*.safetensors"))))

    def original(name):  # MLX path -> name in the original checkpoint
        key = name.replace("language_model.model.", "model.language_model.", 1)
        if key not in st:
            key = name.replace("language_model.", "", 1)
        return mx.array(st.get(key).astype(np.float32))

    def report(label, r, t0):
        results[label] = r
        print(f"{label:42s} KL {r['kl']:.5f}  ppl {r['ppl']:.3f}  top1 {100 * r['top1']:.2f}%  "
              f"{r['bpw']:.3f} bpw  ({time.time() - t0:.0f}s)", flush=True)

    for label, (book, alphas, betas) in GPTQ_CANDIDATES.items():
        if not args.only or not any(s in label for s in args.only):
            continue
        t0 = time.time()
        bpw = run_gptq(model, cal_seqs, original, book, alphas, betas)
        report(label, dict(ref.score(model), bpw=bpw), t0)

    for label, fn in {**CANDIDATES, **LIB_CANDIDATES}.items():
        if args.only and not any(s in label for s in args.only):
            continue
        t0 = time.time()
        bits = n = 0.0
        for name, mod in mods.items():
            W = original(name)
            Wq, bpw = fn(W, moments.get(name))
            mod.weight = Wq.astype(mx.float16)
            mx.eval(mod.weight)
            bits += bpw * W.size
            n += W.size
        report(label, dict(ref.score(model), bpw=bits / n), t0)

    if args.mlx4:
        del model
        mx.clear_cache()
        m4, _ = load(args.mlx4)
        r = dict(ref.score(m4), bpw=4.5)
        results["MLX 4-bit as shipped"] = r
        print(f"{'MLX 4-bit as shipped':42s} KL {r['kl']:.5f}  ppl {r['ppl']:.3f}  top1 {100 * r['top1']:.2f}%  4.500 bpw")

    r = results["reference (bf16 weights)"]
    print(f"{'reference (bf16 weights)':42s} KL {r['kl']:.5f}  ppl {r['ppl']:.3f}  top1 {100 * r['top1']:.2f}%")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
