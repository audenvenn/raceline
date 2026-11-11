#!/usr/bin/env python3
"""Quantize an original Qwen3.5 / Qwen3.8 checkpoint into redline's native format.

Native format: every weight matrix is  s[n] * c[k] * L[q[n][k]]  with q a 4-bit
code the GPU accelerators decode themselves, a scale per output row and a
scale per input column. See tools/quantlib.py for the codebooks and
tools/quant_study.py for how the options compare.

The model is processed one layer at a time, so a checkpoint far larger than
RAM converts fine (Qwen3.8-27B is 55 GB in bf16). For each layer:

  1. run calibration activations through the layer in float and record the
     input second moments of every matrix (H = E[x x^T]);
  2. quantize the matrices with GPTQ on that H;
  3. push the calibration activations through the *quantized* layer, so the
     next layer is calibrated on the inputs it will really see.

A held-out set is pushed through both the float and the quantized model and
the divergence of their next-token distributions is reported at the end.

  tools/quantize.py models/Qwen3.5-4B models/qwen3.5-4b.native.rlm
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
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantlib  # noqa: E402
from convert import GROUP, BLOCK, SafeTensors, Writer  # noqa: E402,F401

T = 512
EVAL_POS = 96
A_GRID, B_GRID = (0.0, 0.25, 0.5, 0.75), (0.0, 0.5, 1.0)
NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")


# ------------------------------------------------------------------- data

def token_sequences(tok, paths, n, skip=0):
    """n sequences of T tokens from the given text files, round-robin."""
    per_file = []
    for p in paths:
        ids = tok.encode(open(p, encoding="utf-8", errors="ignore").read()[:600_000], add_special_tokens=False).ids
        per_file.append([ids[i:i + T] for i in range(skip * T, len(ids) - T, T)])
    out, i = [], 0
    while len(out) < n and any(per_file):
        f = per_file[i % len(per_file)]
        if f:
            out.append(f.pop(0))
        i += 1
    return out


def default_texts():
    """Calibration and evaluation text when none is given: source code from the
    Python standard library plus, if present, the prose files the study uses."""
    py = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    big = [p for p in py if os.path.getsize(p) > 60_000]
    cal, ev = big[:12], big[-6:]
    for name, dst in (("wt2_valid.txt", cal), ("wt2_test.txt", ev), ("zh_readme.md", cal)):
        p = os.path.join("build", "quant", name)
        if os.path.exists(p):
            dst.insert(0, p)
    return cal, ev


# ----------------------------------------------------------------- layers

class Source:
    """Original checkpoint, read lazily, with MLX's weight conventions applied."""

    def __init__(self, src):
        self.st = SafeTensors(sorted(glob.glob(os.path.join(src, "*.safetensors"))))
        self.prefix = "model.language_model."
        conv = [k for k in self.st.index if k.endswith("conv1d.weight")]
        # Original checkpoints store conv weights as [C, 1, taps] and norm weights minus one.
        self.raw = bool(conv) and self.st.index[conv[0]][1]["shape"][-1] != 1

    def get(self, name) -> np.ndarray:
        key = name if name in self.st else self.prefix + name
        w = self.st.get(key).astype(np.float32)
        if self.raw and name.endswith("conv1d.weight"):
            w = np.moveaxis(w, 2, 1)
        if self.raw and (name.endswith(NORM_KEYS) or name == "norm.weight"):
            w = w + 1.0
        return w

    def layer_weights(self, i):
        pre = f"layers.{i}."
        return {k[len(self.prefix) + len(pre):]: self.get(k[len(self.prefix):])
                for k in self.st.index if k.startswith(self.prefix + pre)}


def build_layer(args, i, weights):
    layer = qwen3_5.DecoderLayer(args, i)
    layer.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    mx.eval(layer.parameters())
    return layer


def run_layer(layer, h, batch=8):
    """h [S, T, H] float32 -> layer output, a few sequences at a time."""
    outs = []
    for i in range(0, h.shape[0], batch):
        x = h[i:i + batch]
        mask = None if layer.is_linear else create_attention_mask(x, None)
        y = layer(x, mask=mask, cache=None)
        mx.eval(y)
        outs.append(y)
    return mx.concatenate(outs, axis=0)


def input_moments(layer, h, batch=8):
    """E[x x^T] for the input of every Linear in the layer, by module id."""
    sums = {}
    lin_call = nn.Linear.__call__

    def hooked(self, a):
        a2 = a.astype(mx.float32).reshape(-1, a.shape[-1])
        sums[id(self)] = sums.get(id(self), 0) + a2.T @ a2
        return lin_call(self, a)

    nn.Linear.__call__ = hooked
    try:
        for i in range(0, h.shape[0], batch):
            x = h[i:i + batch]
            mask = None if layer.is_linear else create_attention_mask(x, None)
            mx.eval(layer(x, mask=mask, cache=None), *sums.values())
    finally:
        nn.Linear.__call__ = lin_call
    n = h.shape[0] * h.shape[1]
    return {k: np.array(v) / n for k, v in sums.items()}


def stacks_of(layer):
    """(file name, [(part name, module)]) for every matrix group that shares an input."""
    if layer.is_linear:
        a = layer.linear_attn
        out = [("lin.in_proj", [("qkv", a.in_proj_qkv), ("z", a.in_proj_z), ("b", a.in_proj_b), ("a", a.in_proj_a)]),
               ("lin.out_proj", [("out", a.out_proj)])]
    else:
        a = layer.self_attn
        out = [("attn.qkv", [("q", a.q_proj), ("k", a.k_proj), ("v", a.v_proj)]), ("attn.o_proj", [("out", a.o_proj)])]
    return out + [("mlp.gate_up", [("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj)]),
                  ("mlp.down", [("out", layer.mlp.down_proj)])]


def affine4(W, group=64):
    """MLX's 4-bit format, round to nearest: a minimum and a step per (row, group of 64)."""
    N, K = W.shape
    g = W.reshape(N, K // group, group)
    lo, hi = g.min(axis=-1, keepdims=True), g.max(axis=-1, keepdims=True)
    step = np.maximum((hi - lo) / 15, 1e-12)
    return (np.clip(np.rint((g - lo) / step), 0, 15) * step + lo).reshape(N, K).astype(np.float32)


def quantize_matrix(W, H, book, gptq):
    """-> (dequantized W, None) for the baseline format, else (None, (codes, s, c))."""
    if book == "mlx-affine4":
        return affine4(W), None
    if gptq and H is not None:
        return None, quantlib.quantize_gptq(W, H, book, A_GRID, B_GRID)
    return None, quantlib.quantize_rtn(W, None if H is None else np.diag(H).copy(), book, A_GRID, B_GRID)


def quantize_stack(parts, H, book, gptq):
    """Quantizes the row-stacked matrices jointly: they share the input, hence c."""
    W = np.concatenate([np.array(m.weight.astype(mx.float32)) for _, m in parts], axis=0)
    return quantize_matrix(W, H, book, gptq)


# -------------------------------------------------------------------- main

def log_softmax(z):
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="original (unquantized) model directory")
    ap.add_argument("dst", help="output .rlm file")
    ap.add_argument("--book", choices=sorted(quantlib.LMAX) + ["mlx-affine4"], default="fp4",
                    help="codebook; mlx-affine4 measures MLX's 4-bit format on the same data and writes no file")
    ap.add_argument("--calib", nargs="*", help="text files to calibrate on")
    ap.add_argument("--eval", nargs="*", help="held-out text files for the quality report")
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--eval-seqs", type=int, default=12)
    ap.add_argument("--no-gptq", action="store_true", help="round to nearest (fast, less accurate)")
    ap.add_argument("--layers", type=int, help="stop after this many layers (debugging)")
    cli = ap.parse_args()

    cfg_all = json.load(open(os.path.join(cli.src, "config.json")))
    tc = cfg_all.get("text_config", cfg_all)
    if tc.get("num_experts", 0):
        raise SystemExit("MoE variants are not supported yet")
    args = qwen3_5.TextModelArgs.from_dict(tc)
    tied = bool(tc.get("tie_word_embeddings", cfg_all.get("tie_word_embeddings", False)))
    rope = tc.get("rope_parameters") or {}
    cfg = {k: tc[k] for k in (
        "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
        "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size", "rms_norm_eps", "full_attention_interval")}
    cfg["rope_theta"] = float(rope.get("rope_theta", tc.get("rope_theta", 1e7)))
    cfg["partial_rotary_factor"] = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
    cfg["eos_token_id"] = tc.get("eos_token_id")
    cfg["tie_word_embeddings"] = tied

    src = Source(cli.src)
    tok = Tokenizer.from_file(os.path.join(cli.src, "tokenizer.json"))
    cal_files, ev_files = default_texts()
    cal = token_sequences(tok, cli.calib or cal_files, cli.calib_seqs)
    ev = token_sequences(tok, cli.eval or ev_files, cli.eval_seqs)
    n_layers = min(cli.layers or args.num_hidden_layers, args.num_hidden_layers)
    book, gptq = cli.book, not cli.no_gptq
    print(f"{cli.src}: {n_layers} layers, codebook {book}, {'GPTQ' if gptq else 'round to nearest'}; "
          f"{len(cal)} calibration and {len(ev)} evaluation sequences of {T} tokens")

    t_start = time.time()
    write = book != "mlx-affine4"
    out = Writer(cli.dst) if write else None
    final_norm = src.get("norm.weight")
    cal_np, ev_np = np.asarray(cal), np.asarray(ev)
    ev_ids = mx.array(ev)

    def final(h):  # last-layer hidden -> normalized head input
        return mx.fast.rms_norm(h, mx.array(final_norm), args.rms_norm_eps)

    def second_moment(h):
        x = final(h).reshape(-1, h.shape[-1])
        return np.array(x.T @ x) / x.shape[0]

    def rows(Wq, q, ids):  # dequantized rows of a quantized matrix for token ids
        if q is None:
            return mx.array(Wq[ids])
        codes, s, c = q
        return mx.array(quantlib.dequant(codes[ids], s[ids], c, book))

    # The token lookup table. When it is tied to the output head it has to be
    # quantized before anything else (the quantized stream starts from it), so
    # the head's input statistics come from one float pass over the
    # calibration set. Untied, it is only ever read row by row: round to nearest.
    emb = src.get("embed_tokens.weight")                       # [V, H] float32
    hr = mx.array(emb[ev_np])                                  # evaluation stream, float model
    if tied:
        h = mx.array(emb[cal_np])
        for i in range(n_layers):
            h = run_layer(build_layer(args, i, src.layer_weights(i)), h)
            print(f"\r  float pass for the head statistics: layer {i + 1}/{n_layers}", end="", flush=True)
        print()
        head_wq, head_q = quantize_matrix(emb, second_moment(h), book, gptq)
        emb_wq, emb_q, head_float = head_wq, head_q, emb
    else:
        emb_wq, emb_q = quantize_matrix(emb, None, book, False)
        head_float = None
    hc = rows(emb_wq, emb_q, cal_np)                           # calibration stream, quantized model
    hq = rows(emb_wq, emb_q, ev_np)                            # evaluation stream, quantized model
    mx.eval(hc, hq, hr)
    if write:
        out.begin("embed")
        out.add_native_rows("embed", emb_q[0], emb_q[1], emb_q[2], book)
        out.end()
    del emb, emb_wq, emb_q

    for i in range(n_layers):
        t0 = time.time()
        weights = src.layer_weights(i)
        layer = build_layer(args, i, weights)
        hr = run_layer(layer, hr)
        moments = input_moments(layer, hc)
        stacks = stacks_of(layer)
        results = [quantize_stack(parts, moments[id(parts[0][1])], book, gptq) for _, parts in stacks]

        if write:
            out.begin(f"layer.{i}")
            out.add_float(f"layers.{i}.in_norm", weights["input_layernorm.weight"])
            out.add_float(f"layers.{i}.post_norm", weights["post_attention_layernorm.weight"])
            if layer.is_linear:
                conv = weights["linear_attn.conv1d.weight"]
                out.add_float(f"layers.{i}.lin.conv", conv.reshape(conv.shape[0], conv.shape[1]))
                out.add_float(f"layers.{i}.lin.A_log", weights["linear_attn.A_log"])
                out.add_float(f"layers.{i}.lin.dt_bias", weights["linear_attn.dt_bias"])
                out.add_float(f"layers.{i}.lin.norm", weights["linear_attn.norm.weight"])
            else:
                out.add_float(f"layers.{i}.attn.q_norm", weights["self_attn.q_norm.weight"])
                out.add_float(f"layers.{i}.attn.k_norm", weights["self_attn.k_norm.weight"])
        for (name, parts), (Wq, q) in zip(stacks, results):
            if q is not None:
                codes, s, c = q
                if write:
                    out.add_native(f"layers.{i}.{name}", [(pn, m.weight.shape[0]) for pn, m in parts],
                                   codes, s, c, book)
                Wq = quantlib.dequant(codes, s, c, book)
            row = 0
            for _, m in parts:
                m.weight = mx.array(Wq[row:row + m.weight.shape[0]])
                row += m.weight.shape[0]
        if write:
            out.end()
        mx.eval(layer.parameters())
        hc = run_layer(layer, hc)
        hq = run_layer(layer, hq)
        print(f"  layer {i + 1}/{n_layers} ({'linear' if layer.is_linear else 'attn'}) {time.time() - t0:.0f}s", flush=True)
        del layer, moments, results
        mx.clear_cache()

    if not tied:
        head_float = src.get("lm_head.weight")
        head_wq, head_q = quantize_matrix(head_float, second_moment(hc), book, gptq)
    if head_q is not None and write:
        out.begin("head")
        out.add_float("final_norm", final_norm)
        out.add_native("lm_head", [("out", head_q[0].shape[0])], head_q[0], head_q[1], head_q[2], book)
        out.end()

    # Quality on held-out text: float model vs quantized model, last EVAL_POS
    # positions. The head is applied a slice of the vocabulary at a time: at
    # 27B scale the two head matrices would not fit in memory as float32.
    V = head_float.shape[0]

    def head_rows(v0, v1, quantized):
        if not quantized:
            return head_float[v0:v1]
        if head_q is None:
            return head_wq[v0:v1]
        codes, s, c = head_q
        return quantlib.dequant(codes[v0:v1], s[v0:v1], c, book)

    def log_probs(x, quantized):
        parts = []
        for v0 in range(0, V, 32768):
            parts.append(x @ mx.array(head_rows(v0, min(V, v0 + 32768), quantized)).T)
            mx.eval(parts[-1])
        return log_softmax(mx.concatenate(parts, axis=1))

    sl = slice(T - EVAL_POS - 1, T - 1)
    kl = ce = agree = 0.0
    for j in range(len(ev)):
        lr = log_probs(final(hr[j, sl]), False)
        lq = log_probs(final(hq[j, sl]), True)
        tgt = ev_ids[j, T - EVAL_POS:T]
        kl += mx.sum(mx.exp(lr) * (lr - lq)).item()
        ce += -mx.sum(mx.take_along_axis(lq, tgt[:, None], axis=-1)).item()
        agree += mx.sum(mx.argmax(lq, axis=-1) == mx.argmax(lr, axis=-1)).item()
    n = len(ev) * EVAL_POS
    quality = dict(kl=kl / n, ppl=float(np.exp(ce / n)), top1=agree / n, positions=n)
    print(f"held-out quality vs the float model: KL {quality['kl']:.5f} nats, perplexity {quality['ppl']:.3f}, "
          f"top-1 agreement {100 * quality['top1']:.2f}%")

    if not write:
        return
    cfg["num_hidden_layers"] = n_layers
    out.add_files(cli.src)
    out.finish(dict(format="redline", version=2, arch="qwen3_5", source=os.path.basename(os.path.normpath(cli.src)),
                    config=cfg, quant=dict(mode="native", bits=4, book=book, block=BLOCK, gptq=gptq), quality=quality))
    print(f"wrote {cli.dst}: {os.path.getsize(cli.dst) / 1e9:.2f} GB in {time.time() - t_start:.0f}s")
    if "mtp.fc.weight" in src.st:
        print(f"this checkpoint has an MTP head: tools/add_mtp.py {cli.src} {cli.dst} adds it (about 1.5x faster decoding)")


if __name__ == "__main__":
    sys.exit(main())
