#!/usr/bin/env python3
"""Quantize an original Qwen3.5 / Qwen3.8 checkpoint into raceline's native format.

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

  tools/quantize.py models/Qwen3.5-4B models/qwen3.5-4b.rlm

A large model takes a while (Qwen3.8-27B: close to two hours on an M6). If
the run is interrupted, the same command with --resume picks up after the
last layer that reached the file: finished layers are read back, checked
against the checkpoint and only pushed through the activations again.
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
from convert import GROUP, BLOCK, MAGIC, Layout, SafeTensors, Writer  # noqa: E402,F401
from rlmfile import Native  # noqa: E402

T = 512
EVAL_POS = 96
A_GRID, B_GRID = (0.0, 0.25, 0.5, 0.75), (0.0, 0.5, 1.0)
ACT_ORDER = os.environ.get("RACELINE_ACT_ORDER", "0") == "1"   # GPTQ column order; measured no gain on Qwen3.8-27B
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
        return None, quantlib.quantize_gptq(W, H, book, A_GRID, B_GRID, act_order=ACT_ORDER)
    return None, quantlib.quantize_rtn(W, None if H is None else np.diag(H).copy(), book, A_GRID, B_GRID)


def quantize_stack(parts, H, book, gptq):
    """Quantizes the row-stacked matrices jointly: they share the input, hence c."""
    W = np.concatenate([np.array(m.weight.astype(mx.float32)) for _, m in parts], axis=0)
    return quantize_matrix(W, H, book, gptq)


# ------------------------------------------------------------- rotation
#
# Unless --no-rotate is given, the residual stream is rotated by an orthogonal matrix Q
# (x -> x Q; random signs times blocks of Hadamard matrices, see rotation())
# before anything is quantized. The model computes exactly the same
# function: every RMS norm's weight is first folded into the matrices that read
# its output (the norm itself is then unweighted, and an unweighted RMS norm
# does not change under rotation), every matrix that reads the residual stream
# is multiplied by Q from the right, every matrix that writes it by Q^T from
# the left, and the token table by Q. What changes is the weights: rotated rows
# are spread far more evenly, which is what a format with one scale per row
# and per column, and none per group, needs (the idea of QuaRot and QuIP).
# The rotation is folded into the weights: it costs nothing at run time.

READS_NORM = {"input_layernorm.weight": ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                                         "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
                                         "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                         "self_attn.v_proj.weight"),
              "post_attention_layernorm.weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight")}
WRITES_RESIDUAL = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight")


def hadamard_block(n: int) -> int:
    """Block size of the rotation for width n: the largest power of two dividing n, at most 1024."""
    b = n & -n
    return min(b, 1024)


def rotation_signs(n: int, seed: int = 0) -> np.ndarray:
    return np.where(np.random.default_rng(seed).random(n) < 0.5, -1.0, 1.0).astype(np.float32)


def rotation(n: int, seed: int = 0, kind: str = "hadamard") -> np.ndarray:
    """An orthogonal n x n matrix Q for the residual stream (x -> x Q).

    "hadamard": diag(random signs) times a block-diagonal of normalized
    Sylvester-Hadamard matrices. Applying it, or its inverse, to a vector is a
    sign flip and a fast Walsh-Hadamard transform per block, which is what the
    engine does where the rotation cannot be folded into a weight (the
    draft head's inputs and output). "orthogonal": a uniformly random
    orthogonal matrix, for comparison.
    """
    if kind == "orthogonal":
        A = np.random.default_rng(seed).standard_normal((n, n))
        Qm, R = np.linalg.qr(A)
        return (Qm * np.sign(np.diag(R))[None, :]).astype(np.float32)
    b = hadamard_block(n)
    Hb = np.ones((1, 1), dtype=np.float64)
    while Hb.shape[0] < b:
        Hb = np.block([[Hb, Hb], [Hb, -Hb]])
    Hb /= np.sqrt(b)
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n, b):
        Q[i:i + b, i:i + b] = Hb
    return (rotation_signs(n, seed)[:, None] * Q).astype(np.float32)


def rotate_layer(weights: dict, Q: np.ndarray) -> dict:
    """A layer's weights for the rotated residual stream (norm weights folded in, then set to one)."""
    w = dict(weights)
    for norm, readers in READS_NORM.items():
        g = w[norm]
        for name in readers:
            if name in w:
                w[name] = (w[name] * g[None, :]) @ Q
        w[norm] = np.ones_like(g)
    for name in WRITES_RESIDUAL:
        if name in w:
            w[name] = Q.T @ w[name]
    return w


def rotate_rows(M: np.ndarray, Q: np.ndarray, scale=None, chunk: int = 16384) -> np.ndarray:
    """M <- (M * scale) @ Q, in place and a slice at a time (for the vocabulary-sized matrices)."""
    Qm = mx.array(Q)
    sc = None if scale is None else mx.array(scale)[None, :]
    for i in range(0, M.shape[0], chunk):
        part = mx.array(M[i:i + chunk])
        if sc is not None:
            part = part * sc
        M[i:i + chunk] = np.array(part @ Qm)
    return M


def emit_layer(out, i, layer, weights, stacks, results, book):
    """Writes layer i's section. With results=None, `out` is a Layout and only
    the positions are computed (the same calls, in the same order)."""
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
    for j, (name, parts) in enumerate(stacks):
        named = [(pn, m.weight.shape[0]) for pn, m in parts]
        if results is None:
            out.add_native_layout(f"layers.{i}.{name}", named, parts[0][1].weight.shape[1], book)
        else:
            codes, s, c = results[j][1]
            out.add_native(f"layers.{i}.{name}", named, codes, s, c, book)
    out.end()


# -------------------------------------------------------------------- main

def log_softmax(z):
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def drift(hq, hr) -> float:
    """How far the quantized model's hidden states have moved from the float model's on the
    held-out sequences, relative to their size. Printed per layer: it shows where quality is lost."""
    return (mx.linalg.norm((hq - hr).reshape(-1)) / mx.linalg.norm(hr.reshape(-1))).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="original (unquantized) model directory")
    ap.add_argument("dst", help="output .rlm file")
    ap.add_argument("--book", choices=sorted(quantlib.LMAX) + ["mlx-affine4"], default="fp4",
                    help="codebook; mlx-affine4 (MLX's 4-bit format) and mxfp4 (FP4 with a power-of-two scale "
                         "per 32 weights) are measured on the same data and write no file")
    ap.add_argument("--calib", nargs="*", help="text files to calibrate on")
    ap.add_argument("--eval", nargs="*", help="held-out text files for the quality report")
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--eval-seqs", type=int, default=12)
    ap.add_argument("--no-gptq", action="store_true", help="round to nearest (fast, less accurate)")
    ap.add_argument("--layers", type=int, help="stop after this many layers (debugging)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted conversion of the same checkpoint")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    help="do not rotate the residual stream (see 'rotation' above; rotating is the default: same "
                         "function, better weights)")
    cli = ap.parse_args()

    # Freed buffers are kept for reuse up to this much; without a bound the pool
    # grows to several GB over a layer and pushes a 24 GB machine into swap.
    mx.set_cache_limit(2 << 30)

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
    write = book not in ("mlx-affine4", "mxfp4")     # measured only: the engine does not read these
    # An interrupted run leaves a file without a header. Its contents are found
    # again by replaying the writer's layout (`plan`); `out` opens once the
    # first layer that is not in the file is reached.
    plan = None
    if write and cli.resume and os.path.exists(cli.dst):
        with open(cli.dst, "rb") as f:
            if f.read(4) == MAGIC:
                raise SystemExit(f"{cli.dst} is a finished model file: nothing to resume")
        plan, have = Layout(), os.path.getsize(cli.dst)
        partial = Native(cli.dst, hdr=dict(tensors=plan.tensors, quant=dict(mode="native", book=book)))
    elif cli.resume:
        raise SystemExit(f"--resume: {cli.dst} does not exist")
    out = Writer(cli.dst) if write and plan is None else None

    def reopen():
        """The real writer, positioned after what the plan found in the file."""
        w = Writer.__new__(Writer)
        w.header_room, w.tensors, w.files, w.sections, w._sec = plan.header_room, plan.tensors, {}, plan.sections, None
        w.f = open(cli.dst, "r+b")
        w.f.truncate(plan.pos)
        w.f.seek(plan.pos)
        return w
    final_norm = src.get("norm.weight")
    rot_kind = os.environ.get("RACELINE_ROTATION", "hadamard")
    if cli.rotate and rot_kind != "hadamard" and write:
        raise SystemExit("only the Hadamard rotation can be written: the engine undoes it at run time for the draft head")
    Q = rotation(args.hidden_size, kind=rot_kind) if cli.rotate else None
    final_norm_orig = final_norm
    if Q is not None:
        final_norm = np.ones_like(final_norm)                  # folded into the head below
    # A tied head stops being the token table once the final norm is folded into it.
    separate_head = not tied or Q is not None
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
    if Q is not None:
        rotate_rows(emb, Q)
    hr = mx.array(emb[ev_np])                                  # evaluation stream, float model
    if plan is not None:                                       # the table is the first thing in the file
        plan.begin("embed")
        plan.add_native_rows_layout("embed", emb.shape[0], emb.shape[1], book)
        plan.end()
        if plan.pos > have:
            raise SystemExit(f"{cli.dst} holds nothing to resume from: run without --resume")
        emb_wq, emb_q = None, partial.row_codes("embed")
        err = float(np.linalg.norm(quantlib.dequant(emb_q[0][:4096], emb_q[1][:4096], emb_q[2], book) - emb[:4096])
                    / np.linalg.norm(emb[:4096]))
        if not err < 0.5:
            raise SystemExit(f"{cli.dst} does not look like a conversion of {cli.src} (token table off by {err:.2f})")
        head_wq, head_q, head_float = (None, emb_q, emb) if not separate_head else (None, None, None)
        print(f"  resuming: token table read back from {cli.dst}")
    elif not separate_head:
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
    if write and plan is None:
        out.begin("embed")
        out.add_native_rows("embed", emb_q[0], emb_q[1], emb_q[2], book)
        out.end()
    del emb, emb_wq, emb_q

    for i in range(n_layers):
        t0 = time.time()
        weights = src.layer_weights(i)
        if Q is not None:
            weights = rotate_layer(weights, Q)
        layer = build_layer(args, i, weights)
        hr = run_layer(layer, hr)
        stacks = stacks_of(layer)

        if out is None and plan is not None:
            # Is this layer in the file? Lay it out and see whether its data ends within it.
            mark = (plan.pos, dict(plan.tensors), list(plan.sections))
            emit_layer(plan, i, layer, weights, stacks, None, book)
            last = plan.tensors[f"layers.{i}.{stacks[-1][0]}"]
            if last["c"] + 4 * last["k"] <= have:
                worst = 0.0
                for name, parts in stacks:
                    Wq, row = partial.matrix(f"layers.{i}.{name}"), 0
                    for _, m in parts:
                        W = np.array(m.weight.astype(mx.float32))
                        worst = max(worst, float(np.linalg.norm(Wq[row:row + W.shape[0]] - W) / np.linalg.norm(W)))
                        m.weight = mx.array(Wq[row:row + W.shape[0]])
                        row += W.shape[0]
                if not worst < 0.7:
                    raise SystemExit(f"{cli.dst}: layer {i} does not match {cli.src} (relative error {worst:.2f})")
                mx.eval(layer.parameters())
                hc = run_layer(layer, hc)
                hq = run_layer(layer, hq)
                print(f"  layer {i + 1}/{n_layers} read back from the file (largest deviation from float {worst:.2f}) "
                      f"{time.time() - t0:.0f}s  drift {drift(hq, hr):.4f}", flush=True)
                del layer
                mx.clear_cache()
                continue
            plan.pos, plan.tensors, plan.sections = mark[0], mark[1], mark[2]
            partial.hdr["tensors"] = plan.tensors
            out = reopen()
            print(f"  continuing the conversion at layer {i + 1}", flush=True)

        moments = input_moments(layer, hc)
        results = [quantize_stack(parts, moments[id(parts[0][1])], book, gptq) for _, parts in stacks]
        if write:
            emit_layer(out, i, layer, weights, stacks, results, book)
            out.f.flush()
        for (name, parts), (Wq, q) in zip(stacks, results):
            if q is not None:
                Wq = quantlib.dequant(*q, book)
            row = 0
            for _, m in parts:
                m.weight = mx.array(Wq[row:row + m.weight.shape[0]])
                row += m.weight.shape[0]
        mx.eval(layer.parameters())
        hc = run_layer(layer, hc)
        hq = run_layer(layer, hq)
        print(f"  layer {i + 1}/{n_layers} ({'linear' if layer.is_linear else 'attn'}) {time.time() - t0:.0f}s  "
              f"drift {drift(hq, hr):.4f}", flush=True)
        del layer, moments, results
        mx.clear_cache()

    if write and out is None:          # every layer was already in the file
        out = reopen()
    if separate_head:
        head_float = src.get("lm_head.weight" if not tied else "embed_tokens.weight")
        if Q is not None:
            rotate_rows(head_float, Q, final_norm_orig)
        head_wq, head_q = quantize_matrix(head_float, second_moment(hc), book, gptq)
    if head_q is not None and write:
        out.begin("head")
        out.add_float("final_norm", final_norm)
        if Q is not None:      # what the draft head needs to undo the rotation (tools/add_mtp.py, the engine)
            out.add_float("final_norm.unfolded", final_norm_orig)
            out.add_float("rotation.signs", rotation_signs(args.hidden_size))
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
    out.finish(dict(format="raceline", version=2, arch="qwen3_5", source=os.path.basename(os.path.normpath(cli.src)),
                    config=cfg, quant=dict(mode="native", bits=4, book=book, block=BLOCK, gptq=gptq,
                              **({"rotation": dict(kind="signed-block-hadamard", block=hadamard_block(args.hidden_size),
                                                   seed=0)} if Q is not None else {})),
                    quality=quality))
    print(f"wrote {cli.dst}: {os.path.getsize(cli.dst) / 1e9:.2f} GB in {time.time() - t_start:.0f}s")
    if "mtp.fc.weight" in src.st:
        print(f"this checkpoint has an MTP head: tools/add_mtp.py {cli.src} {cli.dst} adds it (about 1.5x faster decoding)")


if __name__ == "__main__":
    sys.exit(main())
