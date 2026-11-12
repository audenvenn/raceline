#!/usr/bin/env python3
"""Add the model's own multi-token-prediction (MTP) head to a native .rlm file.

Qwen3.5 / Qwen3.8 checkpoints ship a one-layer head that predicts the token
after next from the final hidden state and the next token (see
tools/mtp_study.py). The engine uses it to draft one token ahead and verify
two tokens per pass over the weights. This tool quantizes the head into the
file's own 4-bit format and appends it as a section; the rest of the file is
untouched.

The head is calibrated on what it will really see: hidden states produced by
the engine from the quantized model in the file, and rows of the quantized
token table. It needs the engine's library (cmake --build build/cmake).

  tools/add_mtp.py models/Qwen3.5-4B models/qwen3.5-4b.fp4.rlm
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import quantlib  # noqa: E402
from convert import Writer  # noqa: E402
from rlmfile import Native  # noqa: E402
from quantize import (T, Source, build_layer, default_texts, input_moments, is_moe, quantize_stack,  # noqa: E402
                      rotation, rotation_signs, stack_moments, stacks_of, token_sequences)

NORM_E, NORM_H, NORM_OUT = "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"


def mtp_weights(src: Source):
    """-> (fc [H, 2H], {the three norms}, decoder-layer weights), with MLX conventions (norm weights + 1)."""
    off = 1.0 if src.raw else 0.0
    norms = {k: src.st.get(k).astype(np.float32) + off for k in (NORM_E, NORM_H, NORM_OUT)}
    pre = "mtp.layers.0."
    layer = {k[len(pre):]: src.get(k) for k in src.st.index if k.startswith(pre)}
    if "mlp.experts.0.gate_proj.weight" in layer:
        # A MoE head's experts are stored one by one: fuse them as the model's layers have them,
        # one expert at a time (no second copy of all of them: 3 GB for Qwen3.5-35B-A3B).
        E = 1 + max(int(k.split(".")[2]) for k in layer if k.startswith("mlp.experts."))
        g0, d0 = layer["mlp.experts.0.gate_proj.weight"], layer["mlp.experts.0.down_proj.weight"]
        gate_up = np.empty((E, 2 * g0.shape[0], g0.shape[1]), dtype=np.float32)
        down = np.empty((E,) + d0.shape, dtype=np.float32)
        for e in range(E):
            gate_up[e, :g0.shape[0]] = layer.pop(f"mlp.experts.{e}.gate_proj.weight")
            gate_up[e, g0.shape[0]:] = layer.pop(f"mlp.experts.{e}.up_proj.weight")
            down[e] = layer.pop(f"mlp.experts.{e}.down_proj.weight")
        layer["mlp.experts.gate_up_proj"], layer["mlp.experts.down_proj"] = gate_up, down
    return src.st.get("mtp.fc.weight").astype(np.float32), norms, layer


def engine_hiddens(rlm: str, seqs, hidden: int) -> np.ndarray:
    """Final hidden states (before the final norm) of every position, from the engine."""
    from raceline.engine import _library

    lib = _library()
    e = lib.rl_open(os.path.abspath(rlm).encode(), T, T)
    lib.rl_set_mtp(e, 0)
    out = np.empty((len(seqs), T, hidden), dtype=np.float32)
    for j, ids in enumerate(seqs):
        lib.rl_reset(e)
        lib.rl_forward(e, (ctypes.c_int32 * T)(*ids), T)
        out[j] = np.ctypeslib.as_array(lib.rl_hidden(e), shape=(T, hidden))
        print(f"\r  hidden states from the engine: {j + 1}/{len(seqs)}", end="", flush=True)
    print()
    lib.rl_close(e)
    return out


def fc_inputs(emb_next, hn, norms, eps):
    """Row i pairs the embedding of token i+1 with the normalized final hidden state of position i."""
    return mx.concatenate([mx.fast.rms_norm(emb_next, mx.array(norms[NORM_E]), eps),
                           mx.fast.rms_norm(hn, mx.array(norms[NORM_H]), eps)], axis=-1)


def head_output(fc_w, layer, x, batch=8):
    """The head's residual stream after its decoder layer (before its final norm)."""
    outs = []
    for i in range(0, x.shape[0], batch):
        h = x[i:i + batch] @ mx.array(fc_w).T
        y = layer(h, mask=create_attention_mask(h, None), cache=None)
        mx.eval(y)
        outs.append(y)
    return mx.concatenate(outs, axis=0)


def quantize_head(fc, layer, x_cal, book, gptq=True):
    """The head's decoder layer (float) to 4 bits, in place -> {file tensor name: (parts, (codes, s, c))};
    the layer is left with the dequantized weights. The input projection `fc` is not quantized: it is the
    one matrix of the head that 4 bits hurt (agreement with the model drops by about eight points; the
    others cost a fraction of one), and at [H][2H] it is small enough to keep in half precision."""
    out = {}
    h = x_cal @ mx.array(fc).T
    mx.eval(h)
    moments = input_moments(layer, h)
    for name, parts in stacks_of(layer):
        _, q = quantize_stack(parts, stack_moments(parts, moments), book, gptq)
        out["mtp." + name] = ([(pn, m.weight.shape[0]) for pn, m in parts], q)
        Wq, row = quantlib.dequant(*q, book), 0
        for _, m in parts:
            m.weight = mx.array(Wq[row:row + m.weight.shape[0]])
            row += m.weight.shape[0]
        mx.eval(layer.parameters())
        del Wq
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="original (unquantized) model directory: the source of the mtp.* tensors")
    ap.add_argument("rlm", help="native .rlm file to extend, in place")
    ap.add_argument("--calib-seqs", type=int, default=24)
    ap.add_argument("--eval-seqs", type=int, default=8)
    ap.add_argument("--no-gptq", action="store_true", help="round to nearest (the head agrees far less often)")
    ap.add_argument("--force", action="store_true", help="replace a head the file already has")
    cli = ap.parse_args()
    mx.set_cache_limit(1 << 30)   # freed buffers kept for reuse; unbounded, the pool grows by several GB

    nat = Native(cli.rlm)
    if "mtp.fc" in nat.hdr["tensors"] and not cli.force:
        raise SystemExit(f"{cli.rlm} already has an MTP head (--force replaces it)")
    src = Source(cli.src)
    if "mtp.fc.weight" not in src.st:
        raise SystemExit(f"{cli.src} has no MTP head")
    fc, norms, lw = mtp_weights(src)
    cfg_all = json.load(open(os.path.join(cli.src, "config.json")))
    args = qwen3_5.TextModelArgs.from_dict(cfg_all.get("text_config", cfg_all))
    eps, hidden, book, gptq = args.rms_norm_eps, args.hidden_size, nat.book, not cli.no_gptq
    t_start = time.time()

    tok = Tokenizer.from_file(os.path.join(cli.src, "tokenizer.json"))
    cal_files, ev_files = default_texts()
    seqs = token_sequences(tok, cal_files, cli.calib_seqs) + token_sequences(tok, ev_files, cli.eval_seqs)
    n_cal = min(cli.calib_seqs, len(seqs) - 1)
    ids = np.asarray(seqs)
    # A file converted with rotation (the default) has a rotated residual stream (x Q) and its
    # final norm folded into the output head. The head is calibrated in the
    # original coordinates, so the hidden states and token rows are rotated back.
    rot = nat.hdr["quant"].get("rotation")
    if rot:
        Q = rotation(hidden, seed=rot["seed"])
        if not np.array_equal(rotation_signs(hidden, rot["seed"]), nat.floats("rotation.signs")):
            raise SystemExit("the rotation in the file is not the one this tool reconstructs")
        Qm, wf = mx.array(Q), nat.floats("final_norm.unfolded")
    hid = mx.array(engine_hiddens(cli.rlm, seqs, hidden))
    emb_next = mx.array(nat.rows("embed", ids[:, 1:].reshape(-1)).reshape(len(seqs), T - 1, hidden))
    if rot:
        head_in = mx.fast.rms_norm(hid, None, eps)             # what the output head reads: rotated, norm folded
        hid, emb_next = hid @ Qm.T, emb_next @ Qm.T
        hn = mx.fast.rms_norm(hid, mx.array(wf), eps)
    else:
        hn = mx.fast.rms_norm(hid, mx.array(nat.floats("final_norm")), eps)
        head_in = hn
    x = fc_inputs(emb_next, hn[:, :-1], norms, eps)
    mx.eval(x, hn, head_in)
    del hid, emb_next

    def to_head(u):
        """The head's output as the model's output head reads it."""
        if not rot:
            return mx.fast.rms_norm(u, mx.array(norms[NORM_OUT]), eps)
        return mx.fast.rms_norm(u, mx.array(norms[NORM_OUT] / wf), eps) @ Qm

    fc = fc.astype(np.float16).astype(np.float32)          # as stored
    # One copy of the head's decoder layer: held-out check in float, quantized in place, checked again.
    # Of the float weights, only the small ones are still needed (for the file).
    layer = build_layer(args, args.full_attention_interval - 1, lw)
    lw = {k: v for k, v in lw.items() if v.size < (1 << 22)}

    # Held-out check: how often is the head's top choice the model's own?
    x_ev, head_ev = x[n_cal:], head_in[n_cal:]
    n_ev, skip = x_ev.shape[0], 64
    choice = nat.head_argmax(head_ev.reshape(-1, hidden)).reshape(n_ev, T)     # choice[i]: the pick for token i+1
    agree = {}
    for label in ("float", "4-bit"):
        if label == "4-bit":
            tensors = quantize_head(fc, layer, x[:n_cal], book, gptq)
        out = to_head(head_output(fc, layer, x_ev))
        pred = nat.head_argmax(out.reshape(-1, hidden)).reshape(n_ev, T - 1)   # pred[i]: the guess for token i+2
        agree[label] = float(np.mean(pred[:, skip:T - 2] == choice[:, skip + 1:T - 1]))
    layer_q = layer
    print(f"the head's top choice is the model's own choice: float head {100 * agree['float']:.1f}%, "
          f"{book} head {100 * agree['4-bit']:.1f}%  ({n_ev * (T - 2 - skip)} held-out positions)")

    out, hdr = Writer.append(cli.rlm, drop_from="mtp")
    out.begin("mtp")
    norm_e, fc_file = norms[NORM_E], fc
    if rot:   # the token rows the head reads are rotated: fold its embedding norm and the rotation into fc
        fc_file = fc.copy()
        fc_file[:, :hidden] = (fc[:, :hidden] * norm_e[None, :]) @ Q
        norm_e = np.ones_like(norm_e)
    for name, w in (("mtp.in_norm", lw["input_layernorm.weight"]), ("mtp.post_norm", lw["post_attention_layernorm.weight"]),
                    ("mtp.attn.q_norm", lw["self_attn.q_norm.weight"]), ("mtp.attn.k_norm", lw["self_attn.k_norm.weight"]),
                    ("mtp.norm_e", norm_e), ("mtp.norm_h", norms[NORM_H]), ("mtp.norm", norms[NORM_OUT])):
        out.add_float(name, w)
    out.add_dense("mtp.fc", fc_file)
    for name, (parts, (codes, s, c)) in tensors.items():
        out.add_native(name, parts, codes, s, c, book)
    if is_moe(layer_q):   # the router and the shared expert's gate, the gate/up stack's column scales divided out
        router = np.concatenate([lw["mlp.gate.weight"], lw["mlp.shared_expert_gate.weight"]], axis=0)
        out.add_dense("mtp.moe.router", router / tensors["mtp.moe.gate_up"][1][2][None, :])
    out.end()
    if "tokenizer.json" not in out.files:      # older files, or dropped together with a previous head
        out.add_files(cli.src)
    hdr["mtp"] = dict(gptq=gptq, agreement=agree["4-bit"], agreement_float=agree["float"])
    out.finish(hdr)
    mtp_bytes = next(sec["nbytes"] for sec in out.sections if sec["name"] == "mtp")
    print(f"added the MTP head to {cli.rlm} ({mtp_bytes / 1e6:.0f} MB) in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
