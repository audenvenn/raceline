#!/usr/bin/env python3
"""How good is the multi-token-prediction head that ships inside Qwen3.5 / Qwen3.8?

The original checkpoints carry fifteen `mtp.*` tensors that mlx-lm and
transformers drop on load: a one-layer head that, from the model's final
hidden state at position i and the embedding of token i+1, predicts token i+2.
A draft that is right lets a decoder verify two tokens in one pass over the
weights, so what matters is how often its top choice equals the model's own.

This script measures that on held-out text, in float (MLX), under teacher
forcing, and answers two design questions for the engine:

  - does the head need its attention history, or only the current position?
  - how much agreement is lost when its matrices are stored in 4 bits?

  tools/mtp_study.py models/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantlib  # noqa: E402
from add_mtp import NORM_E, NORM_H, NORM_OUT, fc_inputs, mtp_weights, quantize_head  # noqa: E402
from quantize import A_GRID, B_GRID, T, Source, default_texts, token_sequences  # noqa: E402

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="original (unquantized) model directory")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--calib-seqs", type=int, default=16, help="sequences the 4-bit GPTQ variant is calibrated on")
    ap.add_argument("--tokens", type=int, default=768)
    ap.add_argument("--skip", type=int, default=256, help="positions at the start of each sequence left out")
    cli = ap.parse_args()

    src = Source(cli.model)
    if "mtp.fc.weight" not in src.st:
        raise SystemExit("this checkpoint has no MTP head")
    fc, norms, lw = mtp_weights(src)
    model, tok = load(cli.model)
    lm = model.language_model
    tm, args = lm.model, lm.args
    eps = args.rms_norm_eps
    head = (tm.embed_tokens.weight if args.tie_word_embeddings else lm.lm_head.weight).astype(mx.float32)

    _, files = default_texts()
    seqs = []
    for p in files:
        ids = tok._tokenizer.encode(open(p, encoding="utf-8", errors="ignore").read()[:400_000],
                                    add_special_tokens=False)
        if len(ids) >= cli.tokens:
            seqs.append(ids[:cli.tokens])
        if len(seqs) == cli.seqs:
            break

    # Main model once: embeddings, final hidden states, its own greedy choices.
    runs = []
    for ids in seqs:
        x = mx.array(ids)[None]
        e = tm.embed_tokens(x).astype(mx.float32)
        h = e
        fa = create_attention_mask(h, None)
        for l in tm.layers:
            h = l(h, mask=None if l.is_linear else fa, cache=None)
        hn = tm.norm(h)
        choice = mx.argmax(hn @ head.T, axis=-1)[0]      # choice[i]: the model's pick for token i+1
        mx.eval(e, hn, choice)
        runs.append((x, e, hn, choice))
    cal_files, _ = default_texts()
    xs = []
    for ids in token_sequences(Tokenizer.from_file(os.path.join(cli.model, "tokenizer.json")), cal_files,
                               cli.calib_seqs):
        x = mx.array(ids)[None]
        e = tm.embed_tokens(x).astype(mx.float32)
        h = e
        for l in tm.layers:
            h = l(h, mask=None if l.is_linear else create_attention_mask(h, None), cache=None)
        xs.append(fc_inputs(e[:, 1:], tm.norm(h)[:, :-1], norms, eps))
        mx.eval(xs[-1])
    x_cal = mx.concatenate(xs, axis=0)
    del model
    n_pos = sum(len(ids) - 2 - cli.skip for ids in seqs)
    print(f"{cli.model}: {len(seqs)} sequences, {n_pos} positions")

    layer = qwen3_5.DecoderLayer(args, args.full_attention_interval - 1)   # a full-attention layer
    assert not layer.is_linear

    def agreement(fc_w, layer_w, window, label):
        layer.load_weights([(k, mx.array(v)) for k, v in layer_w.items()])
        n_e, n_h, n_o = (mx.array(norms[k]) for k in (NORM_E, NORM_H, NORM_OUT))
        same = right = 0
        for x, e, hn, choice in runs:
            T = x.shape[1]
            # Position i pairs token i+1 with the hidden state of position i.
            inp = mx.concatenate([mx.fast.rms_norm(e[:, 1:], n_e, eps), mx.fast.rms_norm(hn[:, :-1], n_h, eps)],
                                 axis=-1) @ mx.array(fc_w).T
            if window is None:
                mask = create_attention_mask(inp, None)
            else:
                i = mx.arange(T - 1)
                mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
            out = mx.fast.rms_norm(layer(inp, mask=mask, cache=None), n_o, eps)
            pred = mx.argmax(out @ head.T, axis=-1)[0]   # pred[i]: the head's guess for token i+2
            sl = slice(cli.skip, T - 2)
            same += mx.sum(pred[sl] == choice[1:][sl]).item()
            right += mx.sum(pred[sl] == x[0, 2:][sl]).item()
        print(f"  {label:44s} same as the model's choice {100 * same / n_pos:5.1f}%   "
              f"same as the text {100 * right / n_pos:5.1f}%")

    agreement(fc, lw, None, "float, full history")
    for w in (64, 8, 1):
        agreement(fc, lw, w, f"float, last {w} position{'s' if w > 1 else ''} only")

    def q4(W):
        codes, s, c = quantlib.quantize_rtn(W, None, "fp4", A_GRID, B_GRID)
        return quantlib.dequant(codes, s, c, "fp4")

    def q4_stack(names):
        W = q4(np.concatenate([lw[n] for n in names], axis=0))
        out, row = {}, 0
        for n in names:
            out[n] = W[row:row + lw[n].shape[0]]
            row += lw[n].shape[0]
        return out

    lq = dict(lw)
    for stack in (("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
                  ("self_attn.o_proj.weight",), ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
                  ("mlp.down_proj.weight",)):
        lq.update(q4_stack(stack))
    agreement(q4(fc), lq, None, "4-bit fp4 round-to-nearest, full history")
    agreement(q4(fc), lw, None, "only the input projection fc in 4 bits")
    agreement(fc, {**lq}, None, "only the decoder layer in 4 bits (round to nearest)")
    _, layer_g = quantize_head(args, fc, lw, x_cal, "fp4")
    lg = dict(lw)
    for name in lq:
        if name.endswith("proj.weight"):
            mod = layer_g
            for attr in name.split(".")[:-1]:
                mod = getattr(mod, attr)
            lg[name] = np.array(mod.weight)
    agreement(fc, lg, None, "decoder layer in 4 bits with GPTQ, fc in half precision")

if __name__ == "__main__":
    main()
