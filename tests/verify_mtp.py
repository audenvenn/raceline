#!/usr/bin/env python3
"""Check the engine's MTP head against MLX running the same weights.

The file's weights (main model and head) are decoded to float matrices and
evaluated in MLX with float32 activations: the main model for the hidden
states, then the head as the reference implementation describes it,

    x_i = fc([norm_e(embed(token_{i+1})), norm_h(final_norm(hidden_i))])
    logits_i = lm_head(norm(decoder_layer(x)_i))        # a guess for token i+2

and the logits for the last position are compared with the engine's.

  .venv/bin/python tests/verify_mtp.py models/Qwen3.5-4B models/qwen3.5-4b.fp4.rlm build/ref/tokens_96.i32
"""

import argparse
import os
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_attention_mask

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rlmfile import Native  # noqa: E402
from verify_native import load_decoded  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("float_model")
    ap.add_argument("rlm")
    ap.add_argument("tokens")
    ap.add_argument("--raceline", default="build/cmake/raceline-engine")
    args = ap.parse_args()

    nat = Native(args.rlm)
    if "mtp.fc" not in nat.hdr["tensors"]:
        sys.exit(f"{args.rlm} has no MTP head (tools/add_mtp.py adds one)")
    model = load_decoded(args.float_model, args.rlm)
    lm = model.language_model
    tm, cfg = lm.model, lm.args
    eps = cfg.rms_norm_eps
    tokens = np.fromfile(args.tokens, dtype=np.int32)
    ids = mx.array(tokens.tolist())[None]

    e = tm.embed_tokens(ids).astype(mx.float32)
    h = e[:, :-1]
    for layer in tm.layers:
        h = layer(h, mask=None if layer.is_linear else create_attention_mask(h, None), cache=None)
    hn = tm.norm(h)

    def norm(x, name):
        return mx.fast.rms_norm(x, mx.array(nat.floats(name)), eps)

    head_layer = qwen3_5.DecoderLayer(cfg, cfg.full_attention_interval - 1)
    head_layer.load_weights([(k, mx.array(v)) for k, v in {
        "input_layernorm.weight": nat.floats("mtp.in_norm"),
        "post_attention_layernorm.weight": nat.floats("mtp.post_norm"),
        "self_attn.q_norm.weight": nat.floats("mtp.attn.q_norm"),
        "self_attn.k_norm.weight": nat.floats("mtp.attn.k_norm"),
        "self_attn.q_proj.weight": nat.part("mtp.attn.qkv", "q"),
        "self_attn.k_proj.weight": nat.part("mtp.attn.qkv", "k"),
        "self_attn.v_proj.weight": nat.part("mtp.attn.qkv", "v"),
        "self_attn.o_proj.weight": nat.part("mtp.attn.o_proj", "out"),
        "mlp.gate_proj.weight": nat.part("mtp.mlp.gate_up", "gate"),
        "mlp.up_proj.weight": nat.part("mtp.mlp.gate_up", "up"),
        "mlp.down_proj.weight": nat.part("mtp.mlp.down", "out"),
    }.items()])
    x = mx.concatenate([norm(e[:, 1:], "mtp.norm_e"), norm(hn, "mtp.norm_h")], axis=-1) @ mx.array(nat.matrix("mtp.fc")).T
    out = norm(head_layer(x, mask=create_attention_mask(x, None), cache=None), "mtp.norm")
    head = (tm.embed_tokens.weight if cfg.tie_word_embeddings else lm.lm_head.weight).astype(mx.float32)
    ref = np.array(out[0, -1] @ head.T)
    del model
    mx.clear_cache()

    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        r = subprocess.run([args.raceline, "draft", args.rlm, args.tokens, tmp.name], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"engine failed:\n{r.stdout}\n{r.stderr}")
        eng = np.fromfile(tmp.name, dtype=np.float32)

    top_r, top_e = np.argsort(-ref)[:5], np.argsort(-eng)[:5]
    rel = np.linalg.norm(eng - ref) / np.linalg.norm(ref)
    same = top_r[0] == top_e[0]
    print(f"MTP logits: relative error {rel:.2e}, max|d| {np.abs(eng - ref).max():.3f}, draft {int(top_e[0])} "
          f"({'same' if same else 'DIFFERENT'} as MLX {int(top_r[0])}), top-5 overlap {len(set(top_r) & set(top_e))}/5")
    ok = same and rel < 3e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
