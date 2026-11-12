#!/usr/bin/env python3
"""Check the engine on a mixture-of-experts model file against MLX running the same weights.

Like tests/verify_native.py, but the reference is built from the .rlm file
alone: its header gives the configuration and every tensor the engine uses is
in the file (quantized matrices decoded to float, the router, norms, the
DeltaNet constants). That way a MoE model too large to load in float can be
checked on a truncated conversion of its first layers:

  .venv/bin/python tools/quantize.py models/Qwen3.5-35B-A3B build/moe4.rlm --layers 4
  .venv/bin/python tests/verify_moe.py build/moe4.rlm build/ref/tokens_96.i32
"""

import argparse
import os
import subprocess
import sys
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen3_5

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from rlmfile import Native  # noqa: E402

PARTS = {  # file tensor -> {part: module path below the layer}
    "lin.in_proj": {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "b": "linear_attn.in_proj_b",
                    "a": "linear_attn.in_proj_a"},
    "lin.out_proj": {"out": "linear_attn.out_proj"},
    "attn.qkv": {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj"},
    "attn.o_proj": {"out": "self_attn.o_proj"},
}


def module(root, path):
    for attr in path.split("."):
        root = getattr(root, attr)
    return root


def build(rlm):
    """mlx-lm's Qwen3.5 text model holding exactly what the file holds."""
    nat = Native(rlm)
    cfg = dict(nat.hdr["config"])
    rope = dict(rope_theta=cfg.pop("rope_theta"), partial_rotary_factor=cfg.pop("partial_rotary_factor"),
                type="default")
    cfg.pop("eos_token_id", None)
    args = qwen3_5.TextModelArgs.from_dict(dict(cfg, rope_parameters=rope, tie_word_embeddings=False,
                                                shared_expert_intermediate_size=cfg.get("moe_intermediate_size", 0)))
    lm = qwen3_5.TextModel(args)
    f16 = lambda a: mx.array(np.asarray(a, dtype=np.float32).astype(np.float16))  # noqa: E731
    for i, layer in enumerate(lm.model.layers):
        pre = f"layers.{i}."
        for name, parts in PARTS.items():
            t = nat.hdr["tensors"].get(pre + name)
            if t is None:
                continue
            W = nat.matrix(pre + name)
            for part, path in parts.items():
                p = t["parts"][part]
                module(layer, path).weight = f16(W[p["offset"]:p["offset"] + p["n"]])
        layer.input_layernorm.weight = mx.array(nat.floats(pre + "in_norm"))
        layer.post_attention_layernorm.weight = mx.array(nat.floats(pre + "post_norm"))
        if layer.is_linear:
            a = layer.linear_attn
            conv = nat.floats(pre + "lin.conv").reshape(nat.hdr["tensors"][pre + "lin.conv"]["shape"])
            a.conv1d.weight = mx.array(conv.reshape(conv.shape[0], conv.shape[1], 1))
            a.A_log = mx.array(nat.floats(pre + "lin.A_log"))
            a.dt_bias = mx.array(nat.floats(pre + "lin.dt_bias"))
            a.norm.weight = mx.array(nat.floats(pre + "lin.norm"))
        else:
            layer.self_attn.q_norm.weight = mx.array(nat.floats(pre + "attn.q_norm"))
            layer.self_attn.k_norm.weight = mx.array(nat.floats(pre + "attn.k_norm"))
        moe = layer.mlp
        E, I, H = args.num_experts, args.moe_intermediate_size, args.hidden_size
        gu = nat.matrix(pre + "moe.gate_up")                  # [(E + 1) * 2I, H], the shared expert last
        ex = gu[:E * 2 * I].reshape(E, 2 * I, H)
        moe.switch_mlp.gate_proj.weight, moe.switch_mlp.up_proj.weight = f16(ex[:, :I]), f16(ex[:, I:])
        moe.shared_expert.gate_proj.weight = f16(gu[E * 2 * I:E * 2 * I + I])
        moe.shared_expert.up_proj.weight = f16(gu[E * 2 * I + I:])
        dn = nat.matrix(pre + "moe.down")                     # [(E + 1) * H, I]
        moe.switch_mlp.down_proj.weight = f16(dn[:E * H].reshape(E, H, I))
        moe.shared_expert.down_proj.weight = f16(dn[E * H:])
        # The router reads the norm's output times the gate/up stack's column
        # scales in the engine and has them divided out in the file.
        c = nat.column_scales(pre + "moe.gate_up")
        router = nat.matrix(pre + "moe.router")[:E + 1] * c[None, :]
        moe.gate.weight = f16(router[:E])
        moe.shared_expert_gate.weight = f16(router[E:E + 1])
    lm.model.norm.weight = mx.array(nat.floats("final_norm"))
    lm.model.embed_tokens.weight = f16(nat.matrix("embed"))
    head = nat.matrix("lm_head")
    lm.lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False)
    lm.lm_head.weight = f16(head)
    mx.eval(lm.parameters())
    return lm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rlm")
    ap.add_argument("tokens")
    ap.add_argument("--raceline", default="build/cmake/raceline-engine")
    args = ap.parse_args()

    lm = build(args.rlm)
    tokens = np.fromfile(args.tokens, dtype=np.int32)
    x = mx.array(tokens.tolist())[None]
    ref = np.array(lm(x, input_embeddings=lm.model.embed_tokens(x).astype(mx.float32))[0, -1].astype(mx.float32))
    del lm
    mx.clear_cache()

    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        r = subprocess.run([args.raceline, "logits", args.rlm, args.tokens, tmp.name], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"engine failed:\n{r.stdout}\n{r.stderr}")
        eng = np.fromfile(tmp.name, dtype=np.float32)

    top_r, top_e = np.argsort(-ref)[:5], np.argsort(-eng)[:5]
    rel = np.linalg.norm(eng - ref) / np.linalg.norm(ref)
    same = top_r[0] == top_e[0]
    print(f"logits: relative error {rel:.2e}, max|d| {np.abs(eng - ref).max():.3f}, argmax {int(top_e[0])} "
          f"({'same' if same else 'DIFFERENT'} as MLX {int(top_r[0])}), top-5 overlap {len(set(top_r) & set(top_e))}/5")
    ok = same and rel < 2e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
