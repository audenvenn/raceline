#!/usr/bin/env python3
"""A tiny Qwen3.5 mixture-of-experts checkpoint with random weights, for testing.

Writes a model directory in the original checkpoint's format (fused expert
tensors, a shared expert, norm weights stored minus one, conv weights as
[C, 1, taps]) at a size that converts in a minute: 4 layers (3 Gated DeltaNet,
1 attention), hidden 512, 32 experts with 4 routed per token. The tokenizer and
chat template are copied from a real model directory, so the vocabulary is real.

  .venv/bin/python tests/tiny_moe.py models/Qwen3.5-4B build/tiny-moe
  .venv/bin/python tools/quantize.py build/tiny-moe build/tiny-moe.rlm --calib-seqs 4 --eval-seqs 2
  .venv/bin/python tests/verify_moe.py build/tiny-moe.rlm build/ref/tokens_96.i32
"""

import argparse
import json
import os
import shutil

import numpy as np

H, LAYERS, E, TOPK, I = 512, 4, 32, 4, 256
HK, HV, DK, DV = 2, 4, 128, 128          # linear attention heads and head dims
HEADS, KV, HD = 4, 2, 256                # full attention


def tensors(vocab, rng):
    def w(*shape, scale=None):
        return (rng.standard_normal(shape) * (scale or shape[-1] ** -0.5)).astype(np.float32)

    def norm(n):   # stored minus one, as in the original checkpoints
        return (rng.standard_normal(n) * 0.1).astype(np.float32)

    t = {"model.language_model.embed_tokens.weight": w(vocab, H, scale=0.5),
         "model.language_model.norm.weight": norm(H),
         "lm_head.weight": w(vocab, H)}
    for i in range(LAYERS):
        p = f"model.language_model.layers.{i}."
        t[p + "input_layernorm.weight"] = norm(H)
        t[p + "post_attention_layernorm.weight"] = norm(H)
        if (i + 1) % 4:
            conv = 2 * HK * DK + HV * DV
            t[p + "linear_attn.in_proj_qkv.weight"] = w(conv, H)
            t[p + "linear_attn.in_proj_z.weight"] = w(HV * DV, H)
            t[p + "linear_attn.in_proj_b.weight"] = w(HV, H)
            t[p + "linear_attn.in_proj_a.weight"] = w(HV, H)
            t[p + "linear_attn.conv1d.weight"] = w(conv, 1, 4, scale=0.5)
            t[p + "linear_attn.A_log"] = np.log(rng.uniform(1, 16, HV)).astype(np.float32)
            t[p + "linear_attn.dt_bias"] = np.ones(HV, dtype=np.float32)
            t[p + "linear_attn.norm.weight"] = np.ones(DV, dtype=np.float32)   # this norm is stored as is
            t[p + "linear_attn.out_proj.weight"] = w(H, HV * DV)
        else:
            t[p + "self_attn.q_proj.weight"] = w(2 * HEADS * HD, H)
            t[p + "self_attn.k_proj.weight"] = w(KV * HD, H)
            t[p + "self_attn.v_proj.weight"] = w(KV * HD, H)
            t[p + "self_attn.o_proj.weight"] = w(H, HEADS * HD)
            t[p + "self_attn.q_norm.weight"] = norm(HD)
            t[p + "self_attn.k_norm.weight"] = norm(HD)
        t[p + "mlp.gate.weight"] = w(E, H, scale=0.2)
        t[p + "mlp.experts.gate_up_proj"] = w(E, 2 * I, H)
        t[p + "mlp.experts.down_proj"] = w(E, H, I)
        t[p + "mlp.shared_expert.gate_proj.weight"] = w(I, H)
        t[p + "mlp.shared_expert.up_proj.weight"] = w(I, H)
        t[p + "mlp.shared_expert.down_proj.weight"] = w(H, I)
        t[p + "mlp.shared_expert_gate.weight"] = w(1, H)
    # The MTP head: an attention layer whose MLP is a MoE with its experts stored one by one.
    t["mtp.fc.weight"] = w(H, 2 * H)
    for name in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"):
        t[name] = norm(H)
    p = "mtp.layers.0."
    t[p + "input_layernorm.weight"] = norm(H)
    t[p + "post_attention_layernorm.weight"] = norm(H)
    t[p + "self_attn.q_proj.weight"] = w(2 * HEADS * HD, H)
    t[p + "self_attn.k_proj.weight"] = w(KV * HD, H)
    t[p + "self_attn.v_proj.weight"] = w(KV * HD, H)
    t[p + "self_attn.o_proj.weight"] = w(H, HEADS * HD)
    t[p + "self_attn.q_norm.weight"] = norm(HD)
    t[p + "self_attn.k_norm.weight"] = norm(HD)
    t[p + "mlp.gate.weight"] = w(E, H, scale=0.2)
    for e in range(E):
        t[p + f"mlp.experts.{e}.gate_proj.weight"] = w(I, H)
        t[p + f"mlp.experts.{e}.up_proj.weight"] = w(I, H)
        t[p + f"mlp.experts.{e}.down_proj.weight"] = w(H, I)
    t[p + "mlp.shared_expert.gate_proj.weight"] = w(I, H)
    t[p + "mlp.shared_expert.up_proj.weight"] = w(I, H)
    t[p + "mlp.shared_expert.down_proj.weight"] = w(H, I)
    t[p + "mlp.shared_expert_gate.weight"] = w(1, H)
    return t


def save_safetensors(path, tensors):
    """bfloat16 safetensors, written by hand (no torch here)."""
    header, blobs, offset = {}, [], 0
    for name, a in tensors.items():
        u = a.astype(np.float32).view(np.uint32)
        bf = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)     # round to nearest even
        data = bf.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(a.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer_from", help="a model directory to copy the tokenizer and chat template from")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(args.tokenizer_from, name)):
            shutil.copy(os.path.join(args.tokenizer_from, name), args.out)
    vocab = json.load(open(os.path.join(args.tokenizer_from, "config.json"))).get("text_config", {}).get("vocab_size")
    vocab = vocab or 248320
    text = dict(model_type="qwen3_5_moe_text", hidden_size=H, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                num_key_value_heads=KV, head_dim=HD, linear_num_key_heads=HK, linear_num_value_heads=HV,
                linear_key_head_dim=DK, linear_value_head_dim=DV, linear_conv_kernel_dim=4, vocab_size=vocab,
                rms_norm_eps=1e-6, full_attention_interval=4, tie_word_embeddings=False, attn_output_gate=True,
                num_experts=E, num_experts_per_tok=TOPK, moe_intermediate_size=I, shared_expert_intermediate_size=I,
                mtp_num_hidden_layers=1, eos_token_id=248044,
                rope_parameters=dict(rope_type="default", rope_theta=10000000, partial_rotary_factor=0.25,
                                     mrope_interleaved=True, mrope_section=[11, 11, 10]))
    json.dump(dict(model_type="qwen3_5_moe", architectures=["Qwen3_5MoeForConditionalGeneration"], text_config=text,
                   tie_word_embeddings=False), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    save_safetensors(os.path.join(args.out, "model.safetensors"), tensors(vocab, np.random.default_rng(args.seed)))
    print(f"{args.out}: {LAYERS} layers, {E} experts ({TOPK} per token), hidden {H}, vocabulary {vocab}")


if __name__ == "__main__":
    main()
