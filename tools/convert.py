#!/usr/bin/env python3
"""Convert an MLX 4-bit Qwen3.5 / Qwen3.8 model into a redline model file (.rlm).

The quantized values are carried over unchanged (affine 4-bit, groups of 64:
w = scale * q + bias), so the converted model is numerically the same model MLX
runs. What changes is the memory layout, which is built for the GPU kernels:

  * "blocked" weights: output rows are cut into blocks of 256, and each block is
    stored as [in][256] nibbles (128 bytes per input index). A kernel
    threadgroup then reads one contiguous stream.
  * projections that share an input are stacked into one matrix
    (gate+up, q+k+v, the four linear-attention input projections), so one
    dispatch computes all of them.
  * every section is page aligned so the file can be mmap'd and handed to Metal
    without a copy.

  tools/convert.py models/Qwen3.5-4B-MLX-4bit models/qwen3.5-4b.rlm
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

PAGE = 16384
ALIGN = 128
BLOCK = 256
GROUP = 64
MAGIC = b"RLM1"


# --------------------------------------------------------------- safetensors

class SafeTensors:
    """Minimal reader (mmap) that also understands BF16, which numpy lacks."""

    def __init__(self, paths):
        self.index = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            mm = np.memmap(p, dtype=np.uint8, mode="r", offset=8 + n)
            for name, t in hdr.items():
                if name != "__metadata__":
                    self.index[name] = (mm, t)

    def __contains__(self, name):
        return name in self.index

    def get(self, name) -> np.ndarray:
        mm, t = self.index[name]
        a, b = t["data_offsets"]
        raw = mm[a:b]
        dt, shape = t["dtype"], t["shape"]
        if dt == "BF16":  # widen to float32 exactly
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        np_dt = {"F32": np.float32, "F16": np.float16, "U32": np.uint32, "U8": np.uint8, "I32": np.int32}[dt]
        return np.frombuffer(raw, dtype=np_dt).reshape(shape)


# ------------------------------------------------------------------- packing

def to_f16_exact(x: np.ndarray, what: str) -> np.ndarray:
    y = x.astype(np.float16)
    if not np.array_equal(y.astype(np.float32), x.astype(np.float32)):
        bad = float(np.abs(y.astype(np.float32) - x).max())
        raise SystemExit(f"{what}: values are not exactly representable in float16 (max err {bad:g})")
    return y


def unpack_nibbles(w_u32: np.ndarray) -> np.ndarray:
    """[N, K/8] uint32 -> [N, K] uint8, element i in bits 4i..4i+3 of each word."""
    n = w_u32.shape[0]
    b = w_u32.view(np.uint8).reshape(n, -1)          # little endian bytes
    out = np.empty((n, b.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = b & 0x0F
    out[:, 1::2] = b >> 4
    return out


def pack_blocked(parts):
    """parts: list of (q[N,K] uint8, scales[N,G] f32, biases[N,G] f32) sharing K.

    Returns (w bytes [blocks][K][128], s f16 [blocks][G][256], b f16 [blocks][G][256],
             n_pad, [(row offset, rows) per part]).
    """
    k = parts[0][0].shape[1]
    g = k // GROUP
    offsets, n = [], 0
    for q, _, _ in parts:
        offsets.append((n, q.shape[0]))
        n += q.shape[0]
    n_pad = -(-n // BLOCK) * BLOCK
    q_all = np.zeros((n_pad, k), dtype=np.uint8)
    s_all = np.zeros((n_pad, g), dtype=np.float16)
    b_all = np.zeros((n_pad, g), dtype=np.float16)
    for (off, rows), (q, s, b) in zip(offsets, parts):
        q_all[off:off + rows] = q
        s_all[off:off + rows] = to_f16_exact(s, "scales")
        b_all[off:off + rows] = to_f16_exact(b, "biases")
    blocks = n_pad // BLOCK
    qb = q_all.reshape(blocks, BLOCK, k).transpose(0, 2, 1)             # [blocks][K][256]
    w = (qb[:, :, 0::2] | (qb[:, :, 1::2] << 4)).astype(np.uint8)       # even element in the low nibble
    s = np.ascontiguousarray(s_all.reshape(blocks, BLOCK, g).transpose(0, 2, 1))
    b = np.ascontiguousarray(b_all.reshape(blocks, BLOCK, g).transpose(0, 2, 1))
    return np.ascontiguousarray(w), s, b, n_pad, offsets


def pack_rows(q, s, b):
    """Row-major 4-bit matrix for row lookups (token embeddings)."""
    w = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(w), to_f16_exact(s, "scales"), to_f16_exact(b, "biases")


# -------------------------------------------------------------------- writer

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.header_room = 64 * PAGE          # 1 MiB; the header is patched in at the end
        self.f.write(b"\0" * self.header_room)
        self.tensors = {}
        self.sections = []
        self._sec = None

    def _pad(self, align):
        pos = self.f.tell()
        pad = (-pos) % align
        if pad:
            self.f.write(b"\0" * pad)
        return pos + pad

    def begin(self, name):
        off = self._pad(PAGE)
        self._sec = dict(name=name, offset=off)

    def end(self):
        end = self._pad(PAGE)
        self._sec["nbytes"] = end - self._sec["offset"]
        self.sections.append(self._sec)
        self._sec = None

    def blob(self, arr: np.ndarray) -> int:
        off = self._pad(ALIGN)
        self.f.write(memoryview(np.ascontiguousarray(arr)).cast("B"))
        return off

    def add_float(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.tensors[name] = dict(kind="f32", section=len(self.sections), shape=list(arr.shape), offset=self.blob(arr))

    def add_blocked(self, name, named_parts):
        w, s, b, n_pad, offs = pack_blocked([p for _, p in named_parts])
        k = named_parts[0][1][0].shape[1]
        self.tensors[name] = dict(
            kind="q4a", section=len(self.sections), k=k, n=sum(r for _, r in offs), n_pad=n_pad,
            w=self.blob(w), s=self.blob(s), b=self.blob(b),
            parts={pn: dict(offset=o, n=r) for (pn, _), (o, r) in zip(named_parts, offs)})

    def add_rows(self, name, q, s, b):
        w, s16, b16 = pack_rows(q, s, b)
        self.tensors[name] = dict(kind="q4a_rows", section=len(self.sections), k=q.shape[1], n=q.shape[0],
                                  w=self.blob(w), s=self.blob(s16), b=self.blob(b16))

    def finish(self, meta):
        meta = dict(meta, sections=self.sections, tensors=self.tensors)
        hdr = json.dumps(meta, separators=(",", ":")).encode()
        need = 16 + len(hdr)
        if need > self.header_room:
            raise SystemExit(f"header ({need} bytes) does not fit the reserved {self.header_room}")
        self.f.seek(0)
        self.f.write(MAGIC + struct.pack("<IQ", 1, len(hdr)) + hdr)
        self.f.close()


# ------------------------------------------------------------------- convert

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="MLX model directory (4-bit affine, group 64)")
    ap.add_argument("dst", help="output .rlm file")
    args = ap.parse_args()

    cfg_all = json.load(open(os.path.join(args.src, "config.json")))
    q = cfg_all.get("quantization") or {}
    if (q.get("bits"), q.get("group_size"), q.get("mode", "affine")) != (4, GROUP, "affine"):
        raise SystemExit(f"expected 4-bit affine group-{GROUP} quantization, found {q}")
    tc = cfg_all.get("text_config", cfg_all)
    if tc.get("num_experts", 0):
        raise SystemExit("MoE variants are not supported yet")
    rope = tc.get("rope_parameters") or {}
    cfg = {k: tc[k] for k in (
        "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
        "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size", "rms_norm_eps", "full_attention_interval")}
    cfg["rope_theta"] = float(rope.get("rope_theta", tc.get("rope_theta", 1e7)))
    cfg["partial_rotary_factor"] = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
    cfg["eos_token_id"] = tc.get("eos_token_id")
    tied = bool(tc.get("tie_word_embeddings", cfg_all.get("tie_word_embeddings", False)))
    cfg["tie_word_embeddings"] = tied
    # The 4B checkpoint ships a 1-layer MTP head config too, but MLX conversions drop those weights.

    files = sorted(os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith(".safetensors"))
    st = SafeTensors(files)
    P = "language_model.model."

    def lin(name):
        """(q[N,K], scales[N,G], biases[N,G]) for a quantized linear layer."""
        return unpack_nibbles(st.get(name + ".weight")), st.get(name + ".scales"), st.get(name + ".biases")

    t0 = time.time()
    out = Writer(args.dst)
    n_layers = cfg["num_hidden_layers"]

    out.begin("embed")
    out.add_rows("embed", *lin(P + "embed_tokens"))
    out.end()

    for i in range(n_layers):
        L = f"{P}layers.{i}."
        out.begin(f"layer.{i}")
        out.add_float(f"layers.{i}.in_norm", st.get(L + "input_layernorm.weight"))
        out.add_float(f"layers.{i}.post_norm", st.get(L + "post_attention_layernorm.weight"))
        if (i + 1) % cfg["full_attention_interval"] != 0:
            A = L + "linear_attn."
            out.add_blocked(f"layers.{i}.lin.in_proj", [
                ("qkv", lin(A + "in_proj_qkv")), ("z", lin(A + "in_proj_z")),
                ("b", lin(A + "in_proj_b")), ("a", lin(A + "in_proj_a"))])
            conv = st.get(A + "conv1d.weight")                      # [channels, taps, 1]
            out.add_float(f"layers.{i}.lin.conv", conv.reshape(conv.shape[0], conv.shape[1]))
            out.add_float(f"layers.{i}.lin.A_log", st.get(A + "A_log"))
            out.add_float(f"layers.{i}.lin.dt_bias", st.get(A + "dt_bias"))
            out.add_float(f"layers.{i}.lin.norm", st.get(A + "norm.weight"))
            out.add_blocked(f"layers.{i}.lin.out_proj", [("out", lin(A + "out_proj"))])
        else:
            A = L + "self_attn."
            out.add_blocked(f"layers.{i}.attn.qkv", [
                ("q", lin(A + "q_proj")), ("k", lin(A + "k_proj")), ("v", lin(A + "v_proj"))])
            out.add_float(f"layers.{i}.attn.q_norm", st.get(A + "q_norm.weight"))
            out.add_float(f"layers.{i}.attn.k_norm", st.get(A + "k_norm.weight"))
            out.add_blocked(f"layers.{i}.attn.o_proj", [("out", lin(A + "o_proj"))])
        out.add_blocked(f"layers.{i}.mlp.gate_up", [("gate", lin(L + "mlp.gate_proj")), ("up", lin(L + "mlp.up_proj"))])
        out.add_blocked(f"layers.{i}.mlp.down", [("out", lin(L + "mlp.down_proj"))])
        out.end()
        print(f"\r  layer {i + 1}/{n_layers}", end="", flush=True)
    print()

    out.begin("head")
    out.add_float("final_norm", st.get(P + "norm.weight"))
    head = P + "embed_tokens" if tied else "language_model.lm_head"
    out.add_blocked("lm_head", [("out", lin(head))])
    out.end()

    out.finish(dict(format="redline", version=1, arch="qwen3_5", source=os.path.basename(os.path.normpath(args.src)),
                    config=cfg, quant=dict(bits=4, group=GROUP, block=BLOCK, mode="affine")))
    size = os.path.getsize(args.dst)
    print(f"wrote {args.dst}: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
