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
MODEL_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
               "added_tokens.json", "generation_config.json")


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

def to_bf16(x: np.ndarray, what: str) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16). The source values are bfloat16
    widened to float32, so this must be lossless; anything else is an error."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    if (bits & 0xFFFF).any():
        raise SystemExit(f"{what}: values are not exactly representable in bfloat16")
    return (bits >> 16).astype(np.uint16)


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

    Returns (w bytes [blocks][K][128], s bf16 [blocks][G][256], b bf16 [blocks][G][256],
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
    s_all = np.zeros((n_pad, g), dtype=np.uint16)   # bfloat16 bit patterns
    b_all = np.zeros((n_pad, g), dtype=np.uint16)
    for (off, rows), (q, s, b) in zip(offsets, parts):
        q_all[off:off + rows] = q
        s_all[off:off + rows] = to_bf16(s, "scales")
        b_all[off:off + rows] = to_bf16(b, "biases")
    blocks = n_pad // BLOCK
    qb = q_all.reshape(blocks, BLOCK, k).transpose(0, 2, 1)             # [blocks][K][256]
    w = (qb[:, :, 0::2] | (qb[:, :, 1::2] << 4)).astype(np.uint8)       # even element in the low nibble
    s = np.ascontiguousarray(s_all.reshape(blocks, BLOCK, g).transpose(0, 2, 1))
    b = np.ascontiguousarray(b_all.reshape(blocks, BLOCK, g).transpose(0, 2, 1))
    return np.ascontiguousarray(w), s, b, n_pad, offsets


def pack_rows(q, s, b):
    """Row-major 4-bit matrix for row lookups (token embeddings)."""
    w = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(w), to_bf16(s, "scales"), to_bf16(b, "biases")


# -------------------------------------------------------------------- writer

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.header_room = 64 * PAGE          # 1 MiB; the header is patched in at the end
        self.f.write(b"\0" * self.header_room)
        self.tensors = {}
        self.files = {}
        self.sections = []
        self._sec = None

    @classmethod
    def append(cls, path, drop_from=None):
        """Reopens a finished file to add sections. Returns (writer, header
        without its section, tensor and file tables). `drop_from` names a
        section to discard first, together with everything after it."""
        with open(path, "rb") as f:
            if f.read(4) != MAGIC:
                raise SystemExit(f"{path} is not a redline model file")
            _, n = struct.unpack("<IQ", f.read(12))
            hdr = json.loads(f.read(n))
        w = cls.__new__(cls)
        w.header_room = 64 * PAGE
        w.tensors, w.sections, w.files, w._sec = hdr.pop("tensors"), hdr.pop("sections"), hdr.pop("files", {}), None
        w.f = open(path, "r+b")
        names = [s["name"] for s in w.sections]
        if drop_from in names:
            idx = names.index(drop_from)
            start = w.sections[idx]["offset"]
            w.f.truncate(start)
            w.tensors = {k: t for k, t in w.tensors.items() if t["section"] < idx}
            w.files = {k: f for k, f in w.files.items() if f["offset"] < start}
            del w.sections[idx:]
        w.f.seek(0, 2)
        return w, hdr

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

    def add_native(self, name, named_parts, codes, s, c, book):
        """Scale-free 4-bit matrix: codes [N][K] (uint8, one code per weight),
        a scale per row and a scale per column. Rows are padded to a whole
        block with scale 0, so padding never contributes."""
        n, k = codes.shape
        n_pad = -(-n // BLOCK) * BLOCK
        q = np.zeros((n_pad, k), dtype=np.uint8)
        q[:n] = codes
        blocks = n_pad // BLOCK
        qb = q.reshape(blocks, BLOCK, k).transpose(0, 2, 1)
        w = np.ascontiguousarray((qb[:, :, 0::2] | (qb[:, :, 1::2] << 4)).astype(np.uint8))
        s_pad = np.zeros(n_pad, dtype=np.float32)
        s_pad[:n] = s
        off = 0
        parts = {}
        for pn, rows in named_parts:
            parts[pn] = dict(offset=off, n=rows)
            off += rows
        self.tensors[name] = dict(kind="q4n", section=len(self.sections), book=book, k=k, n=n, n_pad=n_pad,
                                  w=self.blob(w), s=self.blob(s_pad), c=self.blob(c.astype(np.float32)), parts=parts)

    def add_dense(self, name, W):
        """Half-precision matrix [N][K] in the kernel's blocked layout, rows padded with zeros."""
        n, k = W.shape
        n_pad = -(-n // BLOCK) * BLOCK
        full = np.zeros((n_pad, k), dtype=np.float16)
        full[:n] = W
        w = np.ascontiguousarray(full.reshape(n_pad // BLOCK, BLOCK, k).transpose(0, 2, 1))
        self.tensors[name] = dict(kind="f16b", section=len(self.sections), k=k, n=n, n_pad=n_pad, w=self.blob(w),
                                  parts={"out": dict(offset=0, n=n)})

    def add_native_rows(self, name, codes, s, c, book):
        """Row-major variant for row lookups (token embeddings)."""
        w = np.ascontiguousarray((codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8))
        self.tensors[name] = dict(kind="q4n_rows", section=len(self.sections), book=book, k=codes.shape[1],
                                  n=codes.shape[0], w=self.blob(w), s=self.blob(s.astype(np.float32)),
                                  c=self.blob(c.astype(np.float32)))

    def add_rows(self, name, q, s, b):
        w, s16, b16 = pack_rows(q, s, b)
        self.tensors[name] = dict(kind="q4a_rows", section=len(self.sections), k=q.shape[1], n=q.shape[0],
                                  w=self.blob(w), s=self.blob(s16), b=self.blob(b16))

    def add_files(self, src_dir):
        """Embeds the tokenizer, chat template and sampling defaults of the
        model directory, so that the file runs on its own."""
        self.begin("files")
        for name in MODEL_FILES:
            path = os.path.join(src_dir, name)
            if os.path.isfile(path):
                data = np.fromfile(path, dtype=np.uint8)
                self.files[name] = dict(offset=self.blob(data), nbytes=int(data.size))
        self.end()
        if "tokenizer.json" not in self.files:
            raise SystemExit(f"{src_dir} has no tokenizer.json")

    def finish(self, meta):
        meta = dict(meta, sections=self.sections, tensors=self.tensors, files=self.files)
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
    out.add_files(args.src)

    out.finish(dict(format="redline", version=1, arch="qwen3_5", source=os.path.basename(os.path.normpath(args.src)),
                    config=cfg, quant=dict(bits=4, group=GROUP, block=BLOCK, mode="affine", scale_dtype="bf16")))
    size = os.path.getsize(args.dst)
    print(f"wrote {args.dst}: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
