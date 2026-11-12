"""Reads tensors back out of a native .rlm file, as float matrices."""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import numpy as np

import quantlib
from convert import BLOCK, MAGIC


class Native:
    """Reads tensors back out of a native .rlm file."""

    def __init__(self, path, hdr=None):
        """`hdr` supplies the header for a file that has none yet (an interrupted conversion)."""
        if hdr is None:
            with open(path, "rb") as f:
                if f.read(4) != MAGIC:
                    raise SystemExit(f"{path} is not a raceline model file")
                _, n = struct.unpack("<IQ", f.read(12))
                hdr = json.loads(f.read(n))
        self.hdr = hdr
        if self.hdr["quant"].get("mode") != "native":
            raise SystemExit("the MTP head is only supported for native-format files (tools/quantize.py)")
        self.book = self.hdr["quant"]["book"]
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def floats(self, name) -> np.ndarray:
        t = self.hdr["tensors"][name]
        return np.frombuffer(self.mm, dtype=np.float32, count=int(np.prod(t["shape"])), offset=t["offset"]).copy()

    def _scales(self, t, n):
        return (np.frombuffer(self.mm, dtype=np.float32, count=n, offset=t["s"]),
                np.frombuffer(self.mm, dtype=np.float32, count=t["k"], offset=t["c"]))

    def column_scales(self, name) -> np.ndarray:
        """The scale per input column of a native matrix."""
        t = self.hdr["tensors"][name]
        return np.frombuffer(self.mm, dtype=np.float32, count=t["k"], offset=t["c"]).copy()

    def row_codes(self, name):
        """A row-major matrix as stored -> (codes [N][K] uint8, row scales, column scales)."""
        t = self.hdr["tensors"][name]
        b = np.asarray(self.mm[t["w"]:t["w"] + t["n"] * (t["k"] // 2)]).reshape(t["n"], t["k"] // 2)
        codes = np.empty((t["n"], t["k"]), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = b & 15, b >> 4
        s, c = self._scales(t, t["n"])
        return codes, s.copy(), c.copy()

    def rows(self, name, ids) -> np.ndarray:
        """Rows `ids` of a row-major matrix (the token table)."""
        t = self.hdr["tensors"][name]
        half = t["k"] // 2
        ids = np.asarray(ids, dtype=np.int64)
        b = self.mm[t["w"] + ids[:, None] * half + np.arange(half)[None, :]]
        codes = np.empty((len(ids), t["k"]), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = b & 15, b >> 4
        s, c = self._scales(t, t["n"])
        return quantlib.dequant(codes, s[ids], c, self.book)

    def block_rows(self, name, r0, r1) -> np.ndarray:
        """Rows r0..r1 (whole blocks) of a kernel-layout matrix."""
        t = self.hdr["tensors"][name]
        k, b0, b1 = t["k"], r0 // BLOCK, -(-r1 // BLOCK)
        per = k * (BLOCK // 2)
        b = np.asarray(self.mm[t["w"] + b0 * per:t["w"] + b1 * per]).reshape(b1 - b0, k, BLOCK // 2)
        codes = np.empty((b1 - b0, k, BLOCK), dtype=np.uint8)
        codes[:, :, 0::2], codes[:, :, 1::2] = b & 15, b >> 4
        codes = codes.transpose(0, 2, 1).reshape((b1 - b0) * BLOCK, k)
        s, c = self._scales(t, t["n_pad"])
        return quantlib.dequant(codes, s[b0 * BLOCK:b1 * BLOCK], c, self.book)[:r1 - r0]

    def matrix(self, name) -> np.ndarray:
        """A whole matrix [N][K], whatever its storage."""
        t = self.hdr["tensors"][name]
        if t["kind"] == "q4n":
            return self.block_rows(name, 0, t["n"])
        if t["kind"] == "q4n_rows":
            return self.rows(name, np.arange(t["n"]))
        if t["kind"] == "f16b":
            blocks = t["n_pad"] // BLOCK
            w = np.frombuffer(self.mm, dtype=np.float16, count=blocks * t["k"] * BLOCK, offset=t["w"])
            return w.reshape(blocks, t["k"], BLOCK).transpose(0, 2, 1).reshape(t["n_pad"], t["k"])[:t["n"]].astype(np.float32)
        raise ValueError(f"{name}: kind {t['kind']}")

    def part(self, name, part) -> np.ndarray:
        """One of the row-stacked matrices of a tensor."""
        p = self.hdr["tensors"][name]["parts"][part]
        return self.matrix(name)[p["offset"]:p["offset"] + p["n"]]

    def head_argmax(self, x) -> np.ndarray:
        """argmax over the vocabulary of x @ lm_head^T, x [n, H]; the head is decoded a slice at a time."""
        n_rows = self.hdr["tensors"]["lm_head"]["n"]
        best = mx.full((x.shape[0],), -mx.inf)
        arg = mx.zeros((x.shape[0],), dtype=mx.int32)
        for r0 in range(0, n_rows, 128 * BLOCK):
            r1 = min(n_rows, r0 + 128 * BLOCK)
            z = x @ mx.array(self.block_rows("lm_head", r0, r1)).T
            m, a = mx.max(z, axis=-1), mx.argmax(z, axis=-1).astype(mx.int32) + r0
            arg = mx.where(m > best, a, arg)
            best = mx.maximum(best, m)
            mx.eval(arg, best)
        return np.array(arg)
