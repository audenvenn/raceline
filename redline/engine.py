"""Backend for the redline engine (.rlm models), through its C API.

The engine returns logits; sampling happens here in numpy. The tokenizer and
chat template are not part of a .rlm file: they are read from REDLINE_TOKENIZER,
or from the model directory the file was converted from if it sits next to it.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct

import numpy as np

from .backends import Backend, Sampling, Snapshot


def _library() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("REDLINE_LIB"), os.path.join(here, "libredline.dylib"),
                  os.path.join(here, "..", "build", "cmake", "libredline.dylib")]
    for path in candidates:
        if path and os.path.isfile(path):
            lib = ctypes.CDLL(path)
            break
    else:
        raise RuntimeError("libredline.dylib not found: build it with `cmake --build build/cmake` "
                           "or point REDLINE_LIB at it")
    lib.rl_open.restype = ctypes.c_void_p
    lib.rl_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.rl_forward.restype = ctypes.POINTER(ctypes.c_float)
    lib.rl_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.rl_state_bytes.restype = ctypes.c_size_t
    lib.rl_last_gpu_seconds.restype = ctypes.c_double
    for fn in ("rl_close", "rl_vocab", "rl_position", "rl_reset", "rl_argmax", "rl_state_bytes",
               "rl_last_gpu_seconds"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.rl_save_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.rl_load_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return lib


def _tokenizer_dir(model_path: str) -> str:
    with open(model_path, "rb") as f:
        if f.read(4) != b"RLM1":
            raise ValueError(f"{model_path} is not a redline model file")
        _, n = struct.unpack("<IQ", f.read(12))
        source = json.loads(f.read(n)).get("source", "")
    folder = os.path.dirname(os.path.abspath(model_path))
    stem = os.path.basename(model_path).rsplit(".rlm", 1)[0]
    for d in (os.environ.get("REDLINE_TOKENIZER"), os.path.join(folder, stem), os.path.join(folder, source)):
        if d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise RuntimeError(f"no tokenizer for {model_path}: set REDLINE_TOKENIZER to a directory with tokenizer.json "
                       f"and the chat template (looked next to the model for '{stem}/' and '{source}/')")


class RedlineBackend(Backend):
    name = "redline"

    def __init__(self, model_path: str, max_ctx: int = 32768, chunk: int = 1024):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_dir(model_path))
        self.eos_ids = {i for i in (self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                        if i is not None and i >= 0}
        self.model_id = os.path.basename(model_path)
        self._lib = _library()
        self._e = self._lib.rl_open(os.path.abspath(model_path).encode(), max_ctx, chunk)
        self._vocab = self._lib.rl_vocab(self._e)
        self._chunk, self._max_ctx = chunk, max_ctx
        self._logits = None
        self.gpu_seconds = 0.0

    def __del__(self):
        if getattr(self, "_e", None):
            self._lib.rl_close(self._e)
            self._e = None

    @property
    def position(self) -> int:
        return self._lib.rl_position(self._e)

    def reset(self) -> None:
        self._lib.rl_reset(self._e)
        self._logits = None

    def _forward(self, tokens) -> None:
        if self.position + len(tokens) > self._max_ctx:
            raise RuntimeError(f"the context window of {self._max_ctx} tokens is full")
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        ptr = self._lib.rl_forward(self._e, arr, len(tokens))
        self.gpu_seconds += self._lib.rl_last_gpu_seconds(self._e)
        self._logits = np.ctypeslib.as_array(ptr, shape=(self._vocab,))   # a view, valid until the next forward

    def prefill(self, tokens, progress=None) -> None:
        for i in range(0, len(tokens), self._chunk):
            self._forward(tokens[i:i + self._chunk])
            if progress:
                progress(min(i + self._chunk, len(tokens)), len(tokens))

    def generate(self, sampling: Sampling, max_tokens: int, history: list):
        if self._logits is None:
            raise RuntimeError("generate() before any token was fed")
        rng = np.random.default_rng(sampling.seed)
        recent = list(history[-64:])
        for _ in range(max_tokens):
            token = _sample(self._logits, sampling, recent, rng)
            self._forward([token])
            recent.append(token)
            yield token

    def snapshot(self) -> Snapshot:
        state = np.empty(self._lib.rl_state_bytes(self._e), dtype=np.uint8)
        self._lib.rl_save_state(self._e, state.ctypes.data)
        logits = None if self._logits is None else self._logits.copy()
        return Snapshot(self.position, state, logits, state.nbytes)

    def restore(self, snap: Snapshot) -> None:
        self._lib.rl_load_state(self._e, snap.state.ctypes.data, snap.position)
        self._logits = snap.logits


def _sample(logits: np.ndarray, s: Sampling, recent: list, rng) -> int:
    z = logits
    if recent and (s.repetition_penalty != 1.0 or s.presence_penalty):
        z = z.copy()
        idx = np.unique(np.asarray(recent[-64:], dtype=np.int64))
        if s.repetition_penalty != 1.0:
            v = z[idx]
            z[idx] = np.where(v > 0, v / s.repetition_penalty, v * s.repetition_penalty)
        if s.presence_penalty:
            z[idx] -= s.presence_penalty
    if s.temperature <= 0:
        return int(np.argmax(z))
    k = min(s.top_k, z.size) if s.top_k > 0 else min(1024, z.size)   # the tail beyond 1024 carries no usable mass
    top = np.argpartition(z, -k)[-k:]
    top = top[np.argsort(-z[top])]
    p = (z[top] - z[top[0]]).astype(np.float64) / s.temperature
    p = np.exp(p)
    p /= p.sum()
    if 0 < s.top_p < 1:
        keep = int(np.searchsorted(np.cumsum(p), s.top_p)) + 1
        top, p = top[:keep], p[:keep] / p[:keep].sum()
    if s.min_p > 0:
        mask = p >= s.min_p * p[0]
        top, p = top[mask], p[mask] / p[mask].sum()
    return int(rng.choice(top, p=p))
