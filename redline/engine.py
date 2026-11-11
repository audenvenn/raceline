"""Backend for the redline engine (.rlm models), through its C API.

The engine returns logits; sampling happens here in numpy. The tokenizer and
chat template are not part of a .rlm file: they are read from REDLINE_TOKENIZER,
or from the model directory the file was converted from if it sits next to it.

When the file carries the model's multi-token-prediction head, generation is
speculative: the head drafts the token after next, the engine checks the next
token and the draft in one pass over the weights, and a draft that matches
what the model then picks is a second token for the price of one. The output
is exactly what one-token-at-a-time decoding produces. REDLINE_MTP=0 turns
this off.
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import struct
from typing import Optional

import numpy as np

from .backends import Backend, Sampling, Snapshot


@functools.lru_cache(maxsize=1)
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
    lib.rl_set_mtp.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rl_draft_logits.restype = ctypes.POINTER(ctypes.c_float)
    lib.rl_draft_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.rl_forward_draft.restype = ctypes.POINTER(ctypes.c_float)
    lib.rl_forward_draft.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.rl_logits_stride.restype = ctypes.c_size_t
    lib.rl_hidden.restype = ctypes.POINTER(ctypes.c_float)
    for fn in ("rl_has_mtp", "rl_reject_draft", "rl_logits_stride", "rl_hidden", "rl_hidden_size"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.rl_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
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

    def __init__(self, model_path: str, max_ctx: int = 32768, chunk: int = 1024, mtp: Optional[bool] = None):
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
        self._open_draft = False
        if mtp is None:
            mtp = os.environ.get("REDLINE_MTP", "1") != "0"
        self.mtp = bool(self._lib.rl_has_mtp(self._e)) and mtp
        if self._lib.rl_has_mtp(self._e):
            self._lib.rl_set_mtp(self._e, int(self.mtp))
        self._stride = self._lib.rl_logits_stride(self._e)
        self.gpu_seconds = 0.0
        self.counters = dict(passes=0, drafts=0, accepted=0)

    def __del__(self):
        if getattr(self, "_e", None):
            self._lib.rl_close(self._e)
            self._e = None

    @property
    def position(self) -> int:
        return self._lib.rl_position(self._e) - (1 if self._open_draft else 0)

    def reset(self) -> None:
        self._lib.rl_reset(self._e)
        self._logits = None
        self._open_draft = False

    def _settle(self) -> None:
        """Takes back a draft row that nobody confirmed (a generation stopped right after it was fed)."""
        if self._open_draft:
            self._lib.rl_reject_draft(self._e)
            self._open_draft = False

    def _forward(self, tokens) -> None:
        self._settle()
        if self.position + len(tokens) > self._max_ctx:
            raise RuntimeError(f"the context window of {self._max_ctx} tokens is full")
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        ptr = self._lib.rl_forward(self._e, arr, len(tokens))
        self.gpu_seconds += self._lib.rl_last_gpu_seconds(self._e)
        self.counters["passes"] += 1
        self._logits = np.ctypeslib.as_array(ptr, shape=(self._vocab,))   # a view, valid until the next forward

    def prefill(self, tokens, progress=None) -> None:
        for i in range(0, len(tokens), self._chunk):
            self._forward(tokens[i:i + self._chunk])
            if progress:
                progress(min(i + self._chunk, len(tokens)), len(tokens))

    def generate(self, sampling: Sampling, max_tokens: int, history: list):
        if self._logits is None:
            raise RuntimeError("generate() before any token was fed")
        self._settle()
        lib, e, V = self._lib, self._e, self._vocab
        seed = sampling.seed if sampling.seed is not None else int.from_bytes(os.urandom(7), "little")
        recent = list(history[-64:])
        pos = self.position                    # position of the token being chosen
        token = _sample(self._logits, sampling, recent, seed, pos)
        produced = 0
        try:
            while produced < max_tokens:
                if (not self.mtp or token in self.eos_ids or produced + 1 >= max_tokens
                        or self.position + 2 > self._max_ctx):
                    self._forward([token])
                    recent.append(token)
                    produced, pos = produced + 1, pos + 1
                    yield token
                    token = _sample(self._logits, sampling, recent, seed, pos)
                    continue

                # The head's guess for the token after `token`, drawn with the
                # random numbers the real choice at that position will use.
                guess = np.ctypeslib.as_array(lib.rl_draft_logits(e, token), shape=(V,))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                recent.append(token)
                draft = _sample(guess, sampling, recent, seed, pos + 1)
                two = np.ctypeslib.as_array(lib.rl_forward_draft(e, token, draft), shape=(2, self._stride))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                self._open_draft = True
                self._logits = two[0, :V]
                self.counters["passes"] += 1
                self.counters["drafts"] += 1
                produced, pos = produced + 1, pos + 1
                yield token

                nxt = _sample(two[0, :V], sampling, recent, seed, pos)
                if nxt != draft:
                    self._settle()
                    token = nxt
                    continue
                self._open_draft = False       # the draft row is the real next token
                self._logits = two[1, :V]
                self.counters["accepted"] += 1
                recent.append(nxt)
                produced, pos = produced + 1, pos + 1
                yield nxt
                token = _sample(two[1, :V], sampling, recent, seed, pos)
        finally:
            self._settle()

    def snapshot(self) -> Snapshot:
        self._settle()
        state = np.empty(self._lib.rl_state_bytes(self._e), dtype=np.uint8)
        self._lib.rl_save_state(self._e, state.ctypes.data)
        logits = None if self._logits is None else self._logits.copy()
        return Snapshot(self.position, state, logits, state.nbytes)

    def restore(self, snap: Snapshot) -> None:
        self._open_draft = False
        self._lib.rl_load_state(self._e, snap.state.ctypes.data, snap.position)
        self._logits = snap.logits


def _top_k(logits: np.ndarray, k: int):
    """-> (token ids, values) of the k largest logits, descending."""
    k = min(k, logits.size)
    ids, vals = np.empty(k, dtype=np.int32), np.empty(k, dtype=np.float32)
    n = _library().rl_top_k(logits.ctypes.data, logits.size, k, ids.ctypes.data, vals.ctypes.data)
    return ids[:n].astype(np.int64), vals[:n]


def _uniform(seed: int, position: int, ids: np.ndarray) -> np.ndarray:
    """A number in (0, 1) for every (seed, position, token id), by hashing: the
    same token at the same position gets the same number whoever asks."""
    x = ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64((seed * 0x632BE59BD9B4E019 + position) % (1 << 64))
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def _sample(logits: np.ndarray, s: Sampling, recent: list, seed: int, position: int) -> int:
    """Draws the token for `position` from the truncated distribution.

    The draw is a Gumbel-max over the candidates with noise that depends only
    on (seed, position, token id). That is an ordinary sample from the
    distribution, and it has one property speculative decoding wants: two
    distributions that are close (the MTP head's guess and the model's own)
    mostly produce the same token when asked about the same position, so
    drafts keep being accepted at temperatures where the model's pick is not
    simply its most likely token.
    """
    k = 1 if s.temperature <= 0 else (s.top_k if s.top_k > 0 else 1024)   # beyond 1024 the tail carries no usable mass
    top, z = _top_k(logits, k)
    if recent and (s.repetition_penalty != 1.0 or s.presence_penalty):
        # Penalties touch only recently seen tokens, so the penalized top k is
        # among the plain top k + |seen| and the seen tokens themselves.
        seen = np.unique(np.asarray(recent[-64:], dtype=np.int64))
        top = np.union1d(_top_k(logits, k + len(seen))[0], seen)
        z = logits[top].astype(np.float64)
        hit = np.isin(top, seen)
        if s.repetition_penalty != 1.0:
            z[hit] = np.where(z[hit] > 0, z[hit] / s.repetition_penalty, z[hit] * s.repetition_penalty)
        if s.presence_penalty:
            z[hit] -= s.presence_penalty
        order = np.argsort(-z, kind="stable")[:k]
        top, z = top[order], z[order]
    if s.temperature <= 0:
        return int(top[0])
    logp = (z - z[0]).astype(np.float64) / s.temperature
    p = np.exp(logp)
    p /= p.sum()
    keep = len(top)
    if 0 < s.top_p < 1:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), s.top_p)) + 1)
    if s.min_p > 0:
        keep = min(keep, int(np.count_nonzero(p >= s.min_p * p[0])))
    top, logp = top[:keep], logp[:keep]
    gumbel = -np.log(-np.log(_uniform(seed, position, top)))
    return int(top[np.argmax(logp + gumbel)])
