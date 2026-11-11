"""Backend for the redline engine (.rlm models), through its C API.

The engine returns logits; sampling happens here in numpy. The tokenizer and
chat template are not part of a .rlm file: they are read from REDLINE_TOKENIZER,
or from the model directory the file was converted from if it sits next to it.

When the file carries the model's multi-token-prediction head, generation is
speculative: the head drafts the tokens that follow the next one, the engine
checks the next token and the drafts in one pass over the weights, and every
draft the model confirms is a token that cost no pass of its own. The output
is exactly what one-token-at-a-time decoding produces. REDLINE_MTP=0 turns
this off; REDLINE_DRAFTS sets the most drafts per pass.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import shutil
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
    lib.rl_draft_more.restype = ctypes.POINTER(ctypes.c_float)
    lib.rl_draft_more.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.rl_forward_draft.restype = ctypes.POINTER(ctypes.c_float)
    lib.rl_forward_draft.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.rl_keep_drafts.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rl_set_draft_vocab.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rl_logits_stride.restype = ctypes.c_size_t
    lib.rl_hidden.restype = ctypes.POINTER(ctypes.c_float)
    for fn in ("rl_has_mtp", "rl_max_drafts", "rl_draft_vocab", "rl_logits_stride", "rl_hidden", "rl_hidden_size"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.rl_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def _tokenizer_dir(model_path: str) -> str:
    """Directory with the tokenizer and chat template for a .rlm file. Files
    embedded in the model are unpacked once into the cache; older files fall
    back to REDLINE_TOKENIZER or the model directory they were converted from."""
    with open(model_path, "rb") as f:
        if f.read(4) != b"RLM1":
            raise ValueError(f"{model_path} is not a redline model file")
        _, n = struct.unpack("<IQ", f.read(12))
        hdr = json.loads(f.read(n))
        blobs = {}
        for name, entry in (hdr.get("files") or {}).items():
            f.seek(entry["offset"])
            blobs[name] = f.read(entry["nbytes"])
    override = os.environ.get("REDLINE_TOKENIZER")
    if override:
        return override
    if "tokenizer.json" in blobs:
        digest = hashlib.sha1(b"".join(k.encode() + v for k, v in sorted(blobs.items()))).hexdigest()[:16]
        root = os.path.expanduser(os.environ.get("REDLINE_CACHE", "~/.cache/redline"))
        out = os.path.join(root, "files", digest)
        if not os.path.isdir(out):
            tmp = f"{out}.{os.getpid()}.tmp"
            os.makedirs(tmp, exist_ok=True)
            for name, data in blobs.items():
                with open(os.path.join(tmp, name), "wb") as g:
                    g.write(data)
            try:
                os.rename(tmp, out)
            except OSError:          # another process got there first
                shutil.rmtree(tmp, ignore_errors=True)
        return out
    folder = os.path.dirname(os.path.abspath(model_path))
    stem, source = os.path.basename(model_path).rsplit(".rlm", 1)[0], hdr.get("source", "")
    for d in (os.path.join(folder, stem), os.path.join(folder, source)):
        if os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise RuntimeError(f"{model_path} carries no tokenizer: set REDLINE_TOKENIZER to a directory with tokenizer.json "
                       f"and the chat template (also looked next to the model for '{stem}/' and '{source}/')")


class RedlineBackend(Backend):
    name = "redline"

    def __init__(self, model_path: str, max_ctx: int = 32768, chunk: int = 1024, mtp: Optional[bool] = None):
        from transformers import AutoTokenizer

        self.files_dir = _tokenizer_dir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.files_dir)
        self.eos_ids = {i for i in (self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                        if i is not None and i >= 0}
        self.model_id = os.path.basename(model_path)
        self._lib = _library()
        self._e = self._lib.rl_open(os.path.abspath(model_path).encode(), max_ctx, chunk)
        self._vocab = self._lib.rl_vocab(self._e)
        self._chunk, self._max_ctx = chunk, max_ctx
        self.max_context = max_ctx
        self._logits = None
        self._open = None                # [rows fed, rows known to stand] of a speculative pass not settled yet
        if mtp is None:
            mtp = os.environ.get("REDLINE_MTP", "1") != "0"
        self.mtp = bool(self._lib.rl_has_mtp(self._e)) and mtp
        if self._lib.rl_has_mtp(self._e):
            self._lib.rl_set_mtp(self._e, int(self.mtp))
        # Drafting stops early once the chance that the drafts so far all hold
        # (the product of their probabilities under the head) falls below this:
        # a draft costs about a twentieth to a tenth of a pass over the weights.
        # A draft only has to find likely tokens, and low token ids are the
        # frequent ones: its pass over the output matrix covers ids below 64k
        # (98% of English and code), 128k (99% of Chinese too) or everything,
        # whichever the recent context calls for.
        self.draft_vocab = int(os.environ.get("REDLINE_DRAFT_VOCAB", "-1"))     # -1: choose per generation
        self.max_drafts = min(int(os.environ.get("REDLINE_DRAFTS", "6")), self._lib.rl_max_drafts(self._e))
        self.min_confidence = float(os.environ.get("REDLINE_MIN_CONFIDENCE", "0.3"))
        self._stride = self._lib.rl_logits_stride(self._e)
        self.gpu_seconds = 0.0
        self.counters = dict(passes=0, drafts=0, accepted=0)

    def __del__(self):
        if getattr(self, "_e", None):
            self._lib.rl_close(self._e)
            self._e = None

    @property
    def position(self) -> int:
        return self._lib.rl_position(self._e) - (self._open[0] - self._open[1] if self._open else 0)

    def reset(self) -> None:
        self._lib.rl_reset(self._e)
        self._logits = None
        self._open = None

    def _settle(self) -> None:
        """Takes back the draft rows of the last speculative pass that the model did not confirm."""
        if self._open:
            self._lib.rl_keep_drafts(self._e, self._open[1] - 1)
            self.gpu_seconds += self._lib.rl_last_gpu_seconds(self._e)
            self._open = None

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
        if self.mtp:
            rows = self.draft_vocab
            if rows < 0:
                seen_ids = np.asarray(history[-1024:] or [0])
                rows = next((n for n in (65536, 131072) if np.mean(seen_ids >= n) <= 0.01), 0)
            lib.rl_set_draft_vocab(e, rows)
            D = lib.rl_draft_vocab(e)          # logits a draft call returns
        try:
            while produced < max_tokens:
                k = min(self.max_drafts, max_tokens - produced - 1, self._max_ctx - self.position - 1) if self.mtp else 0
                if k < 1 or token in self.eos_ids:
                    self._forward([token])
                    recent.append(token)
                    produced, pos = produced + 1, pos + 1
                    yield token
                    token = _sample(self._logits, sampling, recent, seed, pos)
                    continue

                # The head's guesses for the tokens after `token`, each drawn with
                # the random numbers the model's own choice at that position will use.
                row, seen, confidence = [token], recent + [token], 1.0
                guess = np.ctypeslib.as_array(lib.rl_draft_logits(e, token), shape=(D,))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                while True:
                    draft, p = _sample(guess, sampling, seen, seed, pos + len(row), with_probability=True)
                    confidence *= p
                    row.append(draft)
                    seen.append(draft)
                    if len(row) > k or draft in self.eos_ids or confidence < self.min_confidence:
                        break
                    guess = np.ctypeslib.as_array(lib.rl_draft_more(e, draft), shape=(D,))
                    self.gpu_seconds += lib.rl_last_gpu_seconds(e)

                n = len(row)
                rows = np.ctypeslib.as_array(lib.rl_forward_draft(e, (ctypes.c_int32 * n)(*row), n),
                                             shape=(n, self._stride))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                self._open = [n, 1]
                self._logits = rows[0, :V]
                self.counters["passes"] += 1
                self.counters["drafts"] += n - 1
                recent.append(token)
                produced, pos = produced + 1, pos + 1
                yield token

                # A draft stands if it is what the model picks at its position.
                accepted = 0
                nxt = _sample(rows[0, :V], sampling, recent, seed, pos)
                while accepted < n - 1 and nxt == row[accepted + 1]:
                    accepted += 1
                    self._open[1] = accepted + 1
                    self._logits = rows[accepted, :V]
                    self.counters["accepted"] += 1
                    recent.append(nxt)
                    produced, pos = produced + 1, pos + 1
                    yield nxt
                    nxt = _sample(rows[accepted, :V], sampling, recent, seed, pos)
                self._settle()
                token = nxt
        finally:
            self._settle()

    def snapshot(self) -> Snapshot:
        self._settle()
        state = np.empty(self._lib.rl_state_bytes(self._e), dtype=np.uint8)
        self._lib.rl_save_state(self._e, state.ctypes.data)
        logits = None if self._logits is None else self._logits.copy()
        return Snapshot(self.position, state, logits, state.nbytes)

    def restore(self, snap: Snapshot) -> None:
        self._settle()
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


def _sample(logits: np.ndarray, s: Sampling, recent: list, seed: int, position: int, with_probability: bool = False):
    """Draws the token for `position` from the truncated distribution; with
    `with_probability`, also returns the probability it had there.

    The draw is a Gumbel-max over the candidates with noise that depends only
    on (seed, position, token id). That is an ordinary sample from the
    distribution, and it has one property speculative decoding wants: two
    distributions that are close (the MTP head's guess and the model's own)
    mostly produce the same token when asked about the same position, so
    drafts keep being accepted at temperatures where the model's pick is not
    simply its most likely token.
    """
    greedy = s.temperature <= 0
    k = (8 if with_probability else 1) if greedy else (s.top_k if s.top_k > 0 else 1024)   # 1024: the tail carries no mass
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
    if greedy:
        if not with_probability:
            return int(top[0])
        return int(top[0]), float(1.0 / np.exp((z - z[0]).astype(np.float64)).sum())   # its share among the top few
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
    pick = int(np.argmax(logp + gumbel))
    if not with_probability:
        return int(top[pick])
    return int(top[pick]), float(p[pick] / p[:keep].sum())
