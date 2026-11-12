"""Backend for the raceline engine (.rlm models), through its C API.

The engine returns logits; sampling happens here in numpy. The tokenizer and
chat template are not part of a .rlm file: they are read from RACELINE_TOKENIZER,
or from the model directory the file was converted from if it sits next to it.

When the file carries the model's multi-token-prediction head, generation is
speculative: the head drafts the tokens that follow the next one, the engine
checks the next token and the drafts in one pass over the weights, and every
draft the model confirms is a token that cost no pass of its own. The output
is exactly what one-token-at-a-time decoding produces. RACELINE_MTP=0 turns
this off; RACELINE_DRAFTS sets the most drafts per pass from the head.

Drafts can also come from the context itself. An agent repeats a lot of what
it has read: the lines an edit replaces, a file it rewrites with one change,
code it quotes. When the last few tokens occurred earlier in the context and
what followed them then agrees with the head's first guess, that earlier
continuation is drafted instead, up to 15 tokens at once (RACELINE_LOOKUP sets
the most; 0 turns it off).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import shutil
import struct
import time
from typing import Optional

import numpy as np

from .backends import Backend, Sampling, Snapshot


def unsupported_reason() -> Optional[str]:
    """Why the engine cannot run on this Mac, or None.

    The kernels hand 4-bit weights to the GPU's neural accelerators through
    Metal 4.1 tensor operations: Apple M5 or later, macOS 27 or later. Older
    chips have no such units; whether the operations work there at all has
    not been tried (RACELINE_ANY_GPU=1 skips this check)."""
    import platform
    import re

    from .sysinfo import sysctl_str

    if os.environ.get("RACELINE_ANY_GPU") == "1":
        return None
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "it needs a Mac with Apple Silicon"
    major = int((platform.mac_ver()[0] or "0").split(".")[0])
    if major < 27:
        return f"it needs macOS 27 or later (this Mac runs macOS {platform.mac_ver()[0]})"
    chip = sysctl_str("machdep.cpu.brand_string")
    m = re.match(r"Apple M(\d+)", chip)
    if m and int(m.group(1)) < 5:
        return (f"it needs the GPU neural accelerators of an Apple M5 or later (this Mac has an {chip}). "
                "To try it anyway, set RACELINE_ANY_GPU=1, and please tell us what happens in "
                "https://github.com/audenvenn/raceline/issues/2")
    return None


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    # An installed package carries the kernel sources the engine compiles at run time.
    if os.path.isfile(os.path.join(here, "kernels", "ops.metal")):
        os.environ.setdefault("RL_KERNEL_DIR", os.path.join(here, "kernels"))
    candidates = [os.environ.get("RACELINE_LIB"), os.path.join(here, "libraceline.dylib"),
                  os.path.join(here, "..", "build", "cmake", "libraceline.dylib")]
    for path in candidates:
        if path and os.path.isfile(path):
            lib = ctypes.CDLL(path)
            break
    else:
        raise RuntimeError("libraceline.dylib not found: build it with `cmake --build build/cmake` "
                           "or point RACELINE_LIB at it")
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
    lib.rl_set_draft_tail.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.rl_logits_stride.restype = ctypes.c_size_t
    lib.rl_hidden.restype = ctypes.POINTER(ctypes.c_float)
    for fn in ("rl_has_mtp", "rl_max_drafts", "rl_draft_vocab", "rl_draft_main", "rl_draft_tail", "rl_logits_stride",
               "rl_hidden", "rl_hidden_size"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.rl_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def _tokenizer_dir(model_path: str) -> str:
    """Directory with the tokenizer and chat template for a .rlm file. Files
    embedded in the model are unpacked once into the cache; older files fall
    back to RACELINE_TOKENIZER or the model directory they were converted from."""
    with open(model_path, "rb") as f:
        if f.read(4) != b"RLM1":
            raise ValueError(f"{model_path} is not a raceline model file")
        _, n = struct.unpack("<IQ", f.read(12))
        hdr = json.loads(f.read(n))
        blobs = {}
        for name, entry in (hdr.get("files") or {}).items():
            f.seek(entry["offset"])
            blobs[name] = f.read(entry["nbytes"])
    override = os.environ.get("RACELINE_TOKENIZER")
    if override:
        return override
    if "tokenizer.json" in blobs:
        digest = hashlib.sha1(b"".join(k.encode() + v for k, v in sorted(blobs.items()))).hexdigest()[:16]
        root = os.path.expanduser(os.environ.get("RACELINE_CACHE", "~/.cache/raceline"))
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
    raise RuntimeError(f"{model_path} carries no tokenizer: set RACELINE_TOKENIZER to a directory with tokenizer.json "
                       f"and the chat template (also looked next to the model for '{stem}/' and '{source}/')")


class RacelineBackend(Backend):
    name = "raceline"

    def __init__(self, model_path: str, max_ctx: int = 32768, chunk: int = 1024, mtp: Optional[bool] = None):
        from transformers import AutoTokenizer

        why = unsupported_reason()
        if why:
            raise RuntimeError(f"raceline's engine does not run on this Mac: {why}. MLX models still run here, "
                               "without the engine's speedups: raceline chat <MLX model directory>, for example "
                               "one of mlx-community's Qwen3.5 / Qwen3.8 4-bit models")
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
            mtp = os.environ.get("RACELINE_MTP", "1") != "0"
        self.mtp = bool(self._lib.rl_has_mtp(self._e)) and mtp
        if self._lib.rl_has_mtp(self._e):
            self._lib.rl_set_mtp(self._e, int(self.mtp))
        # A draft only has to find likely tokens, and low token ids are the
        # frequent ones: its pass over the output matrix covers ids below 64k
        # (98% of English and code), 128k (99% of Chinese too) or everything,
        # whichever the recent context calls for (at most 3% of its text
        # tokens left out; an agent's own output is 1 to 1.5% beyond 64k).
        self.draft_vocab = int(os.environ.get("RACELINE_DRAFT_VOCAB", "-1"))     # -1: choose per generation
        # The chat template's special tokens (<|im_start|>, <tool_call>, <think>
        # ...) sit at the top of the id range; they say nothing about which part
        # of the vocabulary the text uses, so they are left out of that choice.
        added = getattr(self.tokenizer, "added_tokens_decoder", None) or {}
        self._text_ids = min(added) if added else self._vocab
        # ... and every draft looks at them anyway: an agent's output turns on
        # them (<tool_call>, </think>, <|im_end|>), and a draft that cannot
        # name them ends its chain there.
        if self.mtp and added:
            self._lib.rl_set_draft_tail(self._e, self._text_ids)
        # Drafting stops early once the chance that the drafts so far all hold
        # (the product of their probabilities under the head) falls below a
        # threshold. Another draft pays off when that chance exceeds what the
        # draft costs relative to a token, so the threshold follows the measured
        # times: three times (time per draft / time per token), which allows
        # for the head being overconfident. RACELINE_MIN_CONFIDENCE fixes it.
        self.max_drafts = min(int(os.environ.get("RACELINE_DRAFTS", "6")), self._lib.rl_max_drafts(self._e))
        fixed = os.environ.get("RACELINE_MIN_CONFIDENCE")
        self.min_confidence = float(fixed) if fixed else None
        self._t_draft = self._t_token = None      # running averages, seconds
        # Drafts copied from the context (see _Lookup): a pass of 16 rows costs
        # a few percent more than a pass of one, so a long copy is drafted whole.
        self.lookup_drafts = min(int(os.environ.get("RACELINE_LOOKUP", "15")), self._lib.rl_max_drafts(self._e))
        self._stride = self._lib.rl_logits_stride(self._e)
        self.gpu_seconds = 0.0
        self.counters = dict(passes=0, drafts=0, accepted=0, lookup_passes=0, lookup_drafts=0, lookup_accepted=0)

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
            limit = self.draft_vocab
            if limit < 0:
                seen_ids = np.asarray(history[-1024:] or [0])
                seen_ids = seen_ids[seen_ids < self._text_ids]
                limit = next((n for n in (65536, 131072) if not seen_ids.size or np.mean(seen_ids >= n) <= 0.03), 0)
            lib.rl_set_draft_vocab(e, limit)
            D = lib.rl_draft_vocab(e)          # logits a draft call returns
            first = lib.rl_draft_tail(e)
            tail = (lib.rl_draft_main(e), first) if first >= 0 else None
        look = _Lookup(history) if self.mtp and self.lookup_drafts > 0 else None
        try:
            while produced < max_tokens:
                room = min(max_tokens - produced - 1, self._max_ctx - self.position - 1)
                k = min(self.max_drafts, room) if self.mtp else 0
                if k < 1 or token in self.eos_ids:
                    self._forward([token])
                    recent.append(token)
                    if look:
                        look.push(token)
                    produced, pos = produced + 1, pos + 1
                    yield token
                    token = _sample(self._logits, sampling, recent, seed, pos)
                    continue

                # The head's guesses for the tokens after `token`, each drawn with
                # the random numbers the model's own choice at that position will use.
                t_cycle = time.perf_counter()
                threshold = self.min_confidence
                if threshold is None:
                    threshold = 0.3 if self._t_token is None else min(0.8, max(0.1, 3.0 * self._t_draft / self._t_token))
                row, seen, confidence = [token], recent + [token], 1.0
                guess = np.ctypeslib.as_array(lib.rl_draft_logits(e, token), shape=(D,))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                draft, p = _sample(guess, sampling, seen, seed, pos + 1, with_probability=True, tail=tail)
                copied = look.propose(token, min(self.lookup_drafts, room)) if look else []
                if len(copied) > 1 and copied[0] == draft:
                    # The context and the head agree on the next token: draft what followed there.
                    end = next((j + 1 for j, t in enumerate(copied) if t in self.eos_ids), len(copied))
                    row += copied[:end]
                else:
                    copied = None
                    while True:
                        confidence *= p
                        row.append(draft)
                        seen.append(draft)
                        if len(row) > k or draft in self.eos_ids or confidence < threshold:
                            break
                        guess = np.ctypeslib.as_array(lib.rl_draft_more(e, draft), shape=(D,))
                        self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                        draft, p = _sample(guess, sampling, seen, seed, pos + len(row), with_probability=True,
                                           tail=tail)
                t_drafts = time.perf_counter() - t_cycle

                n = len(row)
                rows = np.ctypeslib.as_array(lib.rl_forward_draft(e, (ctypes.c_int32 * n)(*row), n),
                                             shape=(n, self._stride))
                self.gpu_seconds += lib.rl_last_gpu_seconds(e)
                self._open = [n, 1]
                self._logits = rows[0, :V]
                self.counters["passes"] += 1
                self.counters["drafts"] += n - 1
                if copied:
                    self.counters["lookup_passes"] += 1
                    self.counters["lookup_drafts"] += n - 1
                recent.append(token)
                if look:
                    look.push(token)
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
                    if copied:
                        self.counters["lookup_accepted"] += 1
                    recent.append(nxt)
                    if look:
                        look.push(nxt)
                    produced, pos = produced + 1, pos + 1
                    yield nxt
                    nxt = _sample(rows[accepted, :V], sampling, recent, seed, pos)
                self._settle()
                token = nxt
                # What a draft from the head and a token cost here, for the next
                # cycles' threshold (time the consumer spends between yields is
                # counted too; it is small). Copied drafts cost nothing to make.
                if not copied:
                    per_draft, per_token = t_drafts / (n - 1), (time.perf_counter() - t_cycle) / (accepted + 1)
                    self._t_draft = per_draft if self._t_draft is None else 0.9 * self._t_draft + 0.1 * per_draft
                    self._t_token = per_token if self._t_token is None else 0.9 * self._t_token + 0.1 * per_token
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


class _Lookup:
    """Continuations copied from the context.

    Remembers, for every run of three tokens in the context, where it last
    occurred. When the newest three tokens occurred before, what followed them
    then is the proposal: the rest of a line being quoted, the code around an
    edit, a file being rewritten. Three tokens catch a quote early; chance
    matches are weeded out by the caller, which drafts a proposal only when
    the head agrees with its first token.
    """

    def __init__(self, tokens):
        self.toks = list(tokens)
        t = self.toks
        self.where = {(t[i - 3], t[i - 2], t[i - 1]): i for i in range(3, len(t))}

    def push(self, token: int) -> None:
        """Appends a token of the context; the run of three before it now has a known continuation."""
        t = self.toks
        t.append(token)
        i = len(t) - 1
        if i >= 3:
            self.where[(t[i - 3], t[i - 2], t[i - 1])] = i

    def propose(self, nxt: int, k: int) -> list:
        """Up to k tokens that followed (the last two tokens, nxt) the most recent time they occurred."""
        t = self.toks
        if len(t) < 2 or k < 1:
            return []
        i = self.where.get((t[-2], t[-1], nxt))
        return [] if i is None else t[i:i + k]


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


def _sample(logits: np.ndarray, s: Sampling, recent: list, seed: int, position: int, with_probability: bool = False,
            tail=None):
    """Draws the token for `position` from the truncated distribution; with
    `with_probability`, also returns the probability it had there.

    `logits` holds one entry per token id, or with `tail` = (main, first) a
    draft's restricted vocabulary: entries for ids 0 .. main-1 followed by
    entries for ids first, first+1, ... Tokens outside it cannot be drawn.

    The draw is a Gumbel-max over the candidates with noise that depends only
    on (seed, position, token id). That is an ordinary sample from the
    distribution, and it has one property speculative decoding wants: two
    distributions that are close (the MTP head's guess and the model's own)
    mostly produce the same token when asked about the same position, so
    drafts keep being accepted at temperatures where the model's pick is not
    simply its most likely token.
    """
    def ids(entries):   # entries of `logits` -> token ids
        if tail is None:
            return entries
        main, first = tail
        return np.where(entries < main, entries, entries - main + first)

    greedy = s.temperature <= 0
    k = (8 if with_probability else 1) if greedy else (s.top_k if s.top_k > 0 else 1024)   # 1024: the tail carries no mass
    top, z = _top_k(logits, k)          # entries, not yet token ids
    if recent and (s.repetition_penalty != 1.0 or s.presence_penalty):
        # Penalties touch only recently seen tokens, so the penalized top k is
        # among the plain top k + |seen| and the seen tokens themselves.
        seen = np.unique(np.asarray(recent[-64:], dtype=np.int64))
        if tail is not None:            # token ids -> entries; ids the logits do not cover drop out
            main, first = tail
            seen = np.concatenate([seen[seen < main], seen[seen >= first] - first + main])
        seen = seen[seen < logits.size]
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
            return int(ids(top[0]))
        return int(ids(top[0])), float(1.0 / np.exp((z - z[0]).astype(np.float64)).sum())   # its share among the top few
    logp = (z - z[0]).astype(np.float64) / s.temperature
    p = np.exp(logp)
    p /= p.sum()
    keep = len(top)
    if 0 < s.top_p < 1:
        keep = min(keep, int(np.searchsorted(np.cumsum(p), s.top_p)) + 1)
    if s.min_p > 0:
        keep = min(keep, int(np.count_nonzero(p >= s.min_p * p[0])))
    top, logp = ids(top[:keep]), logp[:keep]
    gumbel = -np.log(-np.log(_uniform(seed, position, top)))
    pick = int(np.argmax(logp + gumbel))
    if not with_probability:
        return int(top[pick])
    return int(top[pick]), float(p[pick] / p[:keep].sum())
