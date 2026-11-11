"""Model backends.

A backend is a stateful, token-level model: tokens go in, the context grows,
and the next-token distribution comes out. Everything above it (prompt reuse,
chat formatting, tools, the HTTP API) is backend independent, so the same agent
runs on MLX today and on the redline engine where that is faster.

The one capability that matters beyond "append and sample" is `snapshot` /
`restore`. Qwen3.5 / Qwen3.8 are three quarters recurrent layers, whose state
cannot be rewound the way an attention cache can be truncated; a snapshot is
the only way back to an earlier position without recomputing the prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None


@dataclass
class Snapshot:
    position: int
    state: Any          # backend specific
    logits: Any         # distribution after the token at position - 1
    nbytes: int


class Backend:
    """Interface. `position` is the number of tokens in the context."""

    name = "backend"
    position = 0
    max_context: Optional[int] = None   # tokens the context can hold, if bounded
    eos_ids: set = set()
    tokenizer = None    # a transformers tokenizer: encode, decode, apply_chat_template

    def reset(self) -> None:
        raise NotImplementedError

    def prefill(self, tokens: list, progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Appends tokens to the context."""
        raise NotImplementedError

    def generate(self, sampling: Sampling, max_tokens: int, history: list) -> Iterator[int]:
        """Samples up to max_tokens tokens. Every token yielded is already part
        of the context, so the caller may stop consuming at any point."""
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        raise NotImplementedError

    def restore(self, snap: Snapshot) -> None:
        """Returns to an earlier position. Only snapshots taken on the current
        timeline at or before the current position are valid."""
        raise NotImplementedError


class MLXBackend(Backend):
    """mlx-lm models (the MLX 4-bit conversions from mlx-community, for example)."""

    name = "mlx"
    PREFILL_CHUNK = 2048

    def __init__(self, model_dir: str):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self._mx = mx
        self._make_cache = make_prompt_cache
        self.model, wrapper = load(model_dir)
        self.tokenizer = wrapper._tokenizer
        self.eos_ids = set(wrapper.eos_token_ids)
        self.model_id = os.path.basename(os.path.normpath(model_dir))
        self.reset()

    def reset(self) -> None:
        self.cache = self._make_cache(self.model)
        self.position = 0
        self._logits = None

    def prefill(self, tokens, progress=None) -> None:
        mx = self._mx
        for i in range(0, len(tokens), self.PREFILL_CHUNK):
            part = tokens[i:i + self.PREFILL_CHUNK]
            out = self.model(mx.array(part)[None], cache=self.cache)
            self._logits = out[0, -1, :]
            mx.eval(self._logits, [c.state for c in self.cache])
            self.position += len(part)
            mx.clear_cache()
            if progress:
                progress(i + len(part), len(tokens))

    def generate(self, sampling, max_tokens, history):
        mx = self._mx
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        if self._logits is None:
            raise RuntimeError("generate() before any token was fed")
        if sampling.seed is not None:
            mx.random.seed(sampling.seed)
        sampler = make_sampler(temp=sampling.temperature, top_p=sampling.top_p if sampling.top_p < 1 else 0.0,
                               min_p=sampling.min_p, top_k=sampling.top_k)
        processors = make_logits_processors(
            repetition_penalty=sampling.repetition_penalty if sampling.repetition_penalty != 1.0 else None,
            presence_penalty=sampling.presence_penalty or None)
        seen = mx.array(history[-64:] if history else [0])

        def pick(logits, seen):
            z = logits[None]
            for p in processors:
                z = p(seen, z)
            return sampler(z - mx.logsumexp(z, axis=-1, keepdims=True))

        # The token after y is being computed while y is handed to the caller,
        # so the GPU never waits for Python (the mlx-lm generation pattern).
        y = pick(self._logits, seen)
        mx.async_eval(y)
        for n in range(max_tokens):
            out = self.model(y[None], cache=self.cache)
            logits = out[0, -1, :]
            if processors:
                seen = mx.concatenate([seen, y])
            next_y = pick(logits, seen)
            mx.async_eval(next_y, logits)
            token = y.item()
            self.position += 1
            self._logits = logits
            if n % 256 == 255:
                mx.clear_cache()
            yield token
            y = next_y

    def snapshot(self) -> Snapshot:
        state, nbytes = [], 0
        for c in self.cache:
            if c.is_trimmable():            # attention: positions are independent, the offset is enough
                state.append(("kv", c.offset))
            else:                           # recurrent: keep the arrays (immutable, so references suffice)
                arrays, pad, lengths = c.state
                state.append(("arrays", (list(arrays), pad, lengths)))
                nbytes += sum(a.nbytes for a in arrays if a is not None)
        return Snapshot(self.position, state, self._logits, nbytes)

    def restore(self, snap: Snapshot) -> None:
        for c, (kind, value) in zip(self.cache, snap.state):
            if kind == "kv":
                c.trim(c.offset - value)
            else:
                arrays, pad, lengths = value
                c.state = (list(arrays), pad, lengths)
        self.position = snap.position
        self._logits = snap.logits


def load_backend(model: str, backend: Optional[str] = None) -> Backend:
    """`model` is an MLX model directory or a .rlm file (with tokenizer files
    next to it or in the directory given by REDLINE_TOKENIZER)."""
    kind = backend or ("redline" if model.endswith(".rlm") else "mlx")
    if kind == "mlx":
        return MLXBackend(model)
    if kind == "redline":
        from .engine import RedlineBackend
        return RedlineBackend(model)
    raise ValueError(f"unknown backend '{kind}'")
