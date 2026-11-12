"""Chat on top of a session: prompt rendering, reasoning and tool-call parsing.

The model's own chat template is the source of truth for formatting. What the
model then emits is parsed back into three channels: reasoning (between
`<think>` and `</think>`), answer text, and tool calls in Qwen's format

    <tool_call>
    <function=name>
    <parameter=key>
    value
    </parameter>
    </function>
    </tool_call>
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Iterator, Optional

from .backends import Backend, Sampling
from .session import Session

_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def normalize_messages(messages: list) -> list:
    """OpenAI-style messages -> what the chat template expects."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":   # OpenAI's newer name for system instructions
            m["role"] = "system"
        content = m.get("content")
        if isinstance(content, list):  # text parts only
            if any(p.get("type") not in ("text", None) for p in content if isinstance(p, dict)):
                raise ValueError("only text content is supported")
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        m["content"] = content or ""
        calls = []
        for call in m.get("tool_calls") or []:
            fn = dict(call.get("function", call))
            args = fn.get("arguments", {})
            if isinstance(args, str):  # the API carries arguments as a JSON string
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            fn["arguments"] = args
            calls.append({"type": "function", "function": fn})
        if calls:
            m["tool_calls"] = calls
        else:
            m.pop("tool_calls", None)
        out.append(m)
    return out


def parse_tool_call(text: str, tools: Optional[list]) -> list:
    """Body of one <tool_call> block -> [{"name", "arguments"}]. Parameter
    values are text; they are decoded as JSON only where the tool's schema asks
    for something other than a string."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for name, body in _FUNCTION.findall(text):
        name = name.strip()
        args = {}
        for key, value in _PARAMETER.findall(body):
            key = key.strip()
            kind = (schemas.get(name, {}).get(key) or {}).get("type")
            if kind != "string":
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass  # keep the text: better a string than a lost argument
            args[key] = value
        calls.append({"name": name, "arguments": args})
    return calls


class _Decoder:
    """Incremental detokenizer: emits text once it is valid UTF-8."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list = []
        self.sent = 0

    def push(self, token: int) -> str:
        self.ids.append(token)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("\ufffd"):      # a multi-byte character is still incomplete
            return ""
        new = text[self.sent:]
        self.sent = len(text)
        if new.endswith("\n") and len(self.ids) > 64:   # keep decoding cost flat
            self.ids, self.sent = [], 0
        return new

    def flush(self) -> str:
        text = self.tok.decode(self.ids, skip_special_tokens=False) if self.ids else ""
        new = text[self.sent:]
        self.ids, self.sent = [], 0
        return new


class Runtime:
    """A loaded model plus one reusable session. Shared by the server and the terminal agent."""

    def __init__(self, backend: Backend, defaults: Optional[Sampling] = None, park_bytes: Optional[int] = None):
        """`park_bytes`: memory for conversations set aside (Session; default RACELINE_PARK_GB, 2 GB)."""
        self.backend = backend
        if park_bytes is None:
            park_bytes = int(float(os.environ.get("RACELINE_PARK_GB", "2")) * 2**30)
        self.session = Session(backend, max_parked_bytes=park_bytes)
        self.defaults = defaults or Sampling()
        self.lock = threading.Lock()   # one generation at a time: this is a single-user runtime
        tok = backend.tokenizer
        ids = {name: tok.convert_tokens_to_ids(name) for name in
               ("<|im_start|>", "<think>", "</think>", "<tool_call>", "</tool_call>")}
        self._im_start, self._think, self._think_end = ids["<|im_start|>"], ids["<think>"], ids["</think>"]
        self._call, self._call_end = ids["<tool_call>"], ids["</tool_call>"]

    def render(self, messages: list, tools: Optional[list], thinking: bool, effort: Optional[str] = None):
        """-> (prompt tokens, snapshot positions, position to pin). `effort` is the
        reasoning effort for templates that take one (Qwen3.8: low, medium, xhigh)."""
        tok = self.backend.tokenizer
        msgs = normalize_messages(messages)
        extra = {"reasoning_effort": effort} if effort and thinking else {}
        text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=thinking, **extra)
        tokens = tok.encode(text, add_special_tokens=False)
        starts = [i for i, t in enumerate(tokens) if t == self._im_start]
        has_system = bool(tools) or (msgs and msgs[0].get("role") == "system")
        pin = starts[1] if has_system and len(starts) > 1 else None
        # The reply being generated will be re-rendered from here on the next
        # request, so the start of the generation prompt is where it branches.
        points = [p for p in (pin, starts[-1] if starts else None) if p is not None]
        return tokens, points, pin

    def chat(self, messages: list, tools: Optional[list] = None, sampling: Optional[Sampling] = None,
             max_tokens: int = 2048, thinking: bool = True, stop: Optional[list] = None,
             progress=None, effort: Optional[str] = None) -> Iterator[tuple]:
        """Yields ("reasoning", text), ("content", text), ("tool_call", {...}) and
        finally ("done", stats)."""
        with self.lock:
            yield from self._chat(messages, tools, sampling or self.defaults, max_tokens, thinking, stop or [],
                                  progress, effort)

    def _chat(self, messages, tools, sampling, max_tokens, thinking, stop, progress, effort):
        tok = self.backend.tokenizer
        prompt, points, pin = self.render(messages, tools, thinking, effort)
        limit = self.backend.max_context
        if limit is not None:
            if len(prompt) + 8 > limit:
                raise ValueError(f"the conversation is {len(prompt)} tokens; the context window holds {limit}")
            max_tokens = min(max_tokens, limit - len(prompt) - 2)
        counters = dict(getattr(self.backend, "counters", {}))
        experts = getattr(self.backend, "expert_cache", lambda: None)
        experts0 = experts()
        t0 = time.perf_counter()
        usage = self.session.sync(prompt, points, pin, progress)
        t_prefill = time.perf_counter() - t0

        channel = "reasoning" if thinking else "content"
        decoder = _Decoder(tok)
        call_ids: list = []
        answer = ""
        fresh = True            # at the start of a channel: drop leading newlines
        n_out, t_first, finish = 0, None, "length"
        t1 = time.perf_counter()

        def emit(kind, text):
            nonlocal fresh
            if fresh:
                text = text.lstrip("\n")
            if text:
                fresh = False
                return (kind, text)
            return None

        for token in self.session.generate(sampling, max_tokens, self.backend.eos_ids):
            n_out += 1
            if t_first is None:
                t_first = time.perf_counter() - t0
            if token in self.backend.eos_ids:
                if finish != "tool_calls":
                    finish = "stop"
                break
            if channel == "tool":
                if token == self._call_end:
                    for call in parse_tool_call(tok.decode(call_ids, skip_special_tokens=False), tools):
                        call["id"] = "call_" + uuid.uuid4().hex[:12]
                        yield ("tool_call", call)
                        finish = "tool_calls"
                    call_ids, channel, fresh = [], "content", True
                else:
                    call_ids.append(token)
                continue
            if token in (self._think_end, self._think, self._call):
                ev = emit(channel, decoder.flush().rstrip("\n"))
                if ev:
                    yield ev
                channel = {self._think_end: "content", self._think: "reasoning", self._call: "tool"}[token]
                fresh = True
                continue
            text = decoder.push(token)
            if not text:
                continue
            if channel == "content" and stop:
                answer += text
                cut = min((answer.find(s) for s in stop if s in answer), default=-1)
                if cut >= 0:        # a stop string completed: emit only what precedes it
                    keep = len(text) - (len(answer) - cut)
                    if keep > 0:
                        yield ("content", text[:keep])
                    finish = "stop"
                    break
            ev = emit(channel, text)
            if ev:
                yield ev
        if channel == "tool" and call_ids:   # ran out of tokens inside a call: salvage what parses
            for call in parse_tool_call(tok.decode(call_ids, skip_special_tokens=False), tools):
                call["id"] = "call_" + uuid.uuid4().hex[:12]
                yield ("tool_call", call)
                finish = "tool_calls"
        elif channel != "tool":
            ev = emit(channel, decoder.flush().rstrip("\n"))
            if ev:
                yield ev

        t_decode = time.perf_counter() - t1
        fed = usage["prompt_tokens"] - usage["cached_tokens"]
        drafts = {k: v - counters[k] for k, v in getattr(self.backend, "counters", {}).items() if k != "passes"}
        experts1 = experts()
        if experts0 and experts1:            # experts on disk: how often the cache had what a layer needed
            hits, reads = experts1[1] - experts0[1], experts1[2] - experts0[2]
            experts1 = dict(experts_from_cache=hits / max(1, hits + reads), experts_read=reads)
        yield ("done", dict(
            **(experts1 if isinstance(experts1, dict) else {}),
            **({"drafts": drafts["drafts"], "drafts_accepted": drafts["accepted"]} if drafts.get("drafts") else {}),
            **({"copied_drafts": drafts["lookup_drafts"], "copied_drafts_accepted": drafts["lookup_accepted"]}
               if drafts.get("lookup_drafts") else {}),
            finish_reason=finish, prompt_tokens=usage["prompt_tokens"], cached_tokens=usage["cached_tokens"],
            completion_tokens=n_out, prefill_seconds=t_prefill, decode_seconds=t_decode,
            time_to_first_token=t_first if t_first is not None else t_prefill,
            prefill_tokens_per_second=fed / t_prefill if fed and t_prefill > 0 else 0.0,
            decode_tokens_per_second=n_out / t_decode if n_out and t_decode > 0 else 0.0))
