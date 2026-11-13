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
from collections import OrderedDict
from typing import Iterator, Optional

from .backends import Backend, Sampling
from .images import image_keys, picture_of, vision_settings
from .session import Session
from .vision import find_vision

_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


class ContextFull(ValueError):
    """The conversation does not fit the context window."""

    def __init__(self, tokens: int, limit: int):
        super().__init__(f"the conversation is {tokens} tokens; the context window holds {limit}")
        self.tokens, self.limit = tokens, limit


IMAGE_LEFT_OUT = "[image left out: this model runs without image input]"


def _content(content, images: bool):
    """A message's content -> a string, or with images a list of text and image parts
    ({"type": "image", "image": Picture}); without `images`, a note in each image's place."""
    if not isinstance(content, list):
        return content or ""
    parts = []

    def add_text(text):
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += text
        else:
            parts.append({"type": "text", "text": text})

    for p in content:
        kind = p.get("type") if isinstance(p, dict) else None
        if not isinstance(p, dict):
            add_text(str(p))
        elif kind in ("text", "input_text") or (kind is None and "image_url" not in p):
            add_text(p.get("text", ""))
        elif kind == "document":
            add_text("[document left out: this model runs without document input]")
        elif kind in ("image", "image_url", "input_image") or kind is None:
            picture = picture_of(p) if images else None
            parts.append({"type": "image", "image": picture} if picture else {"type": "text", "text": IMAGE_LEFT_OUT})
        else:
            raise ValueError(f"unsupported content part '{kind}'")
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def normalize_messages(messages: list, images: bool = False) -> list:
    """OpenAI-style messages -> what the chat template expects.

    The Qwen templates take one system message, first. Several at the start
    are joined; one that comes later in the conversation (clients insert
    them, such as reminders) becomes a user message at that point. Image
    parts become Pictures where the model takes `images`, else a note."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":   # OpenAI's newer name for system instructions
            m["role"] = "system"
        if m.get("role") == "system" and any(x["role"] != "system" for x in out):
            m["role"] = "user"
        content = _content(m.get("content"), images and m.get("role") != "system")
        if isinstance(content, list) and m.get("role") not in ("user", "tool"):
            raise ValueError(f"images in a {m.get('role')} message: only user messages and tool results take them")
        m["content"] = content
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
        if m["role"] == "system" and out:    # only system messages so far: one system prompt
            out[0]["content"] = (out[0]["content"] + "\n\n" + m["content"]).strip("\n")
            continue
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
        """`park_bytes`: memory for conversations set aside (Session; default RACELINE_PARK_GB, else a
        twelfth of this Mac's memory up to 2 GB: 1.3 GB on a 16 GB Mac, which the model needs more)."""
        self.backend = backend
        if park_bytes is None:
            gb = os.environ.get("RACELINE_PARK_GB")
            if gb is not None:
                park_bytes = int(float(gb) * 2**30)
            else:
                try:
                    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                except (OSError, ValueError, AttributeError):
                    ram = 24 << 30
                park_bytes = min(2 << 30, ram // 12)
        self.session = Session(backend, max_parked_bytes=park_bytes)
        self.defaults = defaults or Sampling()
        # Image input: the model's vision file, its processor settings, and the tower once an image comes.
        path = getattr(backend, "model_path", None)
        self.vision_path = find_vision(path) if path and getattr(backend, "image_token", None) is not None else None
        self.vision = vision_settings(self.vision_path) if self.vision_path else None
        self._tower = None
        self._pictures: OrderedDict = OrderedDict()    # digest -> [Picture, width, rows' vectors or None]
        self.lock = threading.Lock()   # one generation at a time: this is a single-user runtime
        tok = backend.tokenizer
        ids = {name: tok.convert_tokens_to_ids(name) for name in
               ("<|im_start|>", "<think>", "</think>", "<tool_call>", "</tool_call>")}
        self._im_start, self._think, self._think_end = ids["<|im_start|>"], ids["<think>"], ids["</think>"]
        self._call, self._call_end = ids["<tool_call>"], ids["</tool_call>"]

    def render(self, messages: list, tools: Optional[list], thinking: bool, effort: Optional[str] = None,
               keep_reasoning: Optional[bool] = None):
        """-> (prompt tokens, snapshot positions, position to pin). `effort` is the
        reasoning effort for templates that take one (Qwen3.8: low, medium, xhigh).

        `keep_reasoning` sets `preserve_thinking` for templates that take it
        (Qwen3.6 and Qwen3.8; None leaves the template's default). With it, the
        reasoning of earlier turns stays in the prompt as it was written, so
        after a new user message the whole conversation is reused; without it
        (Qwen3.6's default, and every Qwen3.5 template) the previous turn is
        re-rendered without its reasoning and processed again from its first
        reply on, tool outputs included."""
        tok = self.backend.tokenizer
        msgs = normalize_messages(messages, images=self.vision is not None)
        extra = {"reasoning_effort": effort} if effort and thinking else {}
        if keep_reasoning is not None:
            extra["preserve_thinking"] = keep_reasoning
        text = tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=thinking, **extra)
        tokens = tok.encode(text, add_special_tokens=False)
        pictures = [p["image"] for m in msgs if isinstance(m["content"], list) for p in m["content"]
                    if p["type"] == "image"]
        if pictures:
            tokens = self._place(tokens, pictures)
        starts = [i for i, t in enumerate(tokens) if t == self._im_start]
        has_system = bool(tools) or (msgs and msgs[0].get("role") == "system")
        pin = starts[1] if has_system and len(starts) > 1 else None
        # The reply being generated will be re-rendered from here on the next
        # request, so the start of the generation prompt is where it branches.
        points = [p for p in (pin, starts[-1] if starts else None) if p is not None]
        return tokens, points, pin

    def _place(self, tokens: list, pictures: list) -> list:
        """Widens each picture's <|image_pad|> to its rows, keyed by image_keys."""
        pad = self.backend.image_token
        at = [i for i, t in enumerate(tokens) if t == pad]
        if len(at) != len(pictures):
            raise ValueError(f"{len(pictures)} images and {len(at)} places for them in the prompt "
                             "(the text may not contain '<|image_pad|>')")
        out, prev = [], 0
        for i, pic in zip(at, pictures):
            n, width = pic.rows(self.vision)
            entry = self._pictures.setdefault(pic.digest, [pic, width, None])
            if entry[1] != width:                   # another size limit since: its rows change
                self._pictures[pic.digest] = [pic, width, None]
            self._pictures.move_to_end(pic.digest)
            out += tokens[prev:i] + image_keys(pic.digest, n)
            prev = i + 1
        while len(self._pictures) > 64:             # the oldest: a new request would bring them again
            self._pictures.popitem(last=False)
        return out + tokens[prev:]

    def _image_rows(self, digest: int) -> tuple:
        """(the tower's rows of a picture of the current prompt, its width), computed once."""
        entry = self._pictures[digest]
        if entry[2] is None:
            if self._tower is None:
                from .vision import load_tower

                tower = load_tower(self.vision_path)
                source = getattr(self.backend, "source", "")
                if source and tower.source and not source.startswith(tower.source):
                    raise ValueError(f"the vision file {self.vision_path} is {tower.source}'s, the model "
                                     f"{source}")
                self._tower = tower
            vectors, width = self._tower.encode(entry[0].open())
            if width != entry[1]:
                raise RuntimeError(f"the tower made an image {width} rows wide, the prompt has {entry[1]}")
            entry[2] = vectors
            self._tower.mx.clear_cache()            # the tower's working memory back to the system
        return entry[2], entry[1]

    def chat(self, messages: list, tools: Optional[list] = None, sampling: Optional[Sampling] = None,
             max_tokens: int = 2048, thinking: bool = True, stop: Optional[list] = None,
             progress=None, effort: Optional[str] = None, keep_reasoning: Optional[bool] = None) -> Iterator[tuple]:
        """Yields ("reasoning", text), ("content", text), ("tool_call", {...}) and
        finally ("done", stats)."""
        with self.lock:
            yield from self._chat(messages, tools, sampling or self.defaults, max_tokens, thinking, stop or [],
                                  progress, effort, keep_reasoning)

    def _chat(self, messages, tools, sampling, max_tokens, thinking, stop, progress, effort, keep_reasoning):
        tok = self.backend.tokenizer
        prompt, points, pin = self.render(messages, tools, thinking, effort, keep_reasoning)
        limit = self.backend.max_context
        if limit is not None:
            if len(prompt) + 8 > limit:
                raise ContextFull(len(prompt), limit)
            max_tokens = min(max_tokens, limit - len(prompt) - 2)
        counters = dict(getattr(self.backend, "counters", {}))
        experts = getattr(self.backend, "expert_cache", lambda: None)
        experts0 = experts()
        t0 = time.perf_counter()
        usage = self.session.sync(prompt, points, pin, progress, images=self._image_rows)
        self.prompt_usage = usage     # for a reply that reports it before the first token (Anthropic's stream)
        t_prefill = time.perf_counter() - t0

        channel = "reasoning" if thinking else "content"
        decoder = _Decoder(tok)
        call_ids: list = []
        answer = ""
        fresh = True            # at the start of a channel: drop leading newlines
        n_out, t_first, finish, stop_sequence = 0, None, "length", None
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
                cut, matched = min(((answer.find(s), s) for s in stop if s in answer), default=(-1, None))
                if cut >= 0:        # a stop string completed: emit only what precedes it
                    keep = len(text) - (len(answer) - cut)
                    if keep > 0:
                        yield ("content", text[:keep])
                    finish, stop_sequence = "stop", matched
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
            finish_reason=finish, stop_sequence=stop_sequence,
            prompt_tokens=usage["prompt_tokens"], cached_tokens=usage["cached_tokens"],
            completion_tokens=n_out, prefill_seconds=t_prefill, decode_seconds=t_decode,
            time_to_first_token=t_first if t_first is not None else t_prefill,
            prefill_tokens_per_second=fed / t_prefill if fed and t_prefill > 0 else 0.0,
            decode_tokens_per_second=n_out / t_decode if n_out and t_decode > 0 else 0.0))
