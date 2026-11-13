"""Anthropic's Messages API on top of the runtime (`POST /v1/messages`).

Clients written for Anthropic's API, such as the anthropic SDKs or agents that
take a base URL, can use the local model through it. A request is translated
into the runtime's OpenAI-style messages, and the reply back into content
blocks (thinking, text, tool_use), streamed as Anthropic's server-sent events
when the client asks for a stream.

Mapping:
- `system` (a string or text blocks) is the system message; `tools`
  (name, description, input_schema) become functions; server tools that run
  on Anthropic's side (a `type` other than "custom") are left out.
- `thinking: {"type": "enabled" | "adaptive"}` turns the model's reasoning on;
  without it the model answers directly, as the API specifies.
- tool_result blocks become tool messages; thinking blocks of earlier replies
  become their reasoning; a system message inside the conversation becomes a
  user message there (the Qwen templates take one system prompt, first). Images and documents are replaced by a note: the
  model runs without vision input (#28).
- usage: `input_tokens` counts what was processed, `cache_read_input_tokens`
  what the session reused.

Not supported: forcing a tool (`tool_choice` any / tool works as auto) and
the cache_control markers, which are ignored: the session reuses what it can.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _text(content) -> str:
    """A string or a list of blocks -> its text, with a note for anything that is not text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind in ("image", "document"):
            parts.append(f"[{kind} left out: this model runs without {kind} input]")
    return "\n".join(parts)


def _parts(content):
    """User or tool-result content -> its text, or with images a list of text parts and the image
    blocks themselves (raceline.chat.normalize_messages reads them, or notes them for a model
    without image input)."""
    if isinstance(content, str) or not any(isinstance(b, dict) and b.get("type") == "image" for b in content or []):
        return _text(content)
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            parts.append(block)
        else:
            text = _text([block])
            if parts and parts[-1].get("type") == "text":
                parts[-1] = {"type": "text", "text": parts[-1]["text"] + "\n" + text}
            elif text:
                parts.append({"type": "text", "text": text})
    return parts


def to_runtime(body: dict) -> dict:
    """An Anthropic request -> keyword arguments of Runtime.chat (messages, tools, thinking, stop)."""
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": _text(body["system"])})
    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"unknown role '{role}'")
        if isinstance(content, str) or role == "system":   # system: instructions inserted by the client
            messages.append({"role": role, "content": _text(content)})
            continue
        if role == "user":
            rest = []
            for block in content or []:
                if block.get("type") == "tool_result":
                    result = _parts(block.get("content", ""))
                    if block.get("is_error"):
                        result = ("Error: " + result if isinstance(result, str)
                                  else [{"type": "text", "text": "Error: "}] + result)
                    messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": result})
                else:
                    rest.append(block)
            user = _parts(rest)
            if user:
                messages.append({"role": "user", "content": user})
        else:
            text, reasoning, calls = [], [], []
            for block in content or []:
                kind = block.get("type")
                if kind == "text":
                    text.append(block.get("text", ""))
                elif kind == "thinking":
                    reasoning.append(block.get("thinking", ""))
                elif kind == "tool_use":
                    calls.append({"id": block.get("id"), "type": "function", "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}})
            message = {"role": "assistant", "content": "".join(text)}
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            if calls:
                message["tool_calls"] = calls
            messages.append(message)

    tools = None
    if (body.get("tool_choice") or {}).get("type") != "none":
        tools = [{"type": "function", "function": {
                    "name": t["name"], "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
                 for t in body.get("tools") or [] if t.get("type") in (None, "custom")] or None
    thinking = (body.get("thinking") or {}).get("type") in ("enabled", "adaptive")
    stop = body.get("stop_sequences") or None
    return dict(messages=messages, tools=tools, thinking=thinking, stop=stop)


def usage(stats: dict) -> dict:
    return {"input_tokens": stats["prompt_tokens"] - stats["cached_tokens"],
            "cache_read_input_tokens": stats["cached_tokens"], "cache_creation_input_tokens": 0,
            "output_tokens": stats["completion_tokens"]}


def stop_reason(stats: dict) -> str:
    if stats.get("stop_sequence"):
        return "stop_sequence"
    return STOP_REASONS.get(stats["finish_reason"], "end_turn")


def message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def complete(events, model: str) -> tuple:
    """Runtime events -> (the whole message, the final stats)."""
    reasoning, text, blocks, stats = "", "", [], None
    for kind, value in events:
        if kind == "reasoning":
            reasoning += value
        elif kind == "content":
            text += value
        elif kind == "tool_call":
            blocks.append({"type": "tool_use", "id": value["id"], "name": value["name"], "input": value["arguments"]})
        else:
            stats = value
    content = ([{"type": "thinking", "thinking": reasoning, "signature": ""}] if reasoning else []) + \
              ([{"type": "text", "text": text}] if text or not blocks else []) + blocks
    return {"id": message_id(), "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason(stats), "stop_sequence": stats.get("stop_sequence"),
            "usage": usage(stats)}, stats


class Streamer:
    """Runtime events -> Anthropic's stream of (event, data) pairs."""

    def __init__(self, model: str):
        self.model = model
        self.index = -1
        self.open: Optional[str] = None     # type of the content block being streamed

    def start(self, prompt_tokens: int, cached_tokens: int) -> list:
        return [("message_start", {"type": "message_start", "message": {
            "id": message_id(), "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens - cached_tokens, "cache_read_input_tokens": cached_tokens,
                      "cache_creation_input_tokens": 0, "output_tokens": 0}}})]

    def _close(self) -> list:
        out = []
        if self.open == "thinking":
            out.append(("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                "delta": {"type": "signature_delta", "signature": ""}}))
        if self.open is not None:
            out.append(("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        self.open = None
        return out

    def _begin(self, block: dict) -> list:
        out = self._close()
        self.index += 1
        self.open = block["type"]
        out.append(("content_block_start", {"type": "content_block_start", "index": self.index,
                                            "content_block": block}))
        return out

    def event(self, kind: str, value) -> list:
        if kind == "reasoning":
            out = [] if self.open == "thinking" else self._begin({"type": "thinking", "thinking": "", "signature": ""})
            return out + [("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                   "delta": {"type": "thinking_delta", "thinking": value}})]
        if kind == "content":
            out = [] if self.open == "text" else self._begin({"type": "text", "text": ""})
            return out + [("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                   "delta": {"type": "text_delta", "text": value}})]
        if kind == "tool_call":
            out = self._begin({"type": "tool_use", "id": value["id"], "name": value["name"], "input": {}})
            out.append(("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(value["arguments"], ensure_ascii=False)}}))
            return out + self._close()
        return []

    def finish(self, stats: dict) -> list:
        out = self._close()
        if self.index < 0:      # nothing was written: still one (empty) text block
            out += self._begin({"type": "text", "text": ""}) + self._close()
        return out + [
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop_reason(stats), "stop_sequence": stats.get("stop_sequence")},
                               "usage": {"output_tokens": stats["completion_tokens"]}}),
            ("message_stop", {"type": "message_stop"})]
