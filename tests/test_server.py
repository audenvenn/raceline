#!/usr/bin/env python3
"""The OpenAI-compatible server, against a scripted runtime (no GPU, no model).

  python3 tests/test_server.py
"""

import http.client
import json
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import Sampling  # noqa: E402
from raceline.server import Server, make_handler  # noqa: E402

STATS = dict(finish_reason="stop", prompt_tokens=12, cached_tokens=4, completion_tokens=3, prefill_seconds=0.1,
             decode_seconds=0.2, time_to_first_token=0.12, prefill_tokens_per_second=80.0,
             decode_tokens_per_second=15.0)


class ScriptedRuntime:
    """Plays back a list of events per request and records how each generation ended."""

    class backend:
        name = "scripted"

    def __init__(self):
        self.defaults = Sampling()
        self.script, self.calls, self.ended = [], [], []
        self.prompt_usage = {"prompt_tokens": STATS["prompt_tokens"], "cached_tokens": STATS["cached_tokens"]}

    def render(self, messages, tools, thinking, effort=None, keep_reasoning=None):
        return [0] * (7 * len(messages)), [], None

    def chat(self, messages, tools=None, sampling=None, max_tokens=2048, thinking=True, stop=None, progress=None,
             effort=None):
        self.calls.append(dict(messages=messages, tools=tools, sampling=sampling, thinking=thinking, effort=effort))
        try:
            for ev in self.script:
                if isinstance(ev, Exception):
                    raise ev
                if ev == "slow":
                    time.sleep(0.05)
                    continue
                yield ev
            self.ended.append("finished")
        except GeneratorExit:
            self.ended.append("closed")
            raise


def start(rt):
    server = Server(("127.0.0.1", 0), make_handler(rt, "test-model"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def post(port, body, path="/v1/chat/completions"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, r.read().decode()


def sse_of(text):
    """Anthropic's stream -> [(event, data)]."""
    out = []
    for chunk in text.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.strip().splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def events_of(text):
    out = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def test_complete_with_reasoning_and_tool_call():
    rt = ScriptedRuntime()
    rt.script = [("reasoning", "Let me look."), ("content", "Sure"),
                 ("tool_call", {"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}),
                 ("done", dict(STATS, finish_reason="tool_calls"))]
    server, port = start(rt)
    try:
        code, text = post(port, {"messages": [{"role": "developer", "content": "Be brief."},
                                              {"role": "user", "content": "hi"}],
                                 "temperature": 0.2, "reasoning_effort": "high"})
    finally:
        server.shutdown()
    assert code == 200, text
    body = json.loads(text)
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Sure" and msg["reasoning_content"] == "Let me look."
    assert msg["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.py"}'}
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert rt.calls[0]["sampling"].temperature == 0.2 and rt.calls[0]["effort"] == "xhigh"


def test_stream_is_well_formed_and_reports_usage():
    rt = ScriptedRuntime()
    rt.script = [("content", "Hel"), ("content", "lo"), ("done", STATS)]
    server, port = start(rt)
    try:
        code, text = post(port, {"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                 "stream_options": {"include_usage": True}})
    finally:
        server.shutdown()
    assert code == 200
    evs = events_of(text)
    assert evs[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in evs[:-1] if e.get("choices")]
    assert deltas[0]["role"] == "assistant"
    assert "".join(d.get("content", "") for d in deltas) == "Hello"
    finishes = [e["choices"][0]["finish_reason"] for e in evs[:-1] if e.get("choices")]
    assert finishes[-1] == "stop" and all(f is None for f in finishes[:-1])
    assert evs[-2]["usage"]["completion_tokens"] == 3 and evs[-2]["choices"] == []


def test_error_mid_stream_arrives_as_an_event():
    rt = ScriptedRuntime()
    rt.script = [("content", "partial"), ValueError("the context window is full")]
    server, port = start(rt)
    try:
        code, text = post(port, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    finally:
        server.shutdown()
    assert code == 200
    evs = events_of(text)
    assert evs[-1] == "[DONE]" and "context window" in evs[-2]["error"]["message"]


def test_bad_request_and_unknown_path():
    rt = ScriptedRuntime()
    server, port = start(rt)
    try:
        code, text = post(port, {"no": "messages"})
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/v1/models")
        models = json.loads(conn.getresponse().read())
        conn.request("GET", "/nowhere")
        missing = conn.getresponse()
        missing.read()
    finally:
        server.shutdown()
    assert code == 400 and "messages" in text
    assert models["data"][0]["id"] == "test-model" and missing.status == 404


def test_client_that_disconnects_ends_the_generation_at_once():
    rt = ScriptedRuntime()
    rt.script = [("content", "x")] + ["slow", ("content", "y")] * 2000 + [("done", STATS)]   # 100 s if not cut short
    server, port = start(rt)
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        got = b""
        while b'"content": "x"' not in got:   # mid-answer: the generation has started
            data = sock.recv(4096)
            assert data, got
            got += data
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()                            # the client goes away mid-answer (with a reset)
        for _ in range(600):                    # a slow CI machine gets 30 s to notice
            if rt.ended:
                break
            time.sleep(0.05)
    finally:
        server.shutdown()
    assert rt.ended == ["closed"], rt.ended


ANTHROPIC_REQUEST = {
    "model": "local", "max_tokens": 300, "system": [{"type": "text", "text": "Be brief."}],
    "thinking": {"type": "enabled", "budget_tokens": 1000}, "stop_sequences": ["END"],
    "tools": [{"name": "read_file", "description": "Read a file.",
               "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
              {"type": "web_search_20250305", "name": "web_search"}],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Fix a.py."},
                                     {"type": "image", "source": {"type": "base64", "data": "AAAA"}}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "Read it first.", "signature": "s"},
                                          {"type": "text", "text": "Reading."},
                                          {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                           "input": {"path": "a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x = 1",
                                      "is_error": False},
                                     {"type": "text", "text": "Go on."}]}]}


def test_anthropic_request_becomes_runtime_messages():
    from raceline.anthropic import to_runtime

    args = to_runtime(ANTHROPIC_REQUEST)
    roles = [m["role"] for m in args["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"], roles
    assert "image left out" in args["messages"][1]["content"]
    a = args["messages"][2]
    assert a["reasoning_content"] == "Read it first." and a["content"] == "Reading."
    assert json.loads(a["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert args["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "x = 1"}
    assert [t["function"]["name"] for t in args["tools"]] == ["read_file"]    # the server tool is left out
    assert args["thinking"] and args["stop"] == ["END"]
    assert to_runtime(dict(ANTHROPIC_REQUEST, thinking=None, tool_choice={"type": "none"}))["tools"] is None
    inserted = dict(ANTHROPIC_REQUEST, messages=ANTHROPIC_REQUEST["messages"] + [
        {"role": "system", "content": [{"type": "text", "text": "Be careful."}]}])
    assert to_runtime(inserted)["messages"][-1] == {"role": "system", "content": "Be careful."}


def test_anthropic_message_with_thinking_text_and_tool_use():
    rt = ScriptedRuntime()
    rt.script = [("reasoning", "Let me look."), ("content", "Sure"),
                 ("tool_call", {"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}),
                 ("done", dict(STATS, finish_reason="tool_calls"))]
    server, port = start(rt)
    try:
        code, text = post(port, ANTHROPIC_REQUEST, "/v1/messages?beta=true")
    finally:
        server.shutdown()
    assert code == 200, text
    body = json.loads(text)
    assert [b["type"] for b in body["content"]] == ["thinking", "text", "tool_use"]
    assert body["content"][2]["input"] == {"path": "a.py"} and body["stop_reason"] == "tool_use"
    assert body["usage"] == {"input_tokens": 8, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 0,
                             "output_tokens": 3}
    assert rt.calls[0]["thinking"] is True


def test_anthropic_stream_is_well_formed():
    rt = ScriptedRuntime()
    rt.script = [("reasoning", "Hm"), ("content", "Hel"), ("content", "lo"),
                 ("tool_call", {"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}),
                 ("done", dict(STATS, finish_reason="tool_calls"))]
    server, port = start(rt)
    try:
        code, text = post(port, dict(ANTHROPIC_REQUEST, stream=True), "/v1/messages")
    finally:
        server.shutdown()
    assert code == 200, text
    events = sse_of(text)
    names = [e for e, _ in events]
    assert names == ["message_start",
                     "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                     "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
                     "content_block_start", "content_block_delta", "content_block_stop",
                     "message_delta", "message_stop"], names
    assert events[0][1]["message"]["usage"]["input_tokens"] == 8
    starts = [d for e, d in events if e == "content_block_start"]
    assert [(d["index"], d["content_block"]["type"]) for d in starts] == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    text_deltas = [d["delta"]["text"] for e, d in events if e == "content_block_delta" and d["delta"]["type"] == "text_delta"]
    assert "".join(text_deltas) == "Hello"
    json_delta = next(d["delta"]["partial_json"] for e, d in events
                      if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert json.loads(json_delta) == {"path": "a.py"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use" and events[-2][1]["usage"]["output_tokens"] == 3


def test_anthropic_prompt_too_long_is_an_error_before_the_stream():
    from raceline.chat import ContextFull

    rt = ScriptedRuntime()
    server, port = start(rt)
    try:
        for stream in (False, True):
            rt.script = [ContextFull(40000, 32768)]
            code, text = post(port, dict(ANTHROPIC_REQUEST, stream=stream), "/v1/messages")
            assert code == 400, text
            assert json.loads(text)["error"]["message"] == "prompt is too long: 40000 tokens > 32768 maximum"
    finally:
        server.shutdown()


def test_anthropic_count_tokens_and_errors():
    rt = ScriptedRuntime()
    server, port = start(rt)
    try:
        code, text = post(port, ANTHROPIC_REQUEST, "/v1/messages/count_tokens")
        assert code == 200 and json.loads(text) == {"input_tokens": 7 * 5}, text
        code, text = post(port, {"messages": []}, "/v1/messages")       # no max_tokens
        assert code == 400 and json.loads(text)["type"] == "error", text
    finally:
        server.shutdown()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} passed")
