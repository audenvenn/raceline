"""OpenAI-compatible HTTP API for one local user, plus Anthropic's Messages API.

  POST /v1/chat/completions   streaming and non-streaming, tools, reasoning
  POST /v1/messages           the same in Anthropic's format (raceline/anthropic.py)
  POST /v1/messages/count_tokens
  GET  /v1/models
  GET  /health

Standard library only. Generation requests are served one at a time (the
runtime's lock makes one that arrives meanwhile wait its turn): the model has
one context, and reusing it across turns is the point.

Extensions to the OpenAI schema, all optional: `top_k`, `repetition_penalty`,
`enable_thinking` (also read from `chat_template_kwargs`), and in responses
`usage.prompt_tokens_details.cached_tokens` plus a `timings` object.
`reasoning_effort` low / medium / high selects the model's reasoning effort
where its chat template has one (Qwen3.8); none / minimal turns thinking off.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import Runtime


def _sampling(runtime: Runtime, body: dict):
    s = runtime.defaults
    fields = {}
    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "seed"):
        if body.get(key) is not None:
            fields[key] = body[key]
    return replace(s, **fields)


def _thinking(body: dict) -> bool:
    for src in (body, body.get("chat_template_kwargs") or {}):
        if "enable_thinking" in src:
            return bool(src["enable_thinking"])
    return body.get("reasoning_effort") not in ("none", "minimal")


def _effort(body: dict):
    """OpenAI's reasoning_effort in the model's terms (low, medium, xhigh), if the client sent one."""
    value = body.get("reasoning_effort") or (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
    return {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}.get(value)


def _usage(stats: dict) -> dict:
    return {
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"],
        "prompt_tokens_details": {"cached_tokens": stats["cached_tokens"]},
    }


def _timings(stats: dict) -> dict:
    return {k: round(stats[k], 4) for k in ("time_to_first_token", "prefill_seconds", "decode_seconds",
                                            "prefill_tokens_per_second", "decode_tokens_per_second")}


def make_handler(runtime: Runtime, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # one line per request is printed by _log instead
            pass

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str):
            self._json(code, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "created": 0, "owned_by": "raceline"}]})
            elif self.path == "/health":
                self._json(200, {"status": "ok", "backend": runtime.backend.name, "model": model_id})
            else:
                self._error(404, "not found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]      # Anthropic clients add ?beta=true
            if path in ("/v1/messages", "/v1/messages/count_tokens"):
                return self._anthropic(path)
            if path not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, "not found")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                messages = body["messages"]
            except (ValueError, KeyError) as e:
                return self._error(400, f"bad request: {e}")
            tools = body.get("tools") if body.get("tool_choice") != "none" else None
            stop = body.get("stop")
            stop = [stop] if isinstance(stop, str) else stop
            max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 4096
            self._started = False
            events = runtime.chat(messages, tools, _sampling(runtime, body), int(max_tokens), _thinking(body), stop,
                                  effort=_effort(body))
            try:
                if body.get("stream"):
                    self._stream(events, body)
                else:
                    self._complete(events)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away; the session keeps whatever was generated
            except Exception as e:  # report instead of dropping the connection
                self._fail(400 if isinstance(e, ValueError) else 500,
                           str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}")
            finally:
                # Ends the generation now (and releases the runtime's lock), not
                # whenever the abandoned generator happens to be collected.
                events.close()

        def _fail(self, code: int, message: str):
            try:
                if self._started:   # mid-stream: the status line is out, so say it in the stream
                    self._send_event({"error": {"message": message, "type": "server_error"}})
                    self._send_event("[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self._error(code, message)
            except Exception:
                pass

        def _send_event(self, payload):
            data = ("data: " + (payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))
                    + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        # -------------------------------------------------- non-streaming
        def _complete(self, events):
            content, reasoning, calls, stats = "", "", [], None
            for kind, value in events:
                if kind == "content":
                    content += value
                elif kind == "reasoning":
                    reasoning += value
                elif kind == "tool_call":
                    calls.append(value)
                else:
                    stats = value
            message = {"role": "assistant", "content": content or None}
            if reasoning:
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                    for c in calls]
            self._log(stats)
            self._json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": message, "finish_reason": stats["finish_reason"]}],
                "usage": _usage(stats), "timings": _timings(stats)})

        # ------------------------------------------------------ streaming
        def _stream(self, events, body):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._started = True
            rid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            send = self._send_event

            def chunk(delta, finish=None, **extra):
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

            send(chunk({"role": "assistant", "content": ""}))
            n_calls, stats = 0, None
            for kind, value in events:
                if kind == "content":
                    send(chunk({"content": value}))
                elif kind == "reasoning":
                    send(chunk({"reasoning_content": value}))
                elif kind == "tool_call":
                    send(chunk({"tool_calls": [{
                        "index": n_calls, "id": value["id"], "type": "function",
                        "function": {"name": value["name"],
                                     "arguments": json.dumps(value["arguments"], ensure_ascii=False)}}]}))
                    n_calls += 1
                else:
                    stats = value
            send(chunk({}, stats["finish_reason"]))
            if (body.get("stream_options") or {}).get("include_usage"):
                send({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                      "choices": [], "usage": _usage(stats), "timings": _timings(stats)})
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self._log(stats)

        # ------------------------------------------- Anthropic Messages API
        def _anthropic_error(self, code: int, message: str, kind: str = "invalid_request_error"):
            self._json(code, {"type": "error", "error": {"type": kind, "message": message}})

        def _sse(self, event: str, payload: dict):
            data = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def _anthropic(self, path: str):
            from . import anthropic

            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                args = anthropic.to_runtime(body)
                if path.endswith("/count_tokens"):
                    tokens, _, _ = runtime.render(args["messages"], args["tools"], args["thinking"])
                    return self._json(200, {"input_tokens": len(tokens)})
                max_tokens = int(body["max_tokens"])
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self._anthropic_error(400, f"bad request: {e}")
            self._started = False
            events = runtime.chat(args["messages"], args["tools"], _sampling(runtime, body), max_tokens,
                                  args["thinking"], args["stop"])
            try:
                if body.get("stream"):
                    self._anthropic_stream(events, anthropic.Streamer(model_id))
                else:
                    message, stats = anthropic.complete(events, model_id)
                    self._log(stats)
                    self._json(200, message)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                message = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
                try:
                    if self._started:
                        self._sse("error", {"type": "error", "error": {"type": "api_error", "message": message}})
                        self.wfile.write(b"0\r\n\r\n")
                    else:
                        self._anthropic_error(400 if isinstance(e, ValueError) else 500, message,
                                              "invalid_request_error" if isinstance(e, ValueError) else "api_error")
                except Exception:
                    pass
            finally:
                events.close()

        def _anthropic_stream(self, events, streamer):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._started = True
            begun, stats = False, None
            for kind, value in events:
                if not begun:       # the prompt has been read: its size is known
                    u = runtime.prompt_usage
                    for ev in streamer.start(u["prompt_tokens"], u["cached_tokens"]):
                        self._sse(*ev)
                    begun = True
                if kind == "done":
                    stats = value
                else:
                    for ev in streamer.event(kind, value):
                        self._sse(*ev)
            for ev in streamer.finish(stats):
                self._sse(*ev)
            self.wfile.write(b"0\r\n\r\n")
            self._log(stats)

        def _log(self, s):
            print(f"[{time.strftime('%H:%M:%S')}] prompt {s['prompt_tokens']} tok ({s['cached_tokens']} cached, "
                  f"{s['prefill_seconds']:.2f}s)  out {s['completion_tokens']} tok at "
                  f"{s['decode_tokens_per_second']:.1f} tok/s  first token {s['time_to_first_token']:.2f}s  "
                  f"[{s['finish_reason']}]", flush=True)

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return   # a client that went away; nothing to report
        super().handle_error(request, client_address)


def serve(runtime: Runtime, model_id: str, host: str = "127.0.0.1", port: int = 8080):
    server = Server((host, port), make_handler(runtime, model_id))
    print(f"raceline: {model_id} on the {runtime.backend.name} backend")
    print(f"OpenAI-compatible API at http://{host}:{port}/v1, Anthropic's Messages API at /v1/messages "
          "(Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
