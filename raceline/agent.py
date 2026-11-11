"""A terminal agent: the local model, a few tools, and a loop.

Reads are free. Anything that changes the machine (shell commands, file
writes) asks first, unless the user started the agent with --yes.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import re
import subprocess
import sys
import time

from .chat import Runtime

MAX_OUTPUT = 12_000   # characters of tool output kept in the conversation


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = [
    _fn("bash", "Run a shell command in the working directory and return its output.",
        {"command": {"type": "string", "description": "The command line to run."},
         "timeout": {"type": "integer", "description": "Seconds before the command is stopped. Default 60."}},
        ["command"]),
    _fn("read_file", "Read a text file. Returns numbered lines.",
        {"path": {"type": "string", "description": "Path relative to the working directory."},
         "offset": {"type": "integer", "description": "First line to return, 1-based. Default 1."},
         "limit": {"type": "integer", "description": "Maximum number of lines. Default 400."}},
        ["path"]),
    _fn("write_file", "Create a file or replace its whole content.",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _fn("edit_file", "Replace one exact occurrence of old_string in a file with new_string.",
        {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
        ["path", "old_string", "new_string"]),
    _fn("list_dir", "List the entries of a directory.", {"path": {"type": "string"}}, ["path"]),
    _fn("find_files", "Find files by name. Returns paths under the directory that match a glob pattern.",
        {"pattern": {"type": "string", "description": "Glob for the file name, for example *.py or test_*."},
         "path": {"type": "string", "description": "Directory to search. Default: the working directory."}},
        ["pattern"]),
    _fn("search", "Search file contents with a regular expression. Returns matching lines as path:line: text.",
        {"pattern": {"type": "string", "description": "Regular expression (Python syntax)."},
         "path": {"type": "string", "description": "File or directory to search. Default: the working directory."},
         "glob": {"type": "string", "description": "Only search files whose name matches this glob, for example *.py."}},
        ["pattern"]),
]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".mypy_cache", ".pytest_cache"}
MAX_MATCHES = 200

SYSTEM = """You are raceline, an AI assistant running entirely on this Mac. You help with software and system tasks in the user's working directory.

Working directory: {cwd}
Platform: macOS {release} ({machine})
Date: {date}

How to work:
- Paths are relative to the working directory: write src/main.py, not an absolute path.
- Look before you answer: list directories, find and search files, read them. Do not guess at file contents.
- Make changes with edit_file or write_file, not with shell commands. Keep them minimal and focused on the request.
- The shell is macOS zsh with BSD tools (for example, sed -i needs an explicit empty suffix: sed -i '').
- Shell commands and file writes need the user's approval. If one is declined, do not retry it; ask what they want instead.
- Be concise. When a task is done, say what you did in a few sentences."""


def _short(text: str, limit: int = 72) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    head, tail = MAX_OUTPUT * 2 // 3, MAX_OUTPUT // 3
    return f"{text[:head]}\n[... {len(text) - head - tail} characters omitted ...]\n{text[-tail:]}"


class Tools:
    def __init__(self, cwd: str, approve):
        self.cwd = cwd
        self.approve = approve   # (summary: str) -> bool

    def _path(self, path: str) -> str:
        return os.path.abspath(os.path.join(self.cwd, os.path.expanduser(path)))

    def _show(self, full: str) -> str:
        """A path as the model and the user should see it: relative to the working directory when inside it."""
        rel = os.path.relpath(full, self.cwd)
        return full if rel.startswith("..") else rel

    def run(self, name: str, args: dict) -> str:
        try:
            handler = getattr(self, "tool_" + name, None)
            if handler is None:
                return f"Error: there is no tool named '{name}'."
            return _clip(handler(**args))
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def tool_bash(self, command: str, timeout: int = 60) -> str:
        if not self.approve(f"run: {command}"):
            return "The user declined to run this command."
        try:
            r = subprocess.run(command, shell=True, cwd=self.cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"The command was stopped after {timeout} seconds."
        out = (r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")).strip()
        return f"[exit code {r.returncode}]\n{out}" if out else f"[exit code {r.returncode}, no output]"

    def tool_read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        with open(self._path(path), encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        chosen = lines[max(offset, 1) - 1:max(offset, 1) - 1 + limit]
        body = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(chosen, max(offset, 1)))
        more = len(lines) - (max(offset, 1) - 1 + len(chosen))
        return body + (f"\n[{more} more lines]" if more > 0 else "")

    def tool_write_file(self, path: str, content: str) -> str:
        full = self._path(path)
        verb = "overwrite" if os.path.exists(full) else "create"
        if not self.approve(f"{verb} {self._show(full)} ({len(content)} characters)"):
            return "The user declined this write."
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {self._show(full)}."

    def tool_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        full = self._path(path)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        count = text.count(old_string)
        if count != 1:
            return f"Error: old_string occurs {count} times in {self._show(full)}; it must occur exactly once."
        if not self.approve(f"edit {self._show(full)} (replace {len(old_string)} characters with {len(new_string)})"):
            return "The user declined this edit."
        with open(full, "w", encoding="utf-8") as f:
            f.write(text.replace(old_string, new_string, 1))
        return f"Edited {self._show(full)}."

    def tool_list_dir(self, path: str = ".") -> str:
        full = self._path(path)
        entries = sorted(os.listdir(full))
        return "\n".join(e + ("/" if os.path.isdir(os.path.join(full, e)) else "") for e in entries) or "(empty)"

    def _walk(self, path: str):
        full = self._path(path)
        if os.path.isfile(full):
            yield full
            return
        for root, dirs, files in os.walk(full):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(files):
                yield os.path.join(root, name)

    def tool_find_files(self, pattern: str, path: str = ".") -> str:
        hits = [os.path.relpath(f, self.cwd) for f in self._walk(path) if fnmatch.fnmatch(os.path.basename(f), pattern)]
        more = f"\n[{len(hits) - MAX_MATCHES} more]" if len(hits) > MAX_MATCHES else ""
        return ("\n".join(hits[:MAX_MATCHES]) + more) or "No files match."

    def tool_search(self, pattern: str, path: str = ".", glob: str = "") -> str:
        rx = re.compile(pattern)
        hits = []
        for f in self._walk(path):
            if glob and not fnmatch.fnmatch(os.path.basename(f), glob):
                continue
            try:
                if os.path.getsize(f) > 2_000_000:
                    continue
                with open(f, encoding="utf-8") as fh:
                    for n, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(f, self.cwd)}:{n}: {line.rstrip()[:240]}")
                            if len(hits) > MAX_MATCHES:
                                return "\n".join(hits[:MAX_MATCHES]) + "\n[more matches not shown: narrow the search]"
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable
        return "\n".join(hits) or "No matches."


class Agent:
    def __init__(self, runtime: Runtime, cwd: str, auto_approve: bool = False, thinking: bool = True,
                 max_tokens: int = 4096, max_steps: int = 40, interactive: bool = True, color: bool = True,
                 effort: str = "low"):
        self.rt = runtime
        self.cwd = os.path.abspath(cwd)
        self.thinking, self.max_tokens, self.max_steps, self.effort = thinking, max_tokens, max_steps, effort
        self.auto_approve, self.interactive = auto_approve, interactive
        self.c = (lambda code, s: f"\033[{code}m{s}\033[0m") if color else (lambda code, s: s)
        self.tools = Tools(self.cwd, self._approve)
        self.messages = [{"role": "system", "content": SYSTEM.format(
            cwd=self.cwd, release=platform.mac_ver()[0], machine=platform.machine(), date=time.strftime("%Y-%m-%d"))}]

    def _approve(self, summary: str) -> bool:
        if self.auto_approve:
            return True
        if not self.interactive:
            return False
        try:
            reply = input(self.c("33", f"\n  allow {summary}? [y/N/a=always] ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if reply == "a":
            self.auto_approve = True
        return reply in ("y", "yes", "a")

    def reset(self) -> None:
        del self.messages[1:]

    def _make_room(self, used: int) -> None:
        """When the context is three quarters full, drops the bodies of the
        oldest tool results (the model can always run a tool again). What
        precedes the first dropped body stays cached."""
        limit = self.rt.backend.max_context
        if limit is None or used < 0.75 * limit:
            return
        tok = self.rt.backend.tokenizer
        excess = used - limit // 2
        for m in self.messages[:-6]:
            if excess <= 0:
                break
            if m["role"] == "tool" and len(m["content"]) > 200:
                excess -= len(tok.encode(m["content"], add_special_tokens=False))
                m["content"] = "[output dropped to save context; run the tool again if it is needed]"

    def turn(self, text: str) -> dict:
        """Runs one user request to completion (the model may take many tool steps)."""
        self.messages.append({"role": "user", "content": text})
        totals = dict(prompt_tokens=0, cached_tokens=0, completion_tokens=0, steps=0, seconds=0.0)
        out = sys.stdout
        for _ in range(self.max_steps):
            content, reasoning, calls, stats, shown = "", "", [], None, None
            t0 = time.perf_counter()
            try:
                for kind, value in self.rt.chat(self.messages, TOOLS, max_tokens=self.max_tokens,
                                                thinking=self.thinking, effort=self.effort):
                    if kind == "reasoning":
                        reasoning += value
                        if shown != kind:
                            out.write(self.c("2", "\n  thinking: "))
                        out.write(self.c("2", value.replace("\n", "\n  ")))
                    elif kind == "content":
                        content += value
                        if shown != kind:
                            out.write("\n")
                        out.write(value)
                    elif kind == "tool_call":
                        calls.append(value)
                    else:
                        stats = value
                    if kind in ("reasoning", "content"):
                        shown = kind
                        out.flush()
            except KeyboardInterrupt:
                out.write(self.c("33", "\n  [interrupted]\n"))
                self.messages.append({"role": "assistant", "content": content})
                break
            except ValueError as e:      # the conversation no longer fits the context window
                out.write(self.c("31", f"\n  {e}. /reset starts a new conversation.\n"))
                break
            totals["seconds"] += time.perf_counter() - t0
            totals["steps"] += 1
            for k in ("prompt_tokens", "cached_tokens", "completion_tokens"):
                totals[k] += stats[k]
            totals["last"] = stats

            message = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            if calls:
                message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]
            self.messages.append(message)
            self._make_room(stats["prompt_tokens"] + stats["completion_tokens"])
            if not calls:
                break
            for call in calls:
                brief = ", ".join(f"{k}={_short(json.dumps(v, ensure_ascii=False))}" for k, v in call["arguments"].items())
                out.write(self.c("36", f"\n  > {call['name']}({brief})\n"))
                out.flush()
                result = self.tools.run(call["name"], call["arguments"])
                first = result.strip().split("\n")
                out.write(self.c("2", "    " + first[0][:160] + (f"  [+{len(first) - 1} lines]" if len(first) > 1 else "") + "\n"))
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        out.write("\n")
        return totals

    def status(self, totals: dict) -> str:
        last = totals.get("last")
        if not last:
            return ""
        return (f"  [{totals['steps']} step{'s' if totals['steps'] != 1 else ''} · context {last['prompt_tokens']} tok "
                f"({last['cached_tokens']} reused) · {totals['completion_tokens']} out at "
                f"{last['decode_tokens_per_second']:.1f} tok/s"
                + (f", {100 * last['drafts_accepted'] / last['drafts']:.0f}% of drafts accepted" if last.get("drafts") else "")
                + f" · {totals['seconds']:.1f}s]")


def repl(agent: Agent) -> None:
    try:
        import readline  # noqa: F401  (line editing and history for input())
    except ImportError:
        pass
    b = agent.rt.backend
    print(agent.c("1", "raceline") + f" · {getattr(b, 'model_id', '')} · {b.name} backend"
          + (" · speculative decoding" if getattr(b, "mtp", False) else "") + f" · {agent.cwd}")
    print(agent.c("2", "  /reset clears the conversation, /exit quits, Ctrl-C interrupts a reply."))
    while True:
        try:
            text = input(agent.c("1", "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        if text == "/reset":
            agent.reset()
            print(agent.c("2", "  conversation cleared"))
            continue
        totals = agent.turn(text)
        print(agent.c("2", agent.status(totals)))
