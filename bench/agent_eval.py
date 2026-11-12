#!/usr/bin/env python3
"""How often and how fast the agent solves small coding tasks.

Every task of bench/agent_tasks.py runs on a fresh copy of its files, always
at the same path (so the system prompt and its cached state carry over from
task to task, as in a session), with every tool call approved. When the agent
stops, the task's hidden check decides whether it was solved.

Sampling uses the model's defaults with a fixed seed. Drafts never change
what is generated, so two runs that differ only in how drafts are made (for
example RACELINE_LOOKUP=0) write the same tokens, and their times compare
directly.

  bench/agent_eval.py models/qwen3.8-27b.rlm
  RACELINE_LOOKUP=0 bench/agent_eval.py models/qwen3.8-27b.rlm --json out.json
  bench/agent_eval.py models/qwen3.5-4b.rlm --tasks rename,pathlib --transcripts out/

Measure with nothing else using the GPU.
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_tasks import TASKS  # noqa: E402

FIXED_TIME = 1767268800   # 2026-01-01 12:00 UTC: the files' modification time


def warn_if_busy():
    """Speed numbers taken next to other work are worthless: say so up front."""
    load = os.getloadavg()[0]
    if load > 3:
        print(f"warning: the load average is {load:.1f}; other processes are using the machine and the "
              "numbers will be low (`ps -Ao pid,%cpu,command -r | head`)", file=sys.stderr, flush=True)


def check(task, work):
    """-> (solved, reason it was not)."""
    from raceline.agent import run_command

    code = f"ORIGINAL_FILES = {task['files']!r}\n{task['check']}"
    r = run_command([sys.executable, "-c", code], work, 120, shell=False)   # kills what it started, too
    if r is None:
        return False, "check timed out"
    for name in task["keep"]:
        path = os.path.join(work, name)
        if not os.path.isfile(path) or open(path, encoding="utf-8").read() != task["files"][name]:
            return False, f"{name} was changed"
    if r.returncode != 0:
        lines = (r.stderr or r.stdout).strip().splitlines()
        return False, lines[-1][:200] if lines else f"exit code {r.returncode}"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--tasks", help="comma-separated names (default: all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--presence-penalty", type=float)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "xhigh"])
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--transcripts", help="directory to write each task's transcript to")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()

    from raceline.agent import Agent
    from raceline.backends import load_backend
    from raceline.chat import Runtime
    from raceline.cli import _defaults

    warn_if_busy()
    names = args.tasks.split(",") if args.tasks else list(TASKS)
    backend = load_backend(args.model)
    sampling = _defaults(args.model, backend)
    sampling.seed = args.seed
    for key in ("temperature", "top_p", "top_k", "presence_penalty"):
        if getattr(args, key) is not None:
            setattr(sampling, key, getattr(args, key))
    print(f"sampling: temperature {sampling.temperature}, top_p {sampling.top_p}, top_k {sampling.top_k}, "
          f"presence_penalty {sampling.presence_penalty}, seed {sampling.seed}")
    rt = Runtime(backend, sampling)
    work = os.path.join(tempfile.gettempdir(), "raceline-agent-eval")
    for _ in rt.chat([{"role": "user", "content": "Hi"}], thinking=False, max_tokens=8):
        pass          # warm up: compile the kernels, page the weights in
    if args.transcripts:
        os.makedirs(args.transcripts, exist_ok=True)

    results = []
    print(f"{'task':16s} {'result':8s} {'steps':>5s} {'time':>7s} {'tokens':>7s} {'decode':>11s} {'drafts ok':>9s} "
          f"{'copied':>7s}  note")
    for name in names:
        task = TASKS[name]
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        for path, text in task["files"].items():
            with open(os.path.join(work, path), "w", encoding="utf-8") as f:
                f.write(text)
            os.utime(os.path.join(work, path), (FIXED_TIME, FIXED_TIME))
        os.utime(work, (FIXED_TIME, FIXED_TIME))     # so `ls -l` prints the same in every run
        agent = Agent(rt, work, auto_approve=True, thinking=not args.no_think, max_tokens=4096,
                      max_steps=args.max_steps, interactive=False, color=False, effort=args.effort)
        before = dict(getattr(backend, "counters", {}))
        log = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(log):
            totals = agent.turn(task["prompt"])
        seconds = time.perf_counter() - t0
        solved, why = check(task, work)
        c = {k: v - before.get(k, 0) for k, v in getattr(backend, "counters", {}).items()}
        res = dict(task=name, solved=solved, why=why, steps=totals["steps"], seconds=seconds,
                   completion_tokens=totals["completion_tokens"], decode_seconds=totals.get("decode_seconds", 0.0),
                   prefill_seconds=totals.get("prefill_seconds", 0.0), context_tokens=totals["last"]["prompt_tokens"]
                   if totals.get("last") else 0, **c)
        results.append(res)
        if args.transcripts:
            with open(os.path.join(args.transcripts, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"> {task['prompt']}\n{log.getvalue()}\n[{'solved' if solved else 'not solved: ' + why}]\n")
        tps = res["completion_tokens"] / res["decode_seconds"] if res["decode_seconds"] else 0.0
        ok = f"{100 * c['accepted'] / c['drafts']:.0f}%" if c.get("drafts") else "-"
        copied = f"{100 * c['lookup_accepted'] / res['completion_tokens']:.0f}%" if c.get("lookup_drafts") else "-"
        print(f"{name:16s} {'solved' if solved else 'FAILED':8s} {res['steps']:5d} {seconds:6.1f}s "
              f"{res['completion_tokens']:7d} {tps:6.1f} tok/s {ok:>9s} {copied:>7s}  {why}", flush=True)

    n_ok = sum(r["solved"] for r in results)
    out = sum(r["completion_tokens"] for r in results)
    dec = sum(r["decode_seconds"] for r in results)
    total = sum(r["seconds"] for r in results)
    print(f"\n{n_ok}/{len(results)} solved; {total:.0f}s in all; {out} tokens written at {out / dec:.1f} tok/s"
          if dec else f"\n{n_ok}/{len(results)} solved")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(model=args.model, backend=backend.name, sampling=dataclasses.asdict(sampling),
                           thinking=not args.no_think,
                           effort=args.effort, lookup=getattr(backend, "lookup_drafts", None), tasks=results), f, indent=1)


if __name__ == "__main__":
    main()
