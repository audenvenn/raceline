#!/usr/bin/env python3
"""How often and how fast the agent solves small coding tasks.

Every task of bench/agent_tasks.py runs on a fresh copy of its files, always
at the same path (so the system prompt and its cached state carry over from
task to task, as in a session), with every tool call approved. When the agent
stops, the task's hidden check decides whether it was solved.

Sampling is what `raceline chat` uses (raceline.cli.agent_sampling) with a
fixed seed; --temperature and the like override it. Drafts never change
what is generated, so two runs that differ only in how drafts are made (for
example RACELINE_LOOKUP=0) write the same tokens, and their times compare
directly.

  bench/agent_eval.py models/qwen3.8-27b.rlm
  RACELINE_LOOKUP=0 bench/agent_eval.py models/qwen3.8-27b.rlm --json out.json
  bench/agent_eval.py models/qwen3.5-4b.rlm --tasks rename,pathlib --transcripts out/

--followup sends a second request after each task (once its check has run)
and reports how much of the conversation that request processed again: the
cost of a new request in a session. --drop-reasoning leaves earlier replies'
reasoning out of the prompt (the chat template's default for Qwen3.5 / 3.6).

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
    ap.add_argument("--tasks", help="comma-separated names (default: every task of --set)")
    ap.add_argument("--set", default="core", choices=["core", "extended", "all"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--presence-penalty", type=float)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "xhigh"])
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--followup", nargs="?", const="In one sentence: what did you change?",
                    help="a second request after each task (default text: %(const)r)")
    ap.add_argument("--drop-reasoning", action="store_true",
                    help="leave earlier replies' reasoning out of the prompt (no preserve_thinking)")
    ap.add_argument("--transcripts", help="directory to write each task's transcript to")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()

    from raceline.agent import Agent
    from raceline.backends import load_backend
    from raceline.chat import Runtime
    from raceline.cli import agent_sampling

    warn_if_busy()
    names = args.tasks.split(",") if args.tasks else [n for n, t in TASKS.items() if args.set in ("all", t["group"])]
    backend = load_backend(args.model)
    sampling = agent_sampling(args.model, backend, not args.no_think)   # what `raceline chat` uses
    sampling.seed = args.seed
    for key in ("temperature", "top_p", "top_k", "presence_penalty"):
        if getattr(args, key) is not None:
            setattr(sampling, key, getattr(args, key))
    print(f"sampling: temperature {sampling.temperature}, top_p {sampling.top_p}, top_k {sampling.top_k}, "
          f"presence_penalty {sampling.presence_penalty}, seed {sampling.seed}")
    rt = Runtime(backend, sampling, park_bytes=0)   # as `raceline chat`: one conversation at a time
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
            os.makedirs(os.path.dirname(os.path.join(work, path)), exist_ok=True)
            with open(os.path.join(work, path), "w", encoding="utf-8") as f:
                f.write(text)
            os.utime(os.path.join(work, path), (FIXED_TIME, FIXED_TIME))
        os.utime(work, (FIXED_TIME, FIXED_TIME))     # so `ls -l` prints the same in every run
        agent = Agent(rt, work, auto_approve=True, thinking=not args.no_think, max_tokens=4096,
                      max_steps=args.max_steps, interactive=False, color=False, effort=args.effort,
                      keep_reasoning=not args.drop_reasoning)
        before = dict(getattr(backend, "counters", {}))
        experts = getattr(backend, "expert_cache", lambda: None)   # a MoE model with its experts on disk
        e0 = experts()
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
        e1 = experts()
        if e0 and e1:
            hits, reads = e1[1] - e0[1], e1[2] - e0[2]
            res.update(experts_from_cache=hits / max(1, hits + reads), experts_read=reads)
        if args.followup:
            log.write(f"\n> {args.followup}\n")
            t1 = time.perf_counter()
            with contextlib.redirect_stdout(log):
                more = agent.turn(args.followup)
            first = more.get("first") or {}
            res.update(followup_seconds=time.perf_counter() - t1, followup_steps=more["steps"],
                       followup_prompt_tokens=first.get("prompt_tokens", 0),
                       followup_processed=first.get("prompt_tokens", 0) - first.get("cached_tokens", 0),
                       followup_prefill_seconds=first.get("prefill_seconds", 0.0))
        results.append(res)
        if args.transcripts:
            with open(os.path.join(args.transcripts, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"> {task['prompt']}\n{log.getvalue()}\n[{'solved' if solved else 'not solved: ' + why}]\n")
        tps = res["completion_tokens"] / res["decode_seconds"] if res["decode_seconds"] else 0.0
        ok = f"{100 * c['accepted'] / c['drafts']:.0f}%" if c.get("drafts") else "-"
        copied = f"{100 * c['lookup_accepted'] / res['completion_tokens']:.0f}%" if c.get("lookup_drafts") else "-"
        cached = f"  {100 * res['experts_from_cache']:.0f}% experts cached" if "experts_from_cache" in res else ""
        if args.followup:
            cached += (f"  follow-up: {res['followup_processed']} of {res['followup_prompt_tokens']} prompt tokens "
                       f"processed in {res['followup_prefill_seconds']:.2f}s")
        print(f"{name:16s} {'solved' if solved else 'FAILED':8s} {res['steps']:5d} {seconds:6.1f}s "
              f"{res['completion_tokens']:7d} {tps:6.1f} tok/s {ok:>9s} {copied:>7s}{cached}  {why}", flush=True)

    n_ok = sum(r["solved"] for r in results)
    out = sum(r["completion_tokens"] for r in results)
    dec = sum(r["decode_seconds"] for r in results)
    total = sum(r["seconds"] for r in results)
    print(f"\n{n_ok}/{len(results)} solved; {total:.0f}s in all; {out} tokens written at {out / dec:.1f} tok/s"
          if dec else f"\n{n_ok}/{len(results)} solved")
    if args.followup:
        processed = sum(r["followup_processed"] for r in results)
        prompt = sum(r["followup_prompt_tokens"] for r in results)
        print(f"follow-up requests: {processed} of {prompt} prompt tokens processed again "
              f"({sum(r['followup_prefill_seconds'] for r in results):.1f}s of prompt processing)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(model=args.model, backend=backend.name, sampling=dataclasses.asdict(sampling),
                           thinking=not args.no_think, keep_reasoning=not args.drop_reasoning, followup=args.followup,
                           effort=args.effort, lookup=getattr(backend, "lookup_drafts", None), tasks=results), f, indent=1)


if __name__ == "__main__":
    main()
