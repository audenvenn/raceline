#!/usr/bin/env python3
"""A side-by-side replay of two runtimes answering the same request, as an animated GIF.

Each runtime runs in its own process: the same chat request, thinking off,
greedy decoding, and the time at which every piece of text arrived is
recorded. The GIF then replays both streams at their measured speed, next
to each other, so what the numbers in the README mean can be seen.

  bench/race_demo.py record models/Qwen3.8-27B-4bit race_mlx.json
  bench/race_demo.py record models/qwen3.8-27b.rlm race_raceline.json
  bench/race_demo.py render race_mlx.json race_raceline.json race.gif --title "Qwen3.8-27B on a 24 GB M6 Mac mini"

`record-agent` records the terminal agent doing one of the tasks of
bench/agent_tasks.py instead (everything it prints, tools included), and
`render --speed 3` replays at three times the measured speed: that is how
docs/agent_race.gif, the README's first image, was made.

Measure with nothing else using the GPU.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROMPT = ("Write a Python function that parses a log line of the form "
          "'2026-11-11 14:03:22 ERROR [db] connection reset' into a dict with keys "
          "time, level, component and message, and explain it briefly.")


def record(model, out, max_tokens):
    sys.path.insert(0, ROOT)
    from raceline.backends import Sampling, load_backend
    from raceline.chat import Runtime

    backend = load_backend(model)
    rt = Runtime(backend, Sampling(temperature=0.0))
    for _ in rt.chat([{"role": "user", "content": "Hi"}], thinking=False, max_tokens=8):   # warm up
        pass
    events, stats = [], None
    t0 = time.perf_counter()
    for kind, value in rt.chat([{"role": "user", "content": PROMPT}], thinking=False, max_tokens=max_tokens):
        if kind == "content":
            events.append([time.perf_counter() - t0, value])
        elif kind == "done":
            stats = value
    label = {"mlx": "mlx-lm", "raceline": "raceline"}.get(backend.name, backend.name)
    with open(out, "w") as f:
        json.dump(dict(label=label, model=os.path.basename(os.path.normpath(model)), events=events,
                       tokens=stats["completion_tokens"], decode_tps=stats["decode_tokens_per_second"],
                       first_token=stats["time_to_first_token"]), f, indent=1)
    print(f"{label}: {stats['completion_tokens']} tokens, {stats['decode_tokens_per_second']:.1f} tok/s, "
          f"first token after {stats['time_to_first_token']:.2f}s")


def record_agent(model, out, task_name):
    """The agent's whole printed session on one task, each piece with the time it appeared."""
    import shutil
    import tempfile

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    from agent_eval import FIXED_TIME
    from agent_tasks import TASKS
    from raceline.agent import Agent
    from raceline.backends import load_backend
    from raceline.chat import Runtime
    from raceline.cli import agent_sampling

    task = TASKS[task_name]
    backend = load_backend(model)
    sampling = agent_sampling(model, backend, True)
    sampling.seed = 1
    rt = Runtime(backend, sampling)
    for _ in rt.chat([{"role": "user", "content": "Hi"}], thinking=False, max_tokens=8):   # warm up
        pass
    work = os.path.join(tempfile.gettempdir(), "raceline-race")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    for path, text in task["files"].items():
        os.makedirs(os.path.dirname(os.path.join(work, path)), exist_ok=True)
        with open(os.path.join(work, path), "w", encoding="utf-8") as f:
            f.write(text)
        os.utime(os.path.join(work, path), (FIXED_TIME, FIXED_TIME))

    class Recorder:
        def __init__(self):
            self.events, self.t0 = [], time.perf_counter()

        def write(self, text):
            if text:
                self.events.append([time.perf_counter() - self.t0, text])

        def flush(self):
            pass

    agent = Agent(rt, work, auto_approve=True, thinking=True, interactive=False, color=False)
    rec, saved = Recorder(), sys.stdout
    sys.stdout = rec
    try:
        totals = agent.turn(task["prompt"])
    finally:
        sys.stdout = saved
    label = {"mlx": "mlx-lm", "raceline": "raceline"}.get(backend.name, backend.name)
    events = [[0.0, f"> {task['prompt']}\n"]] + rec.events
    with open(out, "w") as f:
        json.dump(dict(label=label, model=os.path.basename(os.path.normpath(model)), events=events,
                       tokens=totals["completion_tokens"],
                       decode_tps=totals["completion_tokens"] / max(totals["decode_seconds"], 1e-9),
                       first_token=0.0), f, indent=1)
    print(f"{label}: {totals['steps']} steps, {totals['completion_tokens']} tokens, {rec.events[-1][0]:.1f}s")


def wrap(text, width):
    """Lines of at most `width` characters, breaking at spaces where possible, keeping newlines."""
    lines = []
    for para in text.split("\n"):
        while len(para) > width:
            cut = para.rfind(" ", 0, width + 1)
            cut = cut if cut > width // 2 else width
            lines.append(para[:cut])
            para = para[cut:].lstrip(" ")
        lines.append(para)
    return lines


def render(runs, out, title, fps=12, width=1280, height=720, hold=2.5, max_seconds=None, speed=1.0,
           subtitle="same request, greedy decoding, measured on this machine and replayed at real speed"):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 17, index=1)
    big = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22, index=1)
    bg, panel, fg, dim, accent = (16, 18, 22), (26, 29, 35), (225, 228, 232), (130, 136, 146), (110, 200, 140)
    char_w = font.getbbox("M")[2]
    line_h = 20
    pad, top = 24, 92
    pane_w = (width - 3 * pad) // 2
    cols = (pane_w - 2 * 16) // char_w
    rows = (height - top - pad - 56) // line_h

    finish = [r["events"][-1][0] if r["events"] else 0.0 for r in runs]
    end = max(finish) if max_seconds is None else min(max(finish), max_seconds)
    n_frames = int((end / speed + hold) * fps)
    frames = []
    for i in range(n_frames):
        t = i / fps * speed
        im = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(im)
        d.text((pad, 24), title, font=big, fill=fg)
        d.text((pad, 56), subtitle, font=font, fill=dim)
        for j, r in enumerate(runs):
            x0 = pad + j * (pane_w + pad)
            d.rounded_rectangle((x0, top, x0 + pane_w, height - pad), radius=10, fill=panel)
            text = "".join(v for ts, v in r["events"] if ts <= t).replace("\t", "  ")
            done = t >= finish[j]
            shown = sum(1 for ts, _ in r["events"] if ts <= t)
            elapsed = min(t, finish[j])
            d.text((x0 + 16, top + 12), r["label"], font=bold, fill=accent if j else fg)
            rate = f"{r['decode_tps']:.1f} tok/s" if done else ""
            status = f"{elapsed:5.1f}s  {rate}" + ("  done" if done else "")
            d.text((x0 + pane_w - 16 - char_w * len(status), top + 14), status, font=font, fill=dim)
            lines = wrap(text, cols)[-rows:]
            for k, line in enumerate(lines):
                d.text((x0 + 16, top + 46 + k * line_h), line, font=font, fill=fg)
            if not done and shown:
                k = len(lines) - 1
                d.rectangle((x0 + 16 + char_w * len(lines[-1]), top + 46 + k * line_h + 2,
                             x0 + 16 + char_w * (len(lines[-1]) + 1), top + 46 + k * line_h + 17), fill=accent)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=32))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{out}: {n_frames} frames, {os.path.getsize(out) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--max-tokens", type=int, default=400)
    p = sub.add_parser("record-agent")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--task", default="inventory")
    p = sub.add_parser("render")
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("out")
    p.add_argument("--title", default="")
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--speed", type=float, default=1.0, help="replay this many times faster than measured")
    p.add_argument("--subtitle")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.model, args.out, args.max_tokens)
    elif args.cmd == "record-agent":
        record_agent(args.model, args.out, args.task)
    else:
        extra = {"subtitle": args.subtitle} if args.subtitle else {}
        render([json.load(open(args.left)), json.load(open(args.right))], args.out, args.title,
               max_seconds=args.max_seconds, speed=args.speed, **extra)


if __name__ == "__main__":
    main()
