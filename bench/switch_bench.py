#!/usr/bin/env python3
"""What it costs a server's client to come back to a conversation after a side request.

A long conversation (a system prompt, then text standing in for a session's
history), a short side request with its own system prompt (as a client sends
for a title or a commit message), then the long conversation's next turn.
Reported: the time to the first token of that turn, and how many of its
prompt tokens were processed again, with conversations set aside
(the server's default) and without (RACELINE_PARK_GB=0).

  bench/switch_bench.py models/qwen3.6-35b-a3b.rlm --history 8000

Measure with nothing else using the GPU.
"""

import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--history", type=int, default=8000, help="tokens of conversation before the side request")
    ap.add_argument("--ctx", type=int, default=32768)
    args = ap.parse_args()

    from raceline.backends import Sampling, load_backend
    from raceline.chat import Runtime

    backend = load_backend(args.model, max_ctx=args.ctx)
    tok = backend.tokenizer
    corpus = open(os.path.join(ROOT, "raceline", "session.py"), encoding="utf-8").read() + \
        open(os.path.join(ROOT, "raceline", "chat.py"), encoding="utf-8").read()
    words = tok.decode(tok.encode(corpus * 20, add_special_tokens=False)[:args.history])
    main = [{"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Here is the code we are working on:\n\n" + words},
            {"role": "assistant", "content": "I have read it."},
            {"role": "user", "content": "What does Session.sync return? One sentence."}]
    side = [{"role": "system", "content": "Write a title of at most six words for the conversation."},
            {"role": "user", "content": "The user asks about Session.sync in a Python inference runtime."}]
    greedy = Sampling(temperature=0.0)

    def ask(rt, messages, n=24):
        t0, first, stats = time.perf_counter(), None, None
        for kind, value in rt.chat(messages, sampling=greedy, max_tokens=n, thinking=False):
            if first is None and kind in ("content", "reasoning", "tool_call"):
                first = time.perf_counter() - t0
            if kind == "done":
                stats = value
        return first, stats

    print(f"{'kept aside':12s} {'first token':>12s} {'processed again':>16s}")
    for park in (2 << 30, 0):
        rt = Runtime(backend, greedy, park_bytes=park)
        rt.session.reset()
        ask(rt, main)                       # the long conversation
        ask(rt, side)                       # the side request
        first, stats = ask(rt, main + [{"role": "assistant", "content": "It returns the prompt's size."},
                                       {"role": "user", "content": "And what does it cost?"}])
        again = stats["prompt_tokens"] - stats["cached_tokens"]
        print(f"{'yes' if park else 'no':12s} {first:11.2f}s {again:9d} of {stats['prompt_tokens']}", flush=True)


if __name__ == "__main__":
    main()
