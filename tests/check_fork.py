#!/usr/bin/env python3
"""A fork after the engine has loaded a model leaves its speed alone (#53).

  .venv/bin/python tests/check_fork.py MODEL.rlm

Decodes the same answer three times; between the second and the third the
process runs a command with subprocess, which forks. Fails if the third
answer is more than 25% slower than the second. (With the model file mapped
private, a fork made every later pass about ten times slower.)
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    from raceline.backends import Sampling
    from raceline.chat import Runtime
    from raceline.engine import RacelineBackend

    rt = Runtime(RacelineBackend(sys.argv[1], max_ctx=8192), Sampling(temperature=0.0), park_bytes=0)
    prompt = [{"role": "user", "content": "Write a short story of about 150 words about a lighthouse keeper."}]
    speeds = []
    for i in range(3):
        if i == 2:
            subprocess.run(["true"], check=True)       # fork + exec
        stats = list(rt.chat(prompt, thinking=False, max_tokens=160))[-1][1]
        speeds.append(stats["decode_tokens_per_second"])
        print(f"answer {i + 1}{' (after a fork)' if i == 2 else ''}: {stats['completion_tokens']} tokens at "
              f"{speeds[-1]:.1f} tok/s")
    ok = speeds[2] >= 0.75 * speeds[1]
    print(f"{'ok  ' if ok else 'FAIL'}  after the fork {speeds[2]:.1f} tok/s, before it {speeds[1]:.1f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
