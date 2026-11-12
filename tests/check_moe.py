#!/usr/bin/env python3
"""Every check of the mixture-of-experts path, on a tiny random MoE model (GPU, about two minutes).

  1. tests/tiny_moe.py writes a 4-layer checkpoint in the original format, with an MTP head;
  2. tools/quantize.py and tools/add_mtp.py convert it;
  3. tests/verify_moe.py: the engine against MLX running the decoded weights;
  4. prompts fed in chunks of 96, 17 and 1 tokens give bit-identical logits,
     with the experts in memory and with them offloaded to the smallest cache
     (RACELINE_EXPERT_CACHE), where nearly every layer reloads experts;
  5. tests/verify_spec.py (speculative passes, abandoned drafts, saved state)
     with the experts in memory and offloaded.

  .venv/bin/python tests/check_moe.py models/Qwen3.5-4B build/ref/tokens_96.i32
"""

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PY = sys.executable


def run(what, cmd, env=None):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    ok = r.returncode == 0
    print(f"{'ok  ' if ok else 'FAIL'}  {what}", flush=True)
    if not ok:
        print((r.stdout + r.stderr)[-2000:])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokenizer_from", help="a model directory with the Qwen tokenizer (e.g. models/Qwen3.5-4B)")
    ap.add_argument("tokens", help="a token file (raw int32), e.g. build/ref/tokens_96.i32")
    ap.add_argument("--engine", default="build/cmake/raceline-engine")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="raceline-moe-")
    src, rlm = os.path.join(work, "tiny-moe"), os.path.join(work, "tiny-moe.rlm")
    small = {"RACELINE_EXPERT_CACHE": "0.001"}
    ok = run("tiny MoE checkpoint", [PY, "tests/tiny_moe.py", args.tokenizer_from, src])
    ok = ok and run("conversion", [PY, "tools/quantize.py", src, rlm, "--calib-seqs", "4", "--eval-seqs", "2"])
    ok = ok and run("MTP head", [PY, "tools/add_mtp.py", src, rlm, "--calib-seqs", "4", "--eval-seqs", "2"])
    if not ok:
        sys.exit(1)
    ok = run("engine against MLX on the decoded weights", [PY, "tests/verify_moe.py", rlm, args.tokens])

    outs = {}
    for label, env in (("in memory", {}), ("offloaded", small)):
        for chunk in (96, 17, 1):
            out = os.path.join(work, f"logits_{label[:3]}_{chunk}.f32")
            ok = run(f"logits, experts {label}, chunks of {chunk}",
                     [args.engine, "logits", rlm, args.tokens, out, "--chunk", str(chunk)], env) and ok
            outs[(label, chunk)] = out
    data = {k: open(v, "rb").read() for k, v in outs.items() if os.path.exists(v)}
    same = len(data) == len(outs) and len(set(data.values())) == 1
    print(f"{'ok  ' if same else 'FAIL'}  all six logits bit-identical")
    ok = ok and same

    ok = run("speculative decoding exact, experts in memory", [PY, "tests/verify_spec.py", rlm, args.tokens]) and ok
    ok = run("speculative decoding exact, experts offloaded", [PY, "tests/verify_spec.py", rlm, args.tokens], small) and ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
