#!/usr/bin/env python3
"""Every check of the mixture-of-experts path, on a tiny random MoE model (GPU, about two minutes).

  1. tests/tiny_moe.py writes a 4-layer checkpoint in the original format, with an MTP head;
  2. tools/quantize.py and tools/add_mtp.py convert it;
  3. tests/verify_moe.py: the engine against MLX running the decoded weights;
  4. prompts fed in chunks of 96, 17 and 1 tokens give bit-identical logits,
     with the experts in memory and offloaded (RACELINE_EXPERT_CACHE) to the
     smallest cache, where nearly every layer reloads experts, and to a cache
     of 70% of them, where prompt chunks prefetch the next layer's experts
     while the GPU computes;
  5. tests/verify_spec.py (speculative passes, abandoned drafts, saved state)
     in all three modes.

  .venv/bin/python tests/check_moe.py models/Qwen3.5-4B build/ref/tokens_96.i32
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_moe import E, H, I, LAYERS  # noqa: E402

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
    ap.add_argument("--keep", action="store_true", help="keep the work directory (it is kept anyway when a check fails)")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="raceline-moe-")
    ok = False
    try:
        ok = checks(args, work)
    finally:
        # The checkpoint, its conversion and the logits are about 800 MB: remove them unless asked to keep
        # them or a check failed (then they are what to look at).
        if ok and not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"work directory kept: {work}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def checks(args, work) -> bool:
    src, rlm = os.path.join(work, "tiny-moe"), os.path.join(work, "tiny-moe.rlm")
    per_expert = 2 * I * H // 2 + H * I // 2 + (2 * I + H) * 4     # 4-bit codes and row scales
    modes = (("in memory", {"RACELINE_EXPERT_CACHE": "0"}),
             ("offloaded, smallest cache", {"RACELINE_EXPERT_CACHE": "0.001"}),
             ("offloaded, 70% cached", {"RACELINE_EXPERT_CACHE": f"{0.7 * LAYERS * (E + 1) * per_expert / 1e9:.6f}"}))
    ok = run("tiny MoE checkpoint", [PY, "tests/tiny_moe.py", args.tokenizer_from, src])
    ok = ok and run("conversion", [PY, "tools/quantize.py", src, rlm, "--calib-seqs", "4", "--eval-seqs", "2"])
    ok = ok and run("MTP head", [PY, "tools/add_mtp.py", src, rlm, "--calib-seqs", "4", "--eval-seqs", "2"])
    if not ok:
        return False
    ok = run("engine against MLX on the decoded weights", [PY, "tests/verify_moe.py", rlm, args.tokens])

    outs = {}
    for m, (label, env) in enumerate(modes):
        for chunk in (96, 17, 1):
            out = os.path.join(work, f"logits_{m}_{chunk}.f32")
            ok = run(f"logits, experts {label}, chunks of {chunk}",
                     [args.engine, "logits", rlm, args.tokens, out, "--chunk", str(chunk)], env) and ok
            outs[(label, chunk)] = out
    data = {k: open(v, "rb").read() for k, v in outs.items() if os.path.exists(v)}
    same = len(data) == len(outs) and len(set(data.values())) == 1
    print(f"{'ok  ' if same else 'FAIL'}  all {len(outs)} logits bit-identical")
    ok = ok and same

    for label, env in modes:
        ok = run(f"speculative decoding exact, experts {label}", [PY, "tests/verify_spec.py", rlm, args.tokens], env) and ok
    return ok


if __name__ == "__main__":
    main()
