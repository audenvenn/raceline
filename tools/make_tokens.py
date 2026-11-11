#!/usr/bin/env python3
"""Turn a text file into a token file for the engine's command line and the verification scripts.

Token files are raw little-endian int32. The tokenizer is taken from a .rlm
file (it is embedded there) or from a model directory with a tokenizer.json.

  tools/make_tokens.py models/qwen3.5-4b.rlm README.md tokens.i32 --count 96
"""

import argparse
import os
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".rlm file or model directory")
    ap.add_argument("text", help="text file to tokenize")
    ap.add_argument("out", help="output .i32 file")
    ap.add_argument("--count", type=int, help="keep only the first N tokens")
    args = ap.parse_args()

    folder = args.model
    if args.model.endswith(".rlm"):
        from raceline.engine import _tokenizer_dir
        folder = _tokenizer_dir(args.model)
    tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    ids = tok.encode(open(args.text, encoding="utf-8", errors="ignore").read(), add_special_tokens=False).ids
    if args.count:
        if len(ids) < args.count:
            raise SystemExit(f"{args.text} has only {len(ids)} tokens")
        ids = ids[:args.count]
    np.asarray(ids, dtype="<i4").tofile(args.out)
    print(f"{args.out}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
