#!/usr/bin/env python3
"""Fails if a tracked file, or a file name, contains CJK text.

The repository is kept in English only (see CONTRIBUTING.md). Run from the
repository root; CI runs it on every push and pull request.
"""

import re
import subprocess
import sys

CJK = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯＀-￯]")


def main():
    files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n") if f]
    bad = []
    for name in files:
        if CJK.search(name):
            bad.append(f"{name}: file name")
        try:
            with open(name, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if CJK.search(line):
                        bad.append(f"{name}:{n}")
                        break
        except (UnicodeDecodeError, IsADirectoryError):
            continue  # binary
    if bad:
        print("non-English text found:\n  " + "\n  ".join(bad))
        return 1
    print(f"{len(files)} files checked: English only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
