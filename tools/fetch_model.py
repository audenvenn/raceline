#!/usr/bin/env python3
"""Download model files from ModelScope or Hugging Face with resume + verification.

Standard library only (shells out to curl for resumable transfers), so it runs
before any Python environment is set up. ModelScope is the default source
because the Hugging Face CDN is unreachable from some networks.

  tools/fetch_model.py Qwen/Qwen3.5-4B
  tools/fetch_model.py unsloth/Qwen3.5-4B-GGUF --include 'Qwen3.5-4B-Q4_K_M.gguf' --dest models/gguf
  tools/fetch_model.py Qwen/Qwen3.5-4B --source hf
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "curl/8"}
SKIP = {".gitattributes", "README.md", "LICENSE", "configuration.json"}


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def list_files(repo: str, source: str, revision: str | None):
    """Return [(path, size, sha256_or_None, url)]."""
    out = []
    if source == "modelscope":
        rev = revision or "master"
        d = _get_json(
            f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
            f"?Revision={rev}&Recursive=true"
        )
        if d.get("Code") != 200:
            raise SystemExit(f"ModelScope: cannot list {repo}: {d.get('Message')}")
        for f in d["Data"]["Files"]:
            if f.get("Type") == "tree":
                continue
            url = f"https://www.modelscope.cn/models/{repo}/resolve/{rev}/{f['Path']}"
            out.append((f["Path"], int(f["Size"]), f.get("Sha256") or None, url))
    else:
        rev = revision or "main"
        d = _get_json(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true")
        for s in d.get("siblings", []):
            sha = (s.get("lfs") or {}).get("sha256")
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{s['rfilename']}"
            out.append((s["rfilename"], int(s.get("size") or 0), sha, url))
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(item, dest: str, verify: bool) -> tuple[str, bool, str]:
    path, size, sha, url = item
    target = os.path.join(dest, path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) == size:
        return path, True, "cached"
    part = target + ".part"
    for attempt in range(6):
        cmd = ["curl", "-sS", "-L", "--fail", "--connect-timeout", "20",
               "--speed-limit", "20000", "--speed-time", "60",
               "-C", "-", "-o", part, url]
        rc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        got = os.path.getsize(part) if os.path.exists(part) else 0
        if got == size:
            break
        if got > size:  # server ignored the range request; start over
            os.remove(part)
        time.sleep(2 + attempt * 3)
    else:
        return path, False, f"incomplete ({got}/{size}): {rc.stderr.strip()[-200:]}"
    if verify and sha and sha256_file(part) != sha:
        os.remove(part)
        return path, False, "sha256 mismatch"
    os.replace(part, target)
    return path, True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    ap.add_argument("--revision")
    ap.add_argument("--include", action="append", help="glob; may repeat (default: everything)")
    ap.add_argument("--dest", help="target directory (default: models/<repo name>)")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 check")
    args = ap.parse_args()

    dest = args.dest or os.path.join("models", args.repo.split("/")[-1])
    files = [f for f in list_files(args.repo, args.source, args.revision) if f[0] not in SKIP]
    if args.include:
        files = [f for f in files if any(fnmatch.fnmatch(f[0], g) for g in args.include)]
    if not files:
        raise SystemExit("nothing matched")
    total = sum(f[1] for f in files)
    print(f"{args.repo} [{args.source}] -> {dest}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)

    t0 = time.time()
    failed = 0
    files.sort(key=lambda f: f[1])  # small files first: config/tokenizer land early
    with ThreadPoolExecutor(args.jobs) as ex:
        for path, ok, msg in ex.map(lambda it: fetch_one(it, dest, not args.no_verify), files):
            print(f"  {'ok ' if ok else 'ERR'} {path}  ({msg})", flush=True)
            failed += not ok
    dt = time.time() - t0
    print(f"done in {dt:.0f}s ({total / 1e6 / max(dt, 1e-9):.1f} MB/s), {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
