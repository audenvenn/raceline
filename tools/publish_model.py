#!/usr/bin/env python3
"""Publish a model cut by tools/release_model.py to the sources `raceline pull` reads.

  tools/release_model.py models/qwen3.5-4b.rlm dist/qwen3.5-4b --name qwen3.5-4b.rlm --card Qwen/Qwen3.5-4B
  tools/publish_model.py dist/qwen3.5-4b qwen3.5-4b --github --huggingface

- GitHub: the parts and the manifest become assets of the release (created if needed).
- Hugging Face: the parts, the manifest and the model card go to the model
  repository <owner>/raceline-<alias> (created if needed).

Needs `gh` logged in, and `hf auth login` for Hugging Face. Each upload is
retried; a source that keeps failing is reported and the other still runs.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def files_of(dist):
    manifest = next(iter(glob.glob(os.path.join(dist, "*.rlm.json"))), None)
    if manifest is None:
        raise SystemExit(f"{dist} has no manifest (*.rlm.json): cut the model with tools/release_model.py first")
    m = json.load(open(manifest))
    parts = [os.path.join(dist, p["name"]) for p in m["parts"]]
    for p, info in zip(parts, m["parts"]):
        if os.path.getsize(p) != info["size"]:
            raise SystemExit(f"{p} does not have the size the manifest records")
    return m, manifest, parts


def retry(what, fn, attempts=4):
    for i in range(attempts):
        try:
            t0 = time.time()
            fn()
            print(f"  {what}: done in {time.time() - t0:.0f}s", flush=True)
            return True
        except Exception as e:  # network errors of every kind
            print(f"  {what}: attempt {i + 1} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(10 * (i + 1))
    return False


def github(dist, release, repo):
    m, manifest, parts = files_of(dist)
    if subprocess.run(["gh", "release", "view", release, "--repo", repo], capture_output=True).returncode:
        subprocess.run(["gh", "release", "create", release, "--repo", repo, "--title", f"Converted models ({release})",
                        "--notes", "Model files for `raceline pull`.", "--latest=false"], check=True)
    ok = True
    for path in parts + [manifest]:     # the manifest last: a download never finds it before its parts
        ok &= retry(f"GitHub {os.path.basename(path)}", lambda path=path: subprocess.run(
            ["gh", "release", "upload", release, path, "--repo", repo, "--clobber"], check=True, capture_output=True))
    return ok


def huggingface(dist, repo_id):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    m, manifest, parts = files_of(dist)
    ok = True
    for path in parts + [os.path.join(dist, "README.md"), manifest]:
        name = os.path.basename(path)
        ok &= retry(f"Hugging Face {name}", lambda path=path, name=name: api.upload_file(
            path_or_fileobj=path, path_in_repo=name, repo_id=repo_id, commit_message=f"Add {name}"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dist", help="directory written by tools/release_model.py (with --card for the hubs)")
    ap.add_argument("alias", help="the model's name for raceline pull, e.g. qwen3.5-4b")
    ap.add_argument("--github", action="store_true")
    ap.add_argument("--huggingface", action="store_true")
    ap.add_argument("--release", default="models-v2", help="GitHub release tag")
    ap.add_argument("--repo", default="audenvenn/raceline", help="GitHub repository")
    ap.add_argument("--owner", default="audenvenn", help="Hugging Face account")
    args = ap.parse_args()

    results = {}
    if args.github:
        print(f"GitHub release {args.release} of {args.repo}", flush=True)
        results["github"] = github(args.dist, args.release, args.repo)
    hub_repo = f"{args.owner}/raceline-{args.alias}"
    if args.huggingface:
        print(f"Hugging Face {hub_repo}", flush=True)
        results["huggingface"] = huggingface(args.dist, hub_repo)
    print("summary: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in results.items()), flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
