"""`raceline pull`: fetch a converted model from the project's releases.

A model is published as a small manifest plus parts of under 2 GB (the size
limit of a release asset). Parts are written straight into place and checked
against their SHA-256, so an interrupted download continues with the first
part that is not complete, and nothing needs twice the disk space.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request

RELEASES = os.environ.get("RACELINE_RELEASES", "https://github.com/audenvenn/raceline/releases/download/models-v1")
MODELS = {  # alias -> file name in the release
    "qwen3.5-4b": "qwen3.5-4b.rlm",
    "qwen3.8-27b": "qwen3.8-27b.rlm",
}
CHUNK = 1 << 20


def models_dir() -> str:
    return os.path.join(os.path.expanduser(os.environ.get("RACELINE_CACHE", "~/.cache/raceline")), "models")


def resolve(model: str) -> str:
    """A model argument as a path: an existing file or directory, or the alias of a pulled model."""
    if os.path.exists(model):
        return model
    name = MODELS.get(model.lower())
    if name:
        path = os.path.join(models_dir(), name)
        if os.path.isfile(path):
            return path
        raise SystemExit(f"{model} is not downloaded yet: raceline pull {model}")
    return model  # left to the backend (an MLX repository id, for instance)


def default_model():
    """The model to use when none is named: $RACELINE_MODEL, or the one pulled model if there is exactly one."""
    if os.environ.get("RACELINE_MODEL"):
        return os.environ["RACELINE_MODEL"]
    d = models_dir()
    found = sorted(f for f in os.listdir(d) if f.endswith(".rlm")) if os.path.isdir(d) else []
    return os.path.join(d, found[0]) if len(found) == 1 else None


def _open(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "raceline"}), timeout=60)


def pull(model: str, dest: str | None = None) -> str:
    name = MODELS.get(model.lower(), model)
    with _open(f"{RELEASES}/{name}.json") as r:
        manifest = json.load(r)
    out = dest or os.path.join(models_dir(), manifest["name"])
    if os.path.isdir(out):
        out = os.path.join(out, manifest["name"])
    if os.path.isfile(out) and os.path.getsize(out) == manifest["size"]:
        print(f"{out} is already there")
        return out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    partial, state_path = out + ".partial", out + ".partial.json"
    done = set(json.load(open(state_path))["done"]) if os.path.isfile(state_path) and os.path.isfile(partial) else set()
    total, got = manifest["size"], sum(p["size"] for i, p in enumerate(manifest["parts"]) if i in done)
    print(f"{manifest['name']}: {total / 1e9:.2f} GB in {len(manifest['parts'])} parts"
          + (f", {got / 1e9:.2f} GB already here" if got else ""))
    t0, fresh = time.time(), 0
    with open(partial, "r+b" if os.path.isfile(partial) else "w+b") as f:
        offset = 0
        for i, part in enumerate(manifest["parts"]):
            if i not in done:
                for attempt in range(5):
                    f.seek(offset)
                    digest, n = hashlib.sha256(), 0
                    try:
                        with _open(f"{RELEASES}/{part['name']}") as r:
                            while True:
                                block = r.read(CHUNK)
                                if not block:
                                    break
                                f.write(block)
                                digest.update(block)
                                n, got, fresh = n + len(block), got + len(block), fresh + len(block)
                                rate = fresh / max(time.time() - t0, 1e-3)
                                sys.stdout.write(f"\r  {100 * got / total:5.1f}%  {rate / 1e6:6.1f} MB/s  "
                                                 f"about {(total - got) / max(rate, 1) / 60:4.0f} min left ")
                                sys.stdout.flush()
                    except OSError as e:
                        print(f"\n  part {i}: {e}; retrying")
                    if n == part["size"] and digest.hexdigest() == part["sha256"]:
                        break
                    got -= n
                else:
                    raise SystemExit(f"\npart {i} of {manifest['name']} could not be downloaded intact; run the command "
                                     "again to continue")
                f.flush()
                done.add(i)
                with open(state_path, "w") as s:
                    json.dump({"done": sorted(done)}, s)
            offset += part["size"]
    os.replace(partial, out)
    os.remove(state_path)
    print(f"\n{out}")
    return out
