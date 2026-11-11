#!/usr/bin/env python3
"""Cut a model file into release assets for `raceline pull`.

Release assets are limited to 2 GB each, so the file is split into parts and
described by a manifest with sizes and SHA-256 sums:

  tools/release_model.py models/qwen3.5-4b.rlm dist/ --name qwen3.5-4b.rlm
  gh release upload models-v1 dist/qwen3.5-4b.rlm.*

Writes <name>.json and <name>.part-00, <name>.part-01, ... into the directory.
"""

import argparse
import hashlib
import json
import os
import struct

PART = 1_900_000_000
BLOCK = 8 << 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".rlm file")
    ap.add_argument("outdir")
    ap.add_argument("--name", help="file name users will get (default: the model's own)")
    ap.add_argument("--part-size", type=int, default=PART)
    args = ap.parse_args()

    name = args.name or os.path.basename(args.model)
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.model, "rb") as f:
        if f.read(4) != b"RLM1":
            raise SystemExit(f"{args.model} is not a raceline model file")
        _, n = struct.unpack("<IQ", f.read(12))
        hdr = json.loads(f.read(n))
        f.seek(0)
        whole, parts, size = hashlib.sha256(), [], os.path.getsize(args.model)
        while f.tell() < size:
            part_name = f"{name}.part-{len(parts):02d}"
            digest, written = hashlib.sha256(), 0
            with open(os.path.join(args.outdir, part_name), "wb") as out:
                while written < args.part_size:
                    block = f.read(min(BLOCK, args.part_size - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    whole.update(block)
                    written += len(block)
            parts.append(dict(name=part_name, size=written, sha256=digest.hexdigest()))
            print(f"  {part_name}  {written / 1e9:.2f} GB")
    manifest = dict(name=name, size=size, sha256=whole.hexdigest(), parts=parts,
                    source=hdr.get("source"), arch=hdr.get("arch"), quant=hdr.get("quant"),
                    quality=hdr.get("quality"), mtp=hdr.get("mtp"), license="Apache-2.0")
    with open(os.path.join(args.outdir, name + ".json"), "w") as out:
        json.dump(manifest, out, indent=1)
    print(f"{name}: {size / 1e9:.2f} GB in {len(parts)} parts, sha256 {manifest['sha256'][:16]}...")


if __name__ == "__main__":
    main()
