#!/usr/bin/env python3
"""Cut a model file into release assets for `raceline pull`.

Release assets are limited to 2 GB each, so the file is split into parts and
described by a manifest with sizes and SHA-256 sums:

  tools/release_model.py models/qwen3.5-4b.rlm dist/ --name qwen3.5-4b.rlm
  gh release upload models-v1 dist/qwen3.5-4b.rlm.*

Writes <name>.json and <name>.part-00, <name>.part-01, ... into the directory.
With --card BASE_MODEL it also writes the model card (README.md), so the
directory can be uploaded to Hugging Face as it is:

  tools/release_model.py models/qwen3.5-4b.rlm dist/ --name qwen3.5-4b.rlm --card Qwen/Qwen3.5-4B
"""

import argparse
import hashlib
import json
import os
import struct

PART = 1_900_000_000
BLOCK = 8 << 20


def cut(model, outdir, name=None, part_size=PART):
    """Splits `model` (a .rlm file, or a vision file of tools/convert_vision.py) into parts in
    `outdir`, writes the manifest there and returns it."""
    name = name or os.path.basename(model)
    os.makedirs(outdir, exist_ok=True)
    with open(model, "rb") as f:
        magic = f.read(4)
        if magic == b"RLM1":
            _, n = struct.unpack("<IQ", f.read(12))
            hdr = json.loads(f.read(n))
        elif model.endswith(".vision.safetensors"):
            f.seek(0)
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = {"vision": json.loads(f.read(n)).get("__metadata__") or {}}
        else:
            raise SystemExit(f"{model} is neither a raceline model file nor a vision file")
        f.seek(0)
        whole, parts, size = hashlib.sha256(), [], os.path.getsize(model)
        while f.tell() < size:
            part_name = f"{name}.part-{len(parts):02d}"
            digest, written = hashlib.sha256(), 0
            with open(os.path.join(outdir, part_name), "wb") as out:
                while written < part_size:
                    block = f.read(min(BLOCK, part_size - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    whole.update(block)
                    written += len(block)
            parts.append(dict(name=part_name, size=written, sha256=digest.hexdigest()))
    if "vision" in hdr:     # the tower of a model's image input
        manifest = dict(name=name, size=size, sha256=whole.hexdigest(), parts=parts, kind="vision",
                        source=hdr["vision"].get("source"), format=hdr["vision"].get("format"), license="Apache-2.0")
        with open(os.path.join(outdir, name + ".json"), "w") as out:
            json.dump(manifest, out, indent=1)
        return manifest
    manifest = dict(name=name, size=size, sha256=whole.hexdigest(), parts=parts,
                    source=hdr.get("source"), arch=hdr.get("arch"), quant=hdr.get("quant"),
                    quality=hdr.get("quality"), mtp=hdr.get("mtp"), license="Apache-2.0")
    # A mixture of experts: how much of the file the experts are, which the
    # engine can leave on disk (`raceline pull` says so instead of warning about memory).
    experts = sum(s["nbytes"] for s in hdr.get("sections", []) if s["name"].startswith("experts."))
    if experts:
        cfg = hdr.get("config", {})
        manifest["moe"] = dict(experts=cfg.get("num_experts"), per_token=cfg.get("num_experts_per_tok"),
                               experts_bytes=experts)
    with open(os.path.join(outdir, name + ".json"), "w") as out:
        json.dump(manifest, out, indent=1)
    return manifest


CARD = """---
license: apache-2.0
base_model: {base}
base_model_relation: quantized
pipeline_tag: {pipeline}
tags:
- raceline
- apple-silicon
- metal
- speculative-decoding
- mtp
- fp4{moe_tag}
---

# {title} for raceline

[{base}](https://huggingface.co/{base}) converted for [raceline](https://github.com/audenvenn/raceline),
a local agent runtime with its own inference engine for Apple Silicon Macs whose GPU has neural
accelerators (M5 and M6 families).

The model file is the engine's native format: 4-bit FP4 weights with a scale per row and per
column ({bits} bits per weight), quantized with GPTQ after a {rotation}; the model's
multi-token-prediction head, which raceline uses for exact speculative decoding; and the tokenizer
and chat template. It is split into parts of under 2 GB described by `{name}.json`. It needs
raceline {needs} or later and is not usable with other runtimes.
{moe_text}{vision_text}
## Use

```sh
raceline pull {alias} --source huggingface    # or without --source: the GitHub release
raceline chat {alias}
```

See the [raceline README](https://github.com/audenvenn/raceline#quick-start) for installation and for
speed measurements against mlx-lm.

## Quality

Measured at conversion time against the float model on held-out text ({positions} positions):
KL divergence {kl:.3f} nats, perplexity {ppl:.3f}, top-1 agreement {top1:.1f}%.
The draft head's top choice is the model's own next-but-one token {agree:.1f}% of the time.

| File | Size | SHA-256 |
|---|---|---|
| `{name}` (assembled) | {size:.2f} GB | `{sha}` |

## License

Apache-2.0, as the original model.
"""


VISION = """
`{vision}` is the model's vision tower in float16 ({size:.2f} GB), for image input (raceline
{needs} or later): `raceline pull` fetches it with the model, and raceline reads images with it
in the terminal agent and through its OpenAI and Anthropic APIs.
"""


def write_card(outdir, manifest, base, alias, needs="0.2.0", vision=None):
    """`vision`: the manifest of the model's vision file, if one is published with it."""
    q, m, quant = manifest["quality"], manifest.get("mtp") or {}, manifest["quant"]
    rot = (quant.get("rotation") or {}).get("kind")
    title = base.split("/")[-1]
    moe = manifest.get("moe")
    moe_text = "" if not moe else (
        f"\nA mixture of experts: each layer routes a token to {moe['per_token']} of its {moe['experts']} experts "
        f"plus a shared one, so a token reads a small part of the weights. The experts are "
        f"{moe['experts_bytes'] / 1e9:.1f} GB of the file. On a Mac whose GPU may not hold the whole model, raceline "
        "leaves them in the file and keeps the ones in use in memory, reading the others from the SSD when a layer "
        "needs them (`--expert-cache GB` sets how much memory that cache takes). The experts' down "
        "projections are rounded to nearest; the other matrices are quantized with GPTQ.\n")
    vision_text = "" if not vision else VISION.format(vision=vision["name"], size=vision["size"] / 1e9,
                                                       needs=vision.get("needs", "0.6.0"))
    text = CARD.format(base=base, title=title, alias=alias, name=manifest["name"], bits=f"{quant['bits']:.1f}",
                       moe_tag="\n- moe" if moe else "", moe_text=moe_text, vision_text=vision_text,
                       pipeline="image-text-to-text" if vision else "text-generation",
                       rotation=("Hadamard rotation of the residual stream that is folded into the weights"
                                 if rot else "search of the row and column scales"),
                       positions=q["positions"], kl=q["kl"], ppl=q["ppl"], top1=100 * q["top1"],
                       agree=100 * m.get("agreement", float("nan")), size=manifest["size"] / 1e9,
                       sha=manifest["sha256"], needs=needs)
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help=".rlm file")
    ap.add_argument("outdir")
    ap.add_argument("--name", help="file name users will get (default: the model's own)")
    ap.add_argument("--part-size", type=int, default=PART)
    ap.add_argument("--card", metavar="BASE_MODEL", help="also write a model card for this base model (e.g. Qwen/Qwen3.5-4B)")
    ap.add_argument("--needs", default="0.2.0", help="the first raceline version that pulls and runs the model")
    args = ap.parse_args()
    manifest = cut(args.model, args.outdir, args.name, args.part_size)
    if args.card:
        write_card(args.outdir, manifest, args.card, manifest["name"].rsplit(".rlm", 1)[0], args.needs)
    for part in manifest["parts"]:
        print(f"  {part['name']}  {part['size'] / 1e9:.2f} GB")
    print(f"{manifest['name']}: {manifest['size'] / 1e9:.2f} GB in {len(manifest['parts'])} parts, "
          f"sha256 {manifest['sha256'][:16]}...")


if __name__ == "__main__":
    main()
