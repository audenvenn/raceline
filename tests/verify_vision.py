#!/usr/bin/env python3
"""raceline/vision.py against transformers' Qwen3.5 vision model (a reference run saved by the caller).

  .venv/bin/python tests/verify_vision.py models/Qwen3.5-4B image.png ref_dir

ref_dir holds pixel_values.npy, grid_thw.npy and embeds.npy from transformers' image processor and
Qwen3_5VisionModel in float32 (see the script that made them). Checks the patches and the tower's output
in float32, then in float16 (what raceline runs).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.vision import load_tower, preprocess  # noqa: E402


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main():
    model_dir, image_path, ref = sys.argv[1:4]
    import mlx.core as mx
    from PIL import Image

    want_px = np.load(os.path.join(ref, "pixel_values.npy"))
    want_grid = tuple(int(v) for v in np.load(os.path.join(ref, "grid_thw.npy"))[0])
    want = np.load(os.path.join(ref, "embeds.npy"))
    px, grid = preprocess(Image.open(image_path))
    ok = grid == want_grid and px.shape == want_px.shape
    print(f"{'ok  ' if ok else 'FAIL'}  patches {px.shape} grid {grid} (reference {want_px.shape} {want_grid})")
    e = rel(px, want_px)
    print(f"{'ok  ' if e < 1e-5 else 'FAIL'}  patches match the processor's: max relative difference {e:.2e}")
    ok &= e < 1e-5
    # float32 on the CPU checks the implementation itself (the GPU's float32 matrix products round more:
    # about 1e-4 relative per product, 1e-2 after 24 blocks); float16 on the GPU is what raceline runs.
    for label, dtype, stream, bound in (("float32 on the CPU", mx.float32, mx.cpu, 1e-4),
                                        ("float16 on the GPU", mx.float16, mx.gpu, 3e-2)):
        with mx.stream(stream):
            out = np.array(load_tower(model_dir, dtype)(want_px, want_grid))
        e = rel(out, want)
        cos = float((out * want).sum() / np.sqrt((out * out).sum() * (want * want).sum()))
        good = out.shape == want.shape and e < bound and cos > 0.9995
        print(f"{'ok  ' if good else 'FAIL'}  tower output {out.shape}, {label}: max relative difference {e:.2e}, "
              f"cosine {cos:.6f}")
        ok &= good
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
