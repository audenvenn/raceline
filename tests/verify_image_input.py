#!/usr/bin/env python3
"""Image input end to end: raceline's engine and vision tower against transformers.

  .venv/bin/python tests/verify_image_input.py MODEL.rlm VISION IMAGE REF_DIR

VISION is the model's vision file or its original checkpoint directory.
REF_DIR comes from tests/reference_image.py, run with PyTorch and transformers
on the same image: a prompt with the image, the model's greedy answer, the
M-RoPE positions and rotary table transformers uses, its logits, and the same
for a prompt without an image.

Checks, in the order the image's way into the model goes:
  1. every position's rotary position (time, height, width) is the one transformers assigns;
  2. the angles attention rotates each position by are transformers' (which
     of the 32 frequencies turns with which of the three);
  3. on the rows that produce the answer, the logits are as close to
     transformers' as the quantized model's are on a prompt without an image
     (KL divergence and top-1 agreement), and much further when the image's
     rows are zeroed, so the image reaches the model as it should.
A small picture and a plain question hardly depend on the positions inside the
image (with sequential positions the answer rows move by a few hundredths of a
nat), which is why 1 and 2 check them exactly. The rows of the prompt's
template (`<|im_start|>assistant ...`) are left out of 3: their next token is
always given, never generated, and their predictions are not trained after an
image (transformers predicts '\\n\\n' after `<|im_start|>` there); they swing
with any small change of the weights.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from raceline.backends import ImageRows  # noqa: E402
from raceline.engine import RacelineBackend  # noqa: E402
from raceline.vision import load_tower, preprocess  # noqa: E402


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def compare(ours, ref):
    """Per row: KL(ref || ours) in nats, and whether the top tokens agree."""
    lr, lo = log_softmax(ref), log_softmax(ours[:, :ref.shape[1]])
    kl = (np.exp(lr) * (lr - lo)).sum(axis=-1)
    return kl, ours[:, :ref.shape[1]].argmax(-1) == ref.argmax(-1)


def run(b, ids, upto, images):
    """Logits of rows upto-1 .. len(ids)-1: the prompt to `upto` in one pass, then a token at a time."""
    b.reset()
    b.prefill(ids[:upto], images=images)
    rows = [b._logits.copy()]
    for t in ids[upto:]:
        b.prefill([t])
        rows.append(b._logits.copy())
    return np.stack(rows)


def main():
    model, vision, image_path, ref = sys.argv[1:5]
    from PIL import Image

    def load(name):
        return np.load(os.path.join(ref, name))

    ids, P = load("ids.npy").tolist(), int(load("prompt_len.npy"))
    start, n = int(load("image_start.npy")), int(load("n_image.npy"))
    px, grid = preprocess(Image.open(image_path))
    vectors = np.array(load_tower(vision)(px, grid))
    if vectors.shape[0] != n:
        sys.exit(f"the image gives {vectors.shape[0]} rows here and {n} in the reference")
    image = ImageRows(start, vectors, grid[2] // 2)

    b = RacelineBackend(model, max_ctx=4096)
    ours = run(b, ids, P, [image])
    pos = np.array([b.rope_position(p) for p in range(len(ids))]).T
    ok = np.array_equal(pos, load("positions.npy"))
    print(f"{'ok  ' if ok else 'FAIL'}  rotary positions of all {len(ids)} positions equal transformers' (the image: "
          f"time {pos[0, start]}, height {pos[1, start]}..{pos[1, start + n - 1]}, width {pos[2, start]}.."
          f"{pos[2, start + n - 1]}; the text after it from {pos[0, start + n]})")
    angles = np.stack([b.rope_angles(p) for p in range(len(ids))])
    want = np.concatenate([load("rope_cos.npy")[:, :32], load("rope_sin.npy")[:, :32]], axis=1)
    e = float(np.abs(angles - want).max())
    print(f"{'ok  ' if e < 1e-4 else 'FAIL'}  rotary angles equal transformers' to {e:.1e}")
    ok &= e < 1e-4

    want = load("logits.npy")[P - 1 - (start + n - 1):]       # rows from the prompt's last token on
    kl, top = compare(ours, want)
    t_ids, t_P = load("text_ids.npy").tolist(), int(load("text_prompt_len.npy"))
    t_kl, t_top = compare(run(b, t_ids, t_P, []), load("text_logits.npy")[t_P - 1:])
    print(f"      answer rows after the image:  KL {kl.mean():.4f} nats, top-1 agreement {top.mean():.1%} "
          f"({len(kl)} rows)")
    print(f"      answer rows, no image:        KL {t_kl.mean():.4f} nats, top-1 agreement {t_top.mean():.1%} "
          f"({len(t_kl)} rows; the quantized model's own distance)")
    good = kl.mean() <= 2 * t_kl.mean() + 0.01 and top.mean() >= t_top.mean() - 0.05
    print(f"{'ok  ' if good else 'FAIL'}  the image costs no more than the quantization does")
    ok &= good
    z_kl, z_top = compare(run(b, ids, P, [ImageRows(start, np.zeros_like(vectors), image.width)]), want)
    s_kl, s_top = compare(run(b, ids, P, [ImageRows(start + k, vectors[k:k + 1], 1) for k in range(n)]), want)
    sharp = z_kl.mean() > 5 * kl.mean()
    print(f"{'ok  ' if sharp else 'FAIL'}  with the image's rows zeroed: KL {z_kl.mean():.4f}, top-1 {z_top.mean():.1%}"
          f" (with sequential positions instead: KL {s_kl.mean():.4f}, top-1 {s_top.mean():.1%})")
    ok &= sharp
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
