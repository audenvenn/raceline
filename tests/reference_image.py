#!/usr/bin/env python3
"""References for the image input checks, from PyTorch and transformers (a separate environment:
`uv venv ref && uv pip install -p ref torch transformers pillow safetensors numpy`).

  ref/bin/python -I tests/reference_image.py --tower CHECKPOINT IMAGE OUT   (for tests/verify_vision.py)
  ref/bin/python -I tests/reference_image.py CHECKPOINT IMAGE OUT           (for tests/verify_image_input.py)

CHECKPOINT is the original Qwen3.5 / 3.6 / 3.8 directory. --tower runs the
image processor and the vision tower alone, in float32: pixel_values.npy,
grid_thw.npy, embeds.npy. Without it, the whole model in bfloat16 on the CPU
(about twice its parameter count in GB of memory): a prompt with the image
and the model's greedy answer, the M-RoPE positions and rotary table it uses,
its logits from the image's last row on and the residual stream after layers
3, 7, 15 and the last; then a prompt without an image and its logits.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def image_processor(checkpoint):
    from transformers import AutoImageProcessor
    # The PIL processor, which raceline.vision follows (the default one needs torchvision).
    return AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)


def tower(checkpoint, image_path, out):
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    batch = image_processor(checkpoint)(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    cfg = json.load(open(os.path.join(checkpoint, "config.json")))
    vcfg = Qwen3_5VisionConfig(**cfg["vision_config"])
    vcfg._attn_implementation = "eager"
    model = Qwen3_5VisionModel(vcfg).eval()
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"]
    state = {}
    for name, shard in sorted(index.items()):
        if name.startswith("model.visual."):
            with safe_open(os.path.join(checkpoint, shard), framework="pt") as f:
                state[name[len("model.visual."):]] = f.get_tensor(name).to(torch.float32)
    model.load_state_dict(state)
    with torch.no_grad():
        embeds = model(batch["pixel_values"].to(torch.float32), grid_thw=batch["image_grid_thw"]).pooler_output
    np.save(f"{out}/pixel_values.npy", batch["pixel_values"].numpy().astype(np.float32))
    np.save(f"{out}/grid_thw.npy", batch["image_grid_thw"].numpy().astype(np.int64))
    np.save(f"{out}/embeds.npy", embeds.numpy().astype(np.float32))
    print("patches", tuple(batch["pixel_values"].shape), "grid", batch["image_grid_thw"].tolist(), "embeds",
          tuple(embeds.shape))


def model_run(checkpoint, image_path, out):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5ForConditionalGeneration,
                                                              Qwen3_5TextRotaryEmbedding)

    tok = AutoTokenizer.from_pretrained(checkpoint)
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    question = "What shapes, colors and text are in this picture? Answer in two sentences."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = image_processor(checkpoint)(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    grid = batch["image_grid_thw"]
    n_img = int(grid.prod()) // 4
    ids = tok(text, add_special_tokens=False)["input_ids"]
    i = ids.index(pad)
    ids = ids[:i] + [pad] * n_img + ids[i + 1:]

    model = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    pixels = batch["pixel_values"].to(torch.bfloat16)
    x = torch.tensor([ids])
    with torch.no_grad():
        gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=48, do_sample=False,
                             pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(x == pad).int())
    answer = gen[0, len(ids):].tolist()
    print("answer:", tok.decode(answer))
    full = torch.tensor([ids + answer])
    mm = (full == pad).int()
    layers = model.model.language_model.layers
    picks = (3, 7, 15, len(layers) - 1)
    hidden = {}
    for li in picks:
        layers[li].register_forward_hook(
            lambda m, a, o, li=li: hidden.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0].float().numpy()))
    with torch.no_grad():
        pos, _ = model.model.get_rope_index(full, mm_token_type_ids=mm, image_grid_thw=grid)
        o = model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pixels, image_grid_thw=grid,
                  mm_token_type_ids=mm)
    last = i + n_img - 1
    np.save(f"{out}/ids.npy", np.array(ids + answer, dtype=np.int32))
    np.save(f"{out}/prompt_len.npy", np.array(len(ids)))
    np.save(f"{out}/image_start.npy", np.array(i))
    np.save(f"{out}/n_image.npy", np.array(n_img))
    np.save(f"{out}/grid.npy", grid[0].numpy())
    np.save(f"{out}/positions.npy", pos[:, 0].numpy().astype(np.int32))
    np.save(f"{out}/logits.npy", o.logits[0, last:].float().numpy())
    for li, h in hidden.items():
        np.save(f"{out}/hidden_{li:02d}.npy", h)
    cfg = json.load(open(os.path.join(checkpoint, "config.json")))
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    rot = Qwen3_5TextRotaryEmbedding(Qwen3_5TextConfig(**cfg["text_config"]))
    cos, sin = rot(torch.zeros(1, dtype=torch.float32), pos)     # float32, as the model computes them
    np.save(f"{out}/rope_cos.npy", cos[0].numpy())
    np.save(f"{out}/rope_sin.npy", sin[0].numpy())

    tmsgs = [{"role": "user", "content": "Describe a red square, a blue circle and a green triangle in two sentences."}]
    ttext = tok.apply_chat_template(tmsgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tids = tok(ttext, add_special_tokens=False)["input_ids"]
    tx = torch.tensor([tids])
    with torch.no_grad():
        tfull = model.generate(input_ids=tx, attention_mask=torch.ones_like(tx), max_new_tokens=48, do_sample=False)
        to = model(input_ids=tfull, attention_mask=torch.ones_like(tfull))
    np.save(f"{out}/text_ids.npy", tfull[0].numpy().astype(np.int32))
    np.save(f"{out}/text_prompt_len.npy", np.array(len(tids)))
    np.save(f"{out}/text_logits.npy", to.logits[0].float().numpy())
    print("text answer:", tok.decode(tfull[0, len(tids):]))


def main():
    args = sys.argv[1:]
    only_tower = args[:1] == ["--tower"]
    checkpoint, image_path, out = args[1:4] if only_tower else args[0:3]
    os.makedirs(out, exist_ok=True)
    (tower if only_tower else model_run)(checkpoint, image_path, out)


if __name__ == "__main__":
    main()
