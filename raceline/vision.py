"""Image input for the Qwen3.5 / 3.6 / 3.8 checkpoints: the vision tower, in MLX (#28).

An image is resized to multiples of 32 pixels, cut into 16x16 patches (each
repeated over two frames, as the tower was trained on video too) and ordered
block by block over 2x2 blocks of patches. The tower is a ViT: a linear patch
embedding plus a learned 48x48 position table resampled to the image's grid,
2D rotary embeddings over (row, column), LayerNorm blocks of non-causal
attention and a GELU MLP, then a merger that turns every 2x2 block into one
vector of the text model's width. The text model reads those vectors in
place of the <|image_pad|> tokens.

This follows transformers' Qwen3_5VisionModel and Qwen2VLImageProcessor;
tests/verify_vision.py checks it against them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Optional

import numpy as np

FORMAT = "raceline-vision-1"     # metadata["format"] of a vision file (tools/convert_vision.py)
# Pixels an image is scaled down to at most (RACELINE_IMAGE_PIXELS overrides). The
# checkpoints accept up to 16.7 million, 16384 rows of context; a megapixel is
# 1024 rows (a 1280 x 800 screenshot: 1000), what an agent can afford per image.
MAX_PIXELS = 1 << 20


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536,
                 max_pixels: int = 16777216) -> tuple:
    """The size an image is resized to: multiples of `factor`, within [min_pixels, max_pixels],
    the aspect ratio kept as closely as possible (the processor's own rule)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width):.0f} is above 200")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def preprocess(image, patch: int = 16, merge: int = 2, temporal: int = 2, min_pixels: int = 65536,
               max_pixels: int = 16777216, mean: float = 0.5, std: float = 0.5) -> tuple:
    """A PIL image -> (patches [grid_h * grid_w, 3 * temporal * patch * patch] float32, (1, grid_h, grid_w))."""
    from PIL import Image

    image = image.convert("RGB")
    h, w = smart_resize(image.height, image.width, patch * merge, min_pixels, max_pixels)
    if (w, h) != image.size:
        image = image.resize((w, h), Image.BICUBIC)
    x = (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)                                            # C, H, W
    gh, gw = h // patch, w // patch
    x = x.reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)                                # blocks, in-block row/col, C, py, px
    x = np.broadcast_to(x[:, :, :, :, :, None], x.shape[:5] + (temporal,) + x.shape[5:])
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * temporal * patch * patch)), (1, gh, gw)


def _block_order(gh: int, gw: int, merge: int) -> tuple:
    """(row, column) of every patch in the tower's order: block by block over merge x merge blocks."""
    rows = np.arange(gh).reshape(gh // merge, 1, merge, 1)
    cols = np.arange(gw).reshape(1, gw // merge, 1, merge)
    r = np.broadcast_to(rows, (gh // merge, gw // merge, merge, merge)).reshape(-1)
    c = np.broadcast_to(cols, (gh // merge, gw // merge, merge, merge)).reshape(-1)
    return r, c


class VisionTower:
    """The tower's weights and its forward pass. `weights` maps the checkpoint's names below
    `model.visual.` (blocks.0.attn.qkv.weight, ...) to arrays; `processor` holds the image
    processor's settings and `tokens` the image token ids (see tools/convert_vision.py)."""

    def __init__(self, config: dict, weights: dict, dtype=None, processor: Optional[dict] = None,
                 tokens: Optional[dict] = None, source: str = ""):
        import mlx.core as mx

        self.mx = mx
        self.dtype = dtype or mx.float16
        self.hidden = config["hidden_size"]
        self.heads = config["num_heads"]
        self.depth = config["depth"]
        self.out_hidden = config.get("out_hidden_size")
        self.merge = config.get("spatial_merge_size", 2)
        self.patch = config.get("patch_size", 16)
        self.temporal = config.get("temporal_patch_size", 2)
        self.side = int(math.isqrt(config.get("num_position_embeddings", 2304)))
        self.theta = float((config.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        self.processor = dict(processor or {})
        self.tokens = dict(tokens or {})
        self.source = source
        # Float16 throughout: 8-bit matrices (groups of 64) cost the output 100 times the error
        # (cosine to a float32 reference 0.998, the worst row 0.97, against 0.99998 and 0.9997).
        w = {k: (v if isinstance(v, mx.array) else mx.array(np.asarray(v, dtype=np.float32))).astype(self.dtype)
             for k, v in weights.items()}
        w["patch_embed.proj.weight"] = w["patch_embed.proj.weight"].reshape(self.hidden, -1)
        self.w = w
        mx.eval(list(w.values()))

    def linear(self, x, name: str):
        """x @ W.T + b for the matrix `name`, in float32 out."""
        w = self.w
        return (x.astype(self.dtype) @ w[name + ".weight"].T + w[name + ".bias"]).astype(self.mx.float32)

    def encode(self, image, max_pixels: Optional[int] = None) -> tuple:
        """A PIL image -> (vectors [rows][text hidden] float32, width in merged patches)."""
        p = self.processor
        if max_pixels is None:
            max_pixels = int(os.environ.get("RACELINE_IMAGE_PIXELS", MAX_PIXELS))
        px, grid = preprocess(image, self.patch, self.merge, self.temporal,
                              min_pixels=int(p.get("min_pixels") or 65536),
                              max_pixels=min(max_pixels, int(p.get("max_pixels") or max_pixels)),
                              mean=float((p.get("image_mean") or [0.5])[0]), std=float((p.get("image_std") or [0.5])[0]))
        return np.array(self(px, grid)), grid[2] // self.merge

    def _pos_embed(self, gh: int, gw: int):
        """The learned side x side table resampled bilinearly (corners aligned) to gh x gw, in block order."""
        mx, side = self.mx, self.side
        r, c = _block_order(gh, gw, self.merge)

        def taps(index, size):
            src = index * (side - 1) / max(size - 1, 1)
            lo = np.floor(src)
            t = np.stack([lo, lo + 1], axis=1)
            wt = np.clip(1 - np.abs(src[:, None] - t), 0, None)
            return np.clip(t, 0, side - 1).astype(np.int64), wt

        hr, wr = taps(r.astype(np.float64), gh)
        hc, wc = taps(c.astype(np.float64), gw)
        index = (hr[:, :, None] * side + hc[:, None, :]).reshape(-1, 4)
        weight = (wr[:, :, None] * wc[:, None, :]).reshape(-1, 4).astype(np.float32)
        table = self.w["pos_embed.weight"].astype(mx.float32)
        return (table[mx.array(index)] * mx.array(weight)[:, :, None]).sum(axis=1)

    def _rotary(self, gh: int, gw: int):
        """cos, sin [N, head_dim]: the first half of the head's dims rotate with the row, the second with the column."""
        mx = self.mx
        head_dim = self.hidden // self.heads
        spatial = head_dim // 2
        inv = 1.0 / (self.theta ** (np.arange(0, spatial, 2, dtype=np.float64) / spatial))
        r, c = _block_order(gh, gw, self.merge)
        freqs = np.concatenate([r[:, None] * inv[None, :], c[:, None] * inv[None, :]], axis=1)
        freqs = np.concatenate([freqs, freqs], axis=1).astype(np.float32)
        return mx.array(np.cos(freqs)), mx.array(np.sin(freqs))

    def __call__(self, patches, grid: tuple):
        """patches [N, 3 * 2 * 16 * 16] (one image), grid (1, grid_h, grid_w) -> [N / 4, text hidden] float32."""
        mx, w = self.mx, self.w
        t, gh, gw = grid
        if t != 1:
            raise ValueError("one image at a time (video is not supported)")
        n, d, heads = gh * gw, self.hidden, self.heads
        hd = d // heads
        x = self.linear(mx.array(np.asarray(patches, dtype=np.float32)), "patch_embed.proj")
        x = x + self._pos_embed(gh, gw)
        cos, sin = self._rotary(gh, gw)

        def rope(v):                                           # v [heads, N, hd] float32
            half = hd // 2
            rot = mx.concatenate([-v[..., half:], v[..., :half]], axis=-1)
            return v * cos[None] + rot * sin[None]

        for i in range(self.depth):
            p = f"blocks.{i}."
            h = mx.fast.layer_norm(x, w[p + "norm1.weight"].astype(mx.float32), w[p + "norm1.bias"].astype(mx.float32), 1e-6)
            qkv = self.linear(h, p + "attn.qkv")
            qkv = qkv.reshape(n, 3, heads, hd).transpose(1, 2, 0, 3)     # 3, heads, N, hd
            q, k, v = rope(qkv[0]), rope(qkv[1]), qkv[2]
            # Queries a block at a time: the scores of a megapixel image (4096 patches) would be 1 GB at once.
            a = mx.concatenate([mx.fast.scaled_dot_product_attention(q[None, :, r:r + 1024], k[None], v[None],
                                                                     scale=hd ** -0.5)[0]
                                for r in range(0, n, 1024)], axis=1)
            a = a.transpose(1, 0, 2).reshape(n, d)
            x = x + self.linear(a, p + "attn.proj")
            h = mx.fast.layer_norm(x, w[p + "norm2.weight"].astype(mx.float32), w[p + "norm2.bias"].astype(mx.float32), 1e-6)
            u = self.linear(h, p + "mlp.linear_fc1")
            u = 0.5 * u * (1 + mx.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u * u * u)))   # gelu, tanh form
            x = x + self.linear(u, p + "mlp.linear_fc2")
            mx.eval(x)
        h = mx.fast.layer_norm(x, w["merger.norm.weight"].astype(mx.float32), w["merger.norm.bias"].astype(mx.float32), 1e-6)
        h = h.reshape(n // (self.merge * self.merge), -1)
        h = self.linear(h, "merger.linear_fc1")
        h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2)))                                          # gelu, exact
        out = self.linear(h, "merger.linear_fc2")
        mx.eval(out)
        return out


def load_tower(path: str, dtype=None) -> Optional[VisionTower]:
    """The tower in a vision file (tools/convert_vision.py), or in an original checkpoint
    directory; None if a checkpoint has none."""
    import mlx.core as mx

    if os.path.isfile(path):
        weights, meta = mx.load(path, return_metadata=True)
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a raceline vision file")
        return VisionTower(json.loads(meta["config"]), weights, dtype, json.loads(meta["processor"]),
                           json.loads(meta["tokens"]), meta.get("source", ""))
    cfg = json.load(open(os.path.join(path, "config.json")))
    if "vision_config" not in cfg:
        return None
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    weights = {}
    for shard in sorted({s for n, s in index.items() if n.startswith("model.visual.")}):
        for name, arr in mx.load(os.path.join(path, shard)).items():
            if name.startswith("model.visual."):
                weights[name[len("model.visual."):]] = arr
    proc = json.load(open(os.path.join(path, "preprocessor_config.json")))
    size = proc.get("size") or {}
    processor = {"min_pixels": size.get("shortest_edge"), "max_pixels": size.get("longest_edge"),
                 "image_mean": proc.get("image_mean"), "image_std": proc.get("image_std")}
    return VisionTower(cfg["vision_config"], weights, dtype, processor, {"image_pad": cfg.get("image_token_id")},
                       os.path.basename(os.path.normpath(path)))


def find_vision(model_path: str) -> Optional[str]:
    """The vision file of a model file: $RACELINE_VISION, or <name>.vision.safetensors next to it,
    variant suffixes of the name (".rot", ".fp4") dropped one at a time."""
    env = os.environ.get("RACELINE_VISION")
    if env:
        return env
    stem = model_path[:-4] if model_path.endswith(".rlm") else model_path
    while True:
        candidate = stem + ".vision.safetensors"
        if os.path.isfile(candidate):
            return candidate
        head, dot, tail = stem.rpartition(".")
        if not dot or not tail.isalnum() or not head or head.endswith("/"):
            return None
        stem = head


def image_digest(data: bytes) -> int:
    """A number naming an image's content (its file bytes), for telling images apart in a context."""
    return int.from_bytes(hashlib.sha256(data).digest()[:6], "big")
