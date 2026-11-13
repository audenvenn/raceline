"""Images in conversations: the content parts clients send, and their rows in the prompt (#28).

Clients send an image as a content part of a message: OpenAI's
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` or
Anthropic's `{"type": "image", "source": {"type": "base64", ...}}`; the
terminal agent puts a Picture in the part itself (`{"type": "image", "image":
picture}`). Parts never name a file: a client of the server cannot make it
read one. Each becomes a Picture. The chat
template marks its place with one `<|image_pad|>`, which the prompt widens to
one row per 2x2 block of 16-pixel patches; those rows get keys of their own
(`image_keys`), negative numbers no token has, so that a context holding
another picture of the same size is never taken for this one.
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
from typing import Optional

from .vision import MAX_PIXELS, image_digest, smart_resize

KEY_ROW_BITS = 15        # rows per picture: below 32768 (16384 at the checkpoints' own limit)


class Picture:
    """An image in a conversation: its file's bytes, its size, and what names it."""

    @classmethod
    def from_file(cls, path: str) -> "Picture":
        with open(os.path.expanduser(path), "rb") as f:
            return cls(f.read())

    def __init__(self, data: bytes):
        from PIL import Image

        try:
            with Image.open(io.BytesIO(data)) as im:
                self.size = im.size                       # width, height
        except Exception as e:  # noqa: BLE001  (PIL raises several kinds for data it cannot read)
            raise ValueError(f"an image part this runtime cannot read: {e}") from None
        self.data = data
        self.digest = image_digest(data)

    def open(self):
        from PIL import Image

        return Image.open(io.BytesIO(self.data))

    def rows(self, processor: dict, max_pixels: Optional[int] = None) -> tuple:
        """(rows of the prompt it takes, its width in merged patches) at the tower's settings."""
        if max_pixels is None:
            max_pixels = int(os.environ.get("RACELINE_IMAGE_PIXELS", MAX_PIXELS))
        factor = processor.get("patch_size", 16) * processor.get("merge_size", 2)
        w, h = self.size
        h2, w2 = smart_resize(h, w, factor, int(processor.get("min_pixels") or 65536),
                              min(max_pixels, int(processor.get("max_pixels") or max_pixels)))
        return (h2 // factor) * (w2 // factor), w2 // factor


def image_keys(digest: int, n: int) -> list:
    """The context keys of a picture's n rows: negative, one per picture and row."""
    base = -1 - (digest << KEY_ROW_BITS)
    return [base - k for k in range(n)]


def key_row(key: int) -> tuple:
    """A row's key -> (the picture's digest, the row)."""
    v = -1 - key
    return v >> KEY_ROW_BITS, v & ((1 << KEY_ROW_BITS) - 1)


def _data_url(url: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url[:120]:
        shown = url if isinstance(url, str) and len(url) < 80 else str(url)[:77] + "..."
        raise ValueError(f"images are taken as base64 data URLs (data:image/png;base64,...), not {shown!r}: "
                         "this server does not fetch from the network")
    return base64.b64decode(url.split(",", 1)[1])


def picture_of(part) -> Optional[Picture]:
    """A message content part -> its Picture, or None if it is not an image."""
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if isinstance(part.get("image"), Picture):
        return part["image"]
    if kind in ("image_url", "input_image") or (kind is None and "image_url" in part):
        url = part.get("image_url")
        return Picture(_data_url(url.get("url") if isinstance(url, dict) else url))
    if kind == "image":
        source = part.get("source") or {}
        if source.get("type") == "base64":
            return Picture(base64.b64decode(source.get("data", "")))
        if source.get("type") == "url":
            return Picture(_data_url(source.get("url")))
        raise ValueError("an image part without its data (base64 source or a data URL)")
    return None


def vision_settings(path: str) -> dict:
    """The image processor's settings in a vision file's metadata, without loading the tower."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get("__metadata__") or {}
    return json.loads(meta.get("processor") or "{}")
