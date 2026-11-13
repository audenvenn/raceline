#!/usr/bin/env python3
"""Image input through the chat runtime, on a real model (#28).

  .venv/bin/python tests/check_image_chat.py MODEL.rlm

The model needs its vision file next to it (raceline.vision.find_vision).
Two pictures of the same size with different shapes, colors and text:
  1. the model reads the first one (its answer names what is in it);
  2. a follow-up question reuses the whole first request: the picture is not
     run through the tower again and its rows are not computed again;
  3. a new conversation with the second picture, the same size, is not
     taken for the first: its rows are computed, and the answer is about it;
  4. back to the first conversation: it was set aside with its picture's
     rows and continues without computing them again.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def picture(path, colors, text):
    from PIL import Image, ImageDraw

    from PIL import ImageFont

    # 896 x 640: 560 rows, so that a conversation with it is long enough to be set aside (Session.min_park_tokens)
    img = Image.new("RGB", (896, 640), (240, 240, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 400, 280], fill=colors[0])
    d.ellipse([480, 80, 840, 400], fill=colors[1])
    d.polygon([(120, 600), (320, 360), (520, 600)], fill=colors[2])
    d.text((580, 500), text, fill=(0, 0, 0), font=ImageFont.load_default(size=40))
    img.save(path)


def main():
    from raceline.backends import Sampling
    from raceline.chat import Runtime
    from raceline.engine import RacelineBackend

    model = sys.argv[1]
    work = tempfile.mkdtemp()
    a, b = os.path.join(work, "a.png"), os.path.join(work, "b.png")
    picture(a, [(200, 40, 40), (40, 90, 200), (40, 160, 70)], "raceline 42")
    picture(b, [(40, 90, 200), (220, 200, 40), (200, 40, 40)], "local 17")

    rt = Runtime(RacelineBackend(model, max_ctx=8192), Sampling(temperature=0.0, top_k=1), park_bytes=1 << 30)
    if rt.vision is None:
        sys.exit(f"{model} has no vision file next to it")
    print("vision file:", rt.vision_path)
    encoded = []
    encode = rt._image_rows

    def counted(digest):
        if rt._pictures[digest][2] is None:
            encoded.append(digest)
        return encode(digest)

    rt._image_rows = counted

    def ask(messages):
        text, stats = "", None
        for kind, value in rt.chat(messages, thinking=False, max_tokens=80):
            if kind == "content":
                text += value
            elif kind == "done":
                stats = value
        return text.strip(), stats

    q = "What shapes, colors and text are in this picture? Answer in one sentence."
    first = [{"role": "user", "content": [{"type": "image", "path": a}, {"type": "text", "text": q}]}]
    answer1, s1 = ask(first)
    print(f"1. {answer1!r}  [{s1['prompt_tokens']} prompt tokens, {s1['prefill_seconds']:.2f} s]")
    low = answer1.lower()
    ok1 = "red" in low and "blue" in low and "42" in low
    print(f"{'ok  ' if ok1 else 'FAIL'}  the answer names the red and blue shapes and the 42")

    follow = first + [{"role": "assistant", "content": answer1},
                      {"role": "user", "content": "What color is the circle? One word."}]
    answer2, s2 = ask(follow)
    print(f"2. {answer2!r}  [{s2['cached_tokens']} of {s2['prompt_tokens']} prompt tokens reused]")
    ok2 = len(encoded) == 1 and s2["cached_tokens"] >= s1["prompt_tokens"] - 8 and "blue" in answer2.lower()
    print(f"{'ok  ' if ok2 else 'FAIL'}  the follow-up reuses the first request (tower runs: {len(encoded)}) "
          "and says blue")

    other = [{"role": "user", "content": [{"type": "image", "path": b}, {"type": "text", "text": q}]}]
    answer3, s3 = ask(other)
    print(f"3. {answer3!r}  [{s3['cached_tokens']} of {s3['prompt_tokens']} prompt tokens reused]")
    low = answer3.lower()
    ok3 = (len(encoded) == 2 and s3["cached_tokens"] < 16 and "17" in low and "42" not in low
           and ("yellow" in low or "red" in low))
    print(f"{'ok  ' if ok3 else 'FAIL'}  the second picture, the same size, is computed and read as itself")

    answer4, s4 = ask(follow)
    print(f"4. {answer4!r}  [{s4['cached_tokens']} of {s4['prompt_tokens']} prompt tokens reused]")
    ok4 = len(encoded) == 2 and s4["cached_tokens"] >= s2["prompt_tokens"] - 8 and answer4 == answer2
    print(f"{'ok  ' if ok4 else 'FAIL'}  back to the first conversation: set aside with its picture, nothing "
          "computed again, the same answer")
    for f in (a, b):
        os.remove(f)
    os.rmdir(work)
    ok = ok1 and ok2 and ok3 and ok4
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
