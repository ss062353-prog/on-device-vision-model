#!/usr/bin/env python3
"""Smoke test: POST a generated test image to the local server."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request

try:
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HOST = os.environ.get("LOCAL_VISION_HOST", "127.0.0.1")
PORT = os.environ.get("LOCAL_VISION_PORT", "8321")
MODEL_ID = os.environ.get(
    "LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct"
)


def _make_test_image():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\segoeui.ttf",
    ):
        if os.path.isfile(candidate):
            try:
                font = ImageFont.truetype(candidate, 64)
                break
            except Exception:
                font = None
    draw.rectangle([0, 0, 639, 239], outline="black", width=6)
    text = "Hello 42"
    draw.text((120, 80), text, fill="black", font=font or ImageFont.load_default())
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    image_b64 = _make_test_image()
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": "图片里写了什么文字？请逐字回答。"},
                ],
            }
        ],
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print("模型回答:", answer)
    print("usage:", json.dumps(usage, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("冒烟测试失败:", exc, file=sys.stderr)
        raise SystemExit(1)
