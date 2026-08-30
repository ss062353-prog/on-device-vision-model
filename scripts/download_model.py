#!/usr/bin/env python3
"""Download the local vision model into local-vision-model/models/.

Uses the HuggingFace mirror (hf-mirror.com) when huggingface.co is blocked.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = os.environ.get(
    "LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct"
)


def main():
    from huggingface_hub import snapshot_download

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(project_dir, "models", os.path.basename(MODEL_ID))
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    print(f"[download] {MODEL_ID} -> {local_dir}", flush=True)
    snapshot_download(
        MODEL_ID,
        local_dir=local_dir,
        max_workers=4,
    )
    print("[download] 完成", flush=True)


if __name__ == "__main__":
    main()
