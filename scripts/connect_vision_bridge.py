#!/usr/bin/env python3
"""Point the vision-bridge skill at the local vision model server.

Writes vision-bridge's config so future media requests go to the local model:
  provider=custom, base_url=http://127.0.0.1:8321/v1, api_key=local
"""

from __future__ import annotations

import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HOST = os.environ.get("LOCAL_VISION_HOST", "127.0.0.1")
PORT = os.environ.get("LOCAL_VISION_PORT", "8321")
MODEL_ID = os.environ.get(
    "LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct"
)


def main():
    vision_py = os.path.join(
        os.path.expanduser("~"),
        ".codex",
        "skills",
        "vision-bridge",
        "scripts",
        "vision.py",
    )
    if not os.path.isfile(vision_py):
        print("找不到 vision-bridge 脚本，请先安装 vision-bridge 技能", file=sys.stderr)
        return 1
    skill_config = os.path.join(
        os.path.expanduser("~"),
        ".codex",
        "skills",
        "vision-bridge",
        "config.json",
    )
    cmd = [
        sys.executable,
        vision_py,
        "configure",
        "--provider",
        "custom",
        "--api-key",
        "local",
        "--base-url",
        f"http://{HOST}:{PORT}/v1",
        "--model",
        MODEL_ID,
    ]
    env = dict(os.environ)
    env["VISION_CONFIG_FILE"] = skill_config
    subprocess.run(cmd, check=True, env=env)
    print(
        f"vision-bridge 已指向本地模型: {MODEL_ID} @ http://{HOST}:{PORT}/v1\n"
        f"配置文件: {skill_config}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
