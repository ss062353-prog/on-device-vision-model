#!/usr/bin/env python3
"""Local Vision Model Server - OpenAI-compatible /v1/chat/completions.

Serves a small vision-language model (default: HuggingFaceTB/SmolVLM-256M-Instruct)
on your own machine with an OpenAI-compatible API, so the vision-bridge and
deepseek-vision skills can use it as a "custom" provider without any cloud key.

Endpoints:
  GET  /health               -> simple health check
  GET  /v1/models            -> OpenAI model list
  POST /v1/chat/completions  -> OpenAI chat completion (text + image_url)
  POST /v1/ocr               -> multilingual OCR: {"images": [...], "lang": "auto"}

Run:
  python server.py --warmup
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OCR_MODELS_DIR = os.path.join(PROJECT_DIR, "ocr_models")

DEFAULT_MODEL_ID = os.environ.get(
    "LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct"
)
DEFAULT_HOST = os.environ.get("LOCAL_VISION_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LOCAL_VISION_PORT", "8321"))
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_NEW_TOKENS_CAP = 1024
MAX_REQUEST_BYTES = 64 * 1024 * 1024

_model = None
_processor = None
_loaded_model_id = ""
_model_lock = threading.Lock()
_infer_lock = threading.Lock()
_ocr_engines = {}
_ocr_lock = threading.Lock()

OCR_KEYWORDS = (
    "提取文字",
    "识别文字",
    "文字",
    "文本",
    "文字内容",
    "文字提取",
    "读取文字",
    "图片中的文字",
    "全部文字",
    "逐字",
    "ocr",
    "转录",
    "转写",
    "extract text",
    "read the text",
    "transcribe",
    "transcript",
    "text recognition",
)

# 用户常用语言名/代码 -> 模型目录名
LANG_ALIASES = {
    "zh": "default", "ch": "default", "cn": "default", "chinese": "default",
    "en": "en", "english": "en",
    "ja": "japan", "jp": "japan", "japan": "japan", "japanese": "japan",
    "ko": "korean", "kr": "korean", "korean": "korean",
    "ru": "cyrillic", "russian": "cyrillic", "cyrillic": "cyrillic",
    "ar": "arabic", "arabic": "arabic",
    "hi": "devanagari", "hindi": "devanagari", "devanagari": "devanagari",
    "th": "th", "thai": "th",
    "el": "el", "greek": "el",
    "fr": "latin", "de": "latin", "es": "latin", "it": "latin",
    "pt": "latin", "vi": "latin", "latin": "latin",
    "cht": "chinese_cht", "traditional": "chinese_cht", "chinese_cht": "chinese_cht",
    "ta": "ta", "tamil": "ta",
    "te": "te", "telugu": "te",
    "ka": "ka", "kannada": "ka",
    "eslav": "eslav", "bulgarian": "eslav", "serbian": "eslav",
    "ukrainian": "eslav", "belarusian": "eslav",
}

# 提问中的语言提示 -> 模型目录名（按顺序匹配，具体语言在前）
LANG_HINTS = [
    (("日语", "日文", "japanese", "japan"), "japan"),
    (("韩语", "韩文", "korean"), "korean"),
    (("俄语", "俄文", "russian", "cyrillic"), "cyrillic"),
    (("阿拉伯", "arabic"), "arabic"),
    (("印地", "梵文", "梵语", "hindi", "devanagari"), "devanagari"),
    (("泰语", "泰文", "thai"), "th"),
    (("希腊", "greek"), "el"),
    (("繁体", "traditional chinese"), "chinese_cht"),
    (("泰米尔", "tamil"), "ta"),
    (("泰卢固", "telugu"), "te"),
    (("卡纳达", "kannada"), "ka"),
    (("保加利亚", "塞尔维亚", "乌克兰", "白俄罗斯", "bulgarian", "serbian", "ukrainian"), "eslav"),
    (
        ("英语", "英文", "法文", "法语", "德文", "德语", "西班牙", "意大利",
         "葡萄牙", "越南", "拉丁", "english", "french", "german", "spanish",
         "italian", "portuguese", "vietnamese", "latin"),
        "latin",
    ),
]


def _local_model_path(model_id):
    """Resolve a local folder when the model was pre-downloaded."""
    if os.path.isdir(model_id):
        return model_id
    candidate = os.path.join(MODELS_DIR, os.path.basename(model_id))
    if os.path.isdir(candidate):
        return candidate
    return model_id


def load_model(model_id):
    global _model, _processor, _loaded_model_id
    with _model_lock:
        if _model is not None:
            return
        resolved = _local_model_path(model_id)
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        except ImportError:  # transformers < 5
            from transformers import AutoModelForVision2Seq

        print(f"[local-vision] 加载模型: {resolved}", flush=True)
        t0 = time.time()
        _processor = AutoProcessor.from_pretrained(resolved)
        _model = AutoModelForVision2Seq.from_pretrained(resolved)
        _model.eval()
        _loaded_model_id = model_id
        print(
            f"[local-vision] 模型就绪，用时 {time.time() - t0:.1f}s",
            flush=True,
        )


def _get_ocr_engine(lang):
    """Return a cached RapidOCR engine for a language directory name."""
    global _ocr_engines
    with _ocr_lock:
        if lang in _ocr_engines:
            return _ocr_engines[lang]
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise ValueError(
                "未安装 rapidocr_onnxruntime，请运行 "
                ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc
        lang_path = os.path.join(OCR_MODELS_DIR, lang)
        if lang == "default" or not (
            os.path.isfile(os.path.join(lang_path, "rec.onnx"))
            and os.path.isfile(os.path.join(lang_path, "dict.txt"))
        ):
            engine = RapidOCR()
            label = "内置 中英"
        else:
            engine = RapidOCR(
                det_model_path=os.path.join(OCR_MODELS_DIR, "det.onnx"),
                cls_model_path=os.path.join(OCR_MODELS_DIR, "cls.onnx"),
                rec_model_path=os.path.join(lang_path, "rec.onnx"),
                rec_keys_path=os.path.join(lang_path, "dict.txt"),
            )
            label = lang
        print(f"[local-vision] 加载 OCR 引擎: {label}", flush=True)
        _ocr_engines[lang] = engine
        return engine


def _installed_ocr_langs():
    """All language dirs with models, plus the bundled default."""
    langs = ["default"]
    if os.path.isdir(OCR_MODELS_DIR):
        for name in sorted(os.listdir(OCR_MODELS_DIR)):
            if os.path.isfile(os.path.join(OCR_MODELS_DIR, name, "rec.onnx")):
                langs.append(name)
    return langs


def _is_ocr_intent(messages):
    """Detect whether the request asks to extract/read text from images."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content.lower()
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            text = " ".join(parts).lower()
        else:
            continue
        if any(keyword in text for keyword in OCR_KEYWORDS):
            return True
    return False


def _detect_ocr_lang(messages):
    """Infer the target OCR language from prompt keywords, or None."""
    text = ""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text += " " + content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text += " " + str(part.get("text", ""))
    text = text.lower()
    for keywords, lang in LANG_HINTS:
        for keyword in keywords:
            if keyword.lower() in text:
                return lang
    return None


def _box_rect(box):
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def _box_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = _box_rect(box_a)
    bx0, by0, bx1, by1 = _box_rect(box_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ocr_images(images, lang="default"):
    """Run OCR with one engine; return text lines in reading order."""
    import numpy as np

    engine = _get_ocr_engine(lang)
    all_lines = []
    for image in images:
        with _ocr_lock:
            result, _ = engine(np.array(image))
        if result:
            items = sorted(
                result,
                key=lambda item: (round(item[0][0][1]), item[0][0][0]),
            )
            for _box, text, _score in items:
                if str(text).strip():
                    all_lines.append(str(text).strip())
    return "\n".join(all_lines)


def _ocr_images_multi(images, langs):
    """Run several language engines and merge boxes by confidence + overlap."""
    import numpy as np

    candidates = []
    for lang in langs:
        engine = _get_ocr_engine(lang)
        for image in images:
            with _ocr_lock:
                result, _ = engine(np.array(image))
            for box, text, score in result or []:
                if str(text).strip():
                    candidates.append((box, str(text).strip(), float(score), lang))
    candidates.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for candidate in candidates:
        if any(_box_iou(candidate[0], other[0]) > 0.35 for other in kept):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (round(min(point[1] for point in item[0])), min(point[0] for point in item[0])))
    used = sorted({lang for _, _, _, lang in kept})
    return "\n".join(text for _, text, _, _ in kept), used


def _load_image_bytes(url):
    """Return raw image bytes from data URL, http(s) URL, or local path."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" in header:
            return base64.b64decode(payload)
        import urllib.parse

        return urllib.parse.unquote_to_bytes(payload)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    if os.path.isfile(url):
        with open(url, "rb") as fh:
            return fh.read()
    raise ValueError("image_url 必须是 data URL、http(s) URL 或本地文件路径")


def _open_image(url):
    from PIL import Image

    raw = _load_image_bytes(url)
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 25 MB 上限")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError(f"图片解码失败: {exc}") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _to_hf_messages(messages):
    """Convert OpenAI messages to HuggingFace chat messages + image list."""
    images = []
    hf_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            hf_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("content 必须是字符串或内容数组")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": str(part)})
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif ptype == "image_url":
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url", "")
                images.append(_open_image(url))
                parts.append({"type": "image"})
            elif ptype == "input_audio":
                raise ValueError(
                    "input_audio 暂不支持：本地模型只能看图，音频请换用其他渠道"
                )
            else:
                parts.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
        if not any(p.get("type") == "image" for p in parts):
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
            hf_messages.append({"role": role, "content": text or "请回答"})
        else:
            hf_messages.append({"role": role, "content": parts})
    return hf_messages, images


def _generate(messages, max_new_tokens, temperature, top_p, prebuilt=None):
    if prebuilt is not None:
        hf_messages, images = prebuilt
    else:
        hf_messages, images = _to_hf_messages(messages)
    if not images:
        # SmolVLM 也能纯文本对话，这里保留兼容
        prompt = _processor.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = _processor(text=[prompt], return_tensors="pt")
    else:
        prompt = _processor.apply_chat_template(
            hf_messages, add_generation_prompt=True, tokenize=False
        )
        inputs = _processor(
            text=[prompt], images=images, return_tensors="pt"
        )

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature is not None and temperature > 0,
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p or 1.0

    with _infer_lock:
        outputs = _model.generate(**inputs, **gen_kwargs)

    prompt_ids = inputs.get("input_ids")
    prompt_len = int(prompt_ids.shape[1]) if prompt_ids is not None else 0
    answer_ids = outputs[0][prompt_len:]
    answer = _processor.decode(answer_ids, skip_special_tokens=True).strip()
    usage = {
        "prompt_tokens": int(prompt_len),
        "completion_tokens": int(len(answer_ids)),
        "total_tokens": int(prompt_len) + int(len(answer_ids)),
    }
    return answer, usage


def _chat_completion(payload):
    if payload.get("stream"):
        raise ValueError("stream 暂不支持")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 不能为空")
    max_tokens = int(payload.get("max_tokens", 512) or 512)
    max_tokens = max(1, min(max_tokens, MAX_NEW_TOKENS_CAP))
    temperature = payload.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
    top_p = payload.get("top_p")
    if top_p is not None:
        top_p = float(top_p)

    prebuilt = _to_hf_messages(messages)
    # OCR 通道：请求明确要求提取/识别文字且图片存在时，走多语言 OCR 引擎
    if _is_ocr_intent(messages):
        _hf, images = prebuilt
        if images:
            detected = _detect_ocr_lang(messages)
            if detected:
                text = _ocr_images(images, detected)
                used = [detected]
            else:
                text, used = _ocr_images_multi(images, _installed_ocr_langs())
            if text.strip():
                chars = len(text)
                return {
                    "id": "chatcmpl-local-ocr-"
                    + base64.urlsafe_b64encode(os.urandom(6)).decode(),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": _loaded_model_id,
                    "engine": "ocr",
                    "languages": used,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": chars,
                        "total_tokens": chars,
                    },
                }
            # OCR 未检出文字时回退视觉模型

    answer, usage = _generate(
        messages, max_tokens, temperature, top_p, prebuilt=prebuilt
    )
    return {
        "id": "chatcmpl-local-" + base64.urlsafe_b64encode(os.urandom(6)).decode(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _loaded_model_id,
        "engine": "vlm",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalVisionModel/0.1"

    def _send_json(self, status, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # 客户端提前断开，忽略即可，避免刷异常堆栈
            pass

    def _send_error_json(self, status, message):
        self._send_json(
            status,
            {
                "error": {
                    "message": str(message),
                    "type": "invalid_request_error",
                    "code": status,
                }
            },
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok", "model": _loaded_model_id})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": _loaded_model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        if self.path.split("?", 1)[0].rstrip("/") in ("", "/"):
            self._send_json(
                200,
                {
                    "service": "local-vision-model",
                    "model": _loaded_model_id,
                    "endpoints": [
                        "GET /v1/models",
                        "POST /v1/chat/completions",
                        "POST /v1/ocr",
                        "GET /health",
                    ],
                },
            )
            return
        self._send_error_json(404, "Not found")

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/v1/ocr":
            self._handle_ocr()
            return
        if path != "/v1/chat/completions":
            self._send_error_json(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_REQUEST_BYTES:
                self._send_error_json(413, "请求体过大（上限 64 MB）")
                return
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            result = _chat_completion(payload)
            self._send_json(200, result)
        except json.JSONDecodeError:
            self._send_error_json(400, "请求体不是合法 JSON")
        except ValueError as exc:
            self._send_error_json(400, str(exc))
        except Exception as exc:
            self._send_error_json(500, f"推理失败: {exc}")

    def _handle_ocr(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_REQUEST_BYTES:
                self._send_error_json(413, "请求体过大（上限 64 MB）")
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image_urls = payload.get("images") or payload.get("image_urls") or []
            if not image_urls:
                raise ValueError("images 不能为空（data URL / http(s) URL / 本地路径）")
            langs = payload.get("languages") or []
            raw_lang = str(
                payload.get("lang") or payload.get("language") or "auto"
            ).strip().lower()
            if not langs:
                if raw_lang in ("", "auto", "any", "all"):
                    langs = _installed_ocr_langs()
                else:
                    canonical = LANG_ALIASES.get(raw_lang)
                    if not canonical:
                        raise ValueError(
                            f"未知语言: {raw_lang}（可选: {', '.join(sorted(set(LANG_ALIASES.values())))}）"
                        )
                    langs = [canonical]
            langs = [
                LANG_ALIASES.get(str(lang).strip().lower(), str(lang).strip().lower())
                for lang in langs
            ]
            images = [_open_image(url) for url in image_urls]
            if len(langs) == 1:
                text = _ocr_images(images, langs[0])
                used = langs
            else:
                text, used = _ocr_images_multi(images, langs)
            self._send_json(
                200,
                {
                    "engine": "ocr",
                    "text": text,
                    "characters": len(text),
                    "languages": used,
                },
            )
        except json.JSONDecodeError:
            self._send_error_json(400, "请求体不是合法 JSON")
        except ValueError as exc:
            self._send_error_json(400, str(exc))
        except Exception as exc:
            self._send_error_json(500, f"OCR 失败: {exc}")

    def log_message(self, fmt, *args):
        sys.stderr.write("[local-vision] %s\n" % (fmt % args))


def main():
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible vision model server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地模型目录")
    parser.add_argument("--warmup", action="store_true", help="启动时预加载模型（默认行为，保留此参数便于脚本兼容）")
    parser.add_argument("--no-warmup", action="store_true", help="启动时不预加载模型")
    args = parser.parse_args()

    if not args.no_warmup:
        load_model(args.model)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"[local-vision] 启动失败: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    print(
        f"[local-vision] 服务已启动: http://{args.host}:{args.port}  "
        f"(模型: {args.model})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[local-vision] 已停止", flush=True)


if __name__ == "__main__":
    main()
