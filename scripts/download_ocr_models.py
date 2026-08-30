#!/usr/bin/env python3
"""Download multilingual OCR models (RapidOCR / PP-OCR) from ModelScope.

Stores files in local-vision-model/ocr_models/{lang}/ so the server can run
OCR offline for: Chinese/English (bundled with rapidocr_onnxruntime) plus
Japanese, Korean, Cyrillic, Arabic, Devanagari, Latin, Traditional Chinese,
Tamil, Telugu, Kannada, Thai, Greek and Eslav (PP-OCRv5 where available).
"""

from __future__ import annotations

import os
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"

# lang -> (rec model path, dict path); PP-OCRv5 preferred, v4 fallback
LANGS = {
    "en": (
        "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt",
    ),
    "japan": (
        "onnx/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile.onnx",
        "paddle/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile/japan_dict.txt",
    ),
    "korean": (
        "onnx/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/korean_PP-OCRv5_rec_mobile/ppocrv5_korean_dict.txt",
    ),
    "cyrillic": (
        "onnx/PP-OCRv5/rec/cyrillic_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/cyrillic_PP-OCRv5_rec_mobile/ppocrv5_cyrillic_dict.txt",
    ),
    "arabic": (
        "onnx/PP-OCRv5/rec/arabic_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/arabic_PP-OCRv5_rec_mobile/ppocrv5_arabic_dict.txt",
    ),
    "devanagari": (
        "onnx/PP-OCRv5/rec/devanagari_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/devanagari_PP-OCRv5_rec_mobile/ppocrv5_devanagari_dict.txt",
    ),
    "latin": (
        "onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile/ppocrv5_latin_dict.txt",
    ),
    "chinese_cht": (
        "onnx/PP-OCRv4/rec/chinese_cht_PP-OCRv3_rec_mobile.onnx",
        "paddle/PP-OCRv4/rec/chinese_cht_PP-OCRv3_rec_mobile/chinese_cht_dict.txt",
    ),
    "ta": (
        "onnx/PP-OCRv4/rec/ta_PP-OCRv4_rec_mobile.onnx",
        "paddle/PP-OCRv4/rec/ta_PP-OCRv4_rec_mobile/ta_dict.txt",
    ),
    "te": (
        "onnx/PP-OCRv4/rec/te_PP-OCRv4_rec_mobile.onnx",
        "paddle/PP-OCRv4/rec/te_PP-OCRv4_rec_mobile/te_dict.txt",
    ),
    "ka": (
        "onnx/PP-OCRv4/rec/ka_PP-OCRv4_rec_mobile.onnx",
        "paddle/PP-OCRv4/rec/kannada_PP-OCRv4_rec_mobile/ka_dict.txt",
    ),
    "th": (
        "onnx/PP-OCRv5/rec/th_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/th_PP-OCRv5_rec_mobile/ppocrv5_th_dict.txt",
    ),
    "el": (
        "onnx/PP-OCRv5/rec/el_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/el_PP-OCRv5_rec_mobile/ppocrv5_el_dict.txt",
    ),
    "eslav": (
        "onnx/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx",
        "paddle/PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile/ppocrv5_eslav_dict.txt",
    ),
}

COMMON = {
    "det": "onnx/PP-OCRv4/det/multi_PP-OCRv3_det_mobile.onnx",
    "cls": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}


def _download(url, dest, label):
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {label}", flush=True)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[get ] {label}", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)
    print(f"[done] {label} ({os.path.getsize(dest) / 1048576:.1f} MB)", flush=True)


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_dir, "ocr_models")
    for name, rel in COMMON.items():
        _download(f"{BASE}/{rel}", os.path.join(out_dir, f"{name}.onnx"), f"common/{name}")
    for lang, (rec_rel, dict_rel) in LANGS.items():
        lang_dir = os.path.join(out_dir, lang)
        _download(f"{BASE}/{rec_rel}", os.path.join(lang_dir, "rec.onnx"), f"{lang}/rec")
        _download(f"{BASE}/{dict_rel}", os.path.join(lang_dir, "dict.txt"), f"{lang}/dict")
    print("全部多语言 OCR 模型下载完成。")


if __name__ == "__main__":
    main()
