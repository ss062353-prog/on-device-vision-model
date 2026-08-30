# On-device Vision Model

> Give text-only models (e.g. DeepSeek) real vision — 100% on your own device.

**On-device Vision Model** is a fully local, offline vision pipeline that lets
text-only coding models see and read images. It combines a small vision-language
model (SmolVLM-256M-Instruct) with a fast multilingual OCR engine (RapidOCR /
PP-OCR) and exposes a standard OpenAI-compatible API on `http://127.0.0.1:8321`.

## Key points

- **Only tokens are needed.** No API key, no subscription, no cloud bill. The
  model runs entirely on your machine and is free forever.
- **Recommended for native speakers of Chinese, English, Japanese and Korean.**
  These four languages have the best recognition accuracy out of the box, and
  Traditional Chinese plus many Latin/Cyrillic/Arabic/Devanagari/Thai/Greek
  scripts are supported as well.
- **OpenAI-compatible interface** (`/v1/chat/completions`, `/v1/models`,
  `/v1/ocr`, `/health`) — drop it into any OpenAI-compatible client or Codex
  skill (e.g. `deepseek-vision`, `vision-bridge`).
- **CPU friendly**: model loads in under a second; a single image is analyzed
  in 1–3 seconds; no GPU required.

## What it can do

- Understand and describe images, photos, screenshots, diagrams and documents.
- Extract text from images (OCR) with per-line, character-accurate output.
- Detect the document language automatically, or force a specific language
  (`ja`, `ko`, `ru`, `ar`, `hi`, `th`, `el`, `fr`, `cht`, ...).
- Return token usage in every response so you always know the cost (which is
  zero — it is all local).

## Languages supported

Chinese (Simplified/Traditional), English, Japanese, Korean, Cyrillic
(Russian etc.), Arabic, Devanagari (Hindi etc.), Thai, Greek, Latin-script
languages (French, German, Spanish, Italian, Portuguese, Vietnamese, ...),
Tamil, Telugu, Kannada and South Slavic languages.

## Quick start

Requires Python 3.10+.

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt        # macOS / Linux

# 2. Download the vision-language model (~1 GB, via HuggingFace mirror)
.venv/Scripts/python scripts/download_model.py

# 3. Download the multilingual OCR models (~130 MB, via ModelScope)
.venv/Scripts/python scripts/download_ocr_models.py

# 4. Start the server
.venv/Scripts/python server.py --warmup
# Or double-click start-vision-model.cmd (Windows)
```

The server listens on `http://127.0.0.1:8321`.

## Usage examples

```bash
# Describe an image (vision-language model)
curl http://127.0.0.1:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"file:///C:/path/to/image.png"}},
        {"type":"text","text":"Describe this image."}}]}'

# Extract text with automatic language detection (OCR engine)
curl http://127.0.0.1:8321/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{"images":["file:///C:/path/to/document.png"],"lang":"auto"}'

# Force Japanese OCR
curl http://127.0.0.1:8321/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{"images":["file:///C:/path/to/ja.png"],"lang":"ja"}'
```

Every OCR response includes the `languages` that were actually used; every chat
response includes `usage` (prompt/completion/total tokens).

## Codex skill integration

Point the `vision-bridge` skill at this server so a text-only model (DeepSeek)
can read images through it:

```bash
.venv/Scripts/python scripts/connect_vision_bridge.py
```

Then just say “extract the text from this image” — the whole chain
(`deepseek-vision` → `vision-bridge` → this model) runs locally.

## Repository layout

```text
on-device-vision-model/
|-- server.py                     OpenAI-compatible local vision server
|-- requirements.txt              Python dependencies
|-- scripts/
|   |-- download_model.py         Downloads the VLM weights
|   |-- download_ocr_models.py    Downloads multilingual OCR models
|   |-- connect_vision_bridge.py  Wires the Codex skill to this server
|   `-- smoke_test.py             End-to-end smoke test
|-- start-vision-model.cmd        Windows launcher
|-- start-background.ps1          Background service launcher (Windows)
`-- README.md
```

Model weights are intentionally **not** stored in this repository; they are
downloaded by the scripts above so the repo stays small and reproducible.

## License

Project code: MIT (see `LICENSE`).

Component licenses:
- SmolVLM-256M-Instruct (HuggingFaceTB): Apache-2.0.
- RapidOCR / PP-OCR models (RapidAI / PaddlePaddle): Apache-2.0.

The model downloads a few hundred MB on first run; media is sent only to your
own local process — nothing leaves your machine.
