# Technical Manual Translator

Automatic translation of English technical illustrations, service manuals and repair guides into Russian using OCR and local LLM inference.

The project recursively scans images, detects English text, translates it using a local language model and renders the translated text back onto the original image while preserving the layout.

## Features

- Recursive processing of `jpg/jpeg` files
- OCR with text block detection and coordinates (PaddleOCR, with optional Tesseract fallback)
- Same-line fragment merging, including dotted-leader pairs (`Clean ......... Replace`)
- Geometry-aware filtering of OCR noise (vertical digit columns, page numbers, stray punctuation)
- Paragraph-aware translation: multi-line text blocks are translated with full paragraph context, then mapped back to individual line rectangles for accurate rendering
- Context-aware translation of labels and annotations
- Local inference (fully offline)
- Translation cache for repeated terms, with automatic invalidation of bad/garbage entries
- Automotive and motorcycle terminology dictionary
- Two-pass rendering (erase all originals, then draw all translations) to prevent overlap artifacts
- Automatic font size fitting
- Preservation of directory structure in output

## Pipeline

```text
Image
  ↓
OCR (PaddleOCR / Tesseract fallback)
  ↓
Same-line fragment merging
  ↓
Geometry + text-quality filtering
  ↓
Paragraph grouping (for LLM context only)
  ↓
Translation (Qwen GGUF, paragraph-aware)
  ↓
Per-line translation alignment
  ↓
Two-pass rendering (erase, then draw)
  ↓
Translated image
```

### Why paragraph grouping is separate from rendering

Earlier iterations merged OCR fragments vertically into multi-line blocks and rendered the translation into the merged bounding box. This caused two classes of bugs: translated text drawn in the wrong place when the merge mis-estimated the box, and partial erasure that left English text visible underneath the Russian text.

The current pipeline merges lines into paragraphs *only* to give the LLM enough context to translate a multi-line block coherently. Each line keeps its own original OCR rectangle. The LLM's paragraph-level translation is split back into one string per line (`align_translation_to_lines`), and rendering always erases and draws into the original per-line rectangle — never an inferred merged one.

## Project Structure

```text
project/
│
├── files/
├── output/
├── logs/                        # per-page debug JSON (OCR blocks, LLM I/O, render plan)
├── models/
│   └── qwen3-14b-instruct-q4_k_m.gguf
├── fonts/
│   └── DejaVuSans.ttf
│
├── main.py                      # process_image() orchestrator + entry point
├── infra.py                     # config, cache, debug logging, I/O, text-quality
│                                 # checks, dictionaries, rect geometry
├── extract.py                   # OCR engine, fragment merging, paragraph grouping
├── translate_render.py          # local LLM client, translation logic, image rendering
│
├── requirements.txt
└── translation_cache.json
```

### Module responsibilities

| Module | Owns |
|---|---|
| `infra.py` | Env-driven config, on-disk translation cache (with sanitization), per-page debug logging, input/output file handling, text-quality predicates (garbage/proper-name/Cyrillic detection), the `TECH_DICT`/`PHRASE_DICT` dictionaries, and rectangle geometry helpers. No business logic. |
| `extract.py` | Turns a page image into a list of translatable text lines: PaddleOCR init and result parsing, Tesseract fallback, same-line fragment merging (including dotted-leader pairs), paragraph grouping for LLM context, and the inverse operation that splits a paragraph translation back into per-line strings. |
| `translate_render.py` | Turns text lines into Russian text drawn back onto the image: local Qwen model init, the dictionary → cache → LLM translation cascade (both single-line and paragraph-aware), and the two-pass erase/draw renderer with automatic font-size fitting. |
| `main.py` | `process_image()` wires the three stages together for one page and records debug events; `main()` walks `files/` and calls it for every image. Contains no extraction/translation/rendering logic itself. |

Dependency direction is strictly linear: `infra → extract → translate_render → main`. No circular imports.

## Requirements

### Hardware

* NVIDIA GPU with CUDA support
* Recommended: RTX 5060 Ti 16GB or better
* 32 GB RAM recommended
* SSD storage

### Software

* Ubuntu 24.04 / Windows 11 (WSL2)
* Python 3.11
* CUDA 12.x+
* llama-cpp-python (CUDA build)

## Installation

Create virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install CUDA version of llama-cpp-python:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" \
pip install --force-reinstall \
--no-cache-dir \
llama-cpp-python
```

## Models

Place GGUF models into:

```text
models/
```

Example:

```text
models/qwen3-14b-instruct-q4_k_m.gguf
```

Set `MODEL_PATH` if using a different file or location.

## Configuration

All behavior toggles are environment variables (see `infra.py`):

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/qwen3-14b-instruct-q4_k_m.gguf` | GGUF model used for translation |
| `READY_OUTPUT_DIR` | `/mnt/e/repos/translations/ready` | Secondary output copy (e.g. for a downstream watch folder) |
| `OCR_PROFILE` | `gpu_locked` | PaddleOCR device/profile selection |
| `USE_TESSERACT_FALLBACK` | `0` | Fall back to Tesseract when PaddleOCR output is weak |
| `USE_PREPROCESS_VARIANTS` | `0` | Try CLAHE/Otsu preprocessing variants before OCR |
| `FORCE_RETRANSLATE_BAD_CACHE` | `1` | Drop and redo cache entries that look wrong (no Cyrillic, mixed languages, unchanged text) |

## Usage

Put images into:

```text
files/
```

Run:

```bash
python main.py
```

Translated images will be written to `output/` (mirroring the input directory structure) and to `READY_OUTPUT_DIR`. A per-page JSON debug log is written to `logs/`, recording the OCR blocks detected, the filtering decisions, the LLM prompts/responses, and the final render plan — useful for diagnosing misplaced or skipped translations.

## Translation Strategy

The translator receives:

1. The page's largest text blocks as document context, given to the LLM alongside every translation request so it can disambiguate domain-specific terms.
2. Individual lines as translation targets — single lines are batched together; lines belonging to the same paragraph/box are translated together (to preserve cross-line meaning) but rendered independently using their own original coordinates.

Three tiers are tried for each piece of text, cheapest and most deterministic first: a fixed dictionary (`TECH_DICT`/`PHRASE_DICT`), the on-disk translation cache, and finally the local LLM. Repeated terms across pages are translated once and reused, with bad cache entries automatically detected and discarded (e.g. translations that still contain Latin script, or that are identical to the English source).

## Known Failure Modes and Mitigations

* **OCR splits a numbered list's digits into one tall narrow block** (e.g. "2345" from list items 1–5 merging vertically) — filtered out by an aspect-ratio + short-text-length check before translation.
* **Dotted-leader lines split into two OCR fragments** ("Clean" far to the right of "Clogged slow circuit") — merged via a relaxed horizontal-gap rule when the right-hand fragment is short and contains no dots itself.
* **Short action words skipped** (Clean, Replace, Adjust, Correct) — the candidate filter no longer enforces a minimum word length; these are also hardcoded into `TECH_DICT` for zero-ambiguity translation.
* **Translated text overlapping un-erased English** — fixed by a strict two-pass render (erase every original region first, then draw every translation), plus a small rect expansion margin during erase to cover anti-aliasing.

## Supported Content

* Motorcycle service manuals
* Automotive repair manuals
* Parts diagrams
* Technical illustrations
* Assembly instructions
* Workshop documentation

## Limitations

* Currently supports English → Russian translation only.
* Optimized for JPG/JPEG images.
* Does not preserve original font family.
* Complex diagrams with overlapping labels may require manual correction.
* Multi-line paragraph translations are aligned back to source lines by line-count match when possible, falling back to proportional character-length splitting otherwise — occasionally imperfect for heavily reworded LLM output.

## Future Plans

* PDF support
* PNG transparency support
* Automatic text color detection
* Batch translation
* GUI application
* Docker image
* Multi-language support
* OCR bounding-box visualization tool as a first-class debugging command (currently ad hoc)

## License

MIT License