# Technical Manual Translator

Automatic translation of English technical illustrations, service manuals and repair guides into Russian using OCR and local LLM inference.

The project recursively scans images, detects English text, translates it using a local language model and renders the translated text back onto the original image while preserving the layout.

## Features

- Recursive processing of `jpg/jpeg` files
- OCR with text block detection and coordinates
- Automatic merging of fragmented OCR blocks
- Context-aware translation of labels and annotations
- Local inference (fully offline)
- Translation cache for repeated terms
- Automotive and motorcycle terminology dictionary
- Automatic font size fitting
- Preservation of directory structure in output

## Pipeline

```text
Image
  ↓
OCR (PaddleOCR)
  ↓
Block merging
  ↓
Context extraction
  ↓
Translation (Qwen GGUF)
  ↓
Text rendering
  ↓
Translated image
````

## Project Structure

```text
project/
│
├── files/
├── output/
├── models/
│   └── qwen3-14b-q4_k_m.gguf
├── fonts/
│   └── NotoSans-Regular.ttf
│
├── main.py
├── requirements.txt
└── translation_cache.json
```

## Requirements

### Hardware

* NVIDIA GPU with CUDA support
* Recommended: RTX 5060 Ti 16GB or better
* 32 GB RAM recommended
* SSD storage

### Software

* Ubuntu 24.04 / Windows 11
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
models/qwen3-14b-q4_k_m.gguf
```

## Usage

Put images into:

```text
files/
```

Run:

```bash
python main.py
```

Translated images will be written to:

```text
output/
```

## Translation Strategy

The translator receives:

1. Large text blocks as document context.
2. Small labels and annotations as translation targets.

This significantly improves translation quality of short technical labels that depend on surrounding descriptions.

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

## Future Plans

* PDF support
* PNG transparency support
* Automatic text color detection
* Batch translation
* GUI application
* Docker image
* Multi-language support

## License

MIT License
