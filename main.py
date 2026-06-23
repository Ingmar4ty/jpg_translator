import os
import re
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paddleocr import PaddleOCR
from llama_cpp import Llama


##############################################################################
# CONFIG
##############################################################################

INPUT_DIR = "files"
OUTPUT_DIR = "output"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/qwen3-14b-q4_k_m.gguf"
)
FONT_PATH = "fonts/NotoSans-Regular.ttf"

CACHE_FILE = "translation_cache.json"

MIN_CONTEXT_AREA = 50000
MAX_LABEL_AREA = 25000

os.makedirs(OUTPUT_DIR, exist_ok=True)

##############################################################################
# TRANSLATION CACHE
##############################################################################

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf8") as f:
        TRANSLATION_CACHE = json.load(f)
else:
    TRANSLATION_CACHE = {}


def save_cache():
    with open(
        CACHE_FILE,
        "w",
        encoding="utf8"
    ) as f:
        json.dump(
            TRANSLATION_CACHE,
            f,
            ensure_ascii=False,
            indent=2
        )


##############################################################################
# TECH DICTIONARY
##############################################################################

TECH_DICT = {
    "hub": "ступица",
    "bearing": "подшипник",
    "seal": "сальник",
    "oil seal": "сальник",
    "shaft": "вал",
    "washer": "шайба",
    "fork": "вилка",
    "spring": "пружина",
    "clutch": "сцепление",
    "crankshaft": "коленчатый вал",
    "bolt": "болт",
    "nut": "гайка",
    "pin": "штифт",
    "gear": "шестерня",
    "brake": "тормоз",
    "chain": "цепь",
    "camshaft": "распредвал",
    "gasket": "прокладка",
    "sprocket": "звезда",
    "piston": "поршень",
    "cylinder": "цилиндр",
    "lever": "рычаг",
    "screw": "винт",
    "cover": "крышка",
    "plate": "пластина",
}


##############################################################################
# OCR
##############################################################################

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)

##############################################################################
# LLM
##############################################################################

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    n_batch=1024,
    verbose=False
)


##############################################################################
# HELPERS
##############################################################################

def polygon_to_rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    return (
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys))
    )


def area(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)


##############################################################################
# OCR BLOCK MERGING
##############################################################################

def merge_horizontal(blocks):
    blocks = sorted(
        blocks,
        key=lambda x: (
            x["rect"][1],
            x["rect"][0]
        )
    )

    result = []

    while blocks:
        current = blocks.pop(0)

        changed = True

        while changed:
            changed = False

            for b in blocks[:]:
                x1, y1, x2, y2 = current["rect"]
                bx1, by1, bx2, by2 = b["rect"]

                h = (
                    (y2 - y1) +
                    (by2 - by1)
                ) / 2

                same_line = abs(
                    (y1 + y2) / 2
                    -
                    (by1 + by2) / 2
                ) < h * 0.5

                gap = bx1 - x2

                if (
                    same_line
                    and 0 <= gap < h * 1.5
                ):
                    current["text"] += " " + b["text"]

                    current["rect"] = (
                        min(x1, bx1),
                        min(y1, by1),
                        max(x2, bx2),
                        max(y2, by2)
                    )

                    current["area"] = area(
                        current["rect"]
                    )

                    blocks.remove(b)
                    changed = True
                    break

        result.append(current)

    return result


def merge_vertical(blocks):
    merged = []
    used = set()

    for i, a in enumerate(blocks):
        if i in used:
            continue

        ax1, ay1, ax2, ay2 = a["rect"]

        current = [a]

        for j, b in enumerate(blocks):
            if i == j:
                continue

            if j in used:
                continue

            bx1, by1, bx2, by2 = b["rect"]

            h1 = ay2 - ay1
            h2 = by2 - by1

            avg_h = (h1 + h2) / 2

            vertical_gap = abs(by1 - ay2)

            overlap = (
                min(ax2, bx2)
                -
                max(ax1, bx1)
            )

            if (
                vertical_gap < avg_h
                and overlap > 0
            ):
                current.append(b)
                used.add(j)

        current = sorted(
            current,
            key=lambda x:
            x["rect"][1]
        )

        text = " ".join(
            x["text"]
            for x in current
        )

        xs = []
        ys = []

        for c in current:
            x1, y1, x2, y2 = c["rect"]

            xs += [x1, x2]
            ys += [y1, y2]

        rect = (
            min(xs),
            min(ys),
            max(xs),
            max(ys)
        )

        merged.append(
            {
                "text": text,
                "rect": rect,
                "area": area(rect)
            }
        )

    return merged


##############################################################################
# TRANSLATION
##############################################################################

def clean_llm_output(text):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.S
    )

    text = re.sub(
        r"```json",
        "",
        text
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    return text.strip()


def translate_labels(context, labels):
    if not labels:
        return []

    result = [""] * len(labels)

    to_translate = []
    mapping = []

    for i, label in enumerate(labels):
        key = label.strip().lower()

        if key in TECH_DICT:
            result[i] = TECH_DICT[key]
            continue

        if key in TRANSLATION_CACHE:
            result[i] = TRANSLATION_CACHE[key]
            continue

        mapping.append(i)
        to_translate.append(label)

    if not to_translate:
        return result

    labels_text = "\n".join(
        f"{i+1}. {x}"
        for i, x in enumerate(to_translate)
    )

    schema = [
        {
            "id": i + 1,
            "translation": ""
        }
        for i in range(len(to_translate))
    ]

    prompt = f"""
Context:
{context}

Translate labels into Russian.

Use technical terminology.

Output ONLY JSON.

Schema:
{json.dumps(schema, ensure_ascii=False)}

Labels:
{labels_text}
"""

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content":
                    "You are a professional technical translator. "
                    "Output JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        top_p=0.8,
        max_tokens=1024
    )

    txt = response["choices"][0]["message"]["content"]
    txt = clean_llm_output(txt)

    try:
        data = json.loads(txt)

        translated = [
            x["translation"]
            for x in data
        ]

    except Exception:
        print("LLM ERROR")
        print(txt)
        translated = to_translate

    for idx, tr in zip(mapping, translated):
        result[idx] = tr
        TRANSLATION_CACHE[
            labels[idx].strip().lower()
        ] = tr

    save_cache()

    return result


##############################################################################
# RENDER
##############################################################################

def fit_text(draw, text, rect):
    x1, y1, x2, y2 = rect

    w = x2 - x1
    h = y2 - y1

    font_size = max(
        12,
        int(h * 0.8)
    )

    while font_size > 8:
        font = ImageFont.truetype(
            FONT_PATH,
            font_size
        )

        chars = max(
            1,
            int(
                w /
                (font_size * 0.55)
            )
        )

        wrapped = textwrap.fill(
            text,
            width=chars
        )

        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped,
            font=font
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw <= w and th <= h:
            return font, wrapped

        font_size -= 1

    return (
        ImageFont.truetype(
            FONT_PATH,
            10
        ),
        text
    )


def draw_translation(img, rect, text):
    x1, y1, x2, y2 = rect

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return img

    bg = tuple(
        int(x)
        for x in crop.mean(
            axis=(0, 1)
        )
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        bg,
        -1
    )

    pil = Image.fromarray(
        cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    )

    draw = ImageDraw.Draw(pil)

    font, wrapped = fit_text(
        draw,
        text,
        rect
    )

    bbox = draw.multiline_textbbox(
        (0, 0),
        wrapped,
        font=font
    )

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2

    draw.multiline_text(
        (tx, ty),
        wrapped,
        fill=(0, 0, 0),
        font=font,
        align="center"
    )

    return cv2.cvtColor(
        np.array(pil),
        cv2.COLOR_RGB2BGR
    )


##############################################################################
# IMAGE PROCESSING
##############################################################################

def process_image(path):
    print("Processing:", path)

    img = cv2.imread(str(path))

    result = ocr.ocr(str(path))

    blocks = []

    for line in result[0]:
        box = line[0]
        text = line[1][0]

        rect = polygon_to_rect(box)

        blocks.append(
            {
                "text": text,
                "rect": rect,
                "area": area(rect)
            }
        )

    blocks = merge_horizontal(blocks)
    blocks = merge_vertical(blocks)

    context_blocks = [
        x["text"]
        for x in blocks
        if x["area"] > MIN_CONTEXT_AREA
    ]

    context = "\n".join(
        context_blocks
    )

    label_blocks = [
        x
        for x in blocks
        if x["area"] < MAX_LABEL_AREA
    ]

    labels = [
        x["text"]
        for x in label_blocks
    ]

    translations = translate_labels(
        context,
        labels
    )

    for block, tr in zip(
        label_blocks,
        translations
    ):
        img = draw_translation(
            img,
            block["rect"],
            tr
        )

    rel = path.relative_to(INPUT_DIR)

    out_path = (
        Path(OUTPUT_DIR)
        / rel
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(out_path),
        img
    )


##############################################################################
# MAIN
##############################################################################

def main():
    files = []

    for ext in (
        "*.jpg",
        "*.jpeg",
        "*.JPG",
        "*.JPEG"
    ):
        files.extend(
            Path(INPUT_DIR).rglob(ext)
        )

    print(f"Found {len(files)} images")

    for f in files:
        try:
            process_image(f)
        except Exception as e:
            print(f"ERROR: {f}")
            print(e)


if __name__ == "__main__":
    main()
