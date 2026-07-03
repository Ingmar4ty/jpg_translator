"""infra.py — configuration, caching, logging, and small pure helpers.

Everything in this module is low-level and side-effect-light (aside from
cache.py's on-disk JSON cache and debug_log.py's log files). Higher-level
modules (extract.py, translate_render.py, main.py) build on top of this.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2



# ----------------------------------------------------------------------------
# config.py
# ----------------------------------------------------------------------------
INPUT_DIR = "files"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"

READY_OUTPUT_DIR = os.getenv("READY_OUTPUT_DIR", "/mnt/e/repos/translations/ready")
MODEL_PATH = os.getenv("MODEL_PATH", "models/qwen3-14b-instruct-q4_k_m.gguf")
FONT_PATH = "fonts/DejaVuSans.ttf"
CACHE_FILE = "translation_cache.json"

OCR_PROFILE = os.getenv("OCR_PROFILE", "gpu_locked")
USE_TESSERACT_FALLBACK = os.getenv("USE_TESSERACT_FALLBACK", "0") == "1"
USE_PREPROCESS_VARIANTS = os.getenv("USE_PREPROCESS_VARIANTS", "0") == "1"
FORCE_RETRANSLATE_BAD_CACHE = os.getenv("FORCE_RETRANSLATE_BAD_CACHE", "1") == "1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)



# ----------------------------------------------------------------------------
# geometry.py
# ----------------------------------------------------------------------------
def polygon_to_rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def rect_area(rect):
    x1, y1, x2, y2 = rect
    return max(0, x2 - x1) * max(0, y2 - y1)


def sanitize_rect(rect, img_w, img_h):
    """Clamp a rect to image bounds; return None if it collapses to nothing."""
    x1, y1, x2, y2 = [int(round(x)) for x in rect]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)



# ----------------------------------------------------------------------------
# text_quality.py
# ----------------------------------------------------------------------------
def is_suspicious_text(text):
    """True if text looks like OCR noise rather than real words."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    alnum = sum(1 for c in t if c.isalnum())
    punct_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    if alnum == 0:
        return True
    # Dots are common in repair manuals: "Clean........." — allow a high dot
    # ratio but still cap other punctuation.
    non_dot_punct = sum(1 for c in t if c in punct_chars and c != '.')
    if non_dot_punct / max(1, len(t)) > 0.40:
        return True
    if alnum / len(t) < 0.20:
        return True
    return False


def has_cyrillic(text):
    return bool(re.search(r"[А-Яа-яЁё]", text or ""))


def has_latin(text):
    return bool(re.search(r"[A-Za-z]", text or ""))


def is_bad_cached_translation(src, tr):
    """True if a cached (src -> tr) pair is unreliable and should be redone."""
    norm = lambda x: re.sub(r"\s+", " ", (x or "").strip().lower())
    if not (tr or "").strip():
        return True
    if norm(src) == norm(tr):
        return True
    if has_cyrillic(tr) and has_latin(tr):
        return True
    if has_latin(src) and not has_cyrillic(tr):
        return True
    return False


def is_proper_name_text(text):
    """True for brand names, model codes, and page numbers we must not translate."""
    t = (text or "").strip()
    if not t:
        return False
    known = ["HONDA", "VT250", "VT250F", "VT250-FII", "SUZUKI", "YAMAHA",
             "KAWASAKI", "TROUBLESHOOTING"]
    upper = t.upper()
    if any(x in upper for x in known):
        return True
    if re.fullmatch(r"\d+[-–]\d+", t):  # page numbers like 23-16
        return True
    return False



# ----------------------------------------------------------------------------
# dictionaries.py
# ----------------------------------------------------------------------------
TECH_DICT = {
    "hub": "ступица", "bearing": "подшипник", "seal": "сальник",
    "shaft": "вал", "washer": "шайба", "fork": "вилка",
    "spring": "пружина", "clutch": "сцепление", "bolt": "болт",
    "nut": "гайка", "pin": "штифт", "gear": "шестерня",
    "brake": "тормоз", "chain": "цепь", "gasket": "прокладка",
    "sprocket": "звезда", "piston": "поршень", "cylinder": "цилиндр",
    "lever": "рычаг", "screw": "винт", "cover": "крышка",
    "clean": "очистить", "replace": "заменить", "adjust": "отрегулировать",
    "correct": "исправить", "change": "заменить", "inspect": "проверить",
    "check": "проверить",
    # Motorcycle body / cooling
    "fairing": "обтекатель", "windshield": "ветровое стекло",
    "headlight": "фара", "headlamp": "фара",
    "radiator": "радиатор", "thermostat": "термостат",
    "tank": "бак", "pump": "насос", "fan": "вентилятор",
    "frame": "рама", "pocket": "карман",
    # Common actions
    "tighten": "затянуть", "connect": "подключить",
    "install": "установить", "remove": "снять",
    "mount": "закрепить", "fix": "закрепить",
    "disassemble": "разобрать",
}

PHRASE_DICT = {
    "ok": "OK",
    "ng": "NG",
    "pilot screw": "регулировочный винт",
    "return spring": "возвратная пружина",
    "throttle valve": "дроссельный клапан",
    "throttle shaft": "дроссельный вал",
    "correct, replace": "исправить, заменить",
    "still not normal": "всё ещё не в норме",
    "adjust pilot screw": "отрегулировать регулировочный винт",
    "adjust": "отрегулировать",
    # Cooling system
    "cooling system": "система охлаждения",
    "cooling fan": "вентилятор охлаждения",
    "motor-driven cooling fan": "электроприводной вентилятор охлаждения",
    "cooling water": "охлаждающая жидкость",
    "cooling water flow": "поток охлаждающей жидкости",
    "water pump": "водяной насос",
    "reserve tank": "расширительный бачок",
    "side frame": "боковая рама",
    "technical features": "технические особенности",
    # Fairing / body
    "position lamp": "позиционный фонарь",
    "side pocket": "боковой карман",
    "self-tapping screw": "самонарезающий винт",
    "self-tapping screws": "самонарезающие винты",
    "fairing stay": "кронштейн обтекателя",
    "windshield mounting bolts": "болты крепления ветрового стекла",
}



# ----------------------------------------------------------------------------
# cache.py
# ----------------------------------------------------------------------------
def _load_raw():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf8") as f:
            return json.load(f)
    return {}


def sanitize_cache(cache):
    cleaned, dropped = {}, 0
    for k, v in cache.items():
        if is_suspicious_text(k) or is_suspicious_text(v):
            dropped += 1
            continue
        if FORCE_RETRANSLATE_BAD_CACHE and is_bad_cached_translation(k, v):
            dropped += 1
            continue
        cleaned[k] = v
    if dropped:
        print(f"Removed {dropped} suspicious cache entries")
    return cleaned, dropped


# Module-level singleton cache, loaded once at import time.
TRANSLATION_CACHE, _dropped = sanitize_cache(_load_raw())


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf8") as f:
        json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)


if _dropped:
    save_cache()



# ----------------------------------------------------------------------------
# debug_log.py
# ----------------------------------------------------------------------------
def new_debug(path):
    return {
        "file": str(path),
        "file_name": path.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_debug(debug, stage, **data):
    debug.setdefault("events", []).append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "data": data,
    })


def write_debug_log(debug):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", debug.get("file_name", "unknown"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = Path(LOGS_DIR) / f"{safe_name}+{stamp}.json"
    with open(log_path, "w", encoding="utf8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    return log_path



# ----------------------------------------------------------------------------
# io_utils.py
# ----------------------------------------------------------------------------
def find_input_files():
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
        files.extend(Path(INPUT_DIR).rglob(ext))
    return files


def save_outputs(img, input_path):
    out_path = Path(OUTPUT_DIR) / input_path.relative_to(INPUT_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    ready_dir = Path(READY_OUTPUT_DIR)
    ready_dir.mkdir(parents=True, exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    ready_path = ready_dir / f"{input_path.stem}+{dt}+.jpg"
    cv2.imwrite(str(ready_path), img)
    return out_path, ready_path