"""translate_render.py — turn extracted text lines into Russian text
drawn back onto the page image.

Three tiers are tried for each piece of text, cheapest first:
  1. Dictionary lookup (TECH_DICT / PHRASE_DICT)
  2. Translation cache (infra.TRANSLATION_CACHE)
  3. Local LLM (Qwen via llama-cpp-python)

translate_paragraphs() is the entry point used by the pipeline: it keeps
multi-line paragraphs together for LLM context, then splits the result
back to one translation per original OCR line (via
extract.align_translation_to_lines) so render_translations() can draw
each line in its own accurate bounding box.

render_translations() does a strict two-pass render: ALL erasing happens
before ANY drawing, so a translation never gets wiped out by a
neighboring block's erase pass after it's already been drawn.
"""

import json
import re

import cv2
import numpy as np
from llama_cpp import Llama
from PIL import Image, ImageDraw, ImageFont

from infra import (
    FONT_PATH,
    FORCE_RETRANSLATE_BAD_CACHE,
    MODEL_PATH,
    PHRASE_DICT,
    TECH_DICT,
    TRANSLATION_CACHE,
    has_cyrillic,
    is_bad_cached_translation,
    is_proper_name_text,
    is_suspicious_text,
    sanitize_rect,
    save_cache,
)
from extract import align_translation_to_lines



# ----------------------------------------------------------------------------
# llm_engine.py
# ----------------------------------------------------------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=8192,
    n_batch=1024,
    chat_format="qwen",
    verbose=False,
)



# ----------------------------------------------------------------------------
# translate.py
# ----------------------------------------------------------------------------
##############################################################################
# CANDIDATE FILTERING
##############################################################################
def should_translate(text):
    t = (text or "").strip()
    if not t or len(t) < 2:
        return False
    if is_proper_name_text(t):
        return False
    alpha = sum(1 for c in t if c.isalpha())
    if alpha == 0:
        return False
    if is_suspicious_text(t):
        return False
    # NG / OK — keep as-is in flowcharts
    if t.upper() in {"OK", "NG"}:
        return False
    # Pure digit strings — these are list numbers, page refs, part numbers
    if re.fullmatch(r"\d+", t):
        return False
    return True


def should_translate_block(block):
    """Extended filter that uses both text AND geometry."""
    text = block["text"].strip()
    if not should_translate(text):
        return False
    x1, y1, x2, y2 = block["rect"]
    bw = x2 - x1
    bh = y2 - y1
    # Reject blocks taller than wide with short text — these are vertical
    # digit columns OCR'd as one block (e.g. "2345" from list numbering)
    if bh > bw * 2.5 and len(text) <= 6:
        return False
    # Reject suspiciously narrow blocks (single char width, multi-char text)
    if bw < 15 and len(text) > 2:
        return False
    return True


##############################################################################
# LLM OUTPUT PARSING
##############################################################################
def clean_llm_output(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.S)
    text = re.sub(r"```json|```", "", text)
    return text.strip()


def parse_llm_json(raw, expected):
    cleaned = clean_llm_output(raw)
    for candidate in [cleaned, re.search(r"\[[\s\S]*\]", cleaned)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group(0)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                items = []
                for item in parsed:
                    if isinstance(item, dict):
                        v = item.get("translation") or item.get("ru") or item.get("text")
                    elif isinstance(item, str):
                        v = item
                    else:
                        v = None
                    if v:
                        # Strip N||| and "N. " numbering artifacts the LLM echoes back
                        v = re.sub(r"^\d+\|\|\|\s*", "", str(v))
                        v = re.sub(r"^\d+\.\s+", "", v).strip()
                    items.append(v if v else "")
                if len(items) < expected:
                    items += [""] * (expected - len(items))
                return items[:expected]
        except Exception:
            continue
    return None


def parse_llm_lines(raw, expected):
    cleaned = clean_llm_output(raw)
    out = [""] * expected
    matched = 0
    for line in cleaned.splitlines():
        m = re.match(r"^(\d+)\s*\|\|\|\s*(.+)$", line.strip())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < expected:
                out[idx] = m.group(2).strip()
                matched += 1
    return out if matched > 0 else None


##############################################################################
# HELPERS
##############################################################################
def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def dict_lookup(text):
    """Try phrase dict then tech dict for a text string."""
    t = text.strip()
    tl = t.lower()
    if tl in PHRASE_DICT:
        return PHRASE_DICT[tl]
    # Dotted-leader phrases: "Description ......... Action"
    m = re.match(r"^(.+?)\s*\.{3,}\s*(.+)$", t)
    if m:
        full_key = t.lower()
        if full_key in PHRASE_DICT:
            return PHRASE_DICT[full_key]
    first_word = tl.split()[0] if tl.split() else ""
    if first_word in TECH_DICT and len(tl.split()) == 1:
        return TECH_DICT[first_word]
    return None


def fallback_dict_translate(text):
    """Word-by-word fallback using TECH_DICT, used when the LLM call fails."""
    parts = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    result, changed = [], False
    for p in parts:
        key = p.lower()
        if key in TECH_DICT:
            result.append(TECH_DICT[key])
            changed = True
        else:
            result.append(p)
    return "".join(result) if changed else text


def cleanup_translation(src, tr):
    if not (tr or "").strip():
        return ""
    # Strip "N|||" and "N. " numbering artifacts that the LLM echoes back
    tr = re.sub(r"^\d+\|\|\|\s*", "", tr)
    tr = re.sub(r"^\d+\.\s+", "", tr)
    if not tr.strip():
        return ""
    words = tr.split()
    if len(words) >= 2 and words[0].lower() == words[1].lower():
        tr = " ".join(words[1:])
    if not has_cyrillic(tr) and normalize(tr) == normalize(src):
        return ""
    return tr.strip()


##############################################################################
# SINGLE-LINE BATCH TRANSLATION
##############################################################################
def translate_texts(texts, context_hint="", debug=None):
    """
    Translate a list of independent English texts to Russian in one LLM call.
    Returns list of same length with Russian translations. Texts that don't
    need translation are returned unchanged.
    """
    results = [""] * len(texts)
    pending_idx = []
    pending_texts = []

    for i, text in enumerate(texts):
        if not should_translate(text):
            results[i] = text
            continue
        d = dict_lookup(text)
        if d:
            results[i] = d
            continue
        key = normalize(text)
        if key in TRANSLATION_CACHE:
            cached = TRANSLATION_CACHE[key]
            if not (FORCE_RETRANSLATE_BAD_CACHE and is_bad_cached_translation(text, cached)):
                results[i] = cached
                continue
            else:
                TRANSLATION_CACHE.pop(key, None)
        pending_idx.append(i)
        pending_texts.append(text)

    if not pending_texts:
        print(f"  All {len(texts)} items from cache/dict")
        return results

    print(f"  Sending {len(pending_texts)} items to LLM...")

    items_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(pending_texts))
    context_section = f"\nPage context:\n{context_hint}\n" if context_hint else ""

    prompt = (
        "You are a professional technical translator (English\u2192Russian).\n"
        "Translate ALL items below from English to Russian.\n"
        "These are labels and text from a Honda VT250F motorcycle repair manual.\n"
        "Rules:\n"
        "- Translate EVERY item including short words: Clean\u2192\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c, "
        "Replace\u2192\u0417\u0430\u043c\u0435\u043d\u0438\u0442\u044c, "
        "Adjust\u2192\u041e\u0442\u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u0442\u044c, "
        "Correct\u2192\u0418\u0441\u043f\u0440\u0430\u0432\u0438\u0442\u044c, "
        "Change\u2192\u0417\u0430\u043c\u0435\u043d\u0438\u0442\u044c, "
        "Inspect\u2192\u041f\u0440\u043e\u0432\u0435\u0440\u0438\u0442\u044c\n"
        "- For 'X ......... Y' format (dotted leader): translate both parts, e.g. "
        "'Clogged slow circuit ........ Clean' \u2192 '\u0417\u0430\u0441\u043e\u0440\u0451\u043d\u043d\u0430\u044f "
        "\u043c\u0435\u0434\u043b\u0435\u043d\u043d\u0430\u044f \u0446\u0435\u043f\u044c ........ \u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c'\n"
        "- Keep OK, NG, brand names (Honda, VT250F) unchanged\n"
        "- Output ONLY a valid JSON array of strings, same count as input, same order\n"
        "- NO explanations, NO extra text, ONLY the JSON array\n"
        f"{context_section}\n"
        f"Items:\n{items_str}\n\n"
        "JSON array:"
    )

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content":
                "Output ONLY a valid JSON array of Russian translation strings. "
                "Never skip items. Never output anything except the JSON array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        max_tokens=2048,
        repeat_penalty=1.1,
    )
    raw = resp["choices"][0]["message"]["content"]
    translated = parse_llm_json(raw, len(pending_texts))

    if translated is None:
        retry_prompt = (
            "Translate each item from English to Russian.\n"
            "Return EXACTLY one line per item: N|||translation\n"
            "Translate ALL items including short words.\n\n"
            f"{items_str}"
        )
        resp2 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Return only lines: N|||translation"},
                {"role": "user", "content": retry_prompt}
            ],
            temperature=0.1, top_p=0.9, max_tokens=2048,
        )
        translated = parse_llm_lines(resp2["choices"][0]["message"]["content"], len(pending_texts))

    if translated is None:
        print("  LLM parse failed, using dict fallback")
        translated = [fallback_dict_translate(t) for t in pending_texts]

    for orig_idx, src, tr in zip(pending_idx, pending_texts, translated):
        cleaned = cleanup_translation(src, tr)
        if not cleaned:
            cleaned = fallback_dict_translate(src)
        results[orig_idx] = cleaned
        key = normalize(src)
        if key and has_cyrillic(cleaned) and normalize(src) != normalize(cleaned):
            TRANSLATION_CACHE[key] = cleaned

    save_cache()
    return results


##############################################################################
# PARAGRAPH-AWARE TRANSLATION (entry point used by the pipeline)
##############################################################################
def translate_paragraphs(paragraphs, context_hint="", debug=None):
    """
    Translate a list of paragraph groups (see merge.group_lines_into_paragraphs).

    Each paragraph is a dict with:
      "lines": list of line blocks (each has "text", "rect", "sub_rects")
      "text":  "\n"-joined text of all lines  (for LLM)

    Returns a list of render items:
      {"src": str, "dst": str, "rect": tuple, "sub_rects": list}
    One item per LINE (not per paragraph), so rects stay accurate.

    Multi-line paragraphs are sent to the LLM as one item with \n separating
    lines, with an explicit instruction to preserve line count. The result is
    split back per-line via align_translation_to_lines(). Single-line
    paragraphs go through the normal translate_texts() batch.
    """
    if not paragraphs:
        return []

    single_idx, single_texts = [], []
    multi_idx = []

    for i, para in enumerate(paragraphs):
        if len(para["lines"]) == 1:
            single_idx.append(i)
            single_texts.append(para["lines"][0]["text"])
        else:
            multi_idx.append(i)

    single_translations = translate_texts(single_texts, context_hint=context_hint, debug=debug)
    single_map = {idx: tr for idx, tr in zip(single_idx, single_translations)}

    multi_paragraphs = [paragraphs[i] for i in multi_idx]
    multi_translations = {}

    if multi_paragraphs:
        pending_para_idx = []
        pending_texts = []

        for i, para in zip(multi_idx, multi_paragraphs):
            key = normalize(para["text"])
            if key in TRANSLATION_CACHE and not (
                FORCE_RETRANSLATE_BAD_CACHE and is_bad_cached_translation(para["text"], TRANSLATION_CACHE[key])
            ):
                cached = TRANSLATION_CACHE[key]
                lines = align_translation_to_lines([l["text"] for l in para["lines"]], cached)
                multi_translations[i] = lines
            else:
                pending_para_idx.append(i)
                pending_texts.append(para["text"])

        if pending_texts:
            print(f"  Translating {len(pending_texts)} multi-line paragraphs individually...")
            for orig_idx, text in zip(pending_para_idx, pending_texts):
                para = paragraphs[orig_idx]
                src_lines = [l["text"] for l in para["lines"]]

                # Flatten multi-line text to a single string for the LLM batch call.
                # This avoids LLM confusion from numbered multi-paragraph batches
                # where item N's translation can bleed into item M's slot.
                flat_text = " ".join(line.strip() for line in text.split("\n"))
                flat_key = normalize(flat_text)

                # Check cache on the flattened key
                if flat_key in TRANSLATION_CACHE and not (
                    FORCE_RETRANSLATE_BAD_CACHE
                    and is_bad_cached_translation(flat_text, TRANSLATION_CACHE[flat_key])
                ):
                    tr_flat = TRANSLATION_CACHE[flat_key]
                else:
                    results = translate_texts(
                        [flat_text], context_hint=context_hint, debug=debug
                    )
                    tr_flat = results[0] if results else ""
                    if flat_key and tr_flat and has_cyrillic(tr_flat):
                        TRANSLATION_CACHE[flat_key] = tr_flat

                line_translations = align_translation_to_lines(src_lines, tr_flat)
                multi_translations[orig_idx] = line_translations

            save_cache()

    # Build render items — one per LINE across all paragraphs
    render_items = []
    for i, para in enumerate(paragraphs):
        if i in single_map:
            line = para["lines"][0]
            tr = single_map[i]
            if tr and normalize(tr) != normalize(line["text"]):
                render_items.append({
                    "src": line["text"],
                    "dst": tr,
                    "rect": line["rect"],
                    "sub_rects": line.get("sub_rects", [line["rect"]]),
                })
        elif i in multi_translations:
            for line, tr in zip(para["lines"], multi_translations[i]):
                if tr and normalize(tr) != normalize(line["text"]):
                    render_items.append({
                        "src": line["text"],
                        "dst": tr,
                        "rect": line["rect"],
                        "sub_rects": line.get("sub_rects", [line["rect"]]),
                    })

    return render_items



# ----------------------------------------------------------------------------
# render.py
# ----------------------------------------------------------------------------
def estimate_bg(source_img, rect):
    """Sample the page background color around a rect (from the pristine,
    pre-edit source image) so erased/redrawn regions blend in.

    Uses a wider border strip (20 px) and the 75th percentile of brightness
    rather than the median: background pixels are the LIGHTEST ones in the
    sample, while text pixels are dark — the 75th percentile skews toward
    the background and away from ink, giving a cleaner fill.
    """
    h, w = source_img.shape[:2]
    x1, y1, x2, y2 = rect
    margin = 20
    samples = []
    for ry1, ry2, rx1, rx2 in [
        (max(0, y1 - margin), y1, x1, x2),
        (y2, min(h, y2 + margin), x1, x2),
        (y1, y2, max(0, x1 - margin), x1),
        (y1, y2, x2, min(w, x2 + margin)),
    ]:
        if ry2 > ry1 and rx2 > rx1:
            reg = source_img[ry1:ry2, rx1:rx2]
            if reg.size:
                samples.append(reg.reshape(-1, 3))
    if not samples:
        inner = source_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if inner.size == 0:
            return (245, 245, 245)
        pixels = inner.reshape(-1, 3)
    else:
        pixels = np.concatenate(samples, axis=0)
    return tuple(int(v) for v in np.percentile(pixels, 75, axis=0))


def erase_rect(img, rect, source_img, expand=4):
    h, w = img.shape[:2]
    s = sanitize_rect(rect, w, h)
    if s is None:
        return img
    x1, y1, x2, y2 = s
    bg = estimate_bg(source_img, s)
    ex1, ey1 = max(0, x1 - expand), max(0, y1 - expand)
    ex2, ey2 = min(w - 1, x2 + expand), min(h - 1, y2 + expand)
    cv2.rectangle(img, (ex1, ey1), (ex2, ey2), bg, -1)
    return img


def fit_text(draw, text, rect, min_size=7):
    """Find the largest font size (and word-wrap) that fits text inside rect."""
    x1, y1, x2, y2 = rect
    rw, rh = x2 - x1, y2 - y1

    def wrap(font, max_w):
        words = text.split()
        if not words:
            return [text]
        lines, cur = [], ""
        for word in words:
            cand = (cur + " " + word).strip()
            bb = draw.textbbox((0, 0), cand, font=font)
            if bb[2] - bb[0] <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines or [text]

    max_size = max(min_size, min(int(rh * 0.90), 32))
    for font_size in range(max_size, min_size - 1, -1):
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except Exception:
            font = ImageFont.load_default()
        lines = wrap(font, max(1, rw - 4))
        bb = draw.textbbox((0, 0), "Ag", font=font)
        lh = bb[3] - bb[1]
        sp = max(1, lh // 6)
        th = lh * len(lines) + sp * max(0, len(lines) - 1)
        mw = max((draw.textbbox((0, 0), l, font=font)[2] for l in lines), default=0)
        if mw <= rw - 2 and th <= rh - 2:
            return font, "\n".join(lines), sp
    try:
        font = ImageFont.truetype(FONT_PATH, min_size)
    except Exception:
        font = ImageFont.load_default()
    return font, text, 1


def draw_text_in_rect(img, rect, text, source_img):
    h, w = img.shape[:2]
    s = sanitize_rect(rect, w, h)
    if s is None:
        return img
    x1, y1, x2, y2 = s
    bg = estimate_bg(source_img, s)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font, wrapped, spacing = fit_text(draw, text, s)

    lines = wrapped.split("\n")
    bbs = [draw.textbbox((0, 0), l, font=font) for l in lines]
    lh = max((bb[3] - bb[1] for bb in bbs), default=10)
    th = lh * len(lines) + spacing * max(0, len(lines) - 1)
    mw = max((bb[2] - bb[0] for bb in bbs), default=10)

    tx = x1 + ((x2 - x1) - mw) // 2
    ty = y1 + ((y2 - y1) - th) // 2

    brightness = sum(bg) / 3.0
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    draw.multiline_text((tx, ty), wrapped, fill=text_color, font=font,
                        align="center", spacing=spacing)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _expand_rect_for_translation(rect, src_text, dst_text, img_w, img_h):
    """Widen the draw rect when the translation is longer than the source.

    Russian text is typically 30–50 % wider than equivalent English at the
    same font size. Without expansion `fit_text` must shrink the font,
    making the translation visually smaller than the surrounding original.
    We expand the rect proportionally to the character-length ratio, keeping
    the left edge fixed (natural reading anchor) and capping at 2× original
    width so we don't clobber diagram elements far to the right.
    """
    x1, y1, x2, y2 = rect
    rw = x2 - x1
    # Strip spaces for a fairer character count
    src_len = max(len((src_text or "").replace(" ", "")), 1)
    dst_len = max(len((dst_text or "").replace(" ", "")), 1)
    ratio = dst_len / src_len
    if ratio <= 1.15:
        return rect  # translation fits with ≤15 % overhead — no expansion
    new_rw = min(int(rw * ratio), rw * 2)  # cap at 2× original width
    new_x2 = min(img_w - 1, x1 + new_rw)
    result = sanitize_rect((x1, y1, new_x2, y2), img_w, img_h)
    return result if result else rect


def render_translations(img, source_img, render_plan):
    """
    Apply a list of {"rect", "sub_rects", "dst"} render items to img:
    pass 1 erases every original-text region (plus any expanded draw area),
    pass 2 draws every translation in the (possibly expanded) rect.
    Doing all erasing before any drawing prevents a later erase from wiping
    out an earlier translation.
    """
    h, w = img.shape[:2]

    # Pre-compute draw rects — expanded where translation is longer than source
    draw_rects = [
        _expand_rect_for_translation(p["rect"], p.get("src", ""), p["dst"], w, h)
        for p in render_plan
    ]

    # Pass 1: erase all original rects + expanded draw rects
    for p, dr in zip(render_plan, draw_rects):
        all_erase = list(p.get("sub_rects", []))
        if p["rect"] not in all_erase:
            all_erase.append(p["rect"])
        if dr not in all_erase:
            all_erase.append(dr)
        for r in all_erase:
            img = erase_rect(img, r, source_img, expand=6)

    # Pass 2: draw all translations in (possibly expanded) rects
    for p, dr in zip(render_plan, draw_rects):
        img = draw_text_in_rect(img, dr, p["dst"], source_img)

    return img