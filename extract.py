"""extract.py — turn a page image into a list of translatable text lines.

Pipeline within this module:
  run_ocr/parse_ocr_result   -> raw OCR fragments with bounding boxes
  merge_same_line_fragments  -> fuse fragments on the same physical line
                                 (including dotted-leader pairs like
                                 "Description ......... Action")
  group_lines_into_paragraphs-> group lines for LLM context only; each
                                 line keeps its own rect for rendering
align_translation_to_lines lives here too since it is the inverse of the
paragraph grouping above.
"""

import inspect
import re

import cv2
import numpy as np
from paddleocr import PaddleOCR

try:
    import pytesseract
except Exception:
    pytesseract = None

from infra import is_suspicious_text, polygon_to_rect, sanitize_rect



# ----------------------------------------------------------------------------
# ocr_engine.py
# ----------------------------------------------------------------------------
def make_ocr():
    """Build a PaddleOCR instance, preferring GPU, falling back to CPU."""
    try:
        sig = inspect.signature(PaddleOCR.__init__)
        params = sig.parameters
        kwargs = {"lang": "en"}
        if "text_det_limit_side_len" in params:
            kwargs["text_det_limit_side_len"] = 4000
        elif "det_limit_side_len" in params:
            kwargs["det_limit_side_len"] = 4000
        if "use_textline_orientation" in params:
            kwargs["use_textline_orientation"] = True
        elif "use_angle_cls" in params:
            kwargs["use_angle_cls"] = True
        if "device" in params:
            kwargs["device"] = "gpu:0"
        elif "use_gpu" in params:
            kwargs["use_gpu"] = True
        return PaddleOCR(**kwargs)
    except Exception as e:
        print(f"OCR GPU init failed: {e}, trying CPU")
        kwargs = {"lang": "en"}
        sig = inspect.signature(PaddleOCR.__init__)
        params = sig.parameters
        if "device" in params:
            kwargs["device"] = "cpu"
        elif "use_gpu" in params:
            kwargs["use_gpu"] = False
        return PaddleOCR(**kwargs)


# Module-level singleton — PaddleOCR model load is expensive, do it once.
ocr = make_ocr()


def run_ocr(image_or_path):
    if hasattr(ocr, "predict"):
        try:
            return ocr.predict(image_or_path)
        except Exception:
            pass
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(image_or_path, cls=True)
        except Exception:
            pass
    return []


def parse_ocr_result(result, img_w, img_h):
    """Normalize PaddleOCR 3.x or 2.x output into a flat list of
    {"text", "rect", "score"} dicts."""
    blocks = []
    if not result:
        return blocks
    first = result[0]

    if hasattr(first, "get"):
        # PaddleOCR 3.x: dict-like OCRResult
        def as_seq(v):
            if v is None:
                return []
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        polys = as_seq(first.get("rec_polys", []))
        boxes = as_seq(first.get("rec_boxes", []))
        texts = as_seq(first.get("rec_texts", []))
        scores = as_seq(first.get("rec_scores", []))
        for i, text in enumerate(texts):
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            if score < 0.20:
                continue
            clean = (text[0] if isinstance(text, (list, tuple)) else str(text)).strip()
            if not clean:
                continue
            rect = None
            if i < len(boxes):
                b = boxes[i]
                if len(b) == 4:
                    rect = (b[0], b[1], b[2], b[3])
            if rect is None:
                poly = polys[i] if i < len(polys) else None
                if poly is None:
                    continue
                rect = polygon_to_rect(poly)
            rect = sanitize_rect(rect, img_w, img_h)
            if rect is None:
                continue
            blocks.append({"text": clean, "rect": rect, "score": score})
        return blocks

    # PaddleOCR 2.x: [[box, (text, conf)], ...]
    for line in first:
        if not line or len(line) < 2:
            continue
        box, text_conf = line[0], line[1]
        text = text_conf[0] if isinstance(text_conf, (list, tuple)) else str(text_conf)
        conf = float(text_conf[1]) if isinstance(text_conf, (list, tuple)) and len(text_conf) > 1 else 1.0
        if conf < 0.20:
            continue
        text = str(text).strip()
        if not text:
            continue
        rect = sanitize_rect(polygon_to_rect(box), img_w, img_h)
        if rect is None:
            continue
        blocks.append({"text": text, "rect": rect, "score": conf})
    return blocks


def score_blocks(blocks):
    """Heuristic quality score used to pick between OCR variants/fallbacks."""
    if not blocks:
        return -9999.0
    suspicious = sum(1 for b in blocks if is_suspicious_text(b["text"]))
    good = len(blocks) - suspicious
    avg_score = sum(float(b.get("score", 0.5)) for b in blocks) / len(blocks)
    return good * 2 + avg_score * 2 - suspicious * 1.5


def run_tesseract_ocr(img):
    """Optional fallback OCR, used only when PaddleOCR output is weak."""
    if pytesseract is None:
        return []
    try:
        data = pytesseract.image_to_data(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )
    except Exception as e:
        print(f"Tesseract unavailable: {e}")
        return []

    h, w = img.shape[:2]
    blocks = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt or is_suspicious_text(txt):
            continue
        try:
            conf = float(data.get("conf", ["-1"] * n)[i])
        except Exception:
            conf = -1
        if conf < 45:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        bw, bh = int(data["width"][i]), int(data["height"][i])
        rect = sanitize_rect((x, y, x + bw, y + bh), w, h)
        if rect is None:
            continue
        blocks.append({"text": txt, "rect": rect, "score": conf / 100.0})
    return blocks



# ----------------------------------------------------------------------------
# merge.py
# ----------------------------------------------------------------------------
def merge_same_line_fragments(blocks):
    """
    Merge OCR fragments that are on the exact same text line (close vertically,
    small horizontal gap, or a dotted-leader pair). Returns merged blocks where
    each block has:
      - "text": combined text
      - "rect": tight bounding box of the merged group
      - "sub_rects": list of original individual rects (for accurate erasing)
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b["rect"][1], b["rect"][0]))
    result = []
    used = set()

    for i, base in enumerate(blocks):
        if i in used:
            continue
        used.add(i)
        group = [base]
        group_rects = [base["rect"]]

        changed = True
        while changed:
            changed = False
            bx1, by1, bx2, by2 = (
                min(r[0] for r in group_rects),
                min(r[1] for r in group_rects),
                max(r[2] for r in group_rects),
                max(r[3] for r in group_rects),
            )
            bh = by2 - by1

            for j, cand in enumerate(blocks):
                if j in used:
                    continue
                cx1, cy1, cx2, cy2 = cand["rect"]
                ch = cy2 - cy1
                avg_h = (bh + ch) / 2

                # Must be on same line
                center_y_diff = abs((by1 + by2) / 2 - (cy1 + cy2) / 2)
                if center_y_diff > avg_h * 0.45:
                    continue

                # Skip vertical-strip blocks (digit columns like "2345" from
                # list numbering OCR'd as one tall narrow block)
                if (cy2 - cy1) > (cx2 - cx1) * 2.5 and len(cand["text"].strip()) <= 6:
                    continue

                # Merge if gap is small, OR if this looks like a dotted-leader
                # pair: "Description ............. Action" — big gap, same line
                gap = cx1 - bx2
                is_dotted_leader_pair = (
                    gap < (bx2 - bx1) * 3.0  # gap < 3x current group width
                    and len(cand["text"].strip().split()) <= 3  # right side is short
                    and not re.search(r"[.]{3,}", cand["text"])  # no dots itself
                )
                if -8 <= gap < avg_h * 1.5 or is_dotted_leader_pair:
                    group.append(cand)
                    group_rects.append(cand["rect"])
                    used.add(j)
                    changed = True

        group = sorted(group, key=lambda b: b["rect"][0])
        merged_text = " ".join(b["text"] for b in group)
        all_rects = [b["rect"] for b in group]
        merged_rect = (
            min(r[0] for r in all_rects),
            min(r[1] for r in all_rects),
            max(r[2] for r in all_rects),
            max(r[3] for r in all_rects),
        )
        result.append({
            "text": merged_text,
            "rect": merged_rect,
            "sub_rects": all_rects,
            "score": max(b.get("score", 1.0) for b in group),
        })

    return result


def _is_component_label(text):
    """True for '(N) LABEL' style diagram annotations."""
    return bool(re.match(r"^\(\d+\)\s*\S", (text or "").strip()))


def group_lines_into_paragraphs(blocks):
    """
    Group consecutive lines that belong to the same flowchart box / paragraph,
    for LLM-context purposes only. Returns groups where each group has:
      - "lines": list of line blocks (each with text, rect, sub_rects)
      - "text": "\n"-joined text of all lines, for sending to the LLM
      - "rect": overall bounding rect of the group (informational only —
                rendering always uses each line's own rect, never this one)

    Lines are grouped only when they are:
      1. Vertically adjacent (gap < 1.8x line height)
      2. Horizontally overlapping substantially (>=40% of the narrower line)
      3. Similar width (same box/column, not a stray line from elsewhere)
      4. No other block occupies the space between them (no line-skipping)
      5. Same content type: component labels (N) LABEL don't mix with prose
    """
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b["rect"][1], b["rect"][0]))
    groups = []
    used = set()

    for i, base in enumerate(blocks):
        if i in used:
            continue
        used.add(i)
        group_lines = [base]

        searching = True
        while searching:
            searching = False
            last = group_lines[-1]
            lx1, ly1, lx2, ly2 = last["rect"]
            lh = ly2 - ly1

            best_next = None
            best_gap = float("inf")

            for j, cand in enumerate(blocks):
                if j in used:
                    continue
                cx1, cy1, cx2, cy2 = cand["rect"]

                vertical_gap = cy1 - ly2
                if vertical_gap < -4 or vertical_gap > lh * 1.8:
                    continue

                overlap = min(lx2, cx2) - max(lx1, cx1)
                min_w = min(lx2 - lx1, cx2 - cx1)
                if min_w <= 0 or overlap / min_w < 0.40:
                    continue

                lw = lx2 - lx1
                cw = cx2 - cx1
                if max(lw, cw) > 0 and min(lw, cw) / max(lw, cw) < 0.45:
                    continue

                # Don't mix (N) LABEL annotations with prose sentences
                if _is_component_label(last["text"]) != _is_component_label(cand["text"]):
                    continue

                # No-interleaving: reject candidate if any unused block sits
                # between the current group and the candidate in reading order.
                # Use centre-y comparison so that OCR boxes with slightly
                # overlapping y-ranges (tight line spacing) are still detected.
                l_cy = (ly1 + ly2) / 2
                c_cy = (cy1 + cy2) / 2
                interleaved = False
                for k, other in enumerate(blocks):
                    if k in used or k == j:
                        continue
                    ox1, oy1, ox2, oy2 = other["rect"]
                    o_cy = (oy1 + oy2) / 2
                    if l_cy < o_cy < c_cy:
                        if min(lx2, ox2) - max(lx1, ox1) > 0:
                            interleaved = True
                            break
                if interleaved:
                    continue

                if vertical_gap < best_gap:
                    best_gap = vertical_gap
                    best_next = (j, cand)

            if best_next is not None:
                j, cand = best_next
                group_lines.append(cand)
                used.add(j)
                searching = True

        all_rects = [l["rect"] for l in group_lines]
        group_rect = (
            min(r[0] for r in all_rects),
            min(r[1] for r in all_rects),
            max(r[2] for r in all_rects),
            max(r[3] for r in all_rects),
        )
        groups.append({
            "lines": group_lines,
            "text": "\n".join(l["text"] for l in group_lines),
            "rect": group_rect,
        })

    return groups


def align_translation_to_lines(src_lines, translated_paragraph):
    """
    Split a translated paragraph back into per-line translations.
    The LLM may return a different number of lines than the source, so we
    align greedily by character-length proportion when counts don't match.

    src_lines: list of original English line texts
    translated_paragraph: Russian text (may have \n or be one block)

    Returns: list of Russian strings, same length as src_lines.
    """
    if not src_lines:
        return []

    tr = (translated_paragraph or "").strip()
    if not tr:
        return [""] * len(src_lines)

    if len(src_lines) == 1:
        return [tr]

    tr_lines = [l.strip() for l in tr.splitlines() if l.strip()]

    if len(tr_lines) == len(src_lines):
        return tr_lines

    # Mismatch: distribute translated text proportionally by source line length
    src_chars = [max(1, len(l)) for l in src_lines]
    total_src = sum(src_chars)

    all_tr = " ".join(tr_lines)
    total_tr = max(1, len(all_tr))

    result = []
    pos = 0
    for i, sc in enumerate(src_chars):
        if i == len(src_chars) - 1:
            result.append(all_tr[pos:].strip())
        else:
            chars = round(sc / total_src * total_tr)
            end = min(pos + chars, total_tr)
            while end > pos and end < total_tr and all_tr[end] != " ":
                end -= 1
            result.append(all_tr[pos:end].strip())
            pos = end + 1  # skip the space

    return result