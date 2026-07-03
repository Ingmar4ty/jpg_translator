"""main.py — orchestrator and entry point.

process_image() wires the extract -> translate -> render stages together for
a single page; it contains no OCR/merge/translate/render logic of its own.
Run with: python main.py
"""
import traceback

import cv2

from infra import (
    USE_TESSERACT_FALLBACK,
    append_debug,
    find_input_files,
    new_debug,
    rect_area,
    save_outputs,
    write_debug_log,
)
from extract import (
    group_lines_into_paragraphs,
    merge_same_line_fragments,
    parse_ocr_result,
    run_ocr,
    run_tesseract_ocr,
    score_blocks,
)
from translate_render import (
    render_translations,
    should_translate_block,
    translate_paragraphs,
)


def process_image(path):
    print(f"\n{'='*50}\nProcessing: {path}")
    debug = new_debug(path)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read: {path}")
        write_debug_log(debug)
        return

    h, w = img.shape[:2]
    source_img = img.copy()
    print(f"Size: {w}x{h}")
    append_debug(debug, "read.ok", width=w, height=h)

    # --- EXTRACT: OCR ---
    raw_result = run_ocr(str(path))
    raw_blocks = parse_ocr_result(raw_result, w, h)
    print(f"OCR raw blocks: {len(raw_blocks)}")
    append_debug(debug, "ocr.raw", count=len(raw_blocks),
                 sample=[b["text"] for b in raw_blocks[:10]])

    if USE_TESSERACT_FALLBACK and len(raw_blocks) < 8:
        tess = run_tesseract_ocr(img)
        if score_blocks(tess) > score_blocks(raw_blocks) + 10:
            raw_blocks = tess
            print("Switched to Tesseract OCR")

    if not raw_blocks:
        print("No text detected, saving as-is")
        save_outputs(img, path)
        write_debug_log(debug)
        return

    # --- EXTRACT: merge same-line fragments ---
    line_blocks = merge_same_line_fragments(raw_blocks)
    print(f"After line merge: {len(line_blocks)} lines")

    # --- FILTER: skip non-translatable (proper names, page numbers, garbage) ---
    translatable_lines = [b for b in line_blocks if should_translate_block(b)]
    skipped = len(line_blocks) - len(translatable_lines)
    print(f"Translatable lines: {len(translatable_lines)} (skipped {skipped})")
    append_debug(debug, "filter.done",
                 translatable=len(translatable_lines), skipped=skipped,
                 texts=[b["text"] for b in translatable_lines[:20]])

    if not translatable_lines:
        print("Nothing to translate")
        save_outputs(img, path)
        write_debug_log(debug)
        return

    # --- BUILD CONTEXT HINT for LLM (largest text blocks give domain context) ---
    sorted_by_area = sorted(translatable_lines,
                            key=lambda b: rect_area(b["rect"]), reverse=True)
    context_hint = "\n".join(b["text"] for b in sorted_by_area[:6])

    # --- EXTRACT: group lines into paragraphs for better multi-line translation ---
    # Single-line groups → translated individually (fast)
    # Multi-line groups  → sent to LLM as one item with \n between lines,
    #                       result split back per-line for accurate rendering
    paragraphs = group_lines_into_paragraphs(translatable_lines)
    multi_count = sum(1 for p in paragraphs if len(p["lines"]) > 1)
    print(f"Paragraph groups: {len(paragraphs)} total, {multi_count} multi-line")
    append_debug(debug, "paragraphs.done",
                 total=len(paragraphs), multi_line=multi_count,
                 sample=[p["text"][:80] for p in paragraphs[:8]])

    # --- TRANSLATE (paragraph-aware) ---
    render_plan = translate_paragraphs(paragraphs, context_hint=context_hint, debug=debug)

    print(f"Rendering {len(render_plan)} translated lines...")
    append_debug(debug, "render.plan", count=len(render_plan),
                 sample=[{"src": p["src"], "dst": p["dst"]} for p in render_plan[:15]])

    # --- RENDER (erase all originals, then draw all translations) ---
    img = render_translations(img, source_img, render_plan)

    out_path, ready_path = save_outputs(img, path)
    print(f"\u2713 Saved: {out_path}")
    print(f"\u2713 Ready: {ready_path}")
    append_debug(debug, "save.done", output=str(out_path), ready=str(ready_path))
    write_debug_log(debug)


def main():
    files = find_input_files()
    print(f"Found {len(files)} images")
    for f in files:
        try:
            process_image(f)
        except Exception as e:
            print(f"Error {f}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()