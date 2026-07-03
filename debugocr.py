"""
Quick diagnostic: run PaddleOCR on an image and dump all detected blocks
with their bounding boxes, sorted top-to-bottom.
Usage: python debug_ocr.py path/to/image.jpg
"""
import sys, json, inspect
import cv2, numpy as np
from paddleocr import PaddleOCR

def make_ocr():
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

def polygon_to_rect(box):
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

def parse(result, img_w, img_h):
    blocks = []
    if not result: return blocks
    first = result[0]

    if hasattr(first, "get"):
        def as_seq(v):
            if v is None: return []
            if isinstance(v, np.ndarray): return v.tolist()
            return v
        polys  = as_seq(first.get("rec_polys", []))
        boxes  = as_seq(first.get("rec_boxes", []))
        texts  = as_seq(first.get("rec_texts", []))
        scores = as_seq(first.get("rec_scores", []))
        for i, text in enumerate(texts):
            if not text: continue
            score = float(scores[i]) if i < len(scores) else 1.0
            clean = (text[0] if isinstance(text,(list,tuple)) else str(text)).strip()
            if not clean: continue
            rect = None
            if i < len(boxes):
                b = boxes[i]
                if len(b) == 4: rect = tuple(int(x) for x in b)
            if rect is None:
                poly = polys[i] if i < len(polys) else None
                if poly is None: continue
                rect = polygon_to_rect(poly)
            x1,y1,x2,y2 = rect
            x1=max(0,min(x1,img_w-1)); x2=max(0,min(x2,img_w-1))
            y1=max(0,min(y1,img_h-1)); y2=max(0,min(y2,img_h-1))
            if x2-x1<4 or y2-y1<4: continue
            blocks.append({"text":clean,"rect":[x1,y1,x2,y2],"score":round(score,3)})
        return blocks

    for line in first:
        if not line or len(line)<2: continue
        box, tc = line[0], line[1]
        text = tc[0] if isinstance(tc,(list,tuple)) else str(tc)
        conf = float(tc[1]) if isinstance(tc,(list,tuple)) and len(tc)>1 else 1.0
        if conf<0.20: continue
        text = str(text).strip()
        if not text: continue
        rect = polygon_to_rect(box)
        x1,y1,x2,y2 = rect
        x1=max(0,min(x1,img_w-1)); x2=max(0,min(x2,img_w-1))
        y1=max(0,min(y1,img_h-1)); y2=max(0,min(y2,img_h-1))
        if x2-x1<4 or y2-y1<4: continue
        blocks.append({"text":text,"rect":[x1,y1,x2,y2],"score":round(conf,3)})
    return blocks

def visualize(img_path, blocks, out_path):
    img = cv2.imread(img_path)
    for b in blocks:
        x1,y1,x2,y2 = b["rect"]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
        # tiny label: score
        cv2.putText(img,f"{b['score']:.2f}",(x1,max(0,y1-2)),
                    cv2.FONT_HERSHEY_PLAIN,0.7,(255,0,0),1)
    cv2.imwrite(out_path, img)
    print(f"Visualization saved: {out_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "files/test.jpg"
    img = cv2.imread(path)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")
    ocr = make_ocr()
    if hasattr(ocr,"predict"):
        result = ocr.predict(path)
    else:
        result = ocr.ocr(path, cls=True)
    blocks = parse(result, w, h)
    blocks_sorted = sorted(blocks, key=lambda b: (b["rect"][1], b["rect"][0]))
    print(f"\nDetected {len(blocks)} blocks:\n")
    for i, b in enumerate(blocks_sorted):
        x1,y1,x2,y2 = b["rect"]
        print(f"[{i:03d}] y={y1:4d}-{y2:4d} x={x1:4d}-{x2:4d} w={x2-x1:4d} h={y2-y1:3d} "
              f"score={b['score']:.2f}  '{b['text']}'")
    out = path.rsplit(".",1)[0] + "_ocr_debug.jpg"
    visualize(path, blocks_sorted, out)
    json_out = path.rsplit(".",1)[0] + "_ocr_blocks.json"
    with open(json_out,"w",encoding="utf8") as f:
        json.dump(blocks_sorted, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {json_out}")