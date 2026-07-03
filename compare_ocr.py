"""
Compare PaddleOCR vs Tesseract bounding boxes on the same image.
Shows which engine gives cleaner, wider per-line rects (better for rendering).
Usage: python compare_ocr.py path/to/image.jpg
"""
import sys, inspect
import cv2, numpy as np
from paddleocr import PaddleOCR

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False

def make_ocr():
    sig = inspect.signature(PaddleOCR.__init__)
    p = sig.parameters
    kw = {"lang": "en"}
    if "text_det_limit_side_len" in p: kw["text_det_limit_side_len"] = 4000
    elif "det_limit_side_len" in p: kw["det_limit_side_len"] = 4000
    if "device" in p: kw["device"] = "gpu:0"
    elif "use_gpu" in p: kw["use_gpu"] = True
    return PaddleOCR(**kw)

def polygon_to_rect(box):
    xs=[p[0] for p in box]; ys=[p[1] for p in box]
    return (int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys)))

def parse_paddle(result, img_w, img_h):
    blocks = []
    if not result: return blocks
    first = result[0]
    if hasattr(first, "get"):
        def s(v): return v.tolist() if isinstance(v,np.ndarray) else (v or [])
        polys=s(first.get("rec_polys",[])); boxes=s(first.get("rec_boxes",[]))
        texts=s(first.get("rec_texts",[])); scores=s(first.get("rec_scores",[]))
        for i,text in enumerate(texts):
            if not text: continue
            score = float(scores[i]) if i<len(scores) else 1.0
            if score < 0.2: continue
            clean = (text[0] if isinstance(text,(list,tuple)) else str(text)).strip()
            if not clean: continue
            rect = None
            if i<len(boxes):
                b=boxes[i]
                if len(b)==4: rect=(b[0],b[1],b[2],b[3])
            if rect is None:
                poly=polys[i] if i<len(polys) else None
                if poly is None: continue
                rect=polygon_to_rect(poly)
            x1,y1,x2,y2=[int(round(x)) for x in rect]
            x1=max(0,min(x1,img_w-1)); x2=max(0,min(x2,img_w-1))
            y1=max(0,min(y1,img_h-1)); y2=max(0,min(y2,img_h-1))
            if x2-x1<4 or y2-y1<4: continue
            blocks.append({"text":clean,"rect":(x1,y1,x2,y2),"score":round(score,3)})
    else:
        for line in first:
            if not line or len(line)<2: continue
            box,tc=line[0],line[1]
            text=tc[0] if isinstance(tc,(list,tuple)) else str(tc)
            conf=float(tc[1]) if isinstance(tc,(list,tuple)) and len(tc)>1 else 1.0
            if conf<0.2: continue
            text=str(text).strip()
            if not text: continue
            r=polygon_to_rect(box)
            x1,y1,x2,y2=r
            x1=max(0,min(x1,img_w-1)); x2=max(0,min(x2,img_w-1))
            y1=max(0,min(y1,img_h-1)); y2=max(0,min(y2,img_h-1))
            if x2-x1<4 or y2-y1<4: continue
            blocks.append({"text":text,"rect":(x1,y1,x2,y2),"score":round(conf,3)})
    return sorted(blocks, key=lambda b:(b["rect"][1],b["rect"][0]))

def parse_tesseract(img):
    if not HAS_TESS: return []
    data = pytesseract.image_to_data(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6 -l eng"
    )
    h,w=img.shape[:2]; blocks=[]
    n=len(data.get("text",[]))
    for i in range(n):
        txt=(data["text"][i] or "").strip()
        if not txt: continue
        try: conf=float(data["conf"][i])
        except: conf=-1
        if conf<45: continue
        x,y,bw,bh=int(data["left"][i]),int(data["top"][i]),int(data["width"][i]),int(data["height"][i])
        rect=(max(0,x),max(0,y),min(w,x+bw),min(h,y+bh))
        if rect[2]-rect[0]<4 or rect[3]-rect[1]<4: continue
        blocks.append({"text":txt,"rect":rect,"score":round(conf/100,3)})
    return sorted(blocks, key=lambda b:(b["rect"][1],b["rect"][0]))

def visualize(img, blocks, color, label, out_path):
    vis = img.copy()
    for b in blocks:
        x1,y1,x2,y2=b["rect"]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,1)
    cv2.imwrite(out_path, vis)
    print(f"  → {out_path}")

path = sys.argv[1] if len(sys.argv)>1 else "files/test.jpg"
img = cv2.imread(path)
h,w=img.shape[:2]
print(f"\nImage: {w}x{h}\n")

ocr=make_ocr()
result = ocr.predict(path) if hasattr(ocr,"predict") else ocr.ocr(path,cls=True)
paddle_blocks = parse_paddle(result, w, h)

print(f"=== PaddleOCR: {len(paddle_blocks)} blocks ===")
for b in paddle_blocks:
    x1,y1,x2,y2=b["rect"]
    print(f"  y={y1:4d}-{y2:4d} x={x1:4d}-{x2:4d} w={x2-x1:4d} h={y2-y1:3d} '{b['text']}'")

stem = path.rsplit(".",1)[0]
visualize(img, paddle_blocks, (0,0,255), "paddle", f"{stem}_paddle.jpg")

if HAS_TESS:
    tess_blocks = parse_tesseract(img)
    print(f"\n=== Tesseract: {len(tess_blocks)} blocks ===")
    for b in tess_blocks:
        x1,y1,x2,y2=b["rect"]
        print(f"  y={y1:4d}-{y2:4d} x={x1:4d}-{x2:4d} w={x2-x1:4d} h={y2-y1:3d} '{b['text']}'")
    visualize(img, tess_blocks, (0,200,0), "tess", f"{stem}_tess.jpg")
else:
    print("\nTesseract not available")

print("\nDone.")
