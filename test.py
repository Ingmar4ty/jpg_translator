import inspect
from paddleocr import PaddleOCR

def make_ocr():
    sig = inspect.signature(PaddleOCR.__init__)
    params = sig.parameters
    kwargs = {"lang": "en"}
    if "use_angle_cls" in params:
        kwargs["use_angle_cls"] = True
    elif "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = True

    if "device" in params:
        try:
            return PaddleOCR(**kwargs, device="gpu:0")
        except Exception:
            return PaddleOCR(**kwargs, device="cpu")
    return PaddleOCR(**kwargs, use_gpu=False)

ocr = make_ocr()
res = ocr.predict("files/645.jpg") if hasattr(ocr, "predict") else ocr.ocr("files/645.jpg")

first = res[0] if res else {}
if hasattr(first, "get"):
    texts = first.get("rec_texts", [])
    scores = first.get("rec_scores", [])
    print(f"Detected blocks: {len(texts)}")
    for i, txt in enumerate(texts[:20]):
        score = float(scores[i]) if i < len(scores) else 1.0
        print(f"{i+1:02d}. [{score:.2f}] {txt}")
else:
    print(f"Detected blocks: {len(first)}")
    for i, line in enumerate(first[:20]):
        txt = line[1][0]
        score = float(line[1][1]) if len(line[1]) > 1 else 1.0
        print(f"{i+1:02d}. [{score:.2f}] {txt}")
