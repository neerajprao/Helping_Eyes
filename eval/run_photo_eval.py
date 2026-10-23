"""
Photo evaluation of the pharmacy photos in eval/data/private/images/ (OCR engines, image checks drawn as boxes, and
every question answered by the local Qwen model through the app's own code).

    .venv/bin/python eval/run_photo_eval.py ocr cv qa      run the three stages in order (each one resumes: a photo
                                                           already done is skipped; --redo does it again)
    .venv/bin/python eval/run_photo_eval.py score report   after markdown references exist (see below)
    .venv/bin/python eval/run_photo_eval.py ocr --only ms01,mb02

Stages:
  ocr     Every photo is decoded and downscaled exactly as /api/capture does (server._decode, 1600 px). Then:
            apple_vision   recognize_text(): the app's reader (Apple Vision, accurate, language correction on)
            app_capture    read_text_enhanced(): what the app really keeps after a capture (best of original /
                           flattened / tone / binarised), the text every question is answered from
            rapidocr       RapidOCR() defaults (the app's reader before commit 1759476)
            tesseract      Tesseract 5, --oem 1 --psm 3 -l eng
          Text, confidence, line boxes and time of each are saved.
  cv      The app's image checks, measured with its own functions and thresholds, and drawn on the photo as boxes of
          different colours (see LEGEND): OCR lines, text region, package outline, glare blobs, blurry and dark
          patches, slant. Also a side-by-side picture of the three OCR engines' boxes.
  qa      QUESTIONS asked about every photo, each one through commands.classify() and a fresh DocAssistant on the
          local Qwen 3B model (provider "qwen3b", as the page's Model menu), exactly as /api/ask does. "Read
          everything" is answered by the app without a model.
  score   Compares each engine and each answer with Claude's reading of the photo (eval/data/private/claude_reference/
          <id>.json, written by looking only at the photo, before any OCR output was read).
  report  markdown/eval_photo_report.md (every photo: pictures with boxes, OCR, checks, all answers; git-ignored)
          and markdown/eval_photo_summary.md (numbers only).

Everything with photo text stays in the git-ignored eval/data/private/photo_eval/.
"""

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "helping_eyes")
sys.path.insert(0, APP)

EVAL = os.path.join(ROOT, "eval")
PRIVATE = os.path.join(EVAL, "data", "private")
IMAGES = os.path.join(PRIVATE, "images")
OUT = os.path.join(PRIVATE, "photo_eval")
REFS = os.path.join(PRIVATE, "claude_reference")
MANIFEST = os.path.join(EVAL, "data", "manifest.csv")
INTAKE = os.path.join(PRIVATE, "intake.json")

QUESTIONS = [
    ("read_all", "Read everything"),
    ("name", "What is the name of this product?"),
    ("use", "What is it used for?"),
    ("ingredients", "What are the ingredients or active ingredients?"),
    ("strength", "What is the strength or dose of the medicine?"),
    ("how_to_use", "How should I take or use it?"),
    ("quantity", "How much is in the pack?"),
    ("expiry", "When does it expire?"),
    ("expired", "Has it expired?"),
    ("mfg", "When was it manufactured?"),
    ("batch", "What is the batch number?"),
    ("price", "What is the price?"),
    ("maker", "Who makes it?"),
    ("storage", "How should I store it?"),
    ("warnings", "Are there any warnings or precautions?"),
    ("children", "Is it safe for children?"),
    ("pregnancy", "Can I use it during pregnancy?"),
    ("prescription", "Does it need a prescription?"),
    ("external", "Is it for external use only?"),
    ("side_effects", "What are the side effects?"),
    ("alcohol", "Can I drink alcohol while using it?"),
]

# BGR colours of the boxes, and what each one means
LEGEND = [
    ("ocr", (60, 200, 60), "Apple Vision text line"),
    ("region", (230, 200, 0), "Text region (all text in view)"),
    ("outline", (0, 220, 255), "Item outline (grown from its text)"),
    ("glare", (255, 0, 255), "Glare blob (saturated light)"),
    ("blur", (0, 0, 255), "Blurry text patch"),
    ("dark", (200, 80, 0), "Too dark patch"),
    ("skew", (0, 140, 255), "Text slant (deskew angle)"),
]
COLOR = {k: c for k, c, _ in LEGEND}
ENGINE_COLOR = {"apple_vision": (60, 200, 60), "rapidocr": (0, 140, 255), "tesseract": (200, 60, 160)}
ENGINE_NAME = {"apple_vision": "Apple Vision", "app_capture": "App capture (Apple Vision + fixes)",
               "rapidocr": "RapidOCR", "tesseract": "Tesseract"}

TILE = 40            # px at the 640 px analysis width: blur and dark patches are measured per tile
TILE_CONTENT = 15    # a tile with less grey-level spread than this is blank (no print to be sharp or blurry)


def ids(only=None):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    order = {"ms": 0, "mb": 1, "pc": 2}
    rows.sort(key=lambda r: (order.get(r["id"][:2], 9), r["id"]))
    return [r["id"] for r in rows if not only or r["id"] in only]


def manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def intake():
    with open(INTAKE, encoding="utf-8") as f:
        return json.load(f)


def load(pid):
    """The photo as the app sees it after a capture: server._decode() (downscaled to 1600 px)."""
    import server
    with open(os.path.join(IMAGES, pid + ".jpg"), "rb") as f:
        return server._decode(f.read())


def path(stage, pid, ext="json"):
    folder = os.path.join(OUT, stage)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{pid}.{ext}")


def save(stage, pid, data):
    with open(path(stage, pid), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def read(stage, pid):
    p = path(stage, pid)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return {"commit": commit, "dirty": dirty}


# ------------------------------------------------------------------ OCR
_rapid = None


def run_rapidocr(frame):
    global _rapid
    if _rapid is None:
        from rapidocr import RapidOCR
        _rapid = RapidOCR()
    r = _rapid(frame)
    lines = []
    if r.boxes is not None:
        for quad, text, score in zip(r.boxes, r.txts, r.scores):
            q = [[float(x), float(y)] for x, y in quad]
            xs, ys = [p[0] for p in q], [p[1] for p in q]
            lines.append({"text": text, "confidence": float(score), "quad": q,
                          "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
    return lines


def run_tesseract(frame):
    with tempfile.TemporaryDirectory() as tmp:
        img = os.path.join(tmp, "in.png")
        cv2.imwrite(img, frame)
        out = subprocess.run(["tesseract", img, "stdout", "--oem", "1", "--psm", "3", "-l", "eng", "tsv"],
                             capture_output=True, text=True).stdout
    groups = {}
    for row in out.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        key = (int(cols[2]), int(cols[3]), int(cols[4]))           # block, paragraph, line
        x, y, w, h, conf = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]), float(cols[10])
        groups.setdefault(key, []).append((x, y, w, h, conf, cols[11]))
    lines = []
    for words in groups.values():
        x1 = min(w[0] for w in words); y1 = min(w[1] for w in words)
        x2 = max(w[0] + w[2] for w in words); y2 = max(w[1] + w[3] for w in words)
        lines.append({"text": " ".join(w[5] for w in words), "confidence": round(sum(w[4] for w in words) / len(words) / 100, 3),
                      "box": [x1, y1, x2, y2], "quad": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
    lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
    return lines


def stage_ocr(pid, redo):
    if read("ocr", pid) and not redo:
        return
    import vision
    frame = load(pid)
    result = {"id": pid, "size": [frame.shape[1], frame.shape[0]], "engines": {}}

    t = time.perf_counter()
    lines = vision.recognize_text(frame)
    result["engines"]["apple_vision"] = {
        "seconds": round(time.perf_counter() - t, 3),
        "lines": [{"text": l.text, "confidence": round(l.confidence, 3), "box": list(l.box),
                   "quad": [[round(x, 1), round(y, 1)] for x, y in (l.quad or vision.box_quad(l.box))]} for l in lines]}

    t = time.perf_counter()
    text, version = vision.read_text_enhanced(frame)
    result["engines"]["app_capture"] = {"seconds": round(time.perf_counter() - t, 3), "version": version,
                                        "lines": [{"text": s} for s in text.split("\n") if s.strip()]}

    t = time.perf_counter()
    lines = run_rapidocr(frame)
    result["engines"]["rapidocr"] = {"seconds": round(time.perf_counter() - t, 3), "lines": lines}

    t = time.perf_counter()
    lines = run_tesseract(frame)
    result["engines"]["tesseract"] = {"seconds": round(time.perf_counter() - t, 3), "lines": lines}
    save("ocr", pid, result)


def text_of(ocr, engine):
    return "\n".join(l["text"] for l in ocr["engines"][engine]["lines"])


# ------------------------------------------------------------------ CV checks, drawn as boxes
def tiles(gray):
    h, w = gray.shape
    for y in range(0, h - TILE // 2, TILE):
        for x in range(0, w - TILE // 2, TILE):
            yield x, y, gray[y:y + TILE, x:x + TILE]


def measure(frame, ocr):
    """Every image check of the app on one photo, with where on the photo each problem is."""
    import vision
    h, w = frame.shape[:2]
    lines = ocr["engines"]["apple_vision"]["lines"]
    region = None
    if sum(len(re.sub(r"\s", "", l["text"])) for l in lines) >= 8:
        region = [min(l["box"][0] for l in lines), min(l["box"][1] for l in lines),
                  max(l["box"][2] for l in lines), max(l["box"][3] for l in lines)]
    q_full = vision.assess_quality(frame)                                   # what the capture path uses
    q_text = vision.assess_quality(frame, tuple(region) if region else None)   # what live guidance uses

    gray = vision._analysis_gray(frame)
    scale = w / gray.shape[1]

    # glare: the same saturated blobs assess_quality counts
    saturated = (gray >= 250).astype(np.uint8)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(saturated)
    glare = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area >= 30:
            glare.append({"box": [int(x * scale), int(y * scale), int((x + bw) * scale), int((y + bh) * scale)],
                          "fraction": round(area / saturated.size, 4)})

    # blurry patches: tiles on a recognised text line whose variance of the Laplacian (the app's sharpness measure)
    # is below BLUR_MIN; dark patches: tiles anywhere darker than the app's DARK_MEAN
    blur, dark, text_tiles = [], [], 0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    line_boxes = [[v / scale for v in l["box"]] for l in lines]
    for x, y, t in tiles(gray):
        box = [int(x * scale), int(y * scale), int(min(gray.shape[1], x + TILE) * scale), int(min(gray.shape[0], y + TILE) * scale)]
        if t.mean() < vision.DARK_MEAN:
            dark.append({"box": box, "mean": round(float(t.mean()), 1)})
        on_text = any(x + TILE > b[0] and x < b[2] and y + TILE > b[1] and y < b[3] for b in line_boxes)
        if on_text and t.std() >= TILE_CONTENT:
            text_tiles += 1
            sharp = float(lap[y:y + TILE, x:x + TILE].var())
            if sharp < vision.BLUR_MIN:
                blur.append({"box": box, "sharpness": round(sharp, 1)})

    boxes = [tuple(l["box"]) for l in lines]
    quad = vision.find_page_quad(frame, text_boxes=boxes)          # the item that carries the text
    skew = vision.estimate_skew(frame)
    return {
        "quality_full": {"sharpness": round(q_full.sharpness, 1), "brightness": round(q_full.brightness, 1),
                         "contrast": round(q_full.contrast, 1), "glare_percent": round(q_full.glare_fraction * 100, 2),
                         "problems": q_full.problems, "hint": q_full.hint()},
        "quality_text": {"sharpness": round(q_text.sharpness, 1), "brightness": round(q_text.brightness, 1),
                         "contrast": round(q_text.contrast, 1), "glare_percent": round(q_text.glare_fraction * 100, 2),
                         "glare_at": [round(v, 2) for v in q_text.glare_at] if q_text.glare_at else None,
                         "problems": q_text.problems, "hint": q_text.hint()},
        "text_region": region,
        "glare_blobs": glare,
        "blurry_patches": blur,
        "text_tiles": text_tiles,
        "dark_patches": dark,
        "page_quad": [[round(float(x)), round(float(y))] for x, y in quad] if quad is not None else None,
        "would_flatten": bool(quad is not None and vision._needs_warp(quad, frame.shape)),
        "skew_degrees": round(skew, 1),
        "page_fill": round(vision.page_fill(frame, boxes), 2),
        "capture_version": ocr["engines"]["app_capture"]["version"],
    }


def _font(size):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial Unicode.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw_cv(frame, ocr, m):
    """The photo with every check as a coloured box, and a legend panel on the right with the measured values."""
    from PIL import Image, ImageDraw
    img = frame.copy()
    h, w = img.shape[:2]
    th = max(2, round(max(h, w) / 500))

    def rect(box, color, width=th):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, width)

    for d in m["dark_patches"]:
        rect(d["box"], COLOR["dark"], max(1, th - 1))
    for b in m["blurry_patches"]:
        rect(b["box"], COLOR["blur"], th)
    for l in ocr["engines"]["apple_vision"]["lines"]:
        cv2.polylines(img, [np.array(l["quad"], np.int32)], True, COLOR["ocr"], th)
    if m["text_region"]:
        rect(m["text_region"], COLOR["region"], th + 1)
    if m["page_quad"]:
        cv2.polylines(img, [np.array(m["page_quad"], np.int32)], True, COLOR["outline"], th + 2)
    for g in m["glare_blobs"]:
        rect(g["box"], COLOR["glare"], th + 1)
    if m["skew_degrees"]:
        cx, cy, L = w // 2, h // 2, int(min(h, w) * 0.35)
        a = np.radians(-m["skew_degrees"])     # the text runs at -angle (OpenCV turns by +angle to level it)
        p1 = (int(cx - L * np.cos(a)), int(cy - L * np.sin(a)))
        p2 = (int(cx + L * np.cos(a)), int(cy + L * np.sin(a)))
        cv2.arrowedLine(img, p1, p2, COLOR["skew"], th + 2, tipLength=0.05)

    fs = max(16, max(h, w) // 62)
    panel_w = int(fs * 22)
    canvas = Image.new("RGB", (w + panel_w, max(h, int(fs * 34))), "white")
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas)
    f, fb = _font(fs), _font(int(fs * 1.15))
    x, y = w + 16, 14
    qt, qf = m["quality_text"], m["quality_full"]
    counts = {"ocr": len(ocr["engines"]["apple_vision"]["lines"]), "region": 1 if m["text_region"] else 0,
              "outline": 1 if m["page_quad"] else 0, "glare": len(m["glare_blobs"]),
              "blur": len(m["blurry_patches"]), "dark": len(m["dark_patches"]), "skew": 1 if m["skew_degrees"] else 0}
    d.text((x, y), "Image checks (boxes)", fill="black", font=fb); y += int(fs * 1.8)
    for key, color, label in LEGEND:
        rgb = tuple(reversed(color))
        d.rectangle([x, y + 2, x + fs, y + fs + 2], outline=rgb, width=4)
        d.text((x + fs + 10, y), f"{label}: {counts[key]}", fill="black", font=f)
        y += int(fs * 1.5)
    y += fs // 2
    d.text((x, y), "Measured (app's own thresholds)", fill="black", font=fb); y += int(fs * 1.8)
    rows = [
        f"Sharpness of the text {qt['sharpness']:.0f} (blurry < {int(_blur_min())})",
        f"Brightness {qf['brightness']:.0f}  (dark < 50)",
        f"Contrast {qf['contrast']:.0f}  (flat < 25)",
        f"Glare {qf['glare_percent']:.1f}% of picture  (glare >= 1%)",
        f"Blurry text patches {len(m['blurry_patches'])} of {m['text_tiles']}",
        f"Slant {m['skew_degrees']:.1f} deg" + ("" if m["skew_degrees"] else "  (level)"),
        f"Package fills {m['page_fill'] * 100:.0f}% of picture",
        "Outline: " + ("found, app would flatten it" if m["would_flatten"] else "found, already flat" if m["page_quad"] else "not found"),
        f"Problems: {', '.join(qt['problems']) or 'none'}",
        f"App would say: {qt['hint'] or '(nothing, ready)'}",
        f"Capture kept: {m['capture_version']} version",
    ]
    for line in rows:
        for part in _wrap(line, d, f, panel_w - 30):
            d.text((x, y), part, fill="black", font=f); y += int(fs * 1.35)
    return canvas


def _blur_min():
    import vision
    return vision.BLUR_MIN


def _wrap(text, d, font, width):
    words, out, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if d.textlength(trial, font=font) <= width:
            cur = trial
        else:
            out.append(cur); cur = wd
    return out + [cur] if cur else out


def draw_engines(frame, ocr):
    """The three OCR engines side by side, each with its own line boxes."""
    from PIL import Image, ImageDraw
    panels = []
    for engine in ("apple_vision", "rapidocr", "tesseract"):
        img = frame.copy()
        th = max(2, round(max(img.shape[:2]) / 450))
        for l in ocr["engines"][engine]["lines"]:
            cv2.polylines(img, [np.array(l["quad"], np.int32)], True, ENGINE_COLOR[engine], th)
        s = 700 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        e = ocr["engines"][engine]
        title = f"{ENGINE_NAME[engine]}: {len(e['lines'])} lines, {e['seconds']:.2f} s"
        top = Image.new("RGB", (pil.width, 40), "white")
        ImageDraw.Draw(top).text((8, 8), title, fill=tuple(reversed(ENGINE_COLOR[engine])), font=_font(22))
        panel = Image.new("RGB", (pil.width, pil.height + 40), "white")
        panel.paste(top, (0, 0)); panel.paste(pil, (0, 40))
        panels.append(panel)
    W = sum(p.width for p in panels) + 20
    out = Image.new("RGB", (W, max(p.height for p in panels)), "white")
    x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 10
    return out


def stage_cv(pid, redo):
    if read("cv", pid) and not redo:
        return
    ocr = read("ocr", pid)
    frame = load(pid)
    m = measure(frame, ocr)
    draw_cv(frame, ocr, m).save(path("annotated", pid + "_checks", "jpg"), quality=82)
    draw_engines(frame, ocr).save(path("annotated", pid + "_ocr", "jpg"), quality=82)
    save("cv", pid, m)


# ------------------------------------------------------------------ questions (local Qwen, the app's own path)
def stage_qa(pid, redo):
    if read("qa", pid) and not redo:
        return
    import assistant
    from commands import classify
    ocr = read("ocr", pid)
    document = text_of(ocr, "app_capture")
    answers = []
    for key, question in QUESTIONS:
        cmd = classify(question)
        t = time.perf_counter()
        if not document.strip():
            answers.append({"key": key, "question": question, "route": cmd.name, "seconds": 0.0, "not_in_text": None,
                            "answer": "I haven't captured any text yet. Hold something up to the camera."})
            continue
        if cmd.name == "read_all":
            answers.append({"key": key, "question": question, "route": "read_all (no model)", "seconds": 0.0,
                            "not_in_text": False, "answer": document})
            continue
        doc = assistant.DocAssistant()
        doc.set_provider("qwen3b")
        doc.set_document(document)
        first, parts = None, []
        for s in doc.ask(cmd.query):
            if first is None:
                first = time.perf_counter() - t
            if s == assistant.READ_ALL:
                parts = [document]
                break
            parts.append(s)
        answers.append({"key": key, "question": question, "route": cmd.name, "answer": " ".join(parts),
                        "not_in_text": doc.last_not_in_text, "first_sentence_s": round(first or 0, 2),
                        "seconds": round(time.perf_counter() - t, 2)})
    save("qa", pid, {"id": pid, "model": assistant.QWEN_MODELS["qwen3b"], "date": datetime.date.today().isoformat(),
                     "document_source": "app_capture", "answers": answers, **git_state()})


STAGES = {"ocr": stage_ocr, "cv": stage_cv, "qa": stage_qa}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=list(STAGES) + ["score", "report"])
    parser.add_argument("--only", default="")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    only = set(filter(None, args.only.split(",")))
    for stage in args.stages:
        if stage in ("score", "report"):
            import photo_report
            getattr(photo_report, stage)(ids(only))
            continue
        todo = ids(only)
        started = time.time()
        for i, pid in enumerate(todo, 1):
            try:
                STAGES[stage](pid, args.redo)
            except Exception as e:                       # one bad photo must not stop the run
                print(f"{stage} {pid}: FAILED {e!r}", flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"{stage}: {i}/{len(todo)} done, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
