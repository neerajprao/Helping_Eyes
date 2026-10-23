"""
One folder per image check, holding only the photos where that check found something, with only that check's
boxes drawn (nothing else on the picture), and a short caption strip under the photo.

    .venv/bin/python eval/make_check_folders.py        (after eval/run_photo_eval.py ocr cv)

Writes eval/data/private/cv_checks/ (git-ignored, the photos are private):
    blur_photos/      red boxes: patches on text lines whose sharpness (variance of the Laplacian) is below 60
    glare_photos/     magenta boxes: saturated glare blobs (grey 250+), the ones assess_quality() counts
    dark_photos/      blue boxes: patches darker than the app's "too dark" level (mean below 50)
    slant_photos/     orange box: the block of text, drawn at the slant estimate_skew() measured
    outline_photos/   yellow outline: the item outline find_page_quad() found (grown from the item's text)
Neighbouring patches are joined into one box, so each box marks one area. Every folder has an index.csv.
"""

import csv
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

import run_photo_eval as R

DEST = os.path.join(R.PRIVATE, "cv_checks")
CHECKS = {
    "blur_photos": ((0, 0, 255), "Blur"),
    "glare_photos": ((255, 0, 255), "Glare"),
    "dark_photos": ((200, 80, 0), "Too dark"),
    "slant_photos": ((0, 140, 255), "Slant"),
    "outline_photos": ((0, 220, 255), "Item outline"),
}


def merge(boxes, gap=4):
    """Join boxes that touch or overlap (within `gap` px) into one box each."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] + gap and o[0] <= b[2] + gap and b[1] <= o[3] + gap and o[1] <= b[3] + gap:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def grow(box, w, h, least=28, pad=6):
    """A tiny box made big enough to see, without leaving the picture."""
    x1, y1, x2, y2 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    if x2 - x1 < least:
        c = (x1 + x2) / 2; x1, x2 = c - least / 2, c + least / 2
    if y2 - y1 < least:
        c = (y1 + y2) / 2; y1, y2 = c - least / 2, c + least / 2
    return [int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2))]


def slant_box(ocr, angle, shape):
    """The four corners of the text block, as a rectangle turned to the measured slant."""
    pts = np.array([p for l in ocr["engines"]["apple_vision"]["lines"] for p in l["quad"]], np.float32)
    if not len(pts):
        return None
    h, w = shape[:2]
    level = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)           # turns the text level
    flat = cv2.transform(pts.reshape(-1, 1, 2), level).reshape(-1, 2)
    x1, y1 = flat.min(0); x2, y2 = flat.max(0)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    back = cv2.invertAffineTransform(level)
    return cv2.transform(corners.reshape(-1, 1, 2), back).reshape(-1, 2).astype(np.int32)


def caption(img, text):
    """The photo with a white strip under it holding one line of text (the photo itself is not covered)."""
    h, w = img.shape[:2]
    fs = max(18, w // 55)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    font = R._font(fs)
    d0 = ImageDraw.Draw(pil)
    lines = R._wrap(text, d0, font, w - 24)
    strip = int(fs * 1.4) * len(lines) + 16
    out = Image.new("RGB", (w, h + strip), "white")
    out.paste(pil, (0, 0))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((12, h + 8 + i * int(fs * 1.4)), line, fill="black", font=font)
    return out


def main():
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)                         # rebuilt from scratch every time, so no stale photo stays
    man, intake = R.manifest(), R.intake()
    index = {k: [] for k in CHECKS}
    for pid in R.ids():
        cvm, ocr = R.read("cv", pid), R.read("ocr", pid)
        if not (cvm and ocr):
            continue
        frame = R.load(pid)
        h, w = frame.shape[:2]
        th = max(3, round(max(h, w) / 300))
        name = intake[pid]["original_name"]
        found = {}
        if cvm["blurry_patches"]:
            boxes = merge([b["box"] for b in cvm["blurry_patches"]])
            low = min(b["sharpness"] for b in cvm["blurry_patches"])
            found["blur_photos"] = (boxes, f"{len(cvm['blurry_patches'])} text patch(es) below sharpness 60 "
                                           f"(lowest {low:.0f}); whole text area {cvm['quality_text']['sharpness']:.0f}", low)
        if cvm["glare_blobs"]:
            boxes = merge([grow(g["box"], w, h) for g in cvm["glare_blobs"]])
            total = sum(g["fraction"] for g in cvm["glare_blobs"]) * 100
            found["glare_photos"] = (boxes, f"{len(cvm['glare_blobs'])} saturated blob(s), {total:.2f} % of the picture "
                                            f"(the app calls glare at 1 %)", round(total, 3))
        if cvm["dark_patches"]:
            boxes = merge([d["box"] for d in cvm["dark_patches"]])
            darkest = min(d["mean"] for d in cvm["dark_patches"])
            found["dark_photos"] = (boxes, f"{len(cvm['dark_patches'])} patch(es) darker than 50 (darkest {darkest:.0f}); "
                                           f"whole picture {cvm['quality_full']['brightness']:.0f}", darkest)
        if cvm["skew_degrees"]:
            quad = slant_box(ocr, cvm["skew_degrees"], frame.shape)
            if quad is not None:
                found["slant_photos"] = ([quad], f"text slanted {abs(cvm['skew_degrees']):.1f}° "
                                                 f"({'clockwise' if cvm['skew_degrees'] > 0 else 'anticlockwise'})",
                                         cvm["skew_degrees"])
        if cvm["page_quad"]:
            quad = np.array(cvm["page_quad"], np.int32)
            found["outline_photos"] = ([quad], f"outline fills {cvm['page_fill'] * 100:.0f} % of the picture; "
                                               f"{'the app would flatten it' if cvm['would_flatten'] else 'not flattened'}",
                                       cvm["page_fill"])
        for check, (boxes, text, value) in found.items():
            color, label = CHECKS[check]
            img = frame.copy()
            for b in boxes:
                if isinstance(b, np.ndarray):
                    cv2.polylines(img, [b], True, color, th + 1)
                else:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, th)
            folder = os.path.join(DEST, check)
            os.makedirs(folder, exist_ok=True)
            caption(img, f"{pid} ({name}) · {label}: {text}").save(os.path.join(folder, pid + ".jpg"), quality=85)
            index[check].append({"id": pid, "original_name": name, "category": man[pid]["category"],
                                 "lighting (Claude)": man[pid]["lighting"], "angle (Claude)": man[pid]["angle"],
                                 "boxes": len(boxes), "value": value, "what": text})
    for check, rows in index.items():
        folder = os.path.join(DEST, check)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "index.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["id", "original_name", "category", "lighting (Claude)", "angle (Claude)",
                                               "boxes", "value", "what"])
            wr.writeheader()
            wr.writerows(rows)
        print(f"{check}: {len(rows)} photos")


if __name__ == "__main__":
    main()
