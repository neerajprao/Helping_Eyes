"""
Checks the collected photos (Checkpoint 1) and prints only problems and counts, never any recognised text, so no
engine output can reach the person who types the ground truth.

    .venv/bin/python eval/tools/check_photos.py

  * manifest: allowed values, ids that match their category, every row has a photo and every photo a row
  * coverage: counts per category, lighting, angle and device; the "difficult" subset (dim, glare or curved)
  * duplicates: identical files (hash) and near-duplicates (64-bit difference hash, Hamming distance <= 6)
  * near-empty: Apple Vision finds fewer than 15 characters, or the picture is almost uniform
  * possible personal details: a face (OpenCV Haar cascade), or text shaped like a phone number, an e-mail address,
    a PIN code or a "Name / Patient / Dr. / Address" field. Only the kind of match is printed. These are hints; the
    photos must still be looked at.
"""

import csv
import hashlib
import os
import re
import sys
from collections import Counter
from itertools import combinations

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "helping_eyes"))
from collect import ANGLE, CATEGORIES, DEVICE, IMAGES, LIGHTING, MANIFEST  # noqa: E402

PERSONAL = {
    "phone number": re.compile(r"(?<!\d)(?:\+?\d[\d\s\-]{8,13}\d)(?!\d)"),
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PIN / postcode": re.compile(r"\b(?:pin(?:\s*code)?|postcode|zip)\b\s*[:\-]?\s*\d{3}\s?\d{3}\b", re.I),
    "name / patient / doctor / address field": re.compile(r"\b(?:name|patient|dr\.|doctor|address|addr\.|mr\.|mrs\.|ms\.)\s*[:\-]", re.I),
}


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> int:
    import vision
    if not os.path.exists(MANIFEST):
        print("No manifest yet: eval/data/manifest.csv")
        return 1
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    problems = []
    ids = [r["id"] for r in rows]
    for pid, n in Counter(ids).items():
        if n > 1:
            problems.append(f"{pid}: appears {n} times in the manifest")
    for r in rows:
        missing = [k for k in ("category", "lighting", "angle", "device") if not r[k]]
        if missing:
            problems.append(f"{r['id']}: details missing ({', '.join(missing)}); set them on the intake page")
        elif r["category"] not in CATEGORIES or r["lighting"] not in LIGHTING or r["angle"] not in ANGLE or r["device"] not in DEVICE:
            problems.append(f"{r['id']}: a value is not allowed ({r['category']}, {r['lighting']}, {r['angle']}, {r['device']})")
        elif not r["id"].startswith(CATEGORIES[r["category"]]):
            problems.append(f"{r['id']}: id prefix does not match category {r['category']}")
        if not os.path.exists(os.path.join(IMAGES, r["id"] + ".jpg")):
            problems.append(f"{r['id']}: photo file missing")
    files = {f[:-4] for f in os.listdir(IMAGES) if f.endswith(".jpg")} if os.path.isdir(IMAGES) else set()
    for pid in sorted(files - set(ids)):
        problems.append(f"{pid}.jpg: photo with no manifest row")

    faces = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    hashes, digests, flags = {}, {}, []
    for r in rows:
        path = os.path.join(IMAGES, r["id"] + ".jpg")
        if not os.path.exists(path):
            continue
        digests.setdefault(hashlib.sha256(open(path, "rb").read()).hexdigest(), []).append(r["id"])
        image = cv2.imread(path)
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hashes[r["id"]] = dhash(gray)
        scale = 1600 / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
        text = "\n".join(l.text for l in vision.recognize_text(small))
        chars = len(re.sub(r"\s", "", text))
        if chars < 15 or gray.std() < 8:
            flags.append(f"{r['id']}: near-empty ({chars} characters found, contrast {gray.std():.0f})")
        if min(h, w) < 600:
            flags.append(f"{r['id']}: small photo ({w}x{h})")
        g = cv2.resize(gray, None, fx=min(1, 800 / max(h, w)), fy=min(1, 800 / max(h, w)))
        if len(faces.detectMultiScale(g, 1.1, 6, minSize=(40, 40))):
            flags.append(f"{r['id']}: POSSIBLE FACE")
        for kind, pattern in PERSONAL.items():
            if pattern.search(text):
                flags.append(f"{r['id']}: possible personal detail ({kind})")

    for same in digests.values():
        if len(same) > 1:
            problems.append("identical files: " + ", ".join(same))
    for a, b in combinations(hashes, 2):
        d = bin(hashes[a] ^ hashes[b]).count("1")
        if d <= 6:
            flags.append(f"{a} and {b}: near-duplicates (hash distance {d})")

    print(f"{len(rows)} photos")
    for col in ("category", "lighting", "angle", "device"):
        print(f"  {col}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r[col] for r in rows).items())))
    hard = [r["id"] for r in rows if r["lighting"] in ("dim", "glare") or r["angle"] == "curved"]
    print(f"  difficult subset (dim, glare or curved): {len(hard)}")
    print(f"  tilted: {sum(r['angle'] == 'tilted' for r in rows)}")
    print("\nProblems (must fix):" if problems else "\nProblems: none")
    for p in problems:
        print("  " + p)
    print("\nTo look at:" if flags else "\nNothing flagged")
    for p in flags:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
