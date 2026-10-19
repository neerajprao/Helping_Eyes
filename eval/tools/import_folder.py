"""
Import a whole folder of photos into the evaluation set (eval/data/private/images/ + eval/data/manifest.csv).

    .venv/bin/python eval/tools/import_folder.py                      imports eval/data/private/inbox/
    .venv/bin/python eval/tools/import_folder.py some/folder --dry-run    only shows what would happen

  * Every JPEG, PNG or HEIC in the folder and its subfolders is imported, sorted by file name.
  * A subfolder named after a category (medicine_strip, medicine_box, food_label, book_page, document,
    sign_or_screen) sets that category; otherwise it is left empty and the photo gets an "xx" id until it is chosen.
  * The device is guessed from the photo's own camera details (read before they are removed): a phone maker means
    "phone". Lighting and angle cannot be guessed: set them, and anything left empty, on the intake page
    (eval/tools/collect.py), where photos missing details are listed first.
  * The stored copy has its EXIF rotation applied and every bit of metadata (GPS location included) removed, as with
    the intake page. The original files are not changed or deleted. Photos already imported (same file) are skipped,
    so the command can be run again after adding more photos.

No OCR is run, so no recognised text is shown before the ground truth is typed.
"""

import argparse
import io
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import CATEGORIES, IMAGES, PRIVATE, _is_heic, save_photo  # noqa: E402

EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
PHONE_MAKERS = ("apple", "samsung", "google", "oneplus", "xiaomi", "redmi", "poco", "oppo", "vivo", "realme",
                "motorola", "huawei", "honor", "nothing", "sony", "nokia", "asus", "lg")


def guess_device(data: bytes, name: str) -> str:
    """'phone' when the camera details name a phone maker (an iPad is not counted), otherwise ''."""
    if _is_heic(data, name):
        return "phone"                     # HEIC comes from iPhones (and some Android phones)
    try:
        exif = Image.open(io.BytesIO(data)).getexif()
    except Exception:
        return ""
    make, model = str(exif.get(0x010F, "")).strip().lower(), str(exif.get(0x0110, "")).strip().lower()
    if "ipad" in model:
        return ""
    return "phone" if "iphone" in model or any(make.startswith(m) for m in PHONE_MAKERS) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=os.path.join(PRIVATE, "inbox"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not os.path.isdir(args.folder):
        print(f"No folder {args.folder}. Put the photos there (subfolders named after a category are optional).")
        return 1
    os.makedirs(IMAGES, exist_ok=True)
    paths = sorted(os.path.join(d, f) for d, _, files in os.walk(args.folder) for f in files
                   if f.lower().endswith(EXTENSIONS))
    added = skipped = 0
    for path in paths:
        rel = os.path.relpath(path, args.folder)
        parent = os.path.basename(os.path.dirname(path))
        category = parent if parent in CATEGORIES else ""
        with open(path, "rb") as f:
            data = f.read()
        device = guess_device(data, path)
        if args.dry_run:
            print(f"{rel}: category {category or '?'}, device {device or '?'}")
            continue
        try:
            pid, why = save_photo(data, rel, category=category, device=device)
        except Exception as e:                     # an unreadable file: report it and go on
            pid, why = None, f"{rel}: could not be read ({e})"
        if pid:
            added += 1
            print(f"{rel} -> {pid}  (category {category or '?'}, device {device or '?'})")
        else:
            skipped += 1
            print(f"skipped  {why}")
    if not args.dry_run:
        print(f"\n{added} imported, {skipped} skipped. Now set the details: .venv/bin/python eval/tools/collect.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
