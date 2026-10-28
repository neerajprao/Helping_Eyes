"""
Book mode on real photos and a real page-turn video (FR-9, FR-10, FR-11 of the requirements).

    .venv/bin/python eval/run_book.py pages                       the book_page photos (after run_photo_eval.py ocr)
    .venv/bin/python eval/run_book.py video <file> --turns N      a phone video of N page turns

pages   For every book_page photo, with Claude's reading of it as the reference (claude_reference/<id>.json:
        "pages": the printed page numbers, left to right, [] when none is printed; "running_title": the header
        text or ""; "key_lines": body lines in reading order, the first one being the first line to be read):
          page numbers   the app's page numbers (the layout's, as book mode announces them) equal the reference's
          running title  the reference's running title is in the app's (letters and digits, case folded)
          first line     reading starts at the first body line: the reference's first key line is found in the
                         first 1.5 x its length (+ 20) characters of the app's text
          reading order  each key line is located in the app's text (best approximate match; found when at most
                         30 % of its characters need an edit); share of found pairs that come in the right order
        both for the layout order alone (XY-cut, no model) and for the paragraphs arranged by Qwen 3B
        (arrange_page, as book mode does it before reading).
video   The video is fed to the app's page-turn detector (BookWatcher, the same object the live stream uses) at
        8 frames a second of video time, as the page sends live frames. Every "changed" event is a page the app
        would read; a book opened and turned N times should give N + 1. The frame of every event is saved to
        eval/data/private/book_video/ so each one can be checked by eye.

Output: eval/results/book.json (numbers and page numbers only; no book text).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.RESULTS, "book.json")
VIDEO_OUT = os.path.join(C.PRIVATE, "book_video")
FPS = 8


def locate(ref, text):
    """(fewest character edits to make `ref` a substring of `text`, end index of that substring)."""
    if not ref:
        return 0, 0
    if not text:
        return len(ref), 0
    t = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    prev = np.zeros(len(t) + 1, dtype=np.int64)
    steps = np.arange(len(t) + 1)
    for i, ch in enumerate(ref, 1):
        best = np.empty_like(prev)
        best[0] = i
        best[1:] = np.minimum(prev[:-1] + (t != ord(ch)), prev[1:] + 1)
        prev = steps + np.minimum.accumulate(best - steps)
    end = int(prev.argmin())
    return int(prev[end]), end


def order_score(key_lines, text):
    import photo_report as P
    text = P.norm(text)
    found = []
    for line in key_lines:
        line = P.norm(line)
        edits, end = locate(line, text)
        if line and edits <= 0.3 * len(line):
            found.append(end)
    pairs = [(a, b) for i, a in enumerate(found) for b in found[i + 1:]]
    first = P.norm(key_lines[0]) if key_lines else ""
    starts = locate(first, text[:int(1.5 * len(first)) + 20])[0] <= 0.3 * len(first) if first else None
    return {"key_lines": len(key_lines), "found": len(found),
            "pairs_in_order": sum(a < b for a, b in pairs), "pairs": len(pairs),
            "order": round(sum(a < b for a, b in pairs) / len(pairs), 4) if pairs else None, "starts_at_first_line": starts}


def pages():
    import photo_report as P
    import run_photo_eval as R
    assistant = C.fix_date()
    import vision
    man = R.manifest()
    rows = []
    for pid in R.ids():
        if man[pid]["category"] != "book_page":
            continue
        ref = C.load_json(os.path.join(R.REFS, pid + ".json"))
        if not ref:
            print(f"{pid}: no reference yet"); continue
        frame = R.load(pid)
        text, layout = vision.read_page(frame, enhance=True)
        doc = assistant.DocAssistant()
        doc.set_provider("qwen3b")
        arranged = vision.arrange_page(layout, doc)
        got_pages = [str(n) for n in layout.page_numbers()]
        want_title = P.alnum(ref.get("running_title") or "")
        row = {"id": pid, "lighting": man[pid]["lighting"], "angle": man[pid]["angle"], "spread": len(ref["pages"]) == 2 or ref.get("spread", False),
               "pages_ref": ref["pages"], "pages_app": got_pages, "pages_split": len(layout.pages),
               "arranged_by_model": arranged is not None, "pages_correct": got_pages == [str(p) for p in ref["pages"]],
               "title_ref": bool(want_title), "title_correct": (want_title in P.alnum(layout.running_title() or "")) if want_title else None,
               "layout": order_score(ref["key_lines"], text),
               "arranged": order_score(ref["key_lines"], arranged[0]) if arranged else None}
        rows.append(row)
        print(pid, row["pages_ref"], row["pages_app"], row["layout"]["order"], (row["arranged"] or {}).get("order"), flush=True)

    def share(vals):
        vals = [v for v in vals if v is not None]
        return {"yes": sum(vals), "of": len(vals), "rate": round(sum(vals) / len(vals), 4) if vals else None}

    def pooled(kind):
        r = [x[kind] for x in rows if x[kind]]
        n, d = sum(x["pairs_in_order"] for x in r), sum(x["pairs"] for x in r)
        return {"pairs_in_order": n, "pairs": d, "rate": round(n / d, 4) if d else None,
                "starts_at_first_line": share([x["starts_at_first_line"] for x in r]),
                "key_lines_found": [sum(x["found"] for x in r), sum(x["key_lines"] for x in r)]}

    data = C.load_json(OUT, {})
    data["pages"] = {"photos": len(rows), "spreads": sum(r["spread"] for r in rows),
                     "page_numbers": share([r["pages_correct"] for r in rows]),
                     "page_numbers_spreads": share([r["pages_correct"] for r in rows if r["spread"]]),
                     "running_title": share([r["title_correct"] for r in rows]),
                     "spreads_split_in_two": share([r["pages_split"] == 2 for r in rows if r["spread"]]),
                     "arranged_by_model": share([r["arranged_by_model"] for r in rows]),
                     "order_layout": pooled("layout"), "order_arranged_qwen3b": pooled("arranged"),
                     "rows": rows, **C.git_state()}
    C.save_json(OUT, data)
    print({k: v for k, v in data["pages"].items() if k != "rows"})


def video(path, turns):
    import cv2
    import vision
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open {path} (convert it with: ffmpeg -i in.mov -vf scale=960:-2 out.mp4)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / FPS))
    watcher = vision.BookWatcher()
    os.makedirs(VIDEO_OUT, exist_ok=True)
    for f in os.listdir(VIDEO_OUT):
        os.remove(os.path.join(VIDEO_OUT, f))
    events, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            h, w = frame.shape[:2]
            s = 960 / max(h, w)
            small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
            d = watcher.detector
            event = d.update(small, now=n / fps)
            if event:
                events.append({"t": round(n / fps, 2), "event": event})
                if event == "changed":
                    cv2.imwrite(os.path.join(VIDEO_OUT, f"page_{sum(e['event'] == 'changed' for e in events):02d}_{n / fps:06.1f}s.jpg"), small)
        n += 1
    changed = sum(e["event"] == "changed" for e in events)
    expected = turns + 1
    data = C.load_json(OUT, {})
    data["video"] = {"seconds": round(n / fps, 1), "fps_fed": FPS, "turns": turns, "pages_expected": expected,
                     "pages_read": changed, "moved_events": sum(e["event"] == "moved" for e in events),
                     "missed": max(0, expected - changed), "extra": max(0, changed - expected),
                     "events": events, **C.git_state()}
    C.save_json(OUT, data)
    print(f"{changed} pages read for {expected} pages shown ({turns} turns); events: {events}")
    print(f"check each one by eye: {os.path.relpath(VIDEO_OUT, C.ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["pages", "video"])
    parser.add_argument("file", nargs="?")
    parser.add_argument("--turns", type=int)
    args = parser.parse_args()
    if args.stage == "pages":
        pages()
    else:
        if not (args.file and args.turns is not None):
            sys.exit("video needs the file and --turns N")
        video(args.file, args.turns)


if __name__ == "__main__":
    main()
