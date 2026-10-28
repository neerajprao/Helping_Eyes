"""
Speed (Phase 5 of markdown/eval_plan.md, NFR-1): the real app (uvicorn server:app) on this Mac, over localhost.

    .venv/bin/python eval/run_latency.py              everything (about 15-20 minutes; Gemini calls are spaced 4.5 s)
    .venv/bin/python eval/run_latency.py --runs 5     a quick try

Run it with nothing else busy on the Mac (no other evaluation running). Every operation gets 1 warm-up run (not
counted) and then --runs timed runs; p50, p90, p95, mean, min and max are reported. Inputs are made the way app.js
makes them (longest side and JPEG quality from frameBlob): live frames 960 px at 0.7, captures 1920 px at 0.92, book
pages 1600 px at 0.9.

  live_label / live_spread  WebSocket /ws/live, normal mode: one frame sent, time until the "guide" reply
                            (client -> server -> client; the browser's own canvas / JPEG step is not included)
  live_book                 the same in book mode (page-turn watcher only, no OCR)
  capture                   POST /api/capture, a label photo
  capture_spread            POST /api/capture with the two-page box ticked (book path)
  book_page_<model>         POST /api/book/page with a new session each time (a new page): OCR plus the paragraph
                            step by the language model, as book mode does it
  ocr_only / preprocess     in-process: recognize_text() alone, and capture_candidates() (the corrected versions)
  ask_<model>               POST /api/ask after a capture: time to the first "sentence" event and to "done"
  tts                       POST /api/tts for a typical answer sentence, a different one each run so the server's
                            voice cache is never used (the server's first voice, Kokoro)
  preprocess                note: capture_candidates() makes corrected versions only when the quality check finds a
                            problem, so on good photos it is quick
  first_word_<model>        question to first spoken word: first sentence event + its /api/tts audio received
                            (browser speech recognition and audio start are not included)

Each request carries its own X-Forwarded-For address so the app's per-client rate limit (30 a minute) does not
throttle the test; nothing in the app is changed. Output: eval/results/latency.json.
"""

import argparse
import glob
import itertools
import json
import os
import platform
import secrets
import subprocess
import sys
import time

import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
OUT = os.path.join(C.RESULTS, "latency.json")
QUESTIONS = ["When does it expire?", "What is the batch number?", "Who makes it?", "What is the price?",
             "What is the strength of the medicine?", "How much is in the pack?", "What is the name of this product?",
             "When was it manufactured?", "Does it need a prescription?", "How should I store it?"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# a different sentence every run, so the server's small voice cache never answers instead of the voice
SENTENCES = [f"This medicine has not expired, and it is good to use until the end of {m} {y}."
             for y in (2027, 2028) for m in MONTHS]
_addr = itertools.count(1)


def client():
    n = next(_addr)
    return {"X-Forwarded-For": f"10.{n // 65536 % 256}.{n // 256 % 256}.{n % 256}"}


def sid():
    return "lat" + secrets.token_hex(8)


def jpeg(path, side, quality):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if max(h, w) > side:
        s = side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, round(quality * 100)])[1].tobytes()


def photos():
    """Label photos (the QA subset's pharmacy photos) and two-page spreads (book_page photos, else the test page)."""
    subset = C.load_json(os.path.join(C.EVAL, "data", "qa_subset.json"), {})
    images = os.path.join(C.PRIVATE, "images")
    labels = [os.path.join(images, p + ".jpg") for p in subset.get("pharmacy", [])]
    if not labels:
        labels = sorted(glob.glob(os.path.join(images, "m*.jpg")))[:20]
    spreads = sorted(glob.glob(os.path.join(images, "bp*.jpg"))) or [os.path.join(C.APP, "data_test", "simon_ocarrigan-sound_of_the_dark-06.png")]
    return labels, spreads


def timed(runs, fn, inputs, label):
    times = []
    for i in range(runs + 1):
        t = fn(inputs[i % len(inputs)])
        if i:                                  # the first run is the warm-up
            times.append(t)
    stats = C.timing(times)
    print(f"{label:22s} p50 {stats['p50']:.3f} s   p90 {stats['p90']:.3f} s   (n={stats['n']})", flush=True)
    return {**stats, "values": [round(t, 4) for t in times]}


def start_server():
    log = open(os.path.join(C.PRIVATE, "latency_server.log"), "w")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(PORT)],
                            cwd=C.APP, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            h = requests.get(BASE + "/api/health", timeout=2).json()
            if h["ocr"] == "ready":
                return proc, h
        except requests.RequestException:
            pass
        time.sleep(0.5)
    proc.terminate()
    sys.exit("The server did not become ready; see eval/data/private/latency_server.log")


async def _ws_times(frames, mode, runs):
    import websockets
    s = sid()
    times = []
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws/live?sid={s}", max_size=None,
                                  additional_headers=client()) as ws:
        await ws.send(json.dumps({"type": "mode", "mode": mode}))
        for i in range(runs + 1):
            t = time.perf_counter()
            await ws.send(frames[i % len(frames)])
            await ws.recv()
            if i:
                times.append(time.perf_counter() - t)
            await ws.send(json.dumps({"type": "rearm"}))
    return times


def live(frames, mode, runs, label):
    import asyncio
    times = asyncio.run(_ws_times(frames, mode, runs))
    stats = C.timing(times)
    print(f"{label:22s} p50 {stats['p50']:.3f} s   p90 {stats['p90']:.3f} s   (n={stats['n']})", flush=True)
    return {**stats, "values": [round(t, 4) for t in times]}


def capture(data, s=None, book=False):
    r = requests.post(BASE + "/api/capture", files={"image": ("photo.jpg", data, "image/jpeg")},
                      data={"sid": s or sid(), "book": str(book).lower()}, headers=client(), timeout=120)
    r.raise_for_status()
    return r.json()


def choose(s, provider):
    """The page's "Answers from" menu: an empty request only sets the provider."""
    requests.post(BASE + "/api/ask", json={"sid": s, "question": "", "provider": provider}, headers=client(), timeout=30).content


def ask(s, question, provider):
    t = time.perf_counter()
    first, text = None, None
    with requests.post(BASE + "/api/ask", json={"sid": s, "question": question, "provider": provider},
                       headers=client(), stream=True, timeout=180) as r:
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "sentence" and first is None:
                first, text = time.perf_counter() - t, ev["text"]
            if ev["type"] == "done":
                break
    return first, time.perf_counter() - t, text


def tts(text):
    t = time.perf_counter()
    r = requests.post(BASE + "/api/tts", json={"text": text}, headers=client(), timeout=60)
    r.raise_for_status()
    assert r.content[:3] == b"ID3" or r.content[:2] == b"\xff\xfb" or len(r.content) > 1000
    return time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--models", default="qwen3b,gemini")
    parser.add_argument("--keep", action="store_true", help="keep the operations of an earlier run that this run does not measure")
    parser.add_argument("--only-models", action="store_true", help="measure only the model operations (with --keep)")
    args = parser.parse_args()
    runs, models = args.runs, [m for m in args.models.split(",") if m]
    labels, spreads = photos()
    proc, health = start_server()
    out = {"runs": runs, "date": time.strftime("%Y-%m-%d %H:%M"), "health": health, **C.git_state(),
           "machine": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                       "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                       "memory_gb": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30)},
           "inputs": {"label_photos": len(labels), "spread_photos": len(spreads)}, "ops": {}}
    if args.keep:          # add to an earlier run (e.g. the Gemini part on another day) instead of replacing it
        out["ops"] = C.load_json(OUT, {}).get("ops", {})
        out["earlier_run_kept"] = True
    ops = out["ops"]
    only_models = args.only_models
    try:
        cap = [jpeg(p, 1920, 0.92) for p in labels]
        page1600 = [jpeg(p, 1600, 0.9) for p in spreads]
        if not only_models:
            measure_app(ops, runs, labels, spreads, cap)
        measure_models(ops, runs, models, cap, page1600)
        if "live_label" in ops:
            ops["live_checks_per_second"] = round(1 / ops["live_label"]["p50"], 1)
    finally:
        proc.terminate()
        proc.wait(10)
        C.save_json(OUT, out)
    print(f"wrote {os.path.relpath(OUT, C.ROOT)}")


def measure_app(ops, runs, labels, spreads, cap):
    if True:
        live960 = [jpeg(p, 960, 0.7) for p in labels]
        spread960 = [jpeg(p, 960, 0.7) for p in spreads]
        spread1920 = [jpeg(p, 1920, 0.92) for p in spreads]

        ops["live_label"] = live(live960, "normal", runs, "live_label")
        ops["live_spread"] = live(spread960, "normal", runs, "live_spread")
        ops["live_book"] = live(spread960, "book", runs, "live_book")
        ops["capture"] = timed(runs, lambda d: _t(lambda: capture(d)), cap, "capture")
        ops["capture_spread"] = timed(runs, lambda d: _t(lambda: capture(d, book=True)), spread1920, "capture_spread")

        # in-process: the corrected versions alone and the reader alone, on the same photos as the server decodes them
        sys.path.insert(0, C.APP)
        import server
        import vision
        frames = [server._decode(d) for d in cap]
        ops["ocr_only"] = timed(runs, lambda f: _t(lambda: vision.recognize_text(f)), frames, "ocr_only")
        ops["preprocess"] = timed(runs, lambda f: _t(lambda: vision.capture_candidates(f)), frames, "preprocess")

        ops["tts"] = timed(runs, lambda t: tts(t), SENTENCES, "tts")


def measure_models(ops, runs, models, cap, page1600):
    if True:
        for model in models:
            def book(d, model=model):
                s = sid()
                choose(s, model)
                t = time.perf_counter()
                r = requests.post(BASE + "/api/book/page", files={"image": ("page.jpg", d, "image/jpeg")},
                                  data={"sid": s, "event": "changed"}, headers=client(), timeout=180)
                r.raise_for_status()
                return time.perf_counter() - t
            ops[f"book_page_{model}"] = timed(runs, book, page1600, f"book_page_{model}")

            firsts, totals, words = [], [], []
            for i in range(runs + 1):
                s = sid()
                capture(cap[i % len(cap)], s)
                if model == "gemini":
                    time.sleep(C.GEMINI_SPACING)
                first, total, text = ask(s, QUESTIONS[i % len(QUESTIONS)], model)
                if not i or first is None:
                    continue
                firsts.append(first); totals.append(total)
                words.append(first + tts(text))
            for name, vals in ((f"ask_first_{model}", firsts), (f"ask_total_{model}", totals), (f"first_word_{model}", words)):
                ops[name] = {**C.timing(vals), "values": [round(v, 4) for v in vals]}
                print(f"{name:22s} p50 {ops[name]['p50']:.3f} s   p90 {ops[name]['p90']:.3f} s   (n={ops[name]['n']})", flush=True)


def _t(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    main()
