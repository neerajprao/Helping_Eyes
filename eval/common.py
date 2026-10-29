"""
Shared helpers for the evaluation scripts (run_qa.py, run_expiry.py, run_latency.py, run_book.py, make_results.py).

  * paths, the app's own modules on sys.path and helping_eyes/.env loaded before the app reads its settings
  * the fixed evaluation date: assistant.date is replaced, for the eval process only, by a date whose today() is
    EVAL_DATE, so prompts, date checks and verdicts come from the app's own code but never change with the calendar
  * ask(): one question through the app's own path (commands.classify -> fresh DocAssistant -> ask), timed, with the
    retry rule of the plan for a busy hosted model (503 / 429: up to 3 retries, 30 s apart, every retry logged; a call
    that still fails is a *service error*, reported apart and never scored as a wrong answer; a reply that has not
    finished after CALL_LIMIT seconds counts as a failed call too)
  * bootstrap confidence intervals over photos, git state, JSON in and out
"""

import contextlib
import datetime
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "helping_eyes")
EVAL = os.path.join(ROOT, "eval")
PRIVATE = os.path.join(EVAL, "data", "private")
RESULTS = os.path.join(EVAL, "results")
MANIFEST = os.path.join(EVAL, "data", "manifest.csv")
sys.path.insert(0, APP)
sys.path.insert(0, EVAL)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(APP, ".env"), override=False)     # before assistant.py reads LLM_API_KEY

EVAL_DATE = datetime.date(2026, 10, 19)                     # decided at Checkpoint 0 (markdown/eval_plan.md)
SEED = 3235
RETRIES, RETRY_PAUSE = 3, 30
GEMINI_SPACING = 4.5                                        # seconds between Gemini calls (free tier, 15 a minute)


class _EvalDate(datetime.date):
    @classmethod
    def today(cls):
        return EVAL_DATE


def fix_date():
    """The app sees EVAL_DATE as today (in its prompt, its date checks and its expiry verdicts)."""
    import assistant
    assistant.date = _EvalDate
    return assistant


def git_state():
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def service_messages():
    """The app's own replies when the model could not be reached or refused (never a real answer)."""
    import assistant
    return {getattr(assistant, n) for n in dir(assistant) if n.startswith("MSG_")}


_last_gemini = [0.0]
CALL_LIMIT = 90        # seconds; a hosted stream that never ends (seen under Gemini's 503 load) counts as a service error


class _Hung(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    """Raise _Hung in the main thread after `seconds` (SIGALRM); a no-op elsewhere."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def fire(*_):
        raise _Hung()
    old = signal.signal(signal.SIGALRM, fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def ask(question, document, provider, facts="", page_label="", log=print):
    """
    One request exactly as /api/ask handles it after a capture: classify(), then a fresh DocAssistant with the text,
    the provider chosen in the page's menu, and ask(). "Read everything" is answered by the app without a model.
    Returns {answer, route, not_in_text, first_s, total_s, retries, service_error}.
    """
    import assistant
    from commands import classify
    cmd = classify(question)
    if cmd.name == "read_all":
        return {"answer": document, "route": "read_all", "not_in_text": False, "first_s": 0.0, "total_s": 0.0,
                "retries": 0, "service_error": False}
    failures = service_messages()
    retries = 0
    while True:
        if provider == "gemini":
            wait = GEMINI_SPACING - (time.time() - _last_gemini[0])
            if wait > 0:
                time.sleep(wait)
            _last_gemini[0] = time.time()
        doc = assistant.DocAssistant()
        doc.set_provider(provider)
        doc.set_document(document, facts, page_label)
        started, first, parts = time.perf_counter(), None, []
        hung = False
        with _deadline(CALL_LIMIT):
            try:
                for s in doc.ask(cmd.query):
                    if first is None:
                        first = time.perf_counter() - started
                    if s == assistant.READ_ALL:
                        parts = [document]
                        break
                    parts.append(s)
            except _Hung:
                hung = True
        total = time.perf_counter() - started
        answer = f"no reply finished within {CALL_LIMIT} s" if hung else " ".join(parts)
        if hung or answer in failures:
            if retries < RETRIES:
                retries += 1
                log(f"    service error ({answer[:60]}), retry {retries}/{RETRIES} in {RETRY_PAUSE} s")
                time.sleep(RETRY_PAUSE)
                continue
            return {"answer": answer, "route": cmd.name, "not_in_text": None, "first_s": None, "total_s": None,
                    "retries": retries, "service_error": True}
        return {"answer": answer, "route": cmd.name, "not_in_text": doc.last_not_in_text,
                "first_s": round(first or 0, 3), "total_s": round(total, 3), "retries": retries, "service_error": False}


def bootstrap(groups, stat, n=10000, seed=SEED):
    """95 % percentile interval of stat(list of groups) over resampled groups (a group = one photo's items)."""
    rng = random.Random(seed)
    groups = list(groups)
    if not groups:
        return None
    values = (stat([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(n))
    values = sorted(v for v in values if v is not None)          # a resample with nothing scored has no value
    if not values:
        return None
    return [round(values[int(0.025 * (len(values) - 1))], 4), round(values[int(0.975 * (len(values) - 1))], 4)]


def rate(groups):
    """Share of True over all scored items (None = not scored) in a list of photos' item lists."""
    items = [x for g in groups for x in g if x is not None]
    return sum(items) / len(items) if items else None


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def timing(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "p50": round(percentile(values, 50), 3), "p90": round(percentile(values, 90), 3),
            "p95": round(percentile(values, 95), 3), "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}
