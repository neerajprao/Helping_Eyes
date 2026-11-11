"""
Helping Eyes: the application (runs on your Mac).

A FastAPI server plus a browser page. The browser handles the camera, speech
recognition and speech output; the server does everything else:

    WS   /ws/live         preview frames in, guidance out (live.py): "Move closer",
                          hold still, auto-capture; page turns in book mode
    POST /api/capture     full-size photo -> text kept for questions
    POST /api/quality     uploaded photo -> sharpness, light, glare and text boxes
    POST /api/book/page   full-size photo of a book view -> what to read, from where
    POST /api/ask         a spoken or typed request -> commands, answers, web lookup,
                          streamed as NDJSON (the routing lives in commands.py)
    POST /api/tts         one sentence -> spoken audio (MP3, commands.py; the page uses the browser voice if it fails)
    GET  /api/health      health check
    GET  /                the page (web/index.html)

Run it:
    uvicorn server:app --port 7860          (then open http://localhost:7860)

Settings (environment or .env):
    LLM_API_KEY   key for the language model API (required)
    LLM_MODEL     model name (default gemini-3.1-flash-lite)
    LLM_BASE_URL  any OpenAI-compatible chat API (default: Google Gemini's)
    ACCESS_CODE   when set, the app asks for this code once per browser (30 days) before anything else works
"""

import hashlib
import hmac
import json
import logging
import os
import re
import resource
import sys
import threading
import time
from collections import OrderedDict, defaultdict, deque
from typing import Dict, Iterator, Optional

from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import commands
import vision
from commands import classify
from assistant import LLM_API_KEY, LLM_MODEL, READ_ALL, DocAssistant, web_lookup_allowed
from vision import (BookReader, BookWatcher, LiveGuide, _norm, _norm_quad, assess_quality, find_text_region, read_page,
                    read_text_enhanced)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("helping_eyes.web")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SESSION_TTL = 30 * 60          # seconds a session (captured item, conversation) is kept
MAX_SESSIONS = 200
MAX_SIDE = 1600                # photos are downscaled to this for OCR speed
RATE_LIMIT = 30                # requests per minute per client
MAX_SOCKETS = 2                # live streams per client
_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
ACCESS_COOKIE = "he_access"
ACCESS_DAYS = 30               # a browser that entered the code is remembered this long
SIGN_IN_LIMIT = 5              # wrong codes per minute per client, so the code cannot be guessed quickly
OPEN_PATHS = {"/login", "/api/health", "/favicon.ico", "/static/style.css", "/static/favicon.svg"}   # the sign-in page's own

app = FastAPI(title="Helping Eyes", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ---------------- access code ----------------
def _access_token() -> str:
    """What the browser keeps once the code was right: derived from the code, so changing the code signs everyone out."""
    return hmac.new(ACCESS_CODE.encode(), b"helping-eyes access", hashlib.sha256).hexdigest()


def _signed_in(cookies) -> bool:
    return not ACCESS_CODE or hmac.compare_digest(cookies.get(ACCESS_COOKIE, ""), _access_token())


@app.middleware("http")
async def ask_for_the_access_code(request: Request, call_next):
    """With ACCESS_CODE set, nothing but the sign-in page and the health check works until the code was entered."""
    if request.url.path in OPEN_PATHS or _signed_in(request.cookies):
        return await call_next(request)
    if request.url.path == "/":
        return RedirectResponse("/login", status_code=303)
    return JSONResponse({"detail": "Please enter the access code first."}, status_code=401)


_sign_ins: Dict[str, deque] = defaultdict(deque)


def _sign_in_page(message: str = "", status: int = 200) -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "login.html"), encoding="utf-8") as f:
        page = f.read()
    alert = f'<p class="alert" role="alert">{message}</p>' if message else ""
    return HTMLResponse(page.replace("<!--MESSAGE-->", alert), status_code=status, headers={"Cache-Control": "no-store"})


@app.get("/login")
def sign_in_page() -> HTMLResponse:
    return _sign_in_page()


@app.post("/login")
def sign_in(request: Request, code: str = Form("")) -> Response:
    if not ACCESS_CODE:
        return RedirectResponse("/", status_code=303)
    client, now = _client(request), time.time()
    tries = _sign_ins[client]
    while tries and now - tries[0] > 60:
        tries.popleft()
    if len(tries) >= SIGN_IN_LIMIT:
        return _sign_in_page("Too many wrong codes. Please wait a minute and try again.", 429)
    if not hmac.compare_digest(code.strip().encode(), ACCESS_CODE.encode()):
        tries.append(now)
        return _sign_in_page("That code is not right. Please try again.", 401)
    response = RedirectResponse("/", status_code=303)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(ACCESS_COOKIE, _access_token(), max_age=ACCESS_DAYS * 86400, httponly=True, samesite="lax",
                        secure=secure)
    return response


@app.middleware("http")
async def revalidate_the_page(request: Request, call_next):
    """The page and its files must be checked with the server on every load (a cheap "not modified" when they have not
    changed). Without this a browser may keep an old index.html for hours next to a new app.js, and the two no longer fit."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# One OCR job at a time: the OCR engines are not meant to be called from several threads at once
_ocr_lock = threading.Lock()


# ---------------- sessions and rate limiting ----------------
class Session:
    """Everything the server remembers about one browser: the captured item and the conversation."""

    def __init__(self):
        self.doc = DocAssistant()
        self.last_question = ""
        self.last_answer = ""
        self.pending_search = ""       # question waiting for a yes / no to "Should I look it up online?"
        self.book_mode = False
        self.guide = LiveGuide()       # normal mode: live guidance
        self.watcher = BookWatcher()   # book mode: page turns
        self.reader = BookReader()     # book mode: what to read
        self.touched = time.time()


_sessions: Dict[str, Session] = {}
_sessions_lock = threading.Lock()
_hits: Dict[str, deque] = defaultdict(deque)
_sockets: Dict[str, int] = defaultdict(int)


def _get_session(sid: str) -> Session:
    if not _SID.match(sid or ""):
        raise HTTPException(400, "Missing or invalid session id.")
    now = time.time()
    with _sessions_lock:
        for key in [k for k, s in _sessions.items() if now - s.touched > SESSION_TTL]:
            del _sessions[key]
        session = _sessions.get(sid)
        if session is None:
            if len(_sessions) >= MAX_SESSIONS:
                raise HTTPException(503, "The server is busy. Please try again in a few minutes.")
            session = _sessions[sid] = Session()
        session.touched = now
        return session


def _client(request) -> str:
    headers, host = request.headers, request.client.host if request.client else "?"
    return headers.get("x-forwarded-for", host).split(",")[0].strip()


def _rate_limit(request: Request) -> None:
    client = _client(request)
    now = time.time()
    hits = _hits[client]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Please wait a moment.")
    hits.append(now)


# ---------------- helpers ----------------
_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _sentences(text: str) -> Iterator[str]:
    """Split text into speakable pieces: paragraphs, then sentences."""
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in _SENTENCES.split(paragraph.strip()):
            if sentence.strip():
                yield sentence.strip()


def _event(kind: str, **data) -> str:
    return json.dumps({"type": kind, **data}) + "\n"


def _stream(events: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(events, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _decode(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "The file is not an image.")
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


# ---------------- memory and start-up status (to see why a small host struggles) ----------------
def _memory_mb() -> Optional[float]:
    """Memory this server is using right now, in MB (None when the system doesn't say)."""
    try:
        with open("/proc/self/status") as f:                      # Linux, which is what a host runs
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # elsewhere: the peak so far (bytes on macOS, KB on Linux)
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _memory_limit_mb() -> Optional[float]:
    """The memory the host allows this container, when it says so (Linux cgroups)."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = open(path).read().strip()
        except OSError:
            continue
        if value.isdigit() and int(value) < 1 << 50:               # "max" or a huge number means no limit
            return round(int(value) / (1024 * 1024))
    return None


_warmup = {"state": "starting"}       # starting -> loading OCR -> ready (or failed)


# ---------------- API ----------------
@app.on_event("startup")
def _warm_up() -> None:
    # Load the OCR models in the background so the first request is fast
    def work():
        # Real text, not a blank image: the OCR models only load when they have something to read,
        # and the first capture would otherwise take many seconds
        sample = np.full((240, 900, 3), 255, np.uint8)
        cv2.putText(sample, "Helping Eyes warm up 123", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
        try:
            logger.info(f"Warm-up: starting, memory {_memory_mb()} MB, limit {_memory_limit_mb()} MB, {os.cpu_count()} CPUs")
            _warmup["state"] = "loading OCR"
            with _ocr_lock:
                find_text_region(sample)
                logger.info(f"Warm-up: live-frame OCR loaded, memory {_memory_mb()} MB")
                read_text_enhanced(sample)
                logger.info(f"Warm-up: photo OCR done, memory {_memory_mb()} MB")
                read_page(sample, split=False, enhance=True)
            _warmup["state"] = "ready"
            logger.info(f"Warm-up finished, memory {_memory_mb()} MB")
        except Exception as e:
            _warmup["state"] = f"failed: {e}"
            logger.error(f"Warm-up failed: {e}")
    threading.Thread(target=work, daemon=True).start()
    threading.Thread(target=commands.warm_up, daemon=True).start()      # load the local voice so the first sentence is quick
    if commands.kokoro_ready():
        vision.SPEECH_CHUNK_CHARS = commands.KOKORO_PIECE              # a page is voiced in pieces the local voice makes quickly


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "favicon.svg"), media_type="image/svg+xml")


@app.get("/api/health")
def health() -> dict:
    # No request to the language model here: free API plans count every call
    return {"status": "ok", "model": LLM_MODEL, "llm_configured": bool(LLM_API_KEY),
            "ocr": _warmup["state"], "memory_mb": _memory_mb(), "memory_limit_mb": _memory_limit_mb(),
            "cpus": os.cpu_count(), "tts": commands.TTS_ENABLED, "tts_engine": commands.engine()}


class Speak(BaseModel):
    text: str = ""
    sid: str = ""
    gen: int = 0               # goes up each time this browser stops talking (Stop, a new item or page, ...)


_tts_hits: Dict[str, deque] = defaultdict(deque)
TTS_RATE_LIMIT = 60            # speech requests per minute per client (one per sentence)
TTS_GIVE_UP_S = 30             # the browser stops waiting for the local voice after this long (giveUpMs in app.js)
# The newest gen seen from each browser. The local voice makes one sentence at a time, so a request that waited
# while its browser moved on is dropped: its audio would never be played, and every later sentence would wait for it.
_voice_gen: "OrderedDict[str, int]" = OrderedDict()


@app.post("/api/tts")
def speak(q: Speak, request: Request) -> Response:
    """Turn one sentence into MP3 audio."""
    if not commands.TTS_ENABLED:
        raise HTTPException(503, "The server voice is switched off.")
    client, now = _client(request), time.time()
    hits = _tts_hits[client]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= TTS_RATE_LIMIT:
        raise HTTPException(429, "Too many speech requests. Please wait a moment.")
    hits.append(now)
    started = time.time()
    if q.sid:
        _voice_gen[q.sid] = max(q.gen, _voice_gen.get(q.sid, q.gen))
        _voice_gen.move_to_end(q.sid)
        while len(_voice_gen) > 1000:
            _voice_gen.popitem(last=False)

    def stale() -> bool:
        return time.time() - started > TTS_GIVE_UP_S or (bool(q.sid) and q.gen < _voice_gen.get(q.sid, q.gen))
    try:
        audio = commands.synthesize(q.text, stale=stale)
        if time.time() - started > 5:
            logger.warning(f"TTS was slow: {time.time() - started:.1f} s for {len(q.text)} characters")
    except commands.TtsCancelled:
        logger.info(f"TTS skipped after {time.time() - started:.1f} s: the browser had moved on")
        raise HTTPException(409, "This speech is no longer wanted.")
    except commands.TtsError as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(502, "The voice service failed.")
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/capture")
async def capture(request: Request, image: UploadFile = File(...), sid: str = Form(""),
                  book: bool = Form(False)) -> dict:
    """Read all text in a photo and keep it for questions."""
    _rate_limit(request)
    session = _get_session(sid)
    data = await image.read()
    logger.info(f"capture: received {len(data) // 1024} KB (book={book}), memory {_memory_mb()} MB")
    frame = _decode(data)

    def work():
        with _ocr_lock:
            if book:    # a two-page spread: split it, find page numbers, order the columns
                text, layout = read_page(frame, split=True, enhance=True)
                return text, layout, "book"
            # a single item: the untouched frame, plus corrected versions when it looks like it needs them
            text, method = read_text_enhanced(frame)
            return text, None, method

    started = time.time()
    text, layout, method = await run_in_threadpool(work)
    seconds = round(time.time() - started, 2)
    logger.info(f"capture: {len(text)} characters in {seconds} s (book={book}, version={method}), memory {_memory_mb()} MB")
    if not text.strip():
        return {"ok": False, "text": "", "seconds": seconds,
                "message": "I can't see any text. Move closer, hold the item steady and try again."}

    session.pending_search = ""
    session.last_answer = ""
    session.doc.set_document(text, layout.summary() if layout else "", layout.label() if layout else "")
    return {"ok": True, "text": text, "seconds": seconds, "method": method,
            "label": layout.label() if layout else "",
            "running_title": (layout.running_title() or "") if layout else "",
            "message": "Got it. What would you like to know?"}


@app.post("/api/quality")
async def quality(request: Request, image: UploadFile = File(...)) -> dict:
    """Sharpness, light and glare of an uploaded photo (what the camera view shows live), plus where the text is."""
    _rate_limit(request)
    frame = _decode(await image.read())
    h, w = frame.shape[:2]

    def work():
        with _ocr_lock:
            box, lines = find_text_region(frame, 8)
        return box, lines, assess_quality(frame, box)

    box, lines, q = await run_in_threadpool(work)
    return {"box": _norm(box, w, h) if box else None, "lines": [_norm(l.box, w, h) for l in lines],
            "quads": [_norm_quad(l.quad, w, h) for l in lines if l.quad],
            "quality": q.summary(), "problems": list(q.problems), "hint": q.hint(),
            "glare_at": list(q.glare_at) if q.glare_at else None}


@app.post("/api/book/page")
async def book_page(request: Request, image: UploadFile = File(...), sid: str = Form(""),
                    event: str = Form("changed"), read_pos: int = Form(0),
                    reading: bool = Form(False), thinking: bool = Form(False)) -> dict:
    """
    Book mode: the page view became steady (a page turn, or the view moved).
    Replies with what to speak and from where, plus the layout for the on-screen overlay.
    """
    _rate_limit(request)
    session = _get_session(sid)
    frame = _decode(await image.read())

    def work():
        with _ocr_lock:
            return session.reader.process(frame, event, read_pos, reading, thinking, session.doc)

    started = time.time()
    result = await run_in_threadpool(work)
    result["seconds"] = round(time.time() - started, 2)
    if result["action"] in ("new", "resume"):
        session.pending_search = ""
        session.last_answer = result["text"][result["from_pos"]:]
    return result


class Question(BaseModel):
    sid: str
    question: str = ""
    provider: str = ""             # language model chosen in the page's menu: "gemini" or "qwen3b" ("" keeps the last one)


@app.post("/api/ask")
def ask(q: Question, request: Request) -> StreamingResponse:
    """
    One spoken or typed request. Replies as NDJSON, one event per line:
        sentence      {text, source?}   speak this (source "document" or "web")
        command       {name}            the page must do something: stop, new_capture, book_on, book_off
        offer_search  {text}            "Should I look it up online?"
        done
    """
    _rate_limit(request)
    session = _get_session(q.sid)
    request_text = q.question.strip()
    session.doc.set_provider(q.provider)

    def sentences(text: str, source: Optional[str] = None) -> Iterator[str]:
        for s in _sentences(text):
            yield _event("sentence", text=s, **({"source": source} if source else {}))

    def web(question: str) -> Iterator[str]:
        yield _event("sentence", text="Let me look that up.")
        answer = []
        for s in session.doc.ask_web(question):
            answer.append(s)
            yield _event("sentence", text=s, source="web")
        if answer:
            session.last_answer = " ".join(answer)

    def events() -> Iterator[str]:
        started = time.time()
        pending, session.pending_search = session.pending_search, ""
        cmd = classify(request_text, offer_pending=bool(pending),
                       judge=lambda t, p: session.doc.judge_intent(t, p, pending or session.last_question))

        if cmd.name == "empty":
            pass
        elif cmd.name == "yes":
            session.doc.cancel()
            yield from web(pending)
        elif cmd.name == "no":
            yield _event("sentence", text="Okay.")
        elif cmd.name == "stop":
            session.doc.cancel()
            yield _event("command", name="stop")
        elif cmd.name == "repeat":
            yield _event("command", name="stop")
            yield from sentences(session.last_answer or "Nothing to repeat yet.")
        elif cmd.name in ("book_on", "book_off"):
            session.doc.cancel()
            session.book_mode = cmd.name == "book_on"
            session.reader.reset()
            session.watcher.reset()
            yield _event("command", name=cmd.name)
            yield _event("sentence", text="Book mode. Hold the book open in front of the camera. I'll read each page, "
                                          "and the next one when you turn it." if session.book_mode else "Normal mode.")
        elif cmd.name == "new_capture":
            session.doc.cancel()
            session.guide.rearm()
            yield _event("command", name="new_capture")
            yield _event("sentence", text="Turn the page and hold it still. I'll read it." if session.book_mode
                         else "Okay. Show me the next thing.")
        elif cmd.name in ("search_for", "search_last"):
            question = cmd.query or session.last_question
            session.doc.cancel()
            if not question:
                yield _event("sentence", text="What should I look up?")
            else:
                # The captured text comes first: go online only when it cannot answer
                from_text = []
                if session.doc.has_document() and web_lookup_allowed(question):
                    from_text = [s for s in session.doc.ask(question) if s != READ_ALL]
                    if session.doc.last_not_in_text:
                        from_text = []
                if from_text:
                    session.last_answer = " ".join(from_text)
                    for s in from_text:
                        yield _event("sentence", text=s)
                else:
                    yield from web(question)
        elif not session.doc.has_document():
            yield _event("sentence", text="I haven't captured any text yet. Hold something up to the camera.")
        else:
            session.doc.cancel()
            if cmd.name == "read_all":
                session.last_answer = session.doc.document
                yield from sentences(session.doc.document, "document")
            else:
                session.last_question = cmd.query
                first, answer = None, []
                for s in session.doc.ask(cmd.query):
                    if first is None:
                        first = round(time.time() - started, 2)
                    if s == READ_ALL:
                        session.last_answer = session.doc.document
                        yield from sentences(session.doc.document, "document")
                        answer = []
                        break
                    answer.append(s)
                    yield _event("sentence", text=s)
                if answer:
                    session.last_answer = " ".join(answer)
                # The text didn't have it: offer to look it up (never for expiry, batch or price)
                if session.doc.last_not_in_text and web_lookup_allowed(cmd.query):
                    session.pending_search = cmd.query
                    yield _event("offer_search", text="Should I look it up online?")
                logger.info(f"ask: first sentence after {first} s, total {time.time() - started:.1f} s")
        yield _event("done", seconds=round(time.time() - started, 2))

    return _stream(events())


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    """
    Live camera stream. The page sends one JPEG preview frame at a time (and waits for the reply):
        text message  {"type": "mode", "mode": "normal" | "book"}
                      {"type": "rearm"}                     ready for a new item
                      {"type": "capture_result", "ok": bool}
        binary        a JPEG frame -> a "guide" reply (normal mode) or a "book" reply (page turns)
    """
    client = _client(ws)
    if not _signed_in(ws.cookies):
        await ws.close(code=1008)
        return
    if _sockets[client] >= MAX_SOCKETS:
        await ws.close(code=1013)
        return
    sid = ws.query_params.get("sid", "")
    try:
        session = _get_session(sid)
    except HTTPException:
        await ws.close(code=1008)
        return
    await ws.accept()
    _sockets[client] += 1
    book = session.book_mode
    session.guide.reset()
    session.watcher.reset()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            session.touched = time.time()
            if message.get("bytes"):
                frame = cv2.imdecode(np.frombuffer(message["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if book:
                    reply = session.watcher.update(frame)
                else:
                    def work():
                        with _ocr_lock:
                            return session.guide.update(frame)
                    reply = await run_in_threadpool(work)
                await ws.send_json(reply)
            elif message.get("text"):
                try:
                    control = json.loads(message["text"])
                except ValueError:
                    continue
                kind = control.get("type")
                if kind == "mode":
                    book = control.get("mode") == "book"
                    session.book_mode = book
                    session.watcher.reset()
                    session.reader.reset()
                    session.guide.reset()
                elif kind == "rearm":
                    session.guide.rearm()
                elif kind == "capture_result":
                    session.guide.capture_result(bool(control.get("ok")))
    except WebSocketDisconnect:
        pass
    finally:
        _sockets[client] = max(0, _sockets[client] - 1)
