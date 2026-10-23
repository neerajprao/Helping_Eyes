"""
Spoken and typed commands handled by the application itself, not the model, plus the
spoken voice (text to speech).

classify() turns one request into a Command. The server routes on it (server.py),
so the browser needs no command logic of its own.

synthesize() turns one sentence into MP3 audio, on this computer with the Kokoro voice (kokoro-onnx: free, no
network, see `python run.py --get-voice`), or, when Kokoro is not set up or fails, with a free Microsoft Edge neural
voice (edge-tts: no account, key or card). It is the last section of this file. The page uses the browser's own
voice if both fail. Settings (environment or .env):
    TTS_ENGINE      kokoro (default) or edge; with kokoro, Edge is still the backup
    KOKORO_VOICE    Kokoro voice name (default af_sarah, American female; also af_heart, af_bella, af_nicole, am_michael, bf_emma)
    KOKORO_SPEED    speaking speed, 1.0 is normal (default 1.0)
    KOKORO_MODEL_DIR  where kokoro-v1.0.onnx and voices-v1.0.bin are (default helping_eyes/voice_models)
    TTS_VOICE       Edge voice name (default en-US-JennyNeural, female; also en-US-AriaNeural, en-US-AvaNeural, en-GB-SoniaNeural)
    TTS_RATE        Edge speaking speed, e.g. +10% or -10% (default +0%)
    TTS_ENABLED     set to 0 to always use the browser voice
"""

import asyncio
import importlib.util
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from assistant import wants_read_all

logger = logging.getLogger(__name__)

_STOP = re.compile(r"^(stop|stop (it|talking|speaking|reading)|be quiet|quiet|shut up|cancel)[.!]*$")
_REPEAT = re.compile(r"^(repeat|repeat that|say (that|it) again|again|pardon|what)[.?!]*$")
_NEW_CAPTURE = re.compile(r"^(next|next page|new page|scan( again)?|capture( again)?|read something else|new (text|item|document))[.!]*$")
_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|please do|do it|go ahead|yes please|look it up|search)[.!]*$")
_NO = re.compile(r"^(no|nope|nah|no thanks|no thank you|don't|never mind|leave it)[.!]*$")
_BOOK_ON = re.compile(r"^(start |turn on |enter |switch to )?(book|reading) mode( on)?[.!]*$|^read (a|my|this) book[.!]*$")
_BOOK_OFF = re.compile(r"^(stop|exit|leave|end|turn off|quit) (book|reading) mode[.!]*$|^(book|reading) mode off[.!]*$|^(normal|regular) mode[.!]*$")
# "look it up", "search online", "google that" -> web search for the last question
_SEARCH_LAST = re.compile(r"^(please )?(look (it|that) up|search|google)( (it|that))?( (online|on the internet|on the web|the web|the internet))?( please)?[.!]*$")
# "search the web for X", "look up X", "google X" -> web search for X
_SEARCH_FOR = re.compile(r"^(?:please )?(?:search (?:the web |online |the internet )?for|look up|google) (.+?)[.?!]*$")


# Words that make a request possibly mean "search the web" (then the language model decides), and wordings that
# plainly agree to or refuse the offer "Should I look it up online?"
_WEB_WORDS = re.compile(r"\b(web|online|internet|google|search|look(ing)? (it|that|this)? ?up|browse|find out|check)\b")
_AGREES = re.compile(r"^(yes|yeah|yep|yup|yea|sure|ok|okay|alright|please|go ahead|do it|go for it|definitely|absolutely|of course|why not|sounds good|fine)\b"
                     r"|\b(look (it|that|this) up|search|google|check online|go online|on the (web|internet))\b")
_REFUSES = re.compile(r"^(no|nope|nah|don't|do not|never mind|nevermind|leave it|skip|forget it|not now|not really)\b")


@dataclass
class Command:
    """name is one of: stop, repeat, book_off, book_on, new_capture, search_last,
    search_for (query holds the phrase), read_all, question, yes, no, empty."""
    name: str
    query: str = ""


def classify(text: str, offer_pending: bool = False, judge=None) -> Command:
    """
    Decide what one request means. offer_pending is True right after the app asked
    "Should I look it up online?": only then do yes / no answer that offer.

    judge(text, offer_pending) -> "yes" | "no" | "search" | "search: <topic>" | "other" is the language model's
    reading of what was said; it is asked only when the wording is not already clear, so any phrasing that
    means "look it up online" works, not just the exact ones listed above.
    """
    request = text.strip()
    lower = request.lower()
    if not request:
        return Command("empty")
    if offer_pending:
        if _YES.match(lower):
            return Command("yes")
        if _NO.match(lower):
            return Command("no")
        if _REFUSES.match(lower):
            return Command("no")
        if _AGREES.search(lower):
            return Command("yes")
        verdict = _judged(judge, request, True)
        if verdict == "yes":
            return Command("yes")
        if verdict == "no":
            return Command("no")
        # anything else is a new request; the offer is dropped by the caller
    if _STOP.match(lower):
        return Command("stop")
    if _REPEAT.match(lower):
        return Command("repeat")
    if _BOOK_OFF.match(lower):
        return Command("book_off")
    if _BOOK_ON.match(lower):
        return Command("book_on")
    if _NEW_CAPTURE.match(lower):
        return Command("new_capture")
    m = _SEARCH_FOR.match(lower)
    if m:
        return Command("search_for", m.group(1))
    if _SEARCH_LAST.match(lower):
        return Command("search_last")
    if wants_read_all(lower):
        return Command("read_all")
    if _WEB_WORDS.search(lower):
        verdict = _judged(judge, request, False)
        if verdict.startswith("search"):
            topic = verdict.partition(":")[2].strip()
            return Command("search_for", topic) if topic else Command("search_last")
    return Command("question", request)


def _judged(judge, text: str, offer_pending: bool) -> str:
    """The language model's reading of the request ("other" when there is no judge or it fails)."""
    if judge is None:
        return "other"
    try:
        return (judge(text, offer_pending) or "other").strip().lower()
    except Exception:
        return "other"


# ---------------------------------------------------------------- spoken voice (text to speech)
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_ENABLED = os.getenv("TTS_ENABLED", "1") != "0"
TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro").strip().lower()
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.0"))
KOKORO_DIR = os.getenv("KOKORO_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_models"))
KOKORO_FILES = {"kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"}
KOKORO_PIECE = 300              # characters per Kokoro call: about 3 s on a laptop CPU, so the first words come quickly
MAX_CHARS = 700                 # one sentence or a short chunk; keeps each call quick
_CACHE_SIZE = 60                # repeated sentences ("Got it...") are not synthesized twice

_cache: "OrderedDict[str, bytes]" = OrderedDict()
_lock = threading.Lock()


class TtsError(Exception):
    pass


# A number above 9999 is nearly always a serial, batch or phone number: say its digits one by one,
# never "twelve thousand three hundred...". Comma-grouped forms (12,345,678) count too.
_LONG_NUMBER = re.compile(r"(?<![\d,])(?:\d{1,3}(?:,\d{3})+|\d{5,})(?!\d)")


def spoken_form(text: str) -> str:
    """The text as it is sent to the voice: numbers above 9999 spelled out digit by digit."""
    def digits(m):
        number = m.group(0).replace(",", "")
        return " ".join(number) if int(number) > 9999 else m.group(0)
    return _LONG_NUMBER.sub(digits, text)


async def _speak(text: str) -> bytes:
    import edge_tts
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


def _pieces(text: str, size: int = MAX_CHARS) -> list:
    """The text in pieces of at most `size` characters, cut after a sentence end, else a comma, else a space,
    so that nothing is dropped from a long list or a long sentence."""
    pieces = []
    while len(text) > size:
        cut = 0
        for marks in ((". ", "? ", "! "), ("; ", ", "), (" ",)):      # best place first
            found = [text.rfind(m, 0, size) + len(m) for m in marks if text.rfind(m, 0, size) >= 0]
            if found and max(found) >= size // 3:
                cut = max(found)
                break
        cut = cut or size
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    return pieces + [text] if text else pieces


def kokoro_ready() -> bool:
    """Is the local Kokoro voice set up: wanted, its two packages installed and its two model files downloaded?"""
    return (TTS_ENGINE == "kokoro" and not _kokoro_failed
            and all(importlib.util.find_spec(m) for m in ("kokoro_onnx", "lameenc"))
            and all(os.path.exists(os.path.join(KOKORO_DIR, name)) for name in KOKORO_FILES))


def engine():
    """The voice the server uses first: "kokoro" (on this computer), "edge" (Microsoft's service), or None (off)."""
    if not TTS_ENABLED:
        return None
    return "kokoro" if kokoro_ready() else "edge"


_kokoro_model = None
_kokoro_failed = False          # the local voice broke once: the rest of this run uses Edge
_kokoro_lock = threading.Lock()


def _kokoro():
    global _kokoro_model
    if _kokoro_model is None:
        from kokoro_onnx import Kokoro
        _kokoro_model = Kokoro(os.path.join(KOKORO_DIR, "kokoro-v1.0.onnx"), os.path.join(KOKORO_DIR, "voices-v1.0.bin"))
    return _kokoro_model


def _mp3(samples, rate: int) -> bytes:
    """Mono audio (floats from -1 to 1) as MP3, which every browser plays."""
    import lameenc
    import numpy as np
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return bytes(encoder.encode(pcm) + encoder.flush())


def _speak_kokoro(text: str) -> bytes:
    with _kokoro_lock:                                  # one call at a time: the model already uses every core
        samples, rate = _kokoro().create(spoken_form(text), voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang="en-us")
    return _mp3(samples, rate)


def warm_up() -> None:
    """Load the local voice and speak one short phrase, so the first real sentence is not slow."""
    global _kokoro_failed
    if TTS_ENABLED and kokoro_ready():
        try:
            _speak_kokoro("Ready.")
            logger.info(f"Voice: Kokoro ({KOKORO_VOICE}) is ready on this computer")
        except Exception as e:
            _kokoro_failed = True
            logger.warning(f"Voice: Kokoro could not start ({type(e).__name__}: {e}); using the Edge voice")
    elif TTS_ENABLED:
        logger.info("Voice: the Edge voice (run  python run.py --get-voice  for the local Kokoro voice)")


def synthesize(text: str) -> bytes:
    """MP3 audio of one sentence (a long text is spoken in pieces, joined). Raises TtsError when no voice can speak it."""
    text = text.strip()
    if not text:
        raise TtsError("empty text")
    pieces = _pieces(text, KOKORO_PIECE if kokoro_ready() else MAX_CHARS)
    if len(pieces) > 1:
        return b"".join(_synthesize_piece(p) for p in pieces)      # MP3 frames can simply follow each other
    return _synthesize_piece(text)


def _synthesize_piece(text: str) -> bytes:
    """MP3 audio of one piece: the Kokoro voice if it is set up, else (or if it fails) the Edge voice."""
    global _kokoro_failed
    with _lock:
        if text in _cache:
            _cache.move_to_end(text)
            return _cache[text]
    audio = b""
    if kokoro_ready():
        try:
            audio = _speak_kokoro(text)
        except Exception as e:                          # a missing file, a broken install, ...: use Edge from now on
            _kokoro_failed = True
            logger.warning(f"Voice: Kokoro failed ({type(e).__name__}: {e}); using the Edge voice")
    if not audio:
        audio = _speak_edge(text)
    with _lock:
        _cache[text] = audio
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return audio


def _speak_edge(text: str) -> bytes:
    """MP3 audio of one piece from the Edge voice service."""
    # A call to the voice service now and then never answers, so each try gets a short time (about 3 s covers a
    # 600-character piece) and a hung one is tried again at once instead of waiting half a minute.
    limit = 4 + len(text) / 100
    error = None
    for attempt in range(2):
        try:
            audio = asyncio.run(asyncio.wait_for(_speak(spoken_form(text)), timeout=limit))
            break
        except Exception as e:                          # no network, service changed, timeout, ...
            error = e
    else:
        raise TtsError(f"{type(error).__name__}: {error}")
    if not audio:
        raise TtsError("the voice service returned no audio")
    return audio
