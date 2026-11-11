"""
Tests for commands.py (no camera, OCR or model needed):

    python tests/test_commands.py        (or: python -m pytest tests/test_commands.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

from commands import classify


def name(text, pending=False):
    return classify(text, offer_pending=pending).name


def test_simple_commands():
    assert name("stop") == "stop"
    assert name("Be quiet.") == "stop"
    assert name("repeat that") == "repeat"
    assert name("next page") == "new_capture"
    assert name("scan again") == "new_capture"
    assert name("Book mode") == "book_on"
    assert name("read my book") == "book_on"
    assert name("stop book mode") == "book_off"
    assert name("normal mode") == "book_off"
    assert name("") == "empty"


def test_read_all_is_not_a_model_question():
    for text in ("read everything", "Read it all", "what does it say", "please read it out loud"):
        assert name(text) == "read_all", text


def test_questions_go_to_the_model():
    cmd = classify("When does it expire?")
    assert cmd.name == "question" and cmd.query == "When does it expire?"
    assert name("read only the dosage") == "question"


def test_yes_and_no_only_answer_an_offer():
    assert name("yes", pending=True) == "yes"
    assert name("no thanks", pending=True) == "no"
    assert name("yes") != "yes"            # with no offer open, "yes" is not special
    assert name("no") == "question"
    # anything else drops the offer and is handled as a new request
    assert name("stop", pending=True) == "stop"
    assert name("what is the dose", pending=True) == "question"


def test_web_search_phrases():
    assert name("look it up") == "search_last"
    assert name("search online") == "search_last"
    cmd = classify("Look up tartrazine allergy")
    assert (cmd.name, cmd.query) == ("search_for", "tartrazine allergy")
    cmd = classify("search the web for ibuprofen dose")
    assert (cmd.name, cmd.query) == ("search_for", "ibuprofen dose")


def test_numbers_above_9999_are_spoken_digit_by_digit():
    from commands import spoken_form
    assert spoken_form("Serial number 123456 ok") == "Serial number 1 2 3 4 5 6 ok"
    assert spoken_form("Batch AB12345X") == "Batch AB1 2 3 4 5X"
    assert spoken_form("Total 12,345,678 units") == "Total 1 2 3 4 5 6 7 8 units"
    assert spoken_form("Take 9999 or 500 or 1,000 mg") == "Take 9999 or 500 or 1,000 mg"    # up to 9999 stays a normal number
    assert spoken_form("Phone 9876543210.") == "Phone 9 8 7 6 5 4 3 2 1 0."


def test_synthesize_reports_a_failing_voice_service_and_caches_good_audio():
    import commands
    calls = []

    async def fake_speak(text):
        calls.append(text)
        if "fail" in text:
            raise ConnectionError("no network")
        return b"mp3-bytes"

    original, engine = commands._speak, commands.TTS_ENGINE
    commands._speak = fake_speak
    commands.TTS_ENGINE = "edge"                                  # this test is about the Edge voice
    commands._cache.clear()
    try:
        assert commands.synthesize("Serial 123456.") == b"mp3-bytes"
        assert calls == ["Serial 1 2 3 4 5 6."]                  # digits spelled out before the voice
        assert commands.synthesize("Serial 123456.") == b"mp3-bytes" and len(calls) == 1      # cached
        try:
            commands.synthesize("please fail")
            raise AssertionError("expected TtsError")
        except commands.TtsError:
            pass
        try:
            commands.synthesize("   ")
            raise AssertionError("expected TtsError")
        except commands.TtsError:
            pass
    finally:
        commands._speak, commands.TTS_ENGINE = original, engine
        commands._cache.clear()


def test_any_wording_that_agrees_to_the_offer_is_a_yes_without_asking_the_model():
    def never(*_):
        raise AssertionError("the model should not be asked")
    for said in ("yes please look that up", "yeah go for it", "why not", "sounds good", "search the web for it",
                 "can you check that on the internet", "go ahead and google it"):
        assert classify(said, True, never).name == "yes", said
    for said in ("no thanks", "nope", "not really", "skip it", "never mind"):
        assert classify(said, True, never).name == "no", said


def test_unclear_wording_is_left_to_the_model():
    asked = []
    def judge(text, pending):
        asked.append((text, pending))
        return {"hmm i suppose that could help": "yes", "find me ibuprofen side effects online": "search: ibuprofen side effects",
                "look into the web": "search"}.get(text, "other")
    assert classify("hmm i suppose that could help", True, judge).name == "yes"
    cmd = classify("find me ibuprofen side effects online", False, judge)
    assert (cmd.name, cmd.query) == ("search_for", "ibuprofen side effects")
    assert classify("look into the web", False, judge).name == "search_last"
    assert classify("what is the dose", False, judge).name == "question"      # no web wording: the model is not asked
    assert ("what is the dose", False) not in asked
    assert classify("what does the web say", False, lambda *_: "other").name == "question"
    assert classify("find me ibuprofen online", False, lambda *_: 1 / 0).name == "question"   # a failing model changes nothing


def _with_voices(kokoro, edge, body):
    """Run body with fake Kokoro and Edge voices (and Kokoro counted as set up)."""
    import commands
    saved = (commands.TTS_ENGINE, commands._speak, commands._speak_kokoro, commands.kokoro_ready, commands._kokoro_failed)
    commands.TTS_ENGINE, commands._speak, commands._speak_kokoro = "kokoro", edge, kokoro
    commands.kokoro_ready = lambda: commands.TTS_ENGINE == "kokoro" and not commands._kokoro_failed
    commands._kokoro_failed = False
    commands._cache.clear()
    try:
        body()
    finally:
        commands.TTS_ENGINE, commands._speak, commands._speak_kokoro, commands.kokoro_ready, commands._kokoro_failed = saved
        commands._cache.clear()


def test_the_local_voice_speaks_first_and_the_edge_voice_is_the_backup():
    import commands
    calls = {"kokoro": [], "edge": []}

    def kokoro(text):
        calls["kokoro"].append(text)
        if "boom" in text:
            raise RuntimeError("broken")
        return b"kokoro-mp3"

    async def edge(text):
        calls["edge"].append(text)
        return b"edge-mp3"

    def body():
        assert commands.synthesize("Hello there.") == b"kokoro-mp3" and calls["edge"] == []
        assert commands.synthesize("Hello there.") == b"kokoro-mp3" and calls["kokoro"] == ["Hello there."]      # cached
        assert commands.synthesize("boom now") == b"edge-mp3"                  # Kokoro broke: Edge speaks this one ...
        assert commands._kokoro_failed is True
        assert commands.synthesize("Another one.") == b"edge-mp3"             # ... and Kokoro is not tried again this run
        assert calls["kokoro"] == ["Hello there.", "boom now"]
    _with_voices(kokoro, edge, body)


def test_the_local_voice_skips_audio_nobody_wants_and_makes_a_repeated_request_only_once():
    import threading
    import time
    import commands
    calls = []

    def kokoro(text):
        calls.append(text)
        time.sleep(0.2)
        return b"k-" + text.encode()

    async def edge(text):
        raise AssertionError("the Edge voice must not be used")

    def body():
        # the browser moved on while this request waited: nothing is made, and nothing falls back to Edge
        try:
            commands.synthesize("Nobody wants this.", stale=lambda: True)
            raise AssertionError("TtsCancelled expected")
        except commands.TtsCancelled:
            pass
        assert calls == []
        # a request and its second copy (the browser's hedge) at the same time: the voice works once
        results = []
        threads = [threading.Thread(target=lambda: results.append(commands.synthesize("Twice asked."))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == [b"k-Twice asked."] * 2 and calls == ["Twice asked."]
    _with_voices(kokoro, edge, body)


def test_the_phonemizer_word_count_warnings_are_not_logged():
    import logging
    import commands      # noqa: F401  (adds the filter)
    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        logging.getLogger("phonemizer").warning("words count mismatch on %s%% of the lines (%s/%s)", 200.0, 2, 1)
        logging.getLogger("phonemizer").warning("something else")
    finally:
        root.removeHandler(handler)
    assert seen == ["something else"]


def test_a_long_text_is_spoken_in_small_pieces_by_the_local_voice():
    import commands
    seen = []

    def kokoro(text):
        seen.append(text)
        return b"k"

    def body():
        text = " ".join(f"Sentence number {i} is here." for i in range(40))      # about 1200 characters
        assert commands.synthesize(text) == b"k" * len(seen) and len(seen) > 3
        assert all(len(t) <= commands.KOKORO_PIECE for t in seen)
        assert "".join(seen).replace(" ", "") == text.replace(" ", "")          # nothing is dropped
    _with_voices(kokoro, None, body)


def test_the_voice_in_use_is_reported():
    import commands
    saved = (commands.TTS_ENABLED, commands.kokoro_ready)
    try:
        commands.TTS_ENABLED, commands.kokoro_ready = True, lambda: True
        assert commands.engine() == "kokoro"
        commands.kokoro_ready = lambda: False
        assert commands.engine() == "edge"
        commands.TTS_ENABLED = False
        assert commands.engine() is None
    finally:
        commands.TTS_ENABLED, commands.kokoro_ready = saved


def test_the_local_voice_audio_is_encoded_as_mp3():
    import numpy as np
    import commands
    audio = commands._mp3(np.sin(np.linspace(0, 400, 24000)).astype(np.float32) * 0.3, 24000)      # one second of tone
    assert len(audio) > 1000 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0       # an MPEG frame header


def test_pieces_of_a_given_size_lose_nothing():
    import commands
    text = "alpha beta gamma delta " * 60
    pieces = commands._pieces(text.strip(), 100)
    assert all(len(p) <= 100 for p in pieces) and " ".join(pieces).split() == text.split()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for test_name, fn in tests:
        try:
            fn()
            print(f"PASS  {test_name}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"FAIL  {test_name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
