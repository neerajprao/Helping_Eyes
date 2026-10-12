"""
Tests for server.py through its real HTTP and WebSocket interface. Apple Vision reads
a rendered label; the language model is not needed (every request below is
answered by the application itself):

    python tests/test_server.py        (or: python -m pytest tests/test_server.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import json
import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient

import server

client = TestClient(server.app)
SID = uuid.uuid4().hex


def label_jpeg(lines=("PARACETAMOL 500 mg TABLETS", "Take two tablets every six hours", "EXP: 04/2020")):
    page = np.full((720, 1280, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(page, text, (80, 200 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 4, cv2.LINE_AA)
    return cv2.imencode(".jpg", page)[1].tobytes()


def ask(text, sid=SID):
    res = client.post("/api/ask", json={"sid": sid, "question": text})
    assert res.status_code == 200
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def spoken(events):
    return " ".join(e["text"] for e in events if e["type"] == "sentence")


def test_health_and_page():
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["model"] and "llm_configured" in health
    assert isinstance(health["memory_mb"], float) and health["memory_mb"] > 10      # how much memory the server uses
    assert "memory_limit_mb" in health and health["cpus"] >= 1
    assert health["ocr"] in ("starting", "loading OCR", "ready")


def test_bad_inputs_are_rejected():
    bad = client.post("/api/capture", files={"image": ("x.jpg", b"not an image", "image/jpeg")}, data={"sid": SID})
    assert bad.status_code == 400
    assert client.post("/api/ask", json={"sid": "x", "question": "hi"}).status_code == 400   # invalid session id


def test_asking_before_capturing():
    events = ask("when does it expire", sid=uuid.uuid4().hex)
    assert "haven't captured" in spoken(events)
    assert events[-1]["type"] == "done"


def test_commands_are_routed_by_the_server():
    sid = uuid.uuid4().hex
    assert {"type": "command", "name": "stop"} in ask("stop", sid)
    events = ask("book mode", sid)
    assert {"type": "command", "name": "book_on"} in events and "Book mode" in spoken(events)
    events = ask("normal mode", sid)
    assert {"type": "command", "name": "book_off"} in events and spoken(events) == "Normal mode."
    events = ask("next", sid)
    assert {"type": "command", "name": "new_capture"} in events and "next thing" in spoken(events)
    assert spoken(ask("repeat", sid)) == "Nothing to repeat yet."
    assert "What should I look up" in spoken(ask("look it up", sid))


def test_capture_then_answers_from_the_text():
    res = client.post("/api/capture", files={"image": ("c.jpg", label_jpeg(), "image/jpeg")}, data={"sid": SID})
    data = res.json()
    assert res.status_code == 200 and data["ok"], data
    assert "PARACETAMOL" in data["text"].upper()

    # read everything: the captured text itself, source "document", no model involved
    events = ask("read everything")
    assert all(e["source"] == "document" for e in events if e["type"] == "sentence")
    assert "PARACETAMOL" in spoken(events).upper()

    # an expiry question goes to the language model (with the date check computed by code); without a model it says so
    answer = spoken(ask("when does it expire?"))
    assert answer and "Yes, it has expired" not in answer, answer

    # repeat gives the same answer back
    assert spoken(ask("repeat")) == answer


def test_live_stream_normal_mode():
    sid = uuid.uuid4().hex
    with client.websocket_connect(f"/ws/live?sid={sid}") as ws:
        ws.send_bytes(label_jpeg())
        reply = ws.receive_json()
        assert reply["type"] == "guide" and reply["box"] is not None, reply
        assert reply["state"] in ("HOLD", "ADJUST", "COACH", "CAPTURING")
        ws.send_bytes(cv2.imencode(".jpg", np.full((360, 640, 3), 255, np.uint8))[1].tobytes())
        assert ws.receive_json()["state"] == "NO_TEXT"
        ws.send_text(json.dumps({"type": "capture_result", "ok": True}))
        ws.send_text(json.dumps({"type": "rearm"}))


def test_live_stream_book_mode_reports_page_turn_state():
    sid = uuid.uuid4().hex
    with client.websocket_connect(f"/ws/live?sid={sid}") as ws:
        ws.send_text(json.dumps({"type": "mode", "mode": "book"}))
        ws.send_bytes(label_jpeg())
        reply = ws.receive_json()
        assert reply["type"] == "book" and reply["state"] in ("TURNING", "SETTLING", "STEADY"), reply


def test_live_stream_rejects_a_bad_session_id():
    try:
        with client.websocket_connect("/ws/live?sid=x") as ws:
            ws.receive_json()
        raise AssertionError("expected the socket to be refused")
    except Exception as e:                           # noqa: BLE001
        assert "AssertionError" not in type(e).__name__


def test_book_page_endpoint_reads_a_page_and_resumes():
    sid = uuid.uuid4().hex
    page = label_jpeg(("Chapter One", "The river was quiet that morning.", "Nobody came down to the water."))
    first = client.post("/api/book/page", files={"image": ("p.jpg", page, "image/jpeg")},
                        data={"sid": sid, "event": "changed", "read_pos": 0, "reading": False, "thinking": False}).json()
    assert first["action"] == "new" and "river" in first["text"].lower(), first
    assert first["chunks"] and first["layout"]["lines"]
    assert all(0 <= v <= 1 for line in first["layout"]["lines"] for v in line["box"])
    # the same page again, while reading and the view only moved: keep reading
    again = client.post("/api/book/page", files={"image": ("p.jpg", page, "image/jpeg")},
                        data={"sid": sid, "event": "moved", "read_pos": 10, "reading": True, "thinking": False}).json()
    assert again["action"] == "keep", again
    # and the conversation is about this page now
    assert "haven't captured" not in spoken(ask("read everything", sid))


def test_the_page_and_its_files_are_always_revalidated_so_the_html_and_the_script_never_get_out_of_step():
    for path in ("/", "/static/app.js", "/static/style.css"):
        res = client.get(path)
        assert res.status_code == 200 and res.headers["cache-control"] == "no-cache", path
    assert 'id="shutter"' in client.get("/").text and "shutter" in client.get("/static/app.js").text


def test_tts_endpoint_returns_audio_and_reports_failure():
    original = server.commands.synthesize
    server.commands.synthesize = lambda text: b"\xff\xf3audio"
    try:
        res = client.post("/api/tts", json={"text": "Hello there."})
        assert res.status_code == 200 and res.headers["content-type"] == "audio/mpeg" and res.content == b"\xff\xf3audio"

        def failing(text):
            raise server.commands.TtsError("no network")
        server.commands.synthesize = failing
        assert client.post("/api/tts", json={"text": "Hello there."}).status_code == 502
    finally:
        server.commands.synthesize = original
    health = client.get("/api/health").json()
    assert health["tts"] is True and health["tts_engine"] in ("kokoro", "edge")


def test_tts_endpoint_is_rate_limited():
    original = server.commands.synthesize
    server.commands.synthesize = lambda text: b"x"
    server._tts_hits.clear()
    try:
        codes = [client.post("/api/tts", json={"text": "Hi."}).status_code for _ in range(server.TTS_RATE_LIMIT + 1)]
    finally:
        server.commands.synthesize = original
        server._tts_hits.clear()
    assert codes[:-1] == [200] * server.TTS_RATE_LIMIT and codes[-1] == 429


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
