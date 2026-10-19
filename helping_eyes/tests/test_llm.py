"""
Tests for the language-model client in assistant.py against a fake OpenAI-compatible
server (no real model, key or internet needed). The client is meant to work with any
provider that speaks this API, so the fake stands in for all of them:

    python tests/test_llm.py        (or: python -m pytest tests/test_llm.py)
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import assistant as da
from assistant import DocAssistant

TEXT = "PARACETAMOL 500 mg TABLETS\nTake two tablets every six hours\nEXP: 04/2020"


class Fake(BaseHTTPRequestHandler):
    """A model server: replies as the test says (Fake.reply / Fake.status) and records what it was sent."""
    reply = ""
    status = 200
    requests = []          # (headers, body) of every call

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.requests.append((dict(self.headers), body))
        if Fake.status != 200:
            self.send_response(Fake.status)
            self.end_headers()
            return
        if not body.get("stream"):                       # e.g. the web search query
            data = json.dumps({"choices": [{"message": {"content": Fake.reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        pieces = [Fake.reply[i:i + 5] for i in range(0, len(Fake.reply), 5)]    # small chunks, like a real stream
        for piece in pieces:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"


def setup(reply="", status=200, key="test-key"):
    da.LLM_BASE_URL, da.LLM_API_KEY, da.LLM_MODEL = BASE, key, "some-model"
    Fake.reply, Fake.status, Fake.requests = reply, status, []
    doc = DocAssistant()
    doc.set_document(TEXT)
    return doc


def test_answer_streams_sentence_by_sentence():
    doc = setup("The dose is two tablets. Take them every six hours. Do not take more than eight.")
    assert list(doc.ask("how much should I take?")) == [
        "The dose is two tablets.", "Take them every six hours.", "Do not take more than eight."]
    headers, body = Fake.requests[0]
    assert headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "some-model" and body["stream"] is True
    assert body["messages"][0]["role"] == "system" and "PARACETAMOL" in body["messages"][0]["content"]
    assert body["messages"][-1] == {"role": "user", "content": "how much should I take?"}


def test_the_model_arranges_the_lines_of_a_page_into_paragraphs():
    doc = setup('Sure. {"paragraphs": [[0, 1, 2], [3, 4]], "skip": [5]}')
    assert doc.arrange_lines("0 a\n1 b\n2 c\n3 d\n4 e\n5 47", 6) == ([[0, 1, 2], [3, 4]], [5])
    body = Fake.requests[0][1]
    assert body["messages"][0]["content"] == da.ARRANGE_PROMPT and body["messages"][1]["content"].startswith("0 a")


def test_an_arrangement_that_would_lose_or_double_a_line_is_not_used():
    assert da.parse_arrangement('{"paragraphs": [[0, 1], [2]], "skip": []}', 3) == ([[0, 1], [2]], [])
    assert da.parse_arrangement('{"paragraphs": [[0, 1]], "skip": []}', 3) is None            # line 2 lost
    assert da.parse_arrangement('{"paragraphs": [[0, 1], [1, 2]], "skip": []}', 3) is None    # line 1 twice
    assert da.parse_arrangement('{"paragraphs": [[0], [2]], "skip": [1, 3]}', 3) is None      # no such line
    assert da.parse_arrangement('{"paragraphs": [[0], [], [1, 2]], "skip": []}', 3) is None    # empty paragraph
    assert da.parse_arrangement('{"paragraphs": [[0]], "skip": [1, 2, 3, 4, 5, 6, 7, 8]}', 9) is None   # most of the page dropped
    assert da.parse_arrangement("I cannot do that", 3) is None


def test_arranging_fails_quietly_without_a_key_or_when_the_model_fails():
    doc = setup('{"paragraphs": [[0, 1]], "skip": []}', key="")
    assert doc.arrange_lines("0 a\n1 b", 2) is None and Fake.requests == []                   # no key: nothing is sent
    doc = setup("", status=500)
    assert doc.arrange_lines("0 a\n1 b", 2) is None
    doc = setup("not json at all")
    assert doc.arrange_lines("0 a\n1 b", 2) is None


def test_conversation_is_remembered():
    doc = setup("Two tablets.")
    list(doc.ask("how many?"))
    Fake.reply = "Every six hours."
    list(doc.ask("and how often?"))
    roles = [m["role"] for m in Fake.requests[1][1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_not_in_text_marker_is_hidden_and_sets_the_flag():
    doc = setup("[NOT_IN_TEXT] The text doesn't say whether it is waterproof.")   # marker split across chunks
    assert list(doc.ask("is it waterproof?")) == ["The text doesn't say whether it is waterproof."]
    assert doc.last_not_in_text is True


def test_read_all_reply_is_passed_on_alone():
    doc = setup("READ_ALL")
    assert list(doc.ask("tell me everything on it please")) == [da.READ_ALL]


def test_expiry_questions_are_answered_by_the_model_from_the_computed_date_check():
    doc = setup("Yes, it ran out in April 2020, so I wouldn't use it.")
    answer = " ".join(doc.ask("has it expired?"))
    assert answer == "Yes, it ran out in April 2020, so I wouldn't use it."          # the model's own words, not a fixed sentence
    system = Fake.requests[0][1]["messages"][0]["content"]
    assert "HAS EXPIRED" in system and "April 2020" in system                         # the code did the date arithmetic
    asked = Fake.requests[0][1]["messages"][-1]["content"]
    assert asked.startswith("has it expired?") and "HAS EXPIRED" in asked              # and the verdict is repeated next to the question


def test_no_key_means_no_authorization_header():
    doc = setup("Two tablets.", key="")                  # e.g. a local server that needs no key
    list(doc.ask("how many?"))
    assert "Authorization" not in Fake.requests[0][0]


def test_provider_problems_are_spoken_plainly():
    for status, message in [(401, da.MSG_BAD_KEY), (403, da.MSG_BAD_KEY), (404, da.MSG_NO_MODEL), (429, da.MSG_BUSY), (500, da.MSG_FAILED)]:
        doc = setup(status=status)
        assert list(doc.ask("how many?")) == [message], status
    for status in (400, 401, 403):                       # no key set: say so, whatever the provider answers
        assert list(setup(status=status, key="").ask("how many?")) == [da.MSG_NO_KEY], status
    doc = setup()
    da.LLM_BASE_URL = "http://127.0.0.1:9"               # nothing is listening there
    assert list(doc.ask("how many?")) == [da.MSG_UNREACHABLE]


def test_the_providers_own_explanation_of_an_error_is_logged():
    import logging
    records = []

    class Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Catch()
    da.logger.addHandler(handler)
    original = Fake.do_POST

    def not_found(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"error": {"message": "models/old-model is not found for API version v1beta"}}'
        self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    Fake.do_POST = not_found
    try:
        assert list(setup().ask("how many?")) == [da.MSG_NO_MODEL]
    finally:
        Fake.do_POST = original
        da.logger.removeHandler(handler)
    assert any("HTTP 404" in m and "some-model" in m and "is not found for API version" in m for m in records), records


def test_web_lookup_searches_then_answers_from_the_results():
    searched = []
    original = da.web_search
    da.web_search = lambda query, max_results=5: searched.append(query) or [
        {"title": "Paracetamol", "body": "Adults may take 500 to 1000 mg every four to six hours."}]
    try:
        doc = setup("According to the web, adults may take up to 1000 mg per dose. Please confirm with a pharmacist.")
        Fake.requests = []
        answer = list(doc.ask_web("what is the maximum dose?"))
    finally:
        da.web_search = original
    assert searched and searched[0]                       # a search query was made by the model call
    assert answer[0].startswith("According to the web")
    prompt = Fake.requests[-1][1]["messages"][0]["content"]
    assert "500 to 1000 mg" in prompt and "what is the maximum dose?" in prompt
    assert "PARACETAMOL" not in prompt                    # the captured text itself is not sent for the lookup


def test_a_new_request_cancels_the_old_answer():
    doc = setup("One sentence. Two sentences. Three sentences. Four sentences.")
    stream = doc.ask("how many?")
    assert next(stream) == "One sentence."
    doc.cancel()
    assert list(stream) == []


def test_the_server_offers_a_web_lookup_and_runs_it_after_yes():
    from fastapi.testclient import TestClient
    import server
    client, sid = TestClient(server.app), "b" * 32
    setup("[NOT_IN_TEXT] The text doesn't say whether it is waterproof.")
    server._get_session(sid).doc.set_document(TEXT)

    def ask(text):
        res = client.post("/api/ask", json={"sid": sid, "question": text})
        return [json.loads(line) for line in res.text.splitlines() if line.strip()]

    events = ask("is it waterproof?")
    assert [e["type"] for e in events] == ["sentence", "offer_search", "done"]
    assert events[0]["text"] == "The text doesn't say whether it is waterproof."

    original = da.web_search
    da.web_search = lambda query, max_results=5: [{"title": "Page", "body": "It is not waterproof."}]
    try:
        Fake.reply = "According to the web, it is not waterproof."
        events = ask("yes")                               # "yes" only means something while the offer is open
    finally:
        da.web_search = original
    web = [e for e in events if e["type"] == "sentence" and e.get("source") == "web"]
    assert web and web[0]["text"].startswith("According to the web")
    assert [e["text"] for e in ask("yes") if e["type"] == "sentence"] != web[0]["text"]   # the offer is gone


def test_the_chosen_model_answers_and_unknown_choices_are_ignored():
    doc = setup("Two tablets.")
    saved = da.QWEN_BASE_URL, dict(da.QWEN_MODELS)
    da.QWEN_BASE_URL, da.QWEN_MODELS["qwen3b"] = BASE, "qwen-small"
    try:
        doc.set_provider("qwen3b")
        list(doc.ask("how many?"))
        headers, body = Fake.requests[-1]
        assert body["model"] == "qwen-small" and "Authorization" not in headers   # no key for the local model
        doc.set_provider("qwen7b")                                                 # no longer offered: ignored
        doc.set_provider("nonsense")
        assert doc.provider == "qwen3b"
        doc.set_provider("gemini")
        list(doc.ask("how many again?"))
        assert Fake.requests[-1][1]["model"] == "some-model"
        da.QWEN_BASE_URL = "http://127.0.0.1:9"                                    # Ollama not running
        doc.set_provider("qwen3b")
        assert list(doc.ask("and now?")) == [da.MSG_QWEN_DOWN]
    finally:
        da.QWEN_BASE_URL = saved[0]
        da.QWEN_MODELS.update(saved[1])


def test_expiry_dates_are_found_in_common_label_wordings():
    from datetime import date
    cases = {"Exp. Date 02/2027": date(2027, 2, 28), "Exp.Date:02/2027": date(2027, 2, 28),
             "Exp Dt. 02/27": date(2027, 2, 28), "EXP DATE 02/2027": date(2027, 2, 28),
             "Expiry Date: 15/04/2028": date(2028, 4, 15), "Exp. 02/2027": date(2027, 2, 28),
             "EXP: APR 2028": date(2028, 4, 30), "Use before FEB 2027": date(2027, 2, 28),
             "Best before 31-12-26": date(2026, 12, 31)}
    for printed, expected in cases.items():
        assert [d for _, d in da.expiry_dates(printed)] == [expected], printed
    assert da.expiry_dates("Mfg. Date 03/2025") == []
    assert "HAS EXPIRED" in da.expiry_checks("Exp. Date 02/2025", today=date(2026, 10, 19))


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
