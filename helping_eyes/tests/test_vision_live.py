"""
Tests for vision.py part D (live guidance): the guidance state machine and the book-page decisions, with
the OCR replaced by fakes (no camera, OCR or model needed):

    python tests/test_vision_live.py        (or: python -m pytest tests/test_vision_live.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import numpy as np

import vision
from vision import PageInfo, PageLayout, QualityReport, TextLine
from assistant import DocAssistant

W, H = 1280, 720
FRAME = np.zeros((H, W, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def guide_with(box_lines, problems=(), clock=None, confidence=0.9):
    """A LiveGuide whose detector always 'sees' box_lines = (box, [line boxes]) or nothing."""
    clock = clock or Clock()
    state = {"seen": box_lines, "problems": list(problems)}

    def detect(frame, min_chars):
        if state["seen"] is None:
            return None, []
        box, line_boxes = state["seen"]
        return box, [TextLine("text", confidence, b) for b in line_boxes]

    def assess(frame, box=None):
        return QualityReport(problems=state["problems"])

    return vision.LiveGuide(detect=detect, assess=assess, clock=clock), clock, state


CENTRED = ((300, 150, 980, 560), [(300, 150 + i * 40, 980, 175 + i * 40) for i in range(8)])


def test_no_text():
    guide, _, _ = guide_with(None)
    reply = guide.update(FRAME)
    assert reply["state"] == "NO_TEXT" and not reply["capture"] and reply["box"] is None


def test_hold_still_then_capture_once():
    guide, clock, _ = guide_with(CENTRED)
    first = guide.update(FRAME)
    assert first["state"] == "HOLD" and not first["capture"] and first["label"].startswith("HOLD")
    clock.advance(0.3)
    assert guide.update(FRAME)["state"] == "HOLD"
    clock.advance(0.3)
    reply = guide.update(FRAME)
    assert reply["state"] == "CAPTURING" and reply["capture"]
    # the hold restarts, so a second capture is not requested straight away
    assert not guide.update(FRAME)["capture"]
    guide.capture_result(True)
    assert guide.update(FRAME)["state"] == "CAPTURED"


def test_new_item_needs_the_old_one_to_leave_the_view():
    guide, clock, state = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    assert guide.update(FRAME)["capture"]
    guide.capture_result(True)
    clock.advance(1)
    assert guide.update(FRAME)["state"] == "CAPTURED"
    state["seen"] = None
    clock.advance(2)                                    # nothing in view for longer than REARM_AFTER
    assert guide.update(FRAME)["state"] == "NO_TEXT"
    state["seen"] = CENTRED
    assert guide.update(FRAME)["state"] == "HOLD"


def test_failed_capture_is_retried():
    guide, clock, _ = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    assert guide.update(FRAME)["capture"]
    guide.capture_result(False)
    assert guide.update(FRAME)["state"] == "HOLD"       # still armed


def test_rearm_on_request():
    guide, clock, _ = guide_with(CENTRED)
    guide.update(FRAME)
    clock.advance(1)
    guide.update(FRAME)
    guide.capture_result(True)
    assert guide.update(FRAME)["state"] == "CAPTURED"
    guide.rearm()
    assert guide.update(FRAME)["state"] == "HOLD"


def test_small_text_the_reader_is_unsure_of_says_move_closer_without_repeating():
    tiny = ((300, 150, 600, 200), [(300, 150 + i * 10, 600, 158 + i * 10) for i in range(4)])
    guide, clock, _ = guide_with(tiny, confidence=0.5)
    assert guide.update(FRAME)["speak"] == "Move closer"
    clock.advance(0.5)
    assert guide.update(FRAME)["speak"] == ""           # same hint, inside the cooldown
    clock.advance(2)
    assert guide.update(FRAME)["speak"] == "Move closer"


def test_small_text_that_is_read_well_never_says_move_closer_and_is_captured():
    tiny = ((300, 150, 600, 200), [(300, 150 + i * 10, 600, 158 + i * 10) for i in range(4)])
    guide, clock, _ = guide_with(tiny, confidence=0.95)
    reply = guide.update(FRAME)
    assert reply["speak"] == "" and reply["state"] == "HOLD"          # no hint: it goes on to hold still and capture
    clock.advance(0.6)
    assert guide.update(FRAME)["capture"]


def sheet_on_desk(fraction):
    """A frame with a bright sheet (page, book or tablet) that covers `fraction` of it, on a dark desk."""
    import cv2
    frame = np.full((H, W, 3), 40, np.uint8)
    w, h = int(W * fraction ** 0.5), int(H * fraction ** 0.5)
    x, y = (W - w) // 2, (H - h) // 2
    frame[y:y + h, x:x + w] = 235
    for i in range(0, h - 30, 28):
        cv2.putText(frame, "lorem ipsum dolor sit amet", (x + 10, y + 20 + i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
    return frame


def test_a_sheet_that_fills_the_picture_never_gets_move_closer_even_if_the_reader_is_unsure():
    tiny = ((300, 150, 600, 200), [(300, 150 + i * 10, 600, 158 + i * 10) for i in range(4)])
    guide, clock, _ = guide_with(tiny, confidence=0.5)                       # small text, not read with confidence ...
    assert guide.update(sheet_on_desk(0.9))["speak"] == ""                   # ... but the sheet fills the picture: enough
    clock.advance(3)
    assert guide.update(sheet_on_desk(0.5))["speak"] == "Move closer"        # a small sheet on the desk: still asked to come closer


def test_page_fill_measures_how_much_of_the_picture_the_sheet_covers():
    for fraction in (0.2, 0.5, 0.8, 0.95):
        assert abs(vision.page_fill(sheet_on_desk(fraction)) - fraction) < 0.06
    assert vision.page_fill(np.zeros((H, W, 3), np.uint8)) == 0.0              # nothing there


def test_move_closer_depends_on_how_well_most_of_the_text_is_read():
    tiny = ((300, 150, 600, 200), [(300, 150 + i * 10, 600, 158 + i * 10) for i in range(4)])
    sure = [TextLine("a long line that was read with confidence", 1.0, tiny[1][0]), TextLine("x", 0.3, tiny[1][1])]
    unsure = [TextLine("a long line that was not read well", 0.4, tiny[1][0]), TextLine("ok", 1.0, tiny[1][1])]
    assert vision.LiveGuide._read_well(sure) and not vision.LiveGuide._read_well(unsure)      # letters count, not lines
    assert not vision.LiveGuide._read_well([])


def test_text_cut_off_at_an_edge_says_which_way_to_move():
    at_right_edge = ((900, 150, 1270, 560), [(900, 150 + i * 40, 1270, 175 + i * 40) for i in range(8)])
    guide, _, _ = guide_with(at_right_edge)
    assert guide.update(FRAME)["speak"] == "Move Left"
    at_left_edge = ((10, 150, 400, 560), [(10, 150 + i * 40, 400, 175 + i * 40) for i in range(8)])
    guide, _, _ = guide_with(at_left_edge)
    assert guide.update(FRAME)["speak"] == "Move Right"


def test_blur_is_coached_then_captured_anyway():
    guide, clock, _ = guide_with(CENTRED, problems=["blurry"])
    guide.update(FRAME)
    clock.advance(0.6)
    reply = guide.update(FRAME)
    assert reply["state"] == "COACH" and reply["speak"] == "Hold still, the image is blurry" and not reply["capture"]
    clock.advance(1)
    assert guide.update(FRAME)["speak"] == ""           # not repeated inside the cooldown
    clock.advance(vision.QUALITY_PATIENCE)
    assert guide.update(FRAME)["capture"]


def test_dark_room_is_announced_when_no_text_is_in_view():
    guide, _, _ = guide_with(None, problems=["dark"])
    assert guide.update(FRAME)["speak"] == "Low light"


# ---------------------------------------------------------------- book pages
PAGE_A = "The river was quiet that morning. Nobody came down to the water."
PAGE_B = "Chapter two began with rain. The streets filled slowly and the lamps came on."


def fake_page(text, number=None):
    layout = PageLayout(paragraphs=[(0, len(text))])
    if number:
        layout.pages.append(PageInfo(number=number))
    return text, layout


def reader_showing(pages):
    """A BookReader that 'reads' each queued page in turn."""
    queue = list(pages)
    original = vision.read_page
    vision.read_page = lambda frame, **kwargs: queue.pop(0)
    return vision.BookReader(), original


def test_book_new_page_then_same_page_then_next_page():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A), fake_page(PAGE_B)])
    doc = DocAssistant()
    try:
        first = reader.process(FRAME, "changed", 0, False, False, doc)
        assert first["action"] == "new" and first["from_pos"] == 0 and doc.document == PAGE_A
        assert first["chunks"][0]["start"] == 0 and first["chunks"][0]["text"] == PAGE_A
        # the same words again after a hand passed over, while reading: carry on silently
        again = reader.process(FRAME, "moved", 20, True, False, doc)
        assert again["action"] == "keep"
        # a different page
        nxt = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert nxt["action"] == "new" and doc.document == PAGE_B
    finally:
        vision.read_page = original


def test_a_vibrating_view_never_restarts_the_page():
    garbled = "The river was qiet thot mornin Nobdy cme dwn t th wtr"          # OCR of a blurred frame: too different to line up
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(garbled), fake_page("")])
    doc = DocAssistant()
    try:
        assert reader.process(FRAME, "changed", 0, False, False, doc)["action"] == "new"
        # the layout looks the same (a "moved" view) but the words are garbled: still this page
        assert reader.process(FRAME, "moved", 10, False, False, doc)["action"] == "keep"
        assert reader.process(FRAME, "moved", 10, False, False, doc)["action"] == "keep"      # even unreadable
        assert doc.document == PAGE_A
    finally:
        vision.read_page = original


def test_book_view_moved_continues_from_the_word_reached():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A)])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        pos = PAGE_A.index("Nobody")
        resume = reader.process(FRAME, "changed", pos, False, False, doc)
        assert resume["action"] == "resume" and resume["from_pos"] == pos
        assert resume["chunks"][0]["text"].startswith("Nobody")
    finally:
        vision.read_page = original


def test_book_page_fully_read_has_nothing_new():
    reader, original = reader_showing([fake_page(PAGE_A), fake_page(PAGE_A)])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        done = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert done["action"] == "nothing_new"
    finally:
        vision.read_page = original


def test_book_different_printed_numbers_mean_a_new_page_even_with_the_same_words():
    reader, original = reader_showing([fake_page(PAGE_A, "12"), fake_page(PAGE_A, "13")])
    doc = DocAssistant()
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        second = reader.process(FRAME, "changed", len(PAGE_A), False, False, doc)
        assert second["action"] == "new" and second["label"] == "Page 13"
    finally:
        vision.read_page = original


def test_book_page_number_and_title_are_introduced():
    reader, original = reader_showing([fake_page(PAGE_A, "47")])
    doc = DocAssistant()
    try:
        first = reader.process(FRAME, "changed", 0, False, False, doc)
        assert first["intro"] == "Page 47."
        assert doc.page_label == "Page 47"
    finally:
        vision.read_page = original


def test_book_no_text():
    reader, original = reader_showing([("", PageLayout())])
    try:
        assert reader.process(FRAME, "changed", 0, False, False, DocAssistant())["action"] == "no_text"
    finally:
        vision.read_page = original


# ---------------------------------------------------------------- paragraphs by the language model
ROWS = [  # a sentence that runs from the bottom of the left column to the top of the right one
    ("The river was quiet that morning and nobody", 50, 100, 400, 120, 0),
    ("came down to the water until the", 50, 130, 400, 150, 0),
    ("bells rang.", 50, 160, 200, 180, 0),            # [short]
    ("Later that day the rain began", 450, 100, 800, 120, 0),
    ("and the streets filled slowly.", 450, 130, 800, 150, 0),
]


def page_of_lines(rows=ROWS):
    """A PageLayout the way read_page builds it, with every line its own paragraph (as the heuristics wrongly did)."""
    text, lines, paragraphs = "", [], []
    for t, x1, y1, x2, y2, page in rows:
        if text:
            text += "\n\n"
        start = len(text)
        text += t
        words, at = [], 0
        for w in t.split():
            a = t.index(w, at)
            words.append((start + a, start + a + len(w), (x1 + a * 8, y1, x1 + (a + len(w)) * 8, y2)))
            at = a + len(w)
        lines.append(vision.LineSpan(start, len(text), (x1, y1, x2, y2), words, text=t, page=page))
        paragraphs.append((start, len(text)))
    return text, PageLayout(lines=lines, paragraphs=paragraphs)


class ArrangingDoc(DocAssistant):
    """A DocAssistant whose model answers with a fixed plan (None: the model is unavailable)."""
    def __init__(self, plan):
        super().__init__()
        self.plan, self.asked = plan, []

    def arrange_lines(self, numbered, count):
        self.asked.append(numbered)
        return self.plan


def test_regroup_joins_a_sentence_that_runs_over_lines_and_keeps_every_word_findable():
    text, layout = page_of_lines()
    new_text, new = vision.regroup(layout, [[0, 1, 2], [3, 4]], [])
    assert new_text == ("The river was quiet that morning and nobody came down to the water until the bells rang."
                        "\n\nLater that day the rain began and the streets filled slowly.")
    assert [new_text[a:b] for a, b in new.paragraphs] == new_text.split("\n\n")
    assert len(layout.paragraphs) == 5                                    # the given layout is not touched
    for span in new.lines:                                                # every line and word still points at its own text
        assert new_text[span.start:span.end] == span.text
        assert [new_text[a:b] for a, b, _ in span.words] == span.text.split()
    assert len(new.blocks) == 2 and new.blocks[0][0] == 50 and new.blocks[0][3] == 180


def test_regroup_leaves_out_skipped_lines_and_joins_hyphenated_words():
    rows = [("The lighthouse keeper was exam-", 50, 100, 400, 120, 0), ("ple enough for anyone.", 50, 130, 300, 150, 0),
            ("47", 200, 600, 220, 620, 0)]
    _, layout = page_of_lines(rows)
    new_text, new = vision.regroup(layout, [[0, 1]], [2])
    assert new_text == "The lighthouse keeper was example enough for anyone." and len(new.lines) == 2


def test_line_hints_mark_columns_gaps_indents_and_short_lines():
    _, layout = page_of_lines()
    hints = vision.line_hints(layout)
    assert "short" in hints[2] and "column" in hints[3] and hints[1] == ""


SIDEBAR_ROWS = [  # main text with a sidebar beside it, the rows level with each other (so the reading order alternates)
    ("The river was quiet that morning and nobody", 50, 100, 400, 120, 0),
    ("Suddenly remembering", 450, 100, 560, 115, 0),
    ("came down to the water until the", 50, 130, 400, 150, 0),
    ("something you pick up", 450, 128, 560, 143, 0),
    ("bells rang over the roofs of the town.", 50, 160, 400, 180, 0),
    ("the book you threw", 450, 156, 560, 171, 0),
]


def test_a_sidebar_beside_the_main_text_is_its_own_block_even_when_the_rows_alternate():
    _, layout = page_of_lines(SIDEBAR_ROWS)
    assert vision.line_blocks(layout) == [0, 1, 0, 1, 0, 1]
    shown = vision.numbered_lines(layout).splitlines()
    assert shown[0] == "Block 1:" and shown[1].startswith("0 ") and shown[2].startswith("2 ") and shown[4] == "Block 2:"
    assert shown[5].startswith("1 ") and "x78-100" in shown[5] and "h0.9" in shown[5]   # where it is, and how big its letters are


def test_regroup_follows_the_models_order_and_reads_a_sidebar_after_the_main_text():
    _, layout = page_of_lines(SIDEBAR_ROWS)
    text, new = vision.regroup(layout, [[0, 2, 4], [1, 3, 5]], [])
    assert text.split("\n\n")[0].startswith("The river") and text.split("\n\n")[1] == "Suddenly remembering something you pick up the book you threw"
    for span in new.lines:
        assert text[span.start:span.end] == span.text


def test_book_page_is_read_with_the_models_paragraphs():
    text, layout = page_of_lines()
    reader, original = reader_showing([(text, layout)])
    doc = ArrangingDoc(([[0, 1, 2], [3, 4]], []))
    try:
        first = reader.process(FRAME, "changed", 0, False, False, doc)
        assert first["action"] == "new" and len(first["chunks"]) == 2                  # two paragraphs, so one pause between
        assert first["chunks"][0]["text"].endswith("until the bells rang.")
        assert doc.document == first["text"] and "0 " in doc.asked[0] and "[column]" in doc.asked[0]
        assert [first["text"][l["start"]:l["end"]] for l in first["layout"]["lines"]][2] == "bells rang."
    finally:
        vision.read_page = original


def test_book_page_keeps_the_layout_paragraphs_when_the_model_is_not_available():
    text, layout = page_of_lines()
    reader, original = reader_showing([(text, layout)])
    try:
        first = reader.process(FRAME, "changed", 0, False, False, ArrangingDoc(None))
        assert len(first["chunks"]) == 5 and first["text"] == text
    finally:
        vision.read_page = original


def test_book_view_moved_still_continues_from_the_word_reached_with_the_models_paragraphs():
    text, layout = page_of_lines()
    text2, layout2 = page_of_lines()
    reader, original = reader_showing([(text, layout), (text2, layout2)])
    doc = ArrangingDoc(([[0, 1, 2], [3, 4]], []))
    try:
        reader.process(FRAME, "changed", 0, False, False, doc)
        pos = doc.document.index("water")
        resume = reader.process(FRAME, "changed", pos, False, False, doc)
        assert resume["action"] == "resume" and resume["text"][resume["from_pos"]:].startswith("water")
        assert len(doc.asked) == 1                                                   # the same page is not sent again
    finally:
        vision.read_page = original


def test_speech_chunks_split_long_paragraphs_at_sentence_ends():
    text = ("Sentence one is here. " * 40).strip()
    chunks = list(vision.speech_chunks(text, [(0, len(text))], 0, max_len=100))
    assert len(chunks) > 5
    assert "".join(t for _, t in chunks).replace(" ", "") == text.replace(" ", "")
    assert all(t.rstrip().endswith(".") for _, t in chunks)
    # starting part-way: nothing before the start is spoken
    assert list(vision.speech_chunks(text, [(0, len(text))], 30, max_len=1000))[0][0] == 30


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
