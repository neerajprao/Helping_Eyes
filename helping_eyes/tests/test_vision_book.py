"""
Tests for vision.py part C (book reading mode) on synthetic pages, with the OCR replaced by
fakes where a test is about layout, not recognition (no camera, key or language model needed):

    python tests/test_vision_book.py        (or: python -m pytest tests/test_vision_book.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import cv2
import numpy as np

import vision
from vision import TextLine


def line(text, x1, y1, x2, y2):
    return TextLine(text, 0.9, (x1, y1, x2, y2))


# ---------------------------------------------------------------- printed page numbers
def test_parse_page_number_forms():
    parse = vision.parse_page_number
    assert parse("47") == ("47", None)
    assert parse("- 47 -") == ("47", None)
    assert parse("Page 47") == ("47", None)
    assert parse("xii") == ("xii", None)
    assert parse("47  THE SILENT RIVER") == ("47", "THE SILENT RIVER")
    assert parse("CHAPTER THREE  48") == ("48", "CHAPTER THREE")
    assert parse("THE END") == (None, "THE END")
    assert parse("") == (None, None)


# ---------------------------------------------------------------- header, footer, headings
def body_lines(first_y=150, count=8, step=40, x1=100, x2=900, prefix="body"):
    return [line(f"{prefix} {i}", x1, first_y + i * step, x2, first_y + i * step + 28) for i in range(count)]


def test_header_with_number_and_title_is_separated_from_the_body():
    header = line("47  THE SILENT RIVER", 100, 20, 900, 48)
    body = body_lines()
    kept, margins, info = vision.split_page_parts([header] + body)
    assert info.number == "47" and info.running_title == "THE SILENT RIVER"
    assert margins == [header] and header not in kept and len(kept) == len(body)


def test_footer_page_number_is_found_and_not_read():
    body = body_lines()
    footer = line("- 48 -", 450, 560 + 80, 550, 560 + 108)
    kept, margins, info = vision.split_page_parts(body + [footer])
    assert info.number == "48" and margins == [footer] and footer not in kept


def test_a_page_without_header_or_footer_is_left_alone():
    body = body_lines()
    kept, margins, info = vision.split_page_parts(body)
    assert kept == body and margins == [] and info.number is None and info.running_title is None


def test_chapter_headings_are_recorded():
    body = body_lines()
    body[0] = line("CHAPTER THREE", 100, 150, 500, 178)
    _, _, info = vision.split_page_parts([line("12", 450, 20, 500, 48)] + body)
    assert "CHAPTER THREE" in info.headings


def test_page_numbers_of_a_spread_are_inferred_from_the_facing_page():
    layout = vision.PageLayout(pages=[vision.PageInfo(number=None), vision.PageInfo(number="47")])
    assert layout.page_numbers() == ["46", "47"] and layout.label() == "Pages 46 and 47"
    layout = vision.PageLayout(pages=[vision.PageInfo(number="46"), vision.PageInfo(number=None)])
    assert layout.label() == "Pages 46 and 47"
    assert vision.PageLayout(pages=[vision.PageInfo(number="9")]).label() == "Page 9"
    assert vision.PageLayout(pages=[vision.PageInfo()]).label() == ""             # no number printed: none announced


# ---------------------------------------------------------------- reading order (XY-cut)
def test_two_columns_are_read_left_before_right():
    lines = []
    for i in range(5):                                       # input is in row order: L1 R1 L2 R2 ...
        lines.append(line(f"L{i}", 100, 150 + i * 40, 400, 178 + i * 40))
        lines.append(line(f"R{i}", 600, 150 + i * 40, 900, 178 + i * 40))
    columns = []
    paragraphs = vision.xy_cut(lines, columns)
    order = [l.text for p in paragraphs for l in p]
    assert order == [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]
    assert len(columns) == 2 and columns[0][2] < columns[1][0]


def test_paragraphs_are_split_at_larger_gaps():
    first = [line(f"a{i}", 100, 100 + i * 40, 900, 128 + i * 40) for i in range(3)]
    second = [line(f"b{i}", 100, 100 + 3 * 40 + 60 + i * 40, 900, 128 + 3 * 40 + 60 + i * 40) for i in range(3)]
    paragraphs = vision.xy_cut(first + second)
    assert [[l.text for l in p] for p in paragraphs] == [["a0", "a1", "a2"], ["b0", "b1", "b2"]]


def test_indented_first_lines_start_paragraphs_in_novel_style():
    lines = [line(f"t{i}", 130 if i in (0, 3) else 100, 100 + i * 40, 900, 128 + i * 40) for i in range(6)]
    paragraphs = vision.xy_cut(lines)
    assert [[l.text for l in p] for p in paragraphs] == [["t0", "t1", "t2"], ["t3", "t4", "t5"]]


def test_a_heading_is_its_own_block_above_the_text():
    heading = line("Chapter One", 100, 100, 500, 150)
    text = body_lines(first_y=230, count=4)
    paragraphs = vision.xy_cut(text + [heading])
    assert [l.text for l in paragraphs[0]] == ["Chapter One"] and len(paragraphs) == 2


# ---------------------------------------------------------------- a whole page with a fake OCR
def read_with_fake_ocr(lines, frame=None):
    original = vision.recognize_text
    vision.recognize_text = lambda image, word_boxes=False: list(lines)
    try:
        return vision.read_page(frame if frame is not None else np.zeros((720, 1280, 3), np.uint8), split=False)
    finally:
        vision.recognize_text = original


def test_read_page_gives_ordered_text_page_facts_and_positions():
    lines = [line("47  THE SILENT RIVER", 100, 20, 900, 48)]
    lines += [line("The river was quiet that exam-", 100, 150, 900, 178),
              line("ple of a morning. Nobody came.", 100, 190, 900, 218),
              line("Second paragraph starts here.", 100, 290, 900, 318),
              line("And continues on this line.", 100, 330, 900, 358)]
    text, layout = read_with_fake_ocr(lines)
    assert layout.label() == "Page 47" and layout.running_title() == "THE SILENT RIVER"
    assert "47" not in text and "SILENT" not in text                       # announced, not read
    assert "example of a morning" in text                                  # hyphenated word re-joined
    assert text.index("The river") < text.index("Second paragraph")
    assert len(layout.paragraphs) == 2 and len(layout.blocks) == 2
    first = layout.paragraphs[0]
    assert text[first[0]:first[1]].startswith("The river") and text[first[0]:first[1]].endswith("Nobody came.")
    # every printed line knows its place in the page text and on the screen
    assert len(layout.lines) == 4 and all(text[s.start:s.end].strip() for s in layout.lines)
    assert layout.margins == [lines[0].box]
    assert "Page 47." in layout.summary() and "THE SILENT RIVER" in layout.summary()


def test_read_page_with_nothing_readable():
    text, layout = read_with_fake_ocr([])
    assert text == "" and layout.label() == "" and layout.lines == []


# ---------------------------------------------------------------- the spine of an open book
def spread(width=1920, height=1080, shadow=True, gap=0):
    """Two pages of text on a dark desk, spine at x = width // 2. gap = blank margin on each side of the spine."""
    image = np.full((height, width, 3), 55, np.uint8)
    x1, x2, y1, y2 = 150, width - 150, 90, height - 90
    image[y1:y2, x1:x2] = 238
    mid = width // 2
    text = "the river was quiet that morning"
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    for text_x in (mid - gap - 40 - text_w, mid + gap + 40):                  # equal lines: the inner margins match
        for row in range(14):
            cv2.putText(image, text, (text_x, y1 + 70 + row * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    if shadow:
        xs = np.arange(width, dtype=np.float32)
        valley = 1.0 - 0.45 * np.exp(-((xs - mid) / 40.0) ** 2)
        image[y1:y2, x1:x2] = (image[y1:y2, x1:x2] * valley[x1:x2].reshape(1, -1, 1)).astype(np.uint8)
    return image, mid


def test_spine_found_from_its_shadow():
    image, mid = spread(shadow=True)
    assert abs(vision.split_spread(image) - mid) <= 25


def test_spine_found_from_the_blank_strip_when_there_is_no_shadow():
    image, mid = spread(shadow=False, gap=70)
    x = vision.split_spread(image)
    assert x is not None and mid - 110 <= x <= mid + 110              # inside the blank strip between the two pages


def test_a_single_portrait_page_is_not_split():
    image = np.full((1080, 1920, 3), 55, np.uint8)
    image[40:1040, 560:1360] = 238
    for row in range(16):
        cv2.putText(image, "the river was quiet that morning", (620, 120 + row * 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    assert vision.split_spread(image) is None


def sheet(columns, gap_px, width=1920, height=1080):
    """One landscape sheet of paper on a dark desk, with `columns` columns of text and gap_px between them."""
    image = np.full((height, width, 3), 55, np.uint8)
    x1, x2, y1, y2 = 150, width - 150, 90, height - 90
    image[y1:y2, x1:x2] = 238
    col_w = (x2 - x1 - 120 - gap_px * (columns - 1)) // columns
    text = "the river was quiet that morning and nobody came"
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    for c in range(columns):
        cx = x1 + 60 + c * (col_w + gap_px)
        for row in range(14):
            cv2.putText(image, text[:int(len(text) * min(1.0, col_w / text_w))], (cx, y1 + 70 + row * 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
    return image


def test_a_single_wide_sheet_of_text_gets_no_spine_line():
    assert vision.split_spread(sheet(columns=1, gap_px=0)) is None


def test_two_columns_on_one_sheet_are_not_mistaken_for_a_spine():
    assert vision.split_spread(sheet(columns=2, gap_px=40)) is None      # a normal gap between columns


def test_an_empty_or_blank_sheet_gets_no_spine_line():
    image = np.full((1080, 1920, 3), 55, np.uint8)
    image[90:990, 150:1770] = 238
    assert vision.split_spread(image) is None


def test_a_spread_with_text_only_on_one_side_gets_no_spine_line():
    image, mid = spread(shadow=True)
    image[:, mid + 60:] = np.where(image[:, mid + 60:] < 100, image[:, mid + 60:], 230)    # wipe the text on the right page
    image[:, mid + 60:][image[:, mid + 60:] < 200] = 230
    assert vision.split_spread(image) is None


# ---------------------------------------------------------------- the book's border, and its centre as the spine
def paper(x1=150, x2=1770, y1=90, y2=990):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def test_with_the_border_known_the_evidence_comes_first_then_the_centre():
    shadowed, mid = spread(shadow=True)
    plain, _ = spread(shadow=False)                                    # no shadow, narrow inner margins: no evidence
    border = paper(x2=1790)                                            # its centre is x = 970, the real spine 960
    assert vision.split_spread(plain) is None
    assert abs(vision.find_spine(shadowed, border) - mid) <= 25        # the shadow is more exact than the centre
    assert vision.find_spine(plain, border) == 970                     # no evidence: the centre of the border


def test_a_border_taller_than_wide_is_a_single_page_and_gets_no_spine():
    image = np.full((1080, 1920, 3), 55, np.uint8)
    assert vision.find_spine(image, paper(x1=560, x2=1360, y1=40, y2=1040)) is None


def test_the_spine_line_runs_between_the_borders_top_and_bottom_sides_and_leans_with_it():
    tilted = np.array([[100, 100], [1100, 140], [1100, 840], [100, 800]], np.float32)
    top, bottom = vision.spine_line(600, tilted, 1080)
    assert top == (600.0, 120.0) and bottom == (600.0, 820.0)
    assert vision.spine_line(600, None, 1080) == [(600.0, 0.0), (600.0, 1079.0)]


def test_a_spread_on_a_light_patterned_cloth_gets_a_border_and_is_split_at_its_centre():
    image, mid = spread(shadow=False)                                  # no evidence of the spine at all
    cloth = np.full(image.shape, (150, 205, 235), np.uint8)            # a light, checked bedspread under the book
    cloth[(np.indices(image.shape[:2]).sum(axis=0) // 60) % 2 == 0] = (120, 185, 225)
    book = np.zeros(image.shape[:2], bool)
    book[90:990, 150:1770] = True
    image[~book] = cloth[~book]
    assert vision.split_spread(image) is None                          # brightness alone finds no spread here
    text, layout = vision.read_page(image, split=True, enhance=True)
    assert layout.book_quad is not None and layout.spine is not None and len(layout.pages) == 2
    xs = [p[0] for p in layout.book_quad]
    assert min(xs) < 250 and max(xs) > 1670                            # the border is round the book, not the cloth
    assert all(abs(x - mid) <= 40 for x, _ in layout.spine)
    shown = vision.layout_json(layout, image.shape[1], image.shape[0])
    assert len(shown["book"]) == 4 and len(shown["spine"]) == 2 and "river" in text


# ---------------------------------------------------------------- page turns
def page_a():
    image = np.full((360, 640, 3), 240, np.uint8)
    for row in range(14):                                    # full-width lines
        cv2.putText(image, "the river was quiet that morning and nobody", (30, 40 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return image


def page_b():
    image = np.full((360, 640, 3), 240, np.uint8)
    for col_x in (30, 340):                                  # two short columns, a different layout
        for row in range(7):
            cv2.putText(image, "rain began early", (col_x, 60 + row * 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return image


class Feed:
    """Feeds frames to a PageTurnDetector with a fake clock and collects the events."""

    def __init__(self):
        self.detector = vision.PageTurnDetector()
        self.t = 100.0
        self.events = []

    def show(self, frame, seconds):
        for _ in range(int(seconds / 0.1)):
            self.t += 0.1
            event = self.detector.update(frame, now=self.t)
            if event:
                self.events.append(event)


def test_a_page_that_shifts_slightly_is_not_a_new_page():
    feed = Feed()
    feed.show(page_a(), 3)
    assert feed.events == ["changed"]
    shifted = np.roll(np.roll(page_a(), 9, axis=1), 6, axis=0)       # the book slid a little (vibration, a nudge)
    feed.show(page_a(), 1)
    feed.show(shifted, 3)
    feed.show(page_a(), 1)
    feed.show(shifted, 4)
    assert "changed" not in feed.events[1:]                          # never mistaken for a new page


def test_first_steady_page_is_read_once():
    feed = Feed()
    feed.show(page_a(), 3)
    assert feed.events == ["changed"] and feed.detector.state == "STEADY"
    feed.show(page_a(), 4)
    assert feed.events == ["changed"]                        # a page that stays still is not read again


def test_nothing_is_read_before_the_page_has_settled():
    feed = Feed()
    feed.show(page_a(), 0.6)
    assert feed.events == [] and feed.detector.state == "SETTLING"


def test_a_hand_passing_over_the_page_is_a_moved_event_not_a_new_page():
    feed = Feed()
    feed.show(page_a(), 3)
    dark = np.zeros((360, 640, 3), np.uint8)
    for _ in range(8):                                       # something moves across the view: every frame differs
        feed.show(dark, 0.1)
        feed.show(page_a(), 0.1)
    assert feed.detector.state == "TURNING"
    feed.show(page_a(), 3)
    assert feed.events == ["changed", "moved"]


def test_a_single_disturbed_frame_does_not_restart_the_hold_still_timer():
    feed = Feed()
    feed.show(page_a(), 0.5)
    feed.show(np.zeros((360, 640, 3), np.uint8), 0.1)       # one bad frame (the median of recent frames ignores it)
    feed.show(page_a(), 3)
    assert feed.events == ["changed"]


def test_turning_to_a_page_with_a_different_layout_is_a_new_page():
    feed = Feed()
    feed.show(page_a(), 3)
    feed.show(page_b(), 4)
    assert feed.events == ["changed", "changed"]


def test_the_noise_floor_is_learned_so_a_noisy_camera_still_settles():
    feed = Feed()
    rng = np.random.default_rng(0)
    for _ in range(40):
        noisy = np.clip(page_a().astype(np.int16) + rng.integers(-6, 7, page_a().shape), 0, 255).astype(np.uint8)
        feed.show(noisy, 0.1)
    assert feed.events == ["changed"] and feed.detector.state == "STEADY"
    assert 0.0 < feed.detector.noise < 2.0                    # learned from the quiet frames, below its starting guess


# ---------------------------------------------------------------- carrying on after the view moved
WORDS = [f"w{i}x" for i in range(60)]


def words(a, b):
    return " ".join(WORDS[a:b])


def test_continue_from_the_word_reached_when_the_view_scrolls():
    old, new = words(0, 40), words(20, 60)                  # the camera moved down: 20 words overlap
    reached = old.index("w30x")
    assert vision.continue_from(old, reached, new) == new.index("w30x")


def test_continue_with_the_new_lines_after_what_was_read():
    old, new = words(0, 40), words(20, 60)
    assert vision.continue_from(old, len(old), new) == new.index("w40x")   # read to the end: the new words follow


def test_continue_when_the_view_is_unchanged():
    page = words(0, 40)
    assert vision.continue_from(page, page.index("w10x"), page) == page.index("w10x")
    assert vision.continue_from(page, len(page), page) == len(page)       # nothing new to read


def test_a_different_page_is_not_a_continuation():
    assert vision.continue_from(words(0, 30), 50, words(30, 60)) is None


def test_the_same_common_words_in_another_order_are_not_the_same_page():
    old = "the cat and the dog of the bird to the fox"
    new = " ".join(reversed(old.split()))                       # the same little words, but no run of three in a row
    assert vision.continue_from(old, 5, new) is None


def test_empty_views_are_never_a_continuation():
    assert vision.continue_from("", 0, words(0, 10)) is None
    assert vision.continue_from(words(0, 10), 0, "") is None


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
