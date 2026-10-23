"""
Tests for vision.py part B (image quality and page geometry) on synthetic pages (no camera, no OCR needed):

    python tests/test_vision_quality.py        (or: python -m pytest tests/test_vision_quality.py)

A text page is rendered, then degraded in a known way, so each layer can be
checked against the truth: perspective, curved lines, blur, glare, darkness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # the application modules

import cv2
import numpy as np

import vision

W, H = 900, 1200
WORDS = "the quick brown fox jumps over a lazy dog while reading books by night".split()


def make_page(seed: int = 0) -> np.ndarray:
    """A white page with ~28 lines of black text."""
    rng = np.random.default_rng(seed)
    page = np.full((H, W, 3), 245, np.uint8)
    for row in range(28):
        text = " ".join(rng.choice(WORDS, 8))
        cv2.putText(page, text, (60, 80 + row * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return page


def on_desk(page: np.ndarray, quad: np.ndarray, size=(1280, 960)) -> np.ndarray:
    """Place the page on a dark desk with its corners at `quad` (perspective)."""
    src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.warpPerspective(page, matrix, size, borderValue=(40, 45, 50))


def curve(page: np.ndarray, amplitude: float = 30) -> np.ndarray:
    """Bend the lines like a book toward its spine: text at column x moves down by a parabola."""
    xs = np.arange(W, dtype=np.float32)
    shift = amplitude * ((xs / W) ** 2)
    map_x = np.tile(xs, (H, 1))
    map_y = np.arange(H, dtype=np.float32).reshape(-1, 1) - shift.reshape(1, -1)   # content moves down
    return cv2.remap(page, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def test_quality_sharp_page_is_ok():
    report = vision.assess_quality(make_page())
    assert report.ok, report.summary()


def test_quality_detects_blur():
    report = vision.assess_quality(cv2.GaussianBlur(make_page(), (0, 0), 4))
    assert "blurry" in report.problems and "Hold still" in report.hint(), report.summary()


def test_quality_detects_glare_and_side():
    page = make_page()
    cv2.circle(page, (760, 300), 110, (255, 255, 255), -1)
    report = vision.assess_quality(page)
    assert "glare" in report.problems, report.summary()
    assert report.glare_at[0] > 0.62 and "on the right" in report.hint(), (report.glare_at, report.hint())


def test_quality_detects_dark():
    dark = (make_page() * 0.15).astype(np.uint8)
    assert "dark" in vision.assess_quality(dark).problems


def test_page_quad_and_warp_flatten_perspective():
    quad = np.array([[250, 120], [1010, 190], [1080, 850], [170, 820]])
    frame = on_desk(make_page(), quad)
    found = vision.find_page_quad(frame)
    assert found is not None, "page not found"
    assert np.abs(found - quad).max() < 25, (found, quad)
    warped, matrix = vision.warp_page(frame)
    assert matrix is not None
    # Flat again: text lines are horizontal, so their row profile is crisp
    assert vision._profile_sharpness(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)) > \
        1.5 * vision._profile_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def test_the_outline_is_the_item_with_the_text_not_a_brighter_floor():
    # An orange blister strip with dark print, lying at an angle on a light floor tile (as in the pharmacy photos):
    # the floor is the brightest region, so the brightness rule outlines the floor; the text-guided outline does not.
    frame = np.full((H, W, 3), 205, np.uint8)
    cv2.line(frame, (0, 600), (W, 640), (60, 60, 60), 10)
    strip = np.array([[480, 160], [800, 190], [770, 560], [450, 530]], np.float32)
    cv2.fillConvexPoly(frame, strip.astype(np.int32), (40, 140, 230))
    flat = np.full((370, 320, 3), 255, np.uint8)
    rows = []
    for i in range(6):
        cv2.putText(flat, "Tablets 15 mg", (20, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [320, 0], [320, 370], [0, 370]]), strip)
    ink = cv2.warpPerspective(255 - flat, matrix, (W, H))
    frame[ink > 128] = 20
    for i in range(6):                                          # each printed line's box, mapped onto the frame
        pts = cv2.perspectiveTransform(np.float32([[[20, 30 + i * 50], [230, 30 + i * 50], [230, 56 + i * 50], [20, 56 + i * 50]]]), matrix)[0]
        rows.append((int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())))
    found = vision.find_page_quad(frame, text_boxes=rows)
    assert found is not None, "item not found"
    assert np.abs(found - strip).max() < 20, (found.round(), strip)
    blind = vision.find_page_quad(frame)                        # the brightness rule alone ...
    assert blind is None or np.abs(blind - strip).max() > 100   # ... does not find the strip


def test_the_outline_ignores_text_on_things_in_the_background():
    frame = np.full((H, W, 3), 90, np.uint8)
    cv2.rectangle(frame, (500, 200), (760, 560), (235, 235, 235), -1)       # the item, in the middle
    rows = []
    for i in range(8):
        cv2.putText(frame, "Each ml contains", (520, 240 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        rows.append((520, 222 + i * 38, 740, 246 + i * 38))
    cv2.rectangle(frame, (20, 600), (300, 700), (220, 220, 220), -1)        # another box at the edge, with small print
    cv2.putText(frame, "other box", (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    rows.append((30, 638, 120, 654))
    found = vision.find_page_quad(frame, text_boxes=rows)
    assert found is not None and np.abs(found - np.float32([[500, 200], [760, 200], [760, 560], [500, 560]])).max() < 15, found


def test_flat_full_frame_page_is_left_alone():
    warped, matrix = vision.warp_page(make_page())
    assert matrix is None and warped.shape == (H, W, 3)


def test_dewarp_straightens_curved_lines_and_maps_boxes_back():
    original = make_page()
    bent = curve(original, amplitude=30)
    model = vision.estimate_dewarp(bent)
    assert model is not None, "curvature not found"
    flat = vision.dewarp(bent, model)
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    assert vision._profile_sharpness(gray(flat)) > 1.3 * vision._profile_sharpness(gray(bent))
    # The fitted shift matches the truth (30 * (x/W)^2, up to a constant) within 4 px
    xs = np.linspace(100, W - 100, 9)
    fitted = model.shift(xs) - model.shift(xs[0])
    truth = 30 * ((xs / W) ** 2 - (xs[0] / W) ** 2)
    assert np.abs(fitted - truth).max() < 4, (fitted, truth)
    # A box found on the flat image maps back onto the bent line
    x1, x2 = 700, 850
    y_flat = 80 + 10 * 38
    back = vision.unwarp_box((x1, y_flat - 20, x2, y_flat + 8), model)
    y_bent = y_flat + 30 * ((775 / W) ** 2)
    assert back[1] - 6 <= y_bent - 6 and back[3] + 6 >= y_bent, (back, y_bent)


def test_dewarp_leaves_a_flat_page_alone():
    assert vision.estimate_dewarp(make_page()) is None


def test_enhance_tone_removes_shadow():
    page = make_page()
    gradient = np.linspace(0.25, 1.0, W, dtype=np.float32).reshape(1, W, 1)     # shadow on the left
    shadowed = (page * gradient).astype(np.uint8)
    fixed = vision.enhance_tone(shadowed)
    left = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :150].mean())
    right = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, -150:].mean())
    assert abs(left(fixed) - right(fixed)) < 0.5 * abs(left(shadowed) - right(shadowed))


def test_capture_candidates_only_add_work_when_needed():
    assert [n for n, _ in vision.capture_candidates(make_page())] == ["original"]
    names = [n for n, _ in vision.capture_candidates((make_page() * 0.15).astype(np.uint8))]
    assert "tone" in names


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
