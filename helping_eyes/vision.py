"""
Everything that works on the camera picture, in four parts (the OCR is Apple Vision; the rest is
classic OpenCV / NumPy with no models):

    PART A  Reading text (OCR)
            picture -> lines of text with their boxes; a quick "is there text?" check for
            live frames; an accurate read that tries corrected versions of a poor frame
    PART B  Image quality and page geometry
            is the picture good enough? (blur, glare, light) -> spoken coaching;
            flatten a tilted page, straighten curved lines, remove shadows
    PART C  Book reading mode
            when to read (page turns), where the pages are (spine), in what order to
            read (columns, paragraphs), and how to carry on when the view moves
    PART D  Live guidance
            the server side of the live camera: hints ("Move closer"), hold still,
            capture, page turns, and what to speak for a book page

Each part builds on the ones before it (D uses C, C uses B, and all use A). The browser
(web/app.js) only draws and speaks; the decisions are made here. All boxes sent to the
browser are normalised to 0-1.

Optional .env settings:
    VISION_ENHANCE = 0 turns the image-correction layer off (default on)
    BLUR_MIN       = sharpness below this is "blurry" (default 60, see assess_quality)
"""

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


##############################################################################
# PART A: READING TEXT (OCR)
#
# Apple Vision (the text recognition built into macOS: runs on this Mac, uses its neural hardware, needs no
# model download, no API key and sends no picture anywhere). The app therefore runs on a Mac.
#
#     recognize_text()      every line of text in an image, optionally with each word's box
#     find_text_region()    quick check of a live preview frame: is there text, and where?
#     read_text_enhanced()  an accurate read that copes with skew, shadows and glare
##############################################################################

@dataclass
class TextLine:
    text: str
    confidence: float
    box: Box
    # (start, end, box) of each word in `text`; filled when word_boxes=True
    words: List[Tuple[int, int, Box]] = field(default_factory=list)
    # The four corners (top left, top right, bottom right, bottom left) of the line as it really lies, slanted
    # when the text is: `box` is only the axis-aligned bounds of these. word_quads is parallel to `words`.
    quad: Optional[List[Tuple[float, float]]] = None
    word_quads: List[List[Tuple[float, float]]] = field(default_factory=list)


def _vision_box(rect, w: int, h: int) -> Box:
    """A Vision bounding box (fractions of the image, origin at the bottom left) as a pixel box."""
    x, y, bw, bh = rect.origin.x, rect.origin.y, rect.size.width, rect.size.height
    return (max(0, int(x * w)), max(0, int((1 - y - bh) * h)), min(w, int((x + bw) * w)), min(h, int((1 - y) * h)))


def _vision_quad(obs, w: int, h: int) -> List[Tuple[float, float]]:
    """The corners of a Vision rectangle observation (a line or a word) as pixel points, top left first."""
    return [(p.x * w, (1 - p.y) * h) for p in (obs.topLeft(), obs.topRight(), obs.bottomRight(), obs.bottomLeft())]


def recognize_text(image: np.ndarray, word_boxes: bool = False, fast: bool = False) -> List[TextLine]:
    """Find and read every line of text in a BGR image, top to bottom, with Apple Vision.
    word_boxes=True also records where each word is. fast=True is Vision's quicker, less careful mode;
    it is not used by the app because the accurate mode is already quick (about 0.1 s) and finds small text."""
    import cv2 as _cv2
    import Foundation
    import Vision

    h, w = image.shape[:2]
    ok, png = _cv2.imencode(".png", image)
    if not ok:
        return []
    data = Foundation.NSData.dataWithBytes_length_(png.tobytes(), len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                                 else Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(not fast)
    request.setRecognitionLanguages_(["en-US"])
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        logger.error(f"Apple Vision failed: {error}")
        return []

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        text = str(candidate.string())
        line = TextLine(text, float(candidate.confidence()), _vision_box(observation.boundingBox(), w, h),
                        quad=_vision_quad(observation, w, h))
        if word_boxes:
            for m in re.finditer(r"\S+", text):
                # Vision counts characters in UTF-16 units
                start = len(text[:m.start()].encode("utf-16-le")) // 2
                length = len(m.group().encode("utf-16-le")) // 2
                box, _ = candidate.boundingBoxForRange_error_(Foundation.NSMakeRange(start, length), None)
                if box is not None:
                    line.words.append((m.start(), m.end(), _vision_box(box.boundingBox(), w, h)))
                    line.word_quads.append(_vision_quad(box, w, h))
        lines.append(line)
    lines.sort(key=lambda l: (l.box[1], l.box[0]))
    return lines


def find_text_region(image: np.ndarray, min_chars: int = 8) -> Tuple[Optional[Box], List[TextLine]]:
    """
    Quick check for readable text. Returns (box around all text, lines),
    or (None, lines) when there are fewer than `min_chars` letters/digits.
    """
    lines = [l for l in recognize_text(image)
             if len(re.sub(r"[^0-9A-Za-z\u0080-￿]", "", l.text)) >= 2]
    chars = sum(len(re.sub(r"\s", "", l.text)) for l in lines)
    if chars < min_chars:
        return None, lines
    x1 = min(l.box[0] for l in lines)
    y1 = min(l.box[1] for l in lines)
    x2 = max(l.box[2] for l in lines)
    y2 = max(l.box[3] for l in lines)
    return (x1, y1, x2, y2), lines


def read_text_enhanced(image: np.ndarray, report=None) -> Tuple[str, str]:
    """
    Accurate read that copes with skew, shadows and glare: Part B offers
    extra versions of the image (flattened page, shadow-corrected, binarised)
    only when the frame looks like it needs them, and the version the OCR is
    most confident about wins (sum of confidence x characters, so a version
    that recovers more text also scores higher). The untouched frame keeps
    the win unless another version is clearly better.
    Returns (text, name of the winning version).
    """
    best_name, best_lines, best_score = "original", [], -1.0
    for name, candidate in capture_candidates(image, report):
        lines = [l for l in recognize_text(candidate) if l.text.strip()]
        score = sum(l.confidence * len(l.text.strip()) for l in lines)
        if name == "original":
            best_lines, best_score = lines, score * 1.05      # small bias towards the untouched frame
        elif score > best_score:
            best_name, best_lines, best_score = name, lines, score
    return "\n".join(l.text for l in best_lines), best_name


##############################################################################
# PART B: IMAGE QUALITY AND PAGE GEOMETRY
# Sits in front of the OCR.
#
#     assess_quality()     blur (variance of the Laplacian), glare blobs, exposure
#                          -> QualityReport.hint(): "Hold still", "Glare on the right..."
#     find_page_quad()     page outline (contour + approxPolyDP)
#     warp_page()          homography to a flat, top-down page
#     estimate_dewarp()    curved lines (a book bending into its spine) modelled as a
#                          vertical shift per column; dewarp() flattens them and
#                          unwarp_box() maps OCR boxes back onto the camera frame
#     enhance_tone()       shadow removal + CLAHE, only applied when the quality is low
#     binarize()           adaptive threshold (a last-resort variant for hard cases)
##############################################################################

ENABLED = os.getenv("VISION_ENHANCE", "1") != "0"
BLUR_MIN = float(os.getenv("BLUR_MIN", "60"))
GLARE_MIN_FRACTION = 0.01      # a saturated blob must cover this much of the image
DARK_MEAN = 50                 # mean brightness (0-255) below this is "too dark"
BRIGHT_MEAN = 205              # ... above this is "too bright"
_ANALYSIS_WIDTH = 640          # quality is measured at this width so it doesn't depend on camera size


# =====================================================================
# 1. Quality scoring
# =====================================================================
@dataclass
class QualityReport:
    sharpness: float = 0.0                       # variance of the Laplacian (higher = sharper)
    brightness: float = 0.0                      # mean grey level, 0-255
    contrast: float = 0.0                        # std of grey levels
    clipped_bright: float = 0.0                  # fraction of near-white pixels
    glare_fraction: float = 0.0                  # fraction of the image inside glare blobs
    glare_at: Optional[Tuple[float, float]] = None   # centre of the glare, 0-1 (x from the left, y from the top)
    problems: List[str] = field(default_factory=list)   # subset of: dark, bright, blurry, glare, flat

    @property
    def ok(self) -> bool:
        return not self.problems

    def hint(self) -> str:
        """One short spoken instruction for the most important problem ("" when fine)."""
        if "dark" in self.problems:
            return "Low light"
        if "glare" in self.problems and self.glare_at is not None:
            gx, gy = self.glare_at
            side = "on the right" if gx > 0.62 else "on the left" if gx < 0.38 else \
                   "at the bottom" if gy > 0.62 else "at the top" if gy < 0.38 else "in the middle"
            return f"Glare {side}. Tilt the page"
        if "blurry" in self.problems:
            return "Hold still, the image is blurry"
        if "bright" in self.problems:
            return "Too bright. Move away from the light"
        return ""

    def summary(self) -> str:
        return (f"sharp {self.sharpness:.0f}  light {self.brightness:.0f}  "
                f"glare {self.glare_fraction * 100:.1f}%" + (f"  [{', '.join(self.problems)}]" if self.problems else ""))


def _analysis_gray(image: np.ndarray, box: Optional[Box] = None) -> np.ndarray:
    """Grey image at a fixed width, optionally cropped to the text box."""
    if box is not None:
        x1, y1, x2, y2 = box
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size and min(crop.shape[:2]) >= 32:
            image = crop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = _ANALYSIS_WIDTH / gray.shape[1]
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def assess_quality(image: np.ndarray, text_box: Optional[Box] = None) -> QualityReport:
    """
    Blur, glare and exposure of a BGR frame. Pass the text box (when known) so
    sharpness is measured where the text is, not on the desk around it.

    Sharpness is the variance of the Laplacian: sharp print has strong
    edges (high variance), blur removes them. It also depends on how much text
    there is, so BLUR_MIN (default 60) may need adjusting for your camera.
    """
    full = _analysis_gray(image)
    text = _analysis_gray(image, text_box)
    report = QualityReport(
        sharpness=float(cv2.Laplacian(text, cv2.CV_64F).var()),
        brightness=float(full.mean()),
        contrast=float(full.std()),
        clipped_bright=float((full > 245).mean()),
    )

    # Glare: large blobs of fully saturated pixels (paper at 250+ is not glare unless
    # it forms a compact, saturated patch, so ignore small specks with an opening)
    saturated = (full >= 250).astype(np.uint8)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(saturated)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        report.glare_fraction = float(saturated.sum()) / saturated.size
        cx, cy = centroids[biggest]
        report.glare_at = (cx / saturated.shape[1], cy / saturated.shape[0])

    if report.brightness < DARK_MEAN:
        report.problems.append("dark")
    elif report.brightness > BRIGHT_MEAN and report.clipped_bright > 0.4:
        report.problems.append("bright")
    if report.glare_fraction >= GLARE_MIN_FRACTION:
        report.problems.append("glare")
    if report.sharpness < BLUR_MIN:
        report.problems.append("blurry")
    if report.contrast < 25:
        report.problems.append("flat")
    return report


# =====================================================================
# 2. Page detection and perspective correction
# =====================================================================
def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_page_quad(frame: np.ndarray, min_area: float = 0.25) -> Optional[np.ndarray]:
    """
    The page outline as 4 corners (top-left, top-right, bottom-right, bottom-left)
    in frame pixels, or None when there is no clear page: paper is the bright
    region, so Otsu threshold, close the text holes, take the biggest contour and
    simplify it with approxPolyDP.
    """
    h, w = frame.shape[:2]
    scale = 480 / max(h, w)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    size = max(3, int(min(small.shape[:2]) * 0.04)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_area * small.shape[0] * small.shape[1]:
        return None

    hull = cv2.convexHull(biggest)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        return None
    quad = _order_corners(quad)
    # The hull of a page must not be wildly non-rectangular (rules out a table edge, a hand ...)
    if cv2.contourArea(quad) < 0.8 * cv2.contourArea(hull):
        return None
    quad = quad / min(scale, 1.0)
    return quad


def page_fill(frame: np.ndarray) -> float:
    """
    How much of the picture the sheet fills, 0 to 1: a page, a book spread or a lit tablet screen. The paper is the
    bright region (Otsu), the print is closed away, and the big bright regions are added up. Rough on purpose, and
    cheap (a small copy of the picture): it only has to tell "the sheet fills the picture" from "the sheet is small".
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA) if w > 320 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = gray.shape[0] * gray.shape[1]
    area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 0.03 * total)
    return min(1.0, area / total)


def _needs_warp(quad: np.ndarray, frame_shape) -> bool:
    """A page that is already square-on and fills the frame doesn't need warping."""
    h, w = frame_shape[:2]
    margin = 0.02 * max(h, w)
    on_edge = sum(1 for x, y in quad if x < margin or y < margin or x > w - margin or y > h - margin)
    if on_edge >= 2:
        return False        # the page runs out of the frame: warping would cut text off
    tl, tr, br, bl = quad
    angles = [abs(np.degrees(np.arctan2(*(b - a)[::-1]))) for a, b in ((tl, tr), (bl, br))]
    tilt = max(min(a, 180 - a) for a in angles)                        # rotation of the top / bottom edges
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    keystone = max(abs(top - bottom) / max(top, bottom), abs(left - right) / max(left, right))
    return tilt > 2.0 or keystone > 0.04


def warp_page(frame: np.ndarray, quad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Flatten the page with a homography. Returns (image, matrix) where matrix maps
    frame -> image (None when nothing was done: no clear page, or already flat).
    """
    quad = find_page_quad(frame) if quad is None else quad
    if quad is None or not _needs_warp(quad, frame.shape):
        return frame, None
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 100 or height < 100:
        return frame, None
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE), matrix


def unproject_box(box: Box, matrix: Optional[np.ndarray]) -> Box:
    """Map a box from the warped image back to the original frame."""
    if matrix is None:
        return box
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    back = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))[0]
    return (int(back[:, 0].min()), int(back[:, 1].min()), int(back[:, 0].max()), int(back[:, 1].max()))


# =====================================================================
# 3. Curved-page dewarping
# =====================================================================
@dataclass
class DewarpModel:
    """shift(x) = how far the text lines have moved down at column x, relative to a flat part of the page."""
    coeffs: np.ndarray
    width: int

    def shift(self, x) -> np.ndarray:
        return np.polyval(self.coeffs, np.asarray(x, dtype=np.float64))


def _ink(gray: np.ndarray) -> np.ndarray:
    """Dark print on a bright page -> float 0/1 mask."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(15, (min(gray.shape) // 20) | 1)
    return (cv2.adaptiveThreshold(blurred, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
            ).astype(np.float32)


def _line_pitch(profile: np.ndarray) -> int:
    """Distance between text lines: the first strong peak of the profile's autocorrelation."""
    p = profile - profile.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    if ac[0] <= 0:
        return 0
    ac = ac / ac[0]
    lo, hi = 6, min(len(p) // 3, 200)
    if hi <= lo + 2:
        return 0
    window = ac[lo:hi]
    peaks = [i for i in range(1, len(window) - 1) if window[i] > window[i - 1] and window[i] >= window[i + 1] and window[i] > 0.15]
    return lo + peaks[0] if peaks else 0


def _best_lag(profile: np.ndarray, ref: np.ndarray, max_lag: int) -> Tuple[int, float]:
    """The vertical shift (+ = lines lower than the reference) that best aligns `profile` to `ref`."""
    a, b = profile - profile.mean(), ref - ref.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    best, best_score = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            score = float(np.dot(a[lag:], b[:len(b) - lag]))
        else:
            score = float(np.dot(a[:lag], b[-lag:]))
        if score > best_score:
            best, best_score = lag, score
    return best, best_score / norm


def estimate_dewarp(image: np.ndarray, strips: int = 16, degree: int = 3,
                    min_gain: float = 1.08, min_shift: float = 2.0) -> Optional[DewarpModel]:
    """
    Fit how the text lines bend across the page. Cut the page into vertical strips,
    take each strip's row profile (ink per row) and slide it against its neighbour
    (starting from the flattest strip and moving outwards, so a shift never exceeds
    half a line pitch); the accumulated shifts are fitted with a polynomial.
    Returns None when the page is already flat or the fit doesn't make the lines
    any crisper (the line structure of the whole page must sharpen by min_gain).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, 900 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    h, w = small.shape
    ink = _ink(small)
    edges = np.linspace(0, w, strips + 1).astype(int)
    profiles, weights = [], []
    for x0, x1 in zip(edges[:-1], edges[1:]):
        p = ink[:, x0:x1].sum(axis=1)
        p = cv2.GaussianBlur(p.reshape(-1, 1), (1, 0), 1.5).ravel()
        profiles.append(p)
        weights.append(float(p.var()))
    weights = np.array(weights)
    if weights.max() < 1e-3 or (weights > 0.25 * weights.max()).sum() < 4:
        return None                                   # too little text to model

    ref_i = int(np.argmax(weights))
    pitch = _line_pitch(profiles[ref_i]) or max(8, h // 30)
    max_lag = max(2, int(pitch * 0.45))

    shifts = np.zeros(strips)
    valid = weights > 0.25 * weights.max()
    for direction in (1, -1):
        prev = ref_i
        for i in range(ref_i + direction, strips if direction > 0 else -1, direction):
            if not valid[i]:
                shifts[i] = shifts[prev]
                continue
            lag, _ = _best_lag(profiles[i], profiles[prev], max_lag)
            shifts[i] = shifts[prev] + lag
            prev = i
    centres = (edges[:-1] + edges[1:]) / 2.0
    if not valid.any() or np.abs(shifts[valid]).max() < min_shift * scale:
        return None

    coeffs = np.polyfit(centres[valid], shifts[valid], min(degree, int(valid.sum()) - 1), w=np.sqrt(weights[valid]))
    small_model = DewarpModel(coeffs, w)

    # Keep it only if the lines really got crisper
    if _profile_sharpness(_apply_shift(small, small_model)) < min_gain * _profile_sharpness(small):
        return None
    # The fit lives in the downscaled image; re-express it in full-size pixels
    xs = np.linspace(0, gray.shape[1], 64)
    return DewarpModel(np.polyfit(xs, small_model.shift(xs * scale) / scale, degree), gray.shape[1])


def _profile_sharpness(gray: np.ndarray) -> float:
    """How crisply text lines stand out: variance of the ink-per-row profile (higher = straighter lines)."""
    profile = _ink(gray).sum(axis=1)
    return float(profile.var()) + 1e-9


def _apply_shift(image: np.ndarray, model: DewarpModel) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    map_x = np.tile(xs, (h, 1))
    map_y = np.arange(h, dtype=np.float32).reshape(-1, 1) + model.shift(xs).astype(np.float32).reshape(1, -1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def dewarp(image: np.ndarray, model: Optional[DewarpModel]) -> np.ndarray:
    """Straighten the text lines: dst(x, y) = src(x, y + shift(x))."""
    return image if model is None else _apply_shift(image, model)


def unwarp_box(box: Box, model: Optional[DewarpModel]) -> Box:
    """Map a box found on the dewarped image back onto the original frame."""
    if model is None:
        return box
    x1, y1, x2, y2 = box
    s1, s2 = float(model.shift(x1)), float(model.shift(x2))
    return (x1, int(round(y1 + min(s1, s2))), x2, int(round(y2 + max(s1, s2))))


# =====================================================================
# 3b. Slanted pages: measure the slant, straighten for the layout analysis, map everything back
# =====================================================================
SKEW_MIN = 1.5      # degrees: a page slanted less than this is read as it is
SKEW_MAX = 25.0


def estimate_skew(image: np.ndarray, max_angle: float = SKEW_MAX) -> float:
    """
    How many degrees to rotate the image (OpenCV's sign: positive turns it counter-clockwise) so that its lines of
    text run level; 0.0 when there is no clear slant. The right angle is the one at which the rows of ink are the
    sharpest (a level line is all ink on a few rows).
    """
    gray = _analysis_gray(image)
    scale = 500 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    if ink.mean() < 2:                       # next to no ink: nothing to measure
        return 0.0
    h, w = ink.shape
    center = (w / 2, h / 2)
    # only the middle of the picture counts, so the corners that rotation leaves empty do not matter
    mask = np.zeros_like(ink)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(min(h, w) / 2), 255, -1)

    def sharpness(angle: float) -> float:
        turned = cv2.warpAffine(ink, cv2.getRotationMatrix2D(center, angle, 1.0), (w, h))
        turned = cv2.bitwise_and(turned, mask)
        rows = turned.sum(axis=1).astype(np.float64)
        return float(np.sum(np.diff(rows) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=sharpness)
    fine = np.arange(best - 1.0, best + 1.01, 0.1)
    best = max(fine, key=sharpness)
    return float(best) if abs(best) >= SKEW_MIN and sharpness(best) > 1.15 * sharpness(0.0) else 0.0


def rotate_expand(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate by `angle` degrees on a canvas big enough to keep everything. Returns (image, 2x3 matrix original -> rotated)."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), matrix


Quad = List[Tuple[float, float]]


def box_quad(box: Box) -> Quad:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def quad_bounds(quad: Quad) -> Box:
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def map_quad(quad: Quad, point) -> Quad:
    """The quad with every corner passed through point(x, y) -> (x, y)."""
    return [point(x, y) for x, y in quad]


# =====================================================================
# 4. Adaptive preprocessing
# =====================================================================
def enhance_tone(image: np.ndarray, clahe_clip: float = 2.0) -> np.ndarray:
    """
    Shadow removal + CLAHE, geometry unchanged (so OCR boxes stay valid).
    The background (paper) brightness is estimated with a dilate + large blur at
    low resolution and divided out; CLAHE then evens the local contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    small = cv2.resize(light, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    size = max(3, int(min(small.shape[:2]) * 0.06)) | 1
    background = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    background = cv2.GaussianBlur(background, (0, 0), size)
    background = cv2.resize(background, (light.shape[1], light.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(light, np.maximum(background, 1), scale=245)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(flat)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def binarize(image: np.ndarray) -> np.ndarray:
    """Adaptive (local) threshold: black print on white, robust to uneven light."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    block = max(15, (min(gray.shape) // 25) | 1)
    binary = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# =====================================================================
# Pipelines used by the app
# =====================================================================
def capture_candidates(frame: np.ndarray, report: Optional[QualityReport] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Images to try reading for a single item (label, page, ...): the untouched frame
    first, then, only when the page is skewed or the quality is poor, a flattened
    (homography) and / or tone-corrected version. The caller keeps the best read.
    """
    candidates = [("original", frame)]
    if not ENABLED:
        return candidates
    report = report or assess_quality(frame)
    warped, matrix = warp_page(frame)
    if matrix is not None:
        candidates.append(("flattened", warped))
    if not report.ok:
        base = warped
        candidates.append(("tone" if matrix is None else "flattened+tone", enhance_tone(base)))
        if "blurry" not in report.problems:
            candidates.append(("binarized", binarize(base)))
    return candidates


def prepare_page(page: np.ndarray, report: Optional[QualityReport] = None) -> Tuple[np.ndarray, Optional[DewarpModel]]:
    """
    One book page for OCR with geometry kept invertible (the reading highlight
    needs boxes in camera coordinates): dewarp curved lines, and tone-correct
    only when the quality is poor. Returns (image, model); unwarp_box() maps boxes back.
    """
    if not ENABLED:
        return page, None
    model = estimate_dewarp(page)
    image = dewarp(page, model)
    if not (report or assess_quality(page)).ok:
        image = enhance_tone(image)
    return image, model


##############################################################################
# PART C: BOOK READING MODE
# Decides WHEN to read a page, WHERE the pages are, and in WHAT ORDER to read the text.
# The OCR only recognises the letters of each line.
#
#     PageTurnDetector   frame differencing + a small state machine
#                        -> fires once per new page, after the page settles
#     split_spread()     finds the book, then its spine: the text-free (smoothest)
#                        vertical strip, or the spine's shadow -> splits a two-page spread
#     split_page_parts() header / footer rows -> printed page number, running title
#     xy_cut()           recursive XY-cut on the recognised lines
#                        -> columns, headings and paragraphs in reading order
#     read_page()        all of the above for one camera frame
#     continue_from()    compares a new view with the page being read: same words
#                        -> carry on from the word reached instead of starting over
##############################################################################


# =====================================================================
# 1. Page-turn detection (frame differencing)
# =====================================================================
class PageTurnDetector:
    """
    Watches the camera for a page turn.

        TURNING   something is moving (a hand, a page in the air)
        SETTLING  motion stopped; waiting for the page to stay still
        STEADY    page has been still for `settle_time`

    update() returns an event:
        ""         nothing to do
        "changed"  the view became STEADY and the page layout differs from the
                   last page read (a new page, or the first page)
        "moved"    the view became STEADY after movement but the layout looks
                   the same: a hand passed over, the book shifted slightly, or a
                   page of very similar layout was turned (dense book pages
                   look alike), so the caller should compare the words

    Thresholds adapt to the camera: sensor noise and small hand tremor give
    every camera a different "still" level, so the detector keeps a running
    estimate of that noise floor and measures motion relative to it.
    """

    def __init__(self, on_margin: float = 3.0, off_margin: float = 1.5,
                 settle_time: float = 0.8, change_min: float = 0.06, recheck_every: float = 1.0):
        self.on_margin = on_margin      # this far above the noise floor = movement
        self.off_margin = off_margin    # within this of the noise floor = still
        self.settle_time = settle_time  # seconds of stillness before reading
        self.change_min = change_min    # fraction of the page layout that must change
        self.recheck_every = recheck_every  # seconds between layout checks while steady
        self.reset()

    @property
    def motion_on(self) -> float:
        return self.noise + self.on_margin

    @property
    def motion_off(self) -> float:
        return self.noise + self.off_margin

    def reset(self) -> None:
        """Start over: the first steady page will be read."""
        self.state = "TURNING"
        self.motion = 0.0
        self.noise = 2.0                # running estimate of the camera's still-scene motion
        self._prev: Optional[np.ndarray] = None
        self._recent: deque = deque(maxlen=5)   # last few raw motion values
        self._still_since: Optional[float] = None
        self._last_signature: Optional[np.ndarray] = None
        self._last_check = 0.0

    @staticmethod
    def _small_gray(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0)

    @staticmethod
    def signature(frame: np.ndarray) -> np.ndarray:
        """
        A small binary 'fingerprint' of the page layout: text lines appear as
        dark bands. Adaptive threshold makes it robust to lighting changes.
        """
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 10)
        # Smear horizontally so a line of text becomes one band; tiny shifts of
        # the book then change the signature much less than a new page does
        b = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        return cv2.resize(b, (80, 45), interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        g = self._small_gray(frame)
        if self._prev is None:
            self._prev = g
            return ""
        # Median of the last 5 frame differences: one noisy frame can't
        # restart the "hold still" timer, but real movement shows immediately
        self._recent.append(float(np.mean(cv2.absdiff(g, self._prev))))
        self._prev = g
        self.motion = float(np.median(self._recent))

        # Learn the noise floor from quiet frames (slow moving average)
        if self.motion < self.motion_off:
            self.noise = 0.95 * self.noise + 0.05 * self.motion

        if self.motion > self.motion_on:
            self.state = "TURNING"
            self._still_since = None
            return ""

        if self.state == "STEADY":
            # A slow change (camera drifting, page sliding) never looks like
            # motion, so re-check the page layout now and then
            if now - self._last_check >= self.recheck_every:
                self._last_check = now
                return "changed" if self._is_new_page(frame) else ""
            return ""

        if self.motion > self.motion_off:
            self._still_since = None     # between the thresholds: not still yet
            return ""

        if self._still_since is None:
            self._still_since = now
            self.state = "SETTLING"
        if now - self._still_since < self.settle_time:
            return ""

        self.state = "STEADY"
        self._last_check = now
        return "changed" if self._is_new_page(frame) else "moved"

    @staticmethod
    def layout_distance(a: np.ndarray, b: np.ndarray, max_shift: int = 4) -> float:
        """
        How different two layout signatures are (0 to 1), after sliding one over the other by up to
        `max_shift` cells in each direction. A book that shifted or vibrated slightly is the same page
        in a new position, so the best alignment is what counts; a new page differs at any alignment.
        """
        a, b = cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)    # shifts of a fraction of a cell
        h, w = a.shape
        best = 1.0
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                ys, ye = max(dy, 0), h + min(dy, 0)
                xs, xe = max(dx, 0), w + min(dx, 0)
                part_a = a[ys:ye, xs:xe]
                part_b = b[ys - dy:ye - dy, xs - dx:xe - dx]
                best = min(best, float(np.mean(cv2.absdiff(part_a, part_b))) / 255.0)
        return best

    def _is_new_page(self, frame: np.ndarray) -> bool:
        """True (and remembered) when the page layout differs from the last page read, however it is shifted."""
        sig = self.signature(frame)
        if self._last_signature is not None and self.layout_distance(sig, self._last_signature) < self.change_min:
            return False                 # same page as before, only moved a little
        self._last_signature = sig
        return True


# =====================================================================
# 2. Two-page spread splitting (gutter detection)
# =====================================================================
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Start/end (exclusive) of each run of True values."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def find_book(image: np.ndarray) -> Box:
    """
    The open book = the bright (paper) regions: Otsu threshold on a blurred
    grayscale image, a closing to fill in the text, then the bounding box of
    all large paper regions together (a spine shadow can cut the paper into
    two halves). Falls back to the whole frame when no paper is found.
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 0.05 * w * h]
    if big and sum(cv2.contourArea(c) for c in big) > 0.2 * w * h:
        x, y, bw, bh = cv2.boundingRect(np.vstack(big))
        return (x, y, x + bw, y + bh)
    return (0, 0, w, h)


def texture_profile(gray: np.ndarray, window: int = 15) -> np.ndarray:
    """
    How "busy" each x column is: the local standard deviation of brightness,
    averaged down the column. Text is busy even when blurred; blank margins
    and the spine are smooth.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    mean = cv2.blur(g, (window, window))
    mean_sq = cv2.blur(g * g, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    h = g.shape[0]
    profile = std[int(h * 0.1):int(h * 0.9)].mean(axis=0)
    k = max(3, g.shape[1] // 60)
    return np.convolve(profile, np.ones(k) / k, mode="same")


def split_spread(image: np.ndarray, min_depth: float = 0.12) -> Optional[int]:
    """
    Find the gutter (spine) of an open book and return its x position, or
    None for a single page. A spine is only reported when the picture shows
    evidence of one; it is never assumed to be in the middle.

    1. Find the book (the paper). A two-page spread is clearly wider than
       tall; a single page is not.
    2. The spine's shadow: a half-open book casts a dark valley in the
       column brightness profile. When it's clear, it's the most precise clue.
    3. The text-free strip: text never crosses the spine, so the two inner
       margins form a smooth vertical strip in the middle of the book. This
       works for a flat, fully open book whose spine casts no shadow. Of the
       smooth strips, the one nearest the middle of the book wins. It must be
       wider than the gap between two columns of one page.
    4. Either way, both sides of the spine must hold a page of text. A single
       sheet, a page with two columns, or a picture that is mostly one block
       of text has no spine, and no line is drawn.
    """
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = find_book(image)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 1.1 * bh:
        return None                      # taller than wide: a single page
    gray = cv2.cvtColor(image[by1:by2, bx1:bx2], cv2.COLOR_BGR2GRAY)
    lo, hi = int(bw * 0.3), int(bw * 0.7)

    tex = texture_profile(gray)
    text_level = float(np.median(tex[int(bw * 0.1):int(bw * 0.9)]))
    busy = tex > 0.6 * text_level

    def text_on_both_sides(x: int) -> bool:
        """Each side of x holds a real block of text (at least a fifth of the book's width)."""
        left, right = busy[:max(x - int(bw * 0.01), 0)], busy[x + int(bw * 0.01):]
        return left.sum() >= 0.2 * bw and right.sum() >= 0.2 * bw

    # --- clue 1: the spine's shadow ---
    profile = gray.astype(np.float32)[int(bh * 0.1):int(bh * 0.9)].mean(axis=0)
    k = max(3, bw // 100)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    x = lo + int(np.argmin(profile[lo:hi]))
    left = np.median(profile[int(bw * 0.15):lo])
    right = np.median(profile[hi:int(bw * 0.85)])
    if profile[x] < (1.0 - min_depth) * min(left, right) and text_on_both_sides(x):
        return bx1 + x

    # --- clue 2: the smooth, text-free strip nearest the middle ---
    quiet = tex[lo:hi] < 0.6 * text_level
    runs = [(a, b) for a, b in _runs(quiet) if b - a >= max(4, bw * 0.07)]     # wider than the gap between columns of one page
    if runs:
        middle = bw / 2 - lo
        a, b = min(runs, key=lambda r: 0 if r[0] <= middle <= r[1] else min(abs(r[0] - middle), abs(r[1] - middle)))
        x = lo + (a + b) // 2
        if text_on_both_sides(x):
            return bx1 + x

    return None                          # no evidence of a spine: a single page


# =====================================================================
# 3. Page structure: header, footer, page number, running title
# =====================================================================
@dataclass
class PageInfo:
    """What a printed page says about itself."""
    number: Optional[str] = None          # printed page number, e.g. "47" or "xii"
    running_title: Optional[str] = None   # book or chapter title in the header/footer
    headings: List[str] = field(default_factory=list)  # larger-type headings in the text


_ROMAN = re.compile(r"(?=[ivxlcdm]+$)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.I)
_NUMBER_ONLY = re.compile(r"^[\s\-–—.()|]*(?:page\s+)?(\d{1,4})[\s\-–—.()|]*$", re.I)
_NUMBER_THEN_TITLE = re.compile(r"^(\d{1,4})\s+(\D.*)$")
_TITLE_THEN_NUMBER = re.compile(r"^(.*\D)\s+(\d{1,4})$")
_HEADING_WORDS = re.compile(r"^(chapter|part|book|section|prologue|epilogue|preface|introduction|contents|appendix)\b", re.I)


def parse_page_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a header/footer line into (page number, remaining title text).
    Handles "47", "- 47 -", "Page 47", "xii", "47  THE SILENT RIVER" and
    "CHAPTER THREE  48".
    """
    t = text.strip()
    m = _NUMBER_ONLY.match(t)
    if m:
        return m.group(1), None
    if _ROMAN.match(t.strip(" -–—.")) and len(t.strip(" -–—.")) <= 7:
        return t.strip(" -–—.").lower(), None
    m = _NUMBER_THEN_TITLE.match(t)
    if m:
        return m.group(1), m.group(2).strip() or None
    m = _TITLE_THEN_NUMBER.match(t)
    if m:
        return m.group(2), m.group(1).strip() or None
    return None, t or None


def _rows(lines: List[TextLine]) -> List[List[TextLine]]:
    """Group lines that sit side by side (overlapping in y) into rows, top to bottom."""
    rows: List[List[TextLine]] = []
    for line in sorted(lines, key=lambda l: l.box[1]):
        cy = (line.box[1] + line.box[3]) / 2
        if rows:
            r = rows[-1]
            top, bottom = min(l.box[1] for l in r), max(l.box[3] for l in r)
            if top <= cy <= bottom:
                r.append(line)
                continue
        rows.append([line])
    for r in rows:
        r.sort(key=lambda l: l.box[0])
    return rows


def _median(values, default: float = 0.0) -> float:
    return float(np.median(values)) if len(values) else default


def split_page_parts(lines: List[TextLine]) -> Tuple[List[TextLine], List[TextLine], PageInfo]:
    """
    Separate the page's header and footer from its body.

    A header (footer) is a row at the very top (bottom) of the page that is set
    apart from the body by a gap clearly larger than the usual line spacing and
    is either short or contains a page number. Its text gives the printed page
    number and the running title. Returns (body lines, header/footer lines, info).
    """
    info = PageInfo()
    rows = _rows(lines)
    if len(rows) < 3:
        return lines, [], info

    heights = [l.box[3] - l.box[1] for l in lines]
    line_h = _median(heights, 1.0)
    gaps = [max(0, min(l.box[1] for l in b) - max(l.box[3] for l in a)) for a, b in zip(rows, rows[1:])]
    usual_gap = _median(gaps, line_h * 0.5)
    widths = [max(l.box[2] for l in r) - min(l.box[0] for l in r) for r in rows]
    usual_width = _median(widths, 1.0)

    def is_margin_row(row, gap):
        text = " ".join(l.text for l in row)
        number, _ = parse_page_number(text)
        width = max(l.box[2] for l in row) - min(l.box[0] for l in row)
        set_apart = gap > max(1.8 * usual_gap, 0.9 * line_h)
        return set_apart and (number is not None or width < 0.6 * usual_width)

    margin_rows = []
    if is_margin_row(rows[0], gaps[0]):
        margin_rows.append(rows[0])
    if is_margin_row(rows[-1], gaps[-1]):
        margin_rows.append(rows[-1])

    titles = []
    for row in margin_rows:
        for line in row:
            number, title = parse_page_number(line.text)
            if number and info.number is None:
                info.number = number
            if title and not number:
                titles.append(title)
            elif title:
                titles.append(title)
    info.running_title = " ".join(titles).strip() or None

    margin = {id(l) for r in margin_rows for l in r}
    body = [l for l in lines if id(l) not in margin]
    info.headings = [l.text.strip() for l in body
                     if (l.box[3] - l.box[1]) >= 1.35 * line_h or _HEADING_WORDS.match(l.text.strip())]
    return body, [l for l in lines if id(l) in margin], info


# =====================================================================
# 4. Reading order: recursive XY-cut on the recognised lines
# =====================================================================
def xy_cut(lines: List[TextLine], columns: Optional[List[Box]] = None) -> List[List[TextLine]]:
    """
    Recursive XY-cut, a classic document layout algorithm, run on the boxes of
    the lines the OCR recognised (reliable whenever the text is readable):

      1. Vertical cut: project the line boxes onto the x axis. An empty band
         between them that is wider than a column gap splits the region into
         a left and a right part (columns), read left first.
      2. Otherwise, horizontal cut: project onto the y axis. A gap clearly
         larger than the usual line spacing splits the region into an upper
         and a lower part (heading / paragraphs), read top first.
      3. Repeat on each part. A part that can't be cut is read line by line
         from the top; an indented line starts a new paragraph.

    Returns the paragraphs in reading order. Column boxes found by vertical
    cuts are appended to `columns` if given.
    """
    if not lines:
        return []
    line_h = _median([l.box[3] - l.box[1] for l in lines], 1.0)
    rows = _rows(lines)
    usual_gap = _median([max(0, min(l.box[1] for l in b) - max(l.box[3] for l in a))
                         for a, b in zip(rows, rows[1:])], line_h * 0.5)

    def gaps_along(spans):
        """Empty intervals between merged [a, b) spans."""
        spans = sorted(spans)
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]

    def cut(group: List[TextLine]) -> List[List[TextLine]]:
        if len(group) <= 1:
            return [group] if group else []
        # 1. columns
        x_gaps = [g for g in gaps_along([(l.box[0], l.box[2]) for l in group]) if g[1] - g[0] >= 1.2 * line_h]
        if x_gaps:
            a, b = max(x_gaps, key=lambda g: g[1] - g[0])
            split = (a + b) / 2
            left = [l for l in group if (l.box[0] + l.box[2]) / 2 < split]
            right = [l for l in group if (l.box[0] + l.box[2]) / 2 >= split]
            if left and right:
                if columns is not None:
                    for side in (left, right):
                        columns.append((min(l.box[0] for l in side), min(l.box[1] for l in side),
                                        max(l.box[2] for l in side), max(l.box[3] for l in side)))
                return cut(left) + cut(right)
        # 2. headings / paragraphs
        y_gaps = [g for g in gaps_along([(l.box[1], l.box[3]) for l in group])
                  if g[1] - g[0] >= max(1.8 * usual_gap, usual_gap + 0.6 * line_h)]
        if y_gaps:
            a, b = max(y_gaps, key=lambda g: g[1] - g[0])
            split = (a + b) / 2
            top = [l for l in group if (l.box[1] + l.box[3]) / 2 < split]
            bottom = [l for l in group if (l.box[1] + l.box[3]) / 2 >= split]
            if top and bottom:
                return cut(top) + cut(bottom)
        # 3. no gap left: split paragraphs at indented first lines (novel style)
        ordered = [l for r in _rows(group) for l in r]
        left_edge = _median([l.box[0] for l in ordered])
        paragraphs = [[ordered[0]]]
        for line in ordered[1:]:
            if line.box[0] - left_edge > max(0.6 * line_h, 6):
                paragraphs.append([line])
            else:
                paragraphs[-1].append(line)
        return paragraphs

    return cut(lines)


# =====================================================================
# 5. Whole frame
# =====================================================================
@dataclass
class LineSpan:
    """Where one printed line ended up in the page text, and where it is on screen."""
    start: int      # character offsets into the page text
    end: int
    box: Box        # full-frame pixel coordinates
    words: List[Tuple[int, int, Box]] = field(default_factory=list)  # same, per word
    quad: Optional[Quad] = None                                       # the line's real (slanted) outline
    word_quads: List[Quad] = field(default_factory=list)              # same, per word
    text: str = ""                                                    # the printed line as read (stripped)
    page: int = 0                                                     # 0 / 1: left / right page of a spread


@dataclass
class PageLayout:
    """Everything found in one frame, in full-frame pixel coordinates (for drawing)."""
    gutter_x: Optional[int] = None
    columns: List[Box] = field(default_factory=list)
    blocks: List[Box] = field(default_factory=list)   # paragraphs, in reading order
    margins: List[Box] = field(default_factory=list)  # headers / footers (not read aloud)
    column_quads: List[Quad] = field(default_factory=list)   # the same three, as outlines that follow a slanted page
    block_quads: List[Quad] = field(default_factory=list)
    margin_quads: List[Quad] = field(default_factory=list)
    lines: List[LineSpan] = field(default_factory=list)
    paragraphs: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) in the page text
    pages: List[PageInfo] = field(default_factory=list)              # left to right

    def page_numbers(self) -> List[str]:
        """Printed page numbers, left to right. A missing number on one side of
        a spread is inferred from the facing page (left = right - 1)."""
        nums = [p.number for p in self.pages]
        if len(nums) == 2:
            left, right = nums
            if left is None and right and right.isdigit() and int(right) > 1:
                nums[0] = str(int(right) - 1)
            elif right is None and left and left.isdigit():
                nums[1] = str(int(left) + 1)
        return [n for n in nums if n]

    def label(self) -> str:
        """'Page 47', 'Pages 46 and 47', or '' when no number is printed."""
        nums = self.page_numbers()
        if len(nums) == 2:
            return f"Pages {nums[0]} and {nums[1]}"
        return f"Page {nums[0]}" if nums else ""

    def running_title(self) -> Optional[str]:
        titles = [p.running_title for p in self.pages if p.running_title]
        return titles[0] if titles else None

    def summary(self) -> str:
        """Page facts for the language model, so it can answer 'what page is this?'."""
        parts = []
        if self.label():
            parts.append(self.label() + ".")
        titles = [p.running_title for p in self.pages if p.running_title]
        if titles:
            parts.append("Running header: " + "; ".join(dict.fromkeys(titles)) + ".")
        headings = [h for p in self.pages for h in p.headings]
        if headings:
            parts.append("Headings: " + "; ".join(headings) + ".")
        return " ".join(parts)


def _remap(line: TextLine, point, box_of) -> TextLine:
    """The same line with every outline moved through point(x, y) -> (x, y); boxes are recomputed by box_of(quad)."""
    quad = map_quad(line.quad or box_quad(line.box), point)
    word_quads = [map_quad(q, point) for q in line.word_quads] if line.word_quads else \
        [map_quad(box_quad(b), point) for _, _, b in line.words]
    words = [(a, b, box_of(q)) for (a, b, _), q in zip(line.words, word_quads)]
    return TextLine(line.text, line.confidence, box_of(quad), words, quad, word_quads)


def _shift(line: TextLine, dx: int) -> TextLine:
    """The same line with its boxes moved dx pixels to the right."""
    return _remap(line, lambda x, y: (x + dx, y), quad_bounds)


def _unwarp(line: TextLine, model) -> TextLine:
    """Boxes found on a dewarped page image, mapped back onto the camera frame."""
    if model is None:
        return line
    # a word or line is as tall as the vertical shifts under it; boxes keep unwarp_box's rule, outlines move point by point
    out = _remap(line, lambda x, y: (x, y + float(model.shift(x))), quad_bounds)
    out.box = unwarp_box(line.box, model)
    out.words = [(a, b, unwarp_box(box, model)) for a, b, box in line.words]
    return out


def _page_back(matrix: Optional[np.ndarray], dx: int):
    """point(x, y) taking a position on a straightened page crop back to the full frame (matrix: crop -> straightened)."""
    inverse = cv2.invertAffineTransform(matrix) if matrix is not None else None

    def point(x: float, y: float) -> Tuple[float, float]:
        if inverse is not None:
            x, y = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2], inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2])
        return x + dx, y
    return point


def _read_flat_page(frame: np.ndarray, split: bool, enhance: bool) -> Tuple[str, PageLayout]:
    """read_page() on one image: split the spread, OCR each page, order the text (see read_page)."""
    h, w = frame.shape[:2]
    layout = PageLayout(gutter_x=split_spread(frame) if split else None)
    pages = [(0, layout.gutter_x), (layout.gutter_x, w)] if layout.gutter_x else [(0, w)]

    report = assess_quality(frame) if enhance else None
    text = ""
    for page_no, (x_start, x_end) in enumerate(pages):
        crop = frame[:, x_start:x_end]
        # A slanted page is straightened for the layout analysis (rows, columns, paragraphs); everything found is
        # then mapped back, so the outlines drawn on the screen follow the slant.
        angle = estimate_skew(crop) if enhance else 0.0
        turned, matrix = rotate_expand(crop, angle) if angle else (crop, None)
        image, model = prepare_page(turned, report) if enhance else (turned, None)
        back = _page_back(matrix, x_start)
        found = [_unwarp(l, model) for l in recognize_text(image, word_boxes=True) if l.text.strip()]
        body, margin_lines, info = split_page_parts(found)
        layout.pages.append(info)
        lines = lambda group: [_remap(l, back, quad_bounds) for l in group]

        for b in [l.box for l in margin_lines]:
            q = map_quad(box_quad(b), back)
            layout.margins.append(quad_bounds(q)); layout.margin_quads.append(q)

        columns = []
        paragraphs = xy_cut(body, columns)
        for b in columns:
            q = map_quad(box_quad(b), back)
            layout.columns.append(quad_bounds(q)); layout.column_quads.append(q)
        for paragraph in paragraphs:
            q = map_quad(box_quad((min(l.box[0] for l in paragraph), min(l.box[1] for l in paragraph),
                                   max(l.box[2] for l in paragraph), max(l.box[3] for l in paragraph))), back)
            layout.blocks.append(quad_bounds(q)); layout.block_quads.append(q)
            para_start = None
            for line in lines(paragraph):
                t = line.text.strip()
                if para_start is None:
                    if text:
                        text += "\n\n"
                    para_start = len(text)
                elif text.endswith("-") and t[:1].islower():
                    text = text[:-1]          # "exam-" + "ple" -> "example"
                else:
                    text += " "
                start = len(text)
                text += t
                lead = len(line.text) - len(line.text.lstrip())   # strip() shifted the offsets
                words = [(start + a - lead, start + b - lead, box) for a, b, box in line.words]
                layout.lines.append(LineSpan(start, len(text), line.box, words, line.quad, line.word_quads, t, page_no))
            layout.paragraphs.append((para_start, len(text)))

    return text, layout


def _unproject_layout(layout: PageLayout, matrix: np.ndarray, flat_height: int) -> None:
    """Map a layout found on the flattened page back onto the camera frame (in place)."""
    inverse = np.linalg.inv(matrix)

    def point(x: float, y: float) -> Tuple[float, float]:
        out = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), inverse)[0, 0]
        return float(out[0]), float(out[1])

    def both(boxes, quads):
        q = [map_quad(qd, point) for qd in (quads or [box_quad(b) for b in boxes])]
        return [quad_bounds(x) for x in q], q

    if layout.gutter_x:
        layout.gutter_x = int(point(layout.gutter_x, flat_height / 2)[0])
    layout.columns, layout.column_quads = both(layout.columns, layout.column_quads)
    layout.blocks, layout.block_quads = both(layout.blocks, layout.block_quads)
    layout.margins, layout.margin_quads = both(layout.margins, layout.margin_quads)
    spans = []
    for l in layout.lines:
        quad = map_quad(l.quad or box_quad(l.box), point)
        wq = [map_quad(q, point) for q in l.word_quads] if l.word_quads else [map_quad(box_quad(b), point) for _, _, b in l.words]
        spans.append(LineSpan(l.start, l.end, quad_bounds(quad), [(a, b, quad_bounds(q)) for (a, b, _), q in zip(l.words, wq)], quad, wq,
                              l.text, l.page))
    layout.lines = spans


def read_page(frame: np.ndarray, split: bool = True, enhance: bool = False) -> Tuple[str, PageLayout]:
    """
    Split the spread, read each page with the OCR, separate headers and
    footers (page number, running title), and order the body text with XY-cut.
    Paragraphs are separated by a blank line; lines within a paragraph are
    joined with spaces (re-joining words hyphenated across lines). The layout
    remembers which characters came from which printed line and word, so the
    app can show exactly where it is reading. split=False treats the image as
    one page (e.g. a single label photographed in the web version).

    enhance=True first prepares the image (Part B): a page photographed at
    an angle is flattened with a homography, curved lines (a page bending into the
    spine) are straightened, and dark / shadowed / glary frames are tone-corrected.
    Boxes are mapped back, so the layout is still in camera coordinates.
    """
    if enhance:
        flat, matrix = warp_page(frame)
        if matrix is not None:
            text, layout = _read_flat_page(flat, split, enhance)
            if text.strip():
                _unproject_layout(layout, matrix, flat.shape[0])
                return text, layout
            # nothing read from the flattened page: fall back to the frame as the camera saw it
    return _read_flat_page(frame, split, enhance)


# =====================================================================
# 5b. Paragraphs by the language model (the lines stay exactly as read)
# =====================================================================
MAX_ARRANGE_LINES = 150     # a page with more lines than this is read as the layout analysis ordered it


def line_hints(layout: PageLayout) -> List[str]:
    """Layout facts about each line, as short tags for the language model: [page] first line of a new page of a
    spread, [column] first line of a new column, [gap] extra space above, [indent] starts indented, [short] ends
    well before the right edge (judged against the other lines of its own column)."""
    lines = layout.lines
    line_h = _median([l.box[3] - l.box[1] for l in lines], 1.0)
    breaks, runs = [], []                    # why each line starts a block, and which column it is in
    run = 0
    for i, l in enumerate(lines):
        previous = lines[i - 1] if i else None
        kind = ""
        if previous is not None:
            if l.page != previous.page:
                kind = "page"
            elif l.box[1] < previous.box[3] - line_h:
                kind = "column"
            elif l.box[1] - previous.box[3] > 0.8 * line_h:
                kind = "gap"
        run += kind in ("page", "column")
        breaks.append(kind)
        runs.append(run)
    left = {r: _median([l.box[0] for l, x in zip(lines, runs) if x == r]) for r in set(runs)}
    right = {r: float(np.percentile([l.box[2] for l, x in zip(lines, runs) if x == r], 80)) for r in set(runs)}
    hints = []
    for l, kind, r in zip(lines, breaks, runs):
        tags = [kind] if kind else []
        if l.box[0] - left[r] > max(0.6 * line_h, 6):
            tags.append("indent")
        if l.box[2] < right[r] - 3 * line_h:
            tags.append("short")
        hints.append("".join(f"[{t}]" for t in tags))
    return hints


def line_blocks(layout: PageLayout) -> List[int]:
    """
    Which block of text each line belongs to, found from the boxes alone: two lines are in the same block when one
    sits right under the other (a gap of no more than about a line), with the same left edge and mostly the same
    horizontal span. A column, a heading, or a sidebar beside the main text each come out as a block, so the
    rows of a sidebar that are level with the main text's rows are not mixed into it. Blocks are numbered in the
    order their first line appears.
    """
    lines = layout.lines
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for page in {l.page for l in lines}:
        members = [i for i, l in enumerate(lines) if l.page == page]
        line_h = _median([lines[i].box[3] - lines[i].box[1] for i in members], 1.0)
        for i in members:
            a = lines[i].box
            for j in members:
                b = lines[j].box
                if b[1] <= a[1]:
                    continue                                   # b must start below a
                gap = b[1] - a[3]
                overlap = min(a[2], b[2]) - max(a[0], b[0])
                if (-0.6 * line_h <= gap <= 1.2 * line_h and abs(a[0] - b[0]) <= 2 * line_h
                        and overlap >= 0.6 * min(a[2] - a[0], b[2] - b[0])):
                    parent[find(j)] = find(i)
    numbers, blocks = {}, []
    for i in range(len(lines)):
        root = find(i)
        numbers.setdefault(root, len(numbers))
        blocks.append(numbers[root])
    return blocks


def numbered_lines(layout: PageLayout) -> str:
    """
    The page's lines as the language model is shown them, grouped by block (see line_blocks). One row per line:
    its number (the line's place in layout.lines), where it is (x from to, y of its top, both in percent of the
    area the text covers, and h, the height of its letters against the usual line: a sidebar or caption has
    smaller letters), the layout tags, then the text.
    """
    lines = layout.lines
    x0, y0 = min(l.box[0] for l in lines), min(l.box[1] for l in lines)
    width = max(max(l.box[2] for l in lines) - x0, 1)
    height = max(max(l.box[3] for l in lines) - y0, 1)
    usual = _median([l.box[3] - l.box[1] for l in lines], 1.0)
    hints, blocks = line_hints(layout), line_blocks(layout)
    out = []
    for block in range(max(blocks) + 1):
        out.append(f"Block {block + 1}:")
        for i, l in enumerate(lines):
            if blocks[i] == block:
                where = (f"x{100 * (l.box[0] - x0) / width:.0f}-{100 * (l.box[2] - x0) / width:.0f} "
                         f"y{100 * (l.box[1] - y0) / height:.0f} h{(l.box[3] - l.box[1]) / usual:.1f}")
                out.append(f"{i} ({where}){hints[i]} {l.text}")
    return "\n".join(out)


def regroup(layout: PageLayout, groups: List[List[int]], skip: List[int]) -> Tuple[str, PageLayout]:
    """
    The page text and layout with the lines put into the given paragraphs (lists of line numbers, in reading
    order); lines in `skip` are left out. Only the grouping changes: every line keeps its words, so the reading
    highlight still finds them. Lines of a paragraph are joined with spaces (a word hyphenated across lines is
    joined), paragraphs with a blank line, as read_page does. Returns a new layout; the given one is not touched.
    """
    text, spans, paragraphs, blocks, block_quads = "", [], [], [], []
    for group in groups:
        members = []
        para_start = None
        for i in group:
            old = layout.lines[i]
            t = old.text
            if para_start is None:
                if text:
                    text += "\n\n"
                para_start = len(text)
            elif text.endswith("-") and t[:1].islower():
                text = text[:-1]          # "exam-" + "ple" -> "example"
            else:
                text += " "
            start = len(text)
            text += t
            words = [(start + a - old.start, start + b - old.start, box) for a, b, box in old.words]
            members.append(replace(old, start=start, end=len(text), words=words))
        spans += members
        paragraphs.append((para_start, len(text)))
        if all(m.quad for m in members):
            corners = np.array([p for m in members for p in m.quad], np.float32)
            quad = [tuple(map(float, p)) for p in cv2.boxPoints(cv2.minAreaRect(corners))]
        else:
            quad = box_quad((min(m.box[0] for m in members), min(m.box[1] for m in members),
                             max(m.box[2] for m in members), max(m.box[3] for m in members)))
        blocks.append(quad_bounds(quad))
        block_quads.append(quad)
    return text, replace(layout, lines=spans, paragraphs=paragraphs, blocks=blocks, block_quads=block_quads)


def arrange_page(layout: PageLayout, doc, cache: Optional[dict] = None) -> Optional[Tuple[str, PageLayout]]:
    """
    Ask the language model (doc.arrange_lines) how the page's lines form paragraphs, and rebuild the page that way:
    a sentence that runs on over a line, column or page is one paragraph, so it is read without a pause. Returns
    (text, layout), or None when the model cannot be used: then the layout analysis' paragraphs stay.
    """
    lines = layout.lines
    if not 3 <= len(lines) <= MAX_ARRANGE_LINES or not all(l.text for l in lines):
        return None
    key = tuple(l.text for l in lines)
    plan = cache.get(key) if cache is not None else None
    if plan is None:
        plan = doc.arrange_lines(numbered_lines(layout), len(lines))
        if plan is None:
            return None
        if cache is not None:
            cache[key] = plan
            while len(cache) > 16:
                cache.pop(next(iter(cache)))
    groups, skip = plan
    return regroup(layout, groups, skip)


# =====================================================================
# 6. Same words after the camera or book moved: carry on reading
# =====================================================================
def _words(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, normalised word) for each word, ignoring punctuation-only tokens."""
    words = []
    for m in re.finditer(r"\S+", text):
        norm = re.sub(r"[^\w]", "", m.group().lower())
        if norm:
            words.append((m.start(), m.end(), norm))
    return words


def word_overlap(old_text: str, new_text: str, min_run: int = 3) -> float:
    """Share (0 to 1) of the shorter text's words that appear, in runs of at least `min_run`, in the other."""
    old_words, new_words = _words(old_text), _words(new_text)
    if not old_words or not new_words:
        return 0.0
    matcher = SequenceMatcher(None, [w[2] for w in old_words], [w[2] for w in new_words], autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= min_run)
    return matched / min(len(old_words), len(new_words))


def continue_from(old_text: str, old_pos: int, new_text: str,
                  min_run: int = 3, min_overlap: float = 0.4) -> Optional[int]:
    """
    Compare the page being read with a new view of it.

    old_pos is how far reading had got in old_text (a character offset; use
    len(old_text) when the page was read to the end).

    Returns the character offset in new_text to carry on reading from:
    the word reading had reached if it is still in view, otherwise the first
    word after the part already read (len(new_text) when nothing new is in
    view). Returns None when the views don't share enough words, i.e. it's a
    different page.

    The words are aligned with difflib's SequenceMatcher. Only matching runs of
    at least `min_run` words count, so common words ("the", "and") on a
    different page don't look like an overlap.
    """
    old_words, new_words = _words(old_text), _words(new_text)
    if not old_words or not new_words:
        return None
    matcher = SequenceMatcher(None, [w[2] for w in old_words], [w[2] for w in new_words], autojunk=False)
    runs = [b for b in matcher.get_matching_blocks() if b.size >= min_run]
    matched = sum(b.size for b in runs)
    if matched < min_overlap * min(len(old_words), len(new_words)):
        return None                              # different page

    # First old word not finished yet (the one being spoken, or the next one)
    reached = next((i for i, w in enumerate(old_words) if w[1] > old_pos), len(old_words))

    for b in runs:                               # that word is still in view
        if b.a <= reached < b.a + b.size:
            return new_words[b.b + reached - b.a][0]

    earlier = [b for b in runs if b.a + b.size <= reached]
    if earlier:                                  # continue after what was already read
        j = earlier[-1].b + earlier[-1].size
        return new_words[j][0] if j < len(new_words) else len(new_text)

    return new_words[runs[0].b][0]               # reading hadn't reached the visible part yet


##############################################################################
# PART D: LIVE GUIDANCE
# The server side of the live camera (see the module docstring).
#
#     LiveGuide    normal mode: is there text, is it in view, hold still, capture
#     BookWatcher  book mode: page-turn detection on the live frames
#     BookReader   book mode: read a page, decide new page / same page, what to speak
##############################################################################

# ---- capture control (same values as the desktop app used) ----
MIN_TEXT_CHARS = 8         # need this many characters to count as text
MIN_TEXT_HEIGHT = 14       # px on a 720-line preview; smaller text -> "Move closer" ...
READ_WELL = 0.8            # ... but only when the OCR is not this sure of it (mean confidence, 0-1): small text that is read well is fine
PAGE_FILLS = 0.75          # ... and never when the sheet (page, book, tablet) fills this much of the picture: that is close enough
HOLD_TIME = 0.5            # seconds the text must stay steady before capturing
REARM_AFTER = 1.5          # seconds with no text in view before a new capture is allowed
QUALITY_PATIENCE = 3.0     # seconds to coach about blur / glare before capturing anyway
MOVED_SAME_PAGE = 0.12     # book mode: a moved view sharing this much text with the page being read is the same page
QUALITY_COOLDOWN = 5.0     # seconds between spoken image-quality hints
LOW_LIGHT_COOLDOWN = 30.0  # "Low light" is said at most once in this many seconds
HINT_COOLDOWN = 2.0        # seconds before the same position hint is spoken again
GUIDE_MARGIN_X = 30 / 1280  # the guide box, as fractions of the frame
GUIDE_MARGIN_Y = 20 / 720
GUIDE = (GUIDE_MARGIN_X, GUIDE_MARGIN_Y, 1 - GUIDE_MARGIN_X, 1 - GUIDE_MARGIN_Y)



def _norm(box: Box, w: int, h: int) -> List[float]:
    return [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]


# =====================================================================
# Normal mode
# =====================================================================
class LiveGuide:
    """
    Feed it preview frames with update(). It says what to draw and speak, and
    when to take the full-size photo ("capture": true). The page then calls
    capture_result(ok) so a successful capture is not repeated until the item
    leaves the view (or rearm() is called: "next", "r").
    """

    def __init__(self, detect: Callable = find_text_region, assess: Callable = assess_quality,
                 clock: Callable[[], float] = time.time):
        self._detect, self._assess, self._clock = detect, assess, clock
        self.reset()

    def reset(self) -> None:
        self.armed = True
        self.is_stable = False
        self.stable_start = 0.0
        self.last_text_seen = self._clock()
        self.last_hint = ""
        self.last_hint_time = 0.0
        self.last_quality_hint_time = 0.0
        self.last_low_light_time = -1e9
        self.quality = QualityReport()

    def rearm(self) -> None:
        self.armed = True
        self.is_stable = False

    def capture_result(self, ok: bool) -> None:
        if ok:
            self.armed = False
        self.is_stable = False

    # ---- helpers (ported from the desktop app) ----
    @staticmethod
    def _inside_guide(box: Box, w: int, h: int) -> float:
        gx1, gy1, gx2, gy2 = GUIDE[0] * w, GUIDE[1] * h, GUIDE[2] * w, GUIDE[3] * h
        x1, y1, x2, y2 = box
        ox1, oy1, ox2, oy2 = max(x1, gx1), max(y1, gy1), min(x2, gx2), min(y2, gy2)
        if ox2 <= ox1 or oy2 <= oy1:
            return 0.0
        area = (x2 - x1) * (y2 - y1)
        return (ox2 - ox1) * (oy2 - oy1) / area if area > 0 else 0.0

    @staticmethod
    def _read_well(lines) -> bool:
        """Did the OCR read the text with confidence? The mean confidence of its lines, each counted by its letters."""
        weights = [max(len(l.text.strip()), 1) for l in lines]
        return bool(lines) and sum(l.confidence * x for l, x in zip(lines, weights)) / sum(weights) >= READ_WELL

    @staticmethod
    def _guidance(box: Box, lines, w: int, h: int, frame: Optional[np.ndarray] = None) -> str:
        """Spoken hint to bring the text fully into view ("" when it's fine). `frame` lets "Move closer" be skipped
        when the sheet already fills the picture."""
        x1, y1, x2, y2 = box
        gx1, gy1, gx2, gy2 = GUIDE[0] * w, GUIDE[1] * h, GUIDE[2] * w, GUIDE[3] * h
        heights = sorted(l.box[3] - l.box[1] for l in lines)
        if (heights and heights[len(heights) // 2] < MIN_TEXT_HEIGHT * h / 720 and not LiveGuide._read_well(lines)
                and not (frame is not None and page_fill(frame) >= PAGE_FILLS)):          # the cheap tests first
            return "Move closer"
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        slack = 5 * h / 720
        near_x_edge = x1 <= gx1 + slack or x2 >= gx2 - slack
        near_y_edge = y1 <= gy1 + slack or y2 >= gy2 - slack
        far = 80 * h / 720
        if near_x_edge and abs(cx - gcx) > far:
            return "Move Left" if cx > gcx else "Move Right"
        if near_y_edge and abs(cy - gcy) > far:
            return "Move Up" if cy > gcy else "Move Down"
        return ""

    def _coach_quality(self, now: float) -> str:
        hint = self.quality.hint()
        if not hint:
            return ""
        if hint == "Low light":
            if now - self.last_low_light_time <= LOW_LIGHT_COOLDOWN:
                return ""
            self.last_low_light_time = now
            self.last_quality_hint_time = now
            return hint
        if now - self.last_quality_hint_time > QUALITY_COOLDOWN:
            self.last_quality_hint_time = now
            return hint
        return ""

    # ---- one frame ----
    def update(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        now = self._clock()
        try:
            box, lines = self._detect(frame, MIN_TEXT_CHARS)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            box, lines = None, []
        self.quality = self._assess(frame, box) if ENABLED else QualityReport()
        if box is not None:
            self.last_text_seen = now

        # Text gone for a moment -> ready for the next item
        if not self.armed and now - self.last_text_seen > REARM_AFTER:
            self.armed = True
            logger.info("Ready for a new capture")

        reply = {"type": "guide", "state": "NO_TEXT", "label": "SHOW TEXT HERE", "speak": "", "capture": False,
                 "box": _norm(box, w, h) if box else None,
                 "lines": [_norm(l.box, w, h) for l in lines],
                 "quads": [_norm_quad(l.quad, w, h) for l in lines if l.quad],
                 "quality": self.quality.summary(), "problems": list(self.quality.problems)}

        if not self.armed:
            # Already captured this item: stay quiet and wait for questions
            reply.update(state="CAPTURED", label="CAPTURED - ASK ME")
        elif box is not None:
            hint = self._guidance(box, lines, w, h, frame)
            if self._inside_guide(box, w, h) >= 0.65 and not hint:
                if not self.is_stable:
                    self.is_stable = True
                    self.stable_start = now
                held = now - self.stable_start
                blocking = {"blurry", "glare"} & set(self.quality.problems)
                if held < HOLD_TIME:
                    reply.update(state="HOLD", label=f"HOLD {HOLD_TIME - held:.1f}s")
                elif blocking and held < HOLD_TIME + QUALITY_PATIENCE:
                    # Coach for a moment; if it doesn't improve, capture anyway
                    # (the enhancement layer then tries corrected versions of the frame)
                    reply.update(state="COACH", label=self.quality.hint().upper(),
                                 speak=self._coach_quality(now))
                else:
                    reply.update(state="CAPTURING", label="CAPTURING...", capture=True)
                    self.is_stable = False
            else:
                self.is_stable = False
                reply.update(state="ADJUST", label="ADJUST POSITION")
                if hint and (hint != self.last_hint or now - self.last_hint_time > HINT_COOLDOWN):
                    reply["speak"] = hint
                    self.last_hint, self.last_hint_time = hint, now
        else:
            self.is_stable = False
            if "dark" in self.quality.problems:
                reply["speak"] = self._coach_quality(now)
        return reply


# =====================================================================
# Book mode
# =====================================================================
class BookWatcher:
    """Page-turn detection on the live frames (PageTurnDetector)."""

    def __init__(self):
        self.detector = PageTurnDetector()

    def reset(self) -> None:
        self.detector.reset()

    def update(self, frame: np.ndarray) -> dict:
        event = self.detector.update(frame)
        d = self.detector
        return {"type": "book", "event": event, "state": d.state,
                "motion": round(d.motion, 1), "still_below": round(d.motion_off, 1)}


SPEECH_CHUNK_CHARS = 600       # the longest piece of a page spoken as one; the server shortens it for the local voice


def speech_chunks(text: str, paragraphs, from_pos: int = 0, max_len: Optional[int] = None):
    """
    (start, chunk) pieces of the page for tracked speech, beginning at
    character `from_pos`: one per paragraph, long paragraphs split at sentence
    ends, so each chunk is spoken as its own utterance.
    """
    max_len = max_len or SPEECH_CHUNK_CHARS
    for a, b in paragraphs:
        if b <= from_pos:
            continue
        start = max(a, from_pos)
        while start < b:
            end = b
            if end - start > max_len:
                cut = max(text.rfind(". ", start, start + max_len),
                          text.rfind("? ", start, start + max_len),
                          text.rfind("! ", start, start + max_len))
                if cut > start:
                    end = cut + 1                      # end of a sentence
                else:
                    space = text.rfind(" ", start, start + max_len)
                    end = space if space > start else start + max_len   # between words
            yield start, text[start:end]
            start = end


def _norm_quad(quad, w: int, h: int) -> List[List[float]]:
    return [[round(x / w, 4), round(y / h, 4)] for x, y in quad]


def layout_json(layout: PageLayout, w: int, h: int) -> dict:
    """The page layout as JSON, with boxes normalised to 0-1 of the frame. `quads` / `quad` are the same shapes as
    outlines that follow a slanted page; the page draws those when they are there."""
    return {
        "gutter": round(layout.gutter_x / w, 4) if layout.gutter_x else None,
        "columns": [_norm(b, w, h) for b in layout.columns],
        "blocks": [_norm(b, w, h) for b in layout.blocks],
        "margins": [_norm(b, w, h) for b in layout.margins],
        "column_quads": [_norm_quad(q, w, h) for q in layout.column_quads],
        "block_quads": [_norm_quad(q, w, h) for q in layout.block_quads],
        "margin_quads": [_norm_quad(q, w, h) for q in layout.margin_quads],
        "lines": [{"start": s.start, "end": s.end, "box": _norm(s.box, w, h),
                   "quad": _norm_quad(s.quad, w, h) if s.quad else None,
                   "words": [[a, b, _norm(box, w, h), _norm_quad(q, w, h) if q else None]
                             for (a, b, box), q in zip(s.words, s.word_quads or [None] * len(s.words))]}
                  for s in layout.lines],
    }


class BookReader:
    """
    What to do with a steady book view. process() reads the page and returns an
    action for the browser:

        new        a different page: speak intro, then the chunks from the start
        resume     the same page seen from a moved view: speak the chunks from `from_pos`
        keep       same page, barely moved while reading: carry on, change nothing
        nothing_new  same page, nothing left to read
        no_text    no text in view
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.page_text = ""
        self.layout: Optional[PageLayout] = None
        self.last_running_title = ""
        self._arranged: dict = {}               # the model's paragraphs for a page, by its lines

    def _arrange(self, text: str, layout: PageLayout, resume: Optional[int], read_pos: int, doc):
        """The page with paragraphs from the language model, when it can give them (see arrange_page)."""
        try:
            arranged = arrange_page(layout, doc, self._arranged)
        except Exception as e:
            logger.warning(f"Could not arrange the page with the language model: {e}")
            arranged = None
        if arranged is None:
            return text, layout, resume
        new_text, new_layout = arranged
        if resume is None:
            return new_text, new_layout, None
        new_resume = continue_from(self.page_text, read_pos, new_text)
        if new_resume is None:                  # the new text cannot be lined up with what was read: keep the plain one
            return text, layout, resume
        return new_text, new_layout, new_resume

    def process(self, frame: np.ndarray, event: str, read_pos: int, reading: bool, thinking: bool, doc) -> dict:
        h, w = frame.shape[:2]
        try:
            text, layout = read_page(frame, enhance=True)
        except Exception as e:
            logger.error(f"Page reading error: {e}")
            text, layout = "", None

        # Same words as before (camera or book moved)? Carry on instead of starting over
        resume = continue_from(self.page_text, read_pos, text) if (self.page_text and text.strip()) else None
        # Different printed page numbers mean a different page, whatever the words say
        if resume is not None and layout and self.layout:
            old_nums, new_nums = set(self.layout.page_numbers()), set(layout.page_numbers())
            if old_nums and new_nums and not (old_nums & new_nums):
                resume = None

        if resume is not None and event == "moved" and reading:
            return {"action": "keep"}                       # a hand passed over: keep reading

        # The view only moved (the layout looks the same) and the words are mostly the same, but too garbled
        # to line up (vibration blur): it is still the same page. Never start it over from the first word.
        if (resume is None and event == "moved" and self.page_text and
                (not text.strip() or word_overlap(self.page_text, text) >= MOVED_SAME_PAGE)):
            logger.info("The view moved but the page is the same: carrying on")
            return {"action": "keep"}

        if resume is not None or text.strip():
            text, layout, resume = self._arrange(text, layout, resume, read_pos, doc)

        if resume is not None:
            self.page_text, self.layout = text, layout
            doc.update_document(text, layout.summary(), layout.label())
            if re.search(r"\w", text[resume:]) and not thinking:
                logger.info(f"Same words in view: continuing from \"{text[resume:resume + 40]}...\"")
                return {"action": "resume", "text": text, "from_pos": resume, "label": layout.label(),
                        "layout": layout_json(layout, w, h),
                        "chunks": [{"start": s, "text": t} for s, t in speech_chunks(text, layout.paragraphs, resume)]}
            logger.info("Same words in view: nothing new to read")
            return {"action": "nothing_new", "text": text, "label": layout.label(), "layout": layout_json(layout, w, h)}

        if text.strip():
            self.page_text, self.layout = text, layout
            doc.set_document(text, layout.summary(), layout.label())
            logger.info(f"{layout.label() or 'Page (no number found)'}, {len(text)} characters. {layout.summary()}")
            intro = []
            if layout.label():
                intro.append(layout.label() + ".")
            title = layout.running_title()
            if title and title.lower() != self.last_running_title.lower():
                intro.append(title + ".")
                self.last_running_title = title
            return {"action": "new", "text": text, "from_pos": 0, "label": layout.label(),
                    "intro": " ".join(intro), "layout": layout_json(layout, w, h),
                    "chunks": [{"start": s, "text": t} for s, t in speech_chunks(text, layout.paragraphs, 0)]}

        return {"action": "no_text"}
