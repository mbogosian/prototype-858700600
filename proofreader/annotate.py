"""
Label image annotation for ProofReader.

Takes the label zone image and LabelFindings and returns an annotated copy
with colored outlines drawn over each located field:
  Green  — PASS
  Orange — WARN
  Red    — FAIL
  Dashed outline — location approximate (curved text; OCR could not localize)
  No box — ABSENT, INDETERMINATE, ERROR (report text only)

Text localization uses EasyOCR to match extracted field text back to pixel
coordinates in the label zone. EasyOCR returns four-point quadrilateral
coordinates; we draw these natively using ImageDraw line segments rather than
converting to axis-aligned rectangles.

For fields where the vision model found text but OCR could not localize it
(e.g. curved text along a design element), we draw a dashed outline around
the full label zone to indicate that the field was found but cannot be
precisely located. This is distinct from ABSENT/INDETERMINATE/ERROR fields,
which receive no annotation.

Text matching normalizes both sides to lowercase alphanumeric tokens and checks
for substring overlap, so minor formatting differences (punctuation, extra
spaces) do not prevent a match. Short tokens (< 3 chars) are skipped to avoid
spurious hits.

Known limitation: multi-line fields (e.g. the government warning statement)
may produce multiple OCR hits that are each annotated individually. This is
intentional — each hit is drawn in its field's color so reviewers can see
the full extent of the relevant text.

Public API:
    annotate(label_zone, findings) -> Image.Image
"""

import logging
import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from proofreader.models import LabelFindings, Verdict
from proofreader.ocr import get_engine, ocr_lock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RGB colors for each annotatable verdict.
_VERDICT_COLORS: dict[Verdict, tuple[int, int, int]] = {
    Verdict.PASS: (34, 139, 34),  # forest green
    Verdict.EXEMPT: (34, 139, 34),  # forest green (same as PASS — absent but not required)
    Verdict.WARN: (255, 140, 0),  # dark orange
    Verdict.FAIL: (200, 30, 30),  # crimson
}

# Pixel width of drawn outlines.
_OUTLINE_WIDTH = 3

# Minimum OCR confidence to use a result for matching.
_OCR_CONF_MIN = 0.5

# Minimum normalized token length to attempt a match (avoids spurious hits
# on short function words like "by", "of", etc.).
_MIN_MATCH_LEN = 3

# Dash and gap lengths (px) for the approximate-location dashed outline.
_DASH_LEN = 12
_GAP_LEN = 6

# Inset (px) for the approximate-location dashed rectangle so it sits inside
# the image border.
_APPROX_INSET = 4

# Guard against pathological image sizes. Label zones at 300 DPI are typically
# ~2400 x 1200 px; images outside these bounds suggest a rendering anomaly and
# are skipped so all fields fall back to dashed approximate outlines.
_OCR_MIN_DIM = 32  # px — shorter side must be at least this
_OCR_MAX_DIM = 4000  # px — longer side must be no more than this


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_ocr(image: Image.Image) -> list[tuple[list[list[float]], str, float]]:
    """Run RapidOCR on image; return (quad, text, confidence) triples.

    quad is a list of four [x, y] points in clockwise order. An empty list
    is returned when OCR finds no text, the result is malformed, or the image
    falls outside the safe size range (_OCR_MIN_DIM / _OCR_MAX_DIM).
    """
    w, h = image.size
    if min(w, h) < _OCR_MIN_DIM or max(w, h) > _OCR_MAX_DIM:
        logger.warning(
            "Skipping OCR: image size %dx%d is outside safe range [%d, %d]",
            w,
            h,
            _OCR_MIN_DIM,
            _OCR_MAX_DIM,
        )
        return []
    engine = get_engine()
    try:
        with ocr_lock:
            result, _ = engine(np.array(image.convert("RGB")))
    except Exception as exc:
        logger.warning("OCR inference failed (%s); skipping text localization", exc)
        return []
    if result is None:
        return []
    return [(bbox, text, float(conf)) for bbox, text, conf in result]


def _normalize(s: str) -> str:
    """Normalize text for matching: lowercase; keep only alphanumeric and spaces.

    >>> _normalize("GOVERNMENT WARNING: (1)")
    'government warning 1'
    >>> _normalize("  Alc. 13.5% Vol.  ")
    'alc 135 vol'
    """
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())


def _find_matching_quads(
    extracted: str,
    ocr_results: list[tuple[list[list[float]], str, float]],
) -> list[list[list[float]]]:
    """Return quads for OCR results that overlap with extracted text.

    Overlap is determined by normalized substring containment: an OCR token
    matches if its normalized form is contained in (or contains) the
    normalized extracted text, subject to _MIN_MATCH_LEN and _OCR_CONF_MIN.
    """
    norm_extracted = _normalize(extracted)
    matched = []
    for quad, text, conf in ocr_results:
        if conf < _OCR_CONF_MIN:
            continue
        norm_text = _normalize(text)
        if len(norm_text) < _MIN_MATCH_LEN:
            continue
        if norm_text in norm_extracted or norm_extracted in norm_text:
            matched.append(quad)
    return matched


def _draw_quad(
    draw: ImageDraw.ImageDraw,
    quad: Sequence[Sequence[int | float]],
    color: tuple[int, int, int],
    width: int = _OUTLINE_WIDTH,
) -> None:
    """Draw a quadrilateral outline by connecting the four points in order."""
    pts = [(round(pt[0]), round(pt[1])) for pt in quad]
    for i in range(len(pts)):
        draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=color, width=width)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    width: int = _OUTLINE_WIDTH,
    dash: int = _DASH_LEN,
    gap: int = _GAP_LEN,
) -> None:
    """Draw a dashed line segment from (x0, y0) to (x1, y1)."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    nx, ny = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    drawing = True
    while pos < length:
        end = min(pos + (dash if drawing else gap), length)
        if drawing:
            draw.line(
                [(x0 + nx * pos, y0 + ny * pos), (x0 + nx * end, y0 + ny * end)],
                fill=color,
                width=width,
            )
        pos = end
        drawing = not drawing


def _draw_dashed_rect(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a dashed rectangle outline defined by (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bbox
    for ax, ay, bx, by in [
        (x0, y0, x1, y0),  # top
        (x1, y0, x1, y1),  # right
        (x1, y1, x0, y1),  # bottom
        (x0, y1, x0, y0),  # left
    ]:
        _draw_dashed_line(draw, ax, ay, bx, by, color)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate(label_zone: Image.Image, findings: LabelFindings) -> Image.Image:
    """Annotate label_zone with per-field verdict outlines.

    Returns an annotated copy. The original image is not modified.

    For each field with a PASS/WARN/FAIL verdict and non-None extracted text:
      - OCR is run on label_zone to locate the text in pixel space.
      - Matching OCR results are outlined in the verdict color.
      - If no OCR match is found (e.g. curved text), a dashed rectangle is
        drawn around the full label zone in the verdict color to indicate
        "found but not precisely locatable."

    ABSENT, INDETERMINATE, and ERROR fields receive no annotation; they are
    described only in the report text.
    """
    out = label_zone.copy()
    draw = ImageDraw.Draw(out)

    annotatable = [f for f in findings.fields if f.verdict in _VERDICT_COLORS and f.extracted]
    if not annotatable:
        return out

    ocr_results = _run_ocr(label_zone)
    approx_bbox = (
        _APPROX_INSET,
        _APPROX_INSET,
        out.width - _APPROX_INSET,
        out.height - _APPROX_INSET,
    )

    for field in annotatable:
        assert field.extracted is not None  # narrowed above
        color = _VERDICT_COLORS[field.verdict]
        quads = _find_matching_quads(field.extracted, ocr_results)
        if quads:
            for quad in quads:
                _draw_quad(draw, quad, color)
        else:
            _draw_dashed_rect(draw, approx_bbox, color)

    return out
