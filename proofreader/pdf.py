"""
TTB Form 5100.31 page-1 extraction.

Responsibilities:
  - Render page 1 of the submission PDF at a canonical DPI
  - Locate the "AFFIX COMPLETE SET OF LABELS BELOW" anchor line to confirm
    the page is the right form and in the correct orientation
  - Crop the label affixing zone for downstream LabelReader analysis
  - Detect the Item 5 product type (Wine / Distilled Spirits / Malt Beverages)
    by pixel-brightness analysis of the three checkbox regions

Both the anchor search and the product-type detection work on the *rendered*
raster image (via EasyOCR) so that scanned PDFs at any capture resolution
are handled the same way as native vector PDFs. For native PDFs, PyMuPDF's
vector search_for() is tried first as a fast path.

## Scale normalization

PDF coordinates are measured in *points* (1 pt = 1/72 inch), regardless of
how the document was scanned or what DPI the embedded raster image uses.
To convert a point coordinate to a pixel position in our working image:

    pixel = point * (RENDER_DPI / 72)

This scale factor (_SCALE) is applied throughout. All form-layout constants
below are in PDF points and remain correct for any input scan resolution.

## Terminal states and PII protection

extract_page1() always returns a Page1Result. When Page1Result.reason is
non-None the pipeline has reached a terminal state and automation stops.

label_zone is withheld (None) whenever reason is non-None, structurally
preventing potential PII from being sent to an external API when we cannot
confirm the submission is a valid Form 5100.31 in the correct orientation.
See models.REASON_* constants for the full set of terminal reason codes.

Public API:
    extract_page1(pdf_path) -> Page1Result
"""

import logging
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from proofreader.models import Page1Result, Reason
from proofreader.ocr import get_engine, ocr_lock

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants derived from form analysis
# (see reference/f510031.pdf inspection in development notes)
# ---------------------------------------------------------------------------

# We always render to this DPI for our working image, regardless of the
# input PDF's capture resolution. See module docstring for scale math.
# 300 DPI is chosen for label zone quality: government warning text can be as
# small as 1-2 mm, which needs ~12-16 px/mm for reliable vision-model reading.
# Anchor detection and Item 5 checkbox analysis work fine at any DPI above ~100.
RENDER_DPI = 300
_SCALE = RENDER_DPI / 72.0  # pt * (DPI/72) -> pixel; see module docstring

ANCHOR_TEXT = "AFFIX COMPLETE SET OF LABELS BELOW"

# Label affixing zone in PDF points (from form white-fill drawing rect)
_LABEL_BOX_PTS = pymupdf.Rect(24.6, 681.2, 589.4, 979.1)

# Expected y-range of the anchor text in PDF points (sanity check)
_ANCHOR_Y_MIN = 660.0
_ANCHOR_Y_MAX = 700.0

# Individual checkbox rects in PDF points (9.3x9.3 pt unfilled squares)
_CHECKBOX_PTS: dict[str, pymupdf.Rect] = {
    "wine": pymupdf.Rect(147.06, 170.10, 156.36, 179.40),
    "distilled_spirits": pymupdf.Rect(147.06, 181.14, 156.36, 190.44),
    "malt_beverage": pymupdf.Rect(147.06, 192.12, 156.36, 201.42),
}

# A checkbox is considered "checked" when the mean pixel brightness of its
# region is below this threshold (0 = black, 255 = white).
_CHECKBOX_DARK_THRESHOLD = 200

# Minimum OCR confidence to trust a text hit
_OCR_CONF_MIN = 0.6



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pts_to_px(val: float) -> int:
    return round(val * _SCALE)


def _rect_to_box(rect: pymupdf.Rect) -> tuple[int, int, int, int]:
    """Convert a PDF-point Rect to a pixel crop box at RENDER_DPI."""
    return (
        _pts_to_px(rect.x0),
        _pts_to_px(rect.y0),
        _pts_to_px(rect.x1),
        _pts_to_px(rect.y1),
    )


def _render_page1(page: pymupdf.Page) -> Image.Image:
    # Default background with alpha=False is white
    pix = page.get_pixmap(matrix=pymupdf.Matrix(_SCALE, _SCALE), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _find_anchor_vector(page: pymupdf.Page) -> bool:
    """Return True if the anchor text is found at the expected location via
    PyMuPDF vector search (fast path for native PDFs)."""
    hits = page.search_for(ANCHOR_TEXT)
    if not hits:
        return False
    y0 = hits[0].y0
    return _ANCHOR_Y_MIN < y0 < _ANCHOR_Y_MAX


def _has_selectable_text(page: pymupdf.Page) -> bool:
    """Return True if the page contains any selectable (vector) text.

    Used to short-circuit the OCR fallback: if a PDF already has native text
    but doesn't contain the anchor, there is no point running OCR — the page
    is definitively not a Form 5100.31 anchor page.  This also avoids running
    PaddleOCR on dense instruction pages, which is slow.
    """
    return bool(page.get_text().strip())


def _find_anchor_ocr(page_img: Image.Image) -> bool:
    """Return True if 'AFFIX' appears near its expected location via OCR
    (fallback for scanned PDFs with no selectable text).

    We search a narrow horizontal strip around the known anchor y-position
    rather than the full page — this keeps the OCR input small and fast.
    """
    # Anchor expected near y≈679 pts → y≈_pts_to_px(679) pixels.
    # Margin: ~15 pt for the text line height + ~40 pt physical tolerance for
    # scan misalignment. Both expressed in points so _pts_to_px scales correctly
    # with RENDER_DPI (a raw pixel constant would not).
    cy = _pts_to_px((_ANCHOR_Y_MIN + _ANCHOR_Y_MAX) / 2)
    margin = _pts_to_px(15 + 40)  # form label height + scan misalignment tolerance
    y0 = max(0, cy - margin)
    y1 = min(page_img.height, cy + margin)
    strip = page_img.crop((0, y0, page_img.width, y1))

    engine = get_engine()
    try:
        with ocr_lock:
            result, _ = engine(np.array(strip.convert("RGB")))
    except Exception as exc:
        logger.warning("OCR inference failed in anchor search (%s); treating as not found", exc)
        return False
    if result is None:
        return False
    for _bbox, text, conf in result:
        if "AFFIX" in text.upper() and conf >= _OCR_CONF_MIN:
            return True
    return False


def _detect_product_type_checkbox(page_img: Image.Image) -> str | None:
    """Check pixel brightness in each Item 5 checkbox region.
    Returns a product type if one checkbox is clearly darker than the others,
    or None if no checkbox appears filled."""
    grey = np.array(page_img.convert("L"))
    brightness: dict[str, float] = {}
    for product, rect in _CHECKBOX_PTS.items():
        box = _rect_to_box(rect)
        region = grey[box[1] : box[3], box[0] : box[2]]
        if region.size > 0:
            brightness[product] = float(region.mean())

    if not brightness:
        return None

    darkest = min(brightness, key=lambda k: brightness[k])
    if brightness[darkest] < _CHECKBOX_DARK_THRESHOLD:
        return darkest

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_page1(pdf_path: Path) -> Page1Result:
    """Extract product type and label zone from page 1 of a TTB Form 5100.31.

    Always returns a Page1Result. When result.reason is non-None the pipeline
    has reached a terminal state and the submission should be flagged for human
    review. See models.REASON_* constants for the reason codes and their meanings.

    When result.reason is None, extraction succeeded and the anchor was confirmed:
      - label_zone is the cropped label affixing area, safe to send to LabelReader
      - product_type is "wine" | "distilled_spirits" | "malt_beverage" | None
        (None means Item 5 was indeterminate; non-terminal)
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Could not open %s: %s", pdf_path, exc)
        return Page1Result(
            reason=Reason.PDF_UNREADABLE,
            product_type=None,
            label_zone=None,
            page1_image=None,
        )

    if len(doc) == 0:
        logger.warning("Empty PDF: %s", pdf_path)
        return Page1Result(
            reason=Reason.PDF_EMPTY,
            product_type=None,
            label_zone=None,
            page1_image=None,
        )

    page = doc[0]
    page_img = _render_page1(page)

    # OCR fallback is only useful for scanned (raster) PDFs that have no
    # selectable text.  If the page already has native text but the anchor
    # wasn't found by vector search, OCR won't help and would be slow.
    if _find_anchor_vector(page):
        anchor_found = True
    elif not _has_selectable_text(page):
        anchor_found = _find_anchor_ocr(page_img)
    else:
        anchor_found = False
    if not anchor_found:
        logger.info(
            "Anchor text not found in %s — page may be rotated, missing, or not a Form 5100.31",
            pdf_path,
        )
        # label_zone withheld: cannot confirm this crop is label artwork, not PII.
        return Page1Result(
            reason=Reason.ANCHOR_NOT_FOUND,
            product_type=None,
            label_zone=None,
            page1_image=page_img,
        )

    label_zone = page_img.crop(_rect_to_box(_LABEL_BOX_PTS))

    # Detect Item 5 product type via checkbox brightness.
    # OCR fallback intentionally omitted: all three product-type labels are
    # always printed on the form, so text alone cannot identify the selection.
    product_type = _detect_product_type_checkbox(page_img)
    if product_type is None:
        logger.info("Item 5 product type indeterminate in %s", pdf_path)

    return Page1Result(
        reason=None,
        product_type=product_type,
        label_zone=label_zone,
        page1_image=page_img,
    )
