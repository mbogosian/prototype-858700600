"""
Generate synthetic test PDFs for proofreader development.

Takes the blank TTB Form F 5100.31, rasterizes page 1 at a configurable DPI
(default 150; use "scan_dpi" in a test spec to vary), composites label images
into the lower zone using Pillow, and saves the result as a PDF.
Pages 2-5 (form instructions) are appended unchanged.

Label source images come from two places:
  - tests/sample_labels/allowable-revision-comparison-*.jpg  (TTB comparison slides)
  - tests/sample_labels/allowable-revisions-pdf-*.jpeg       (extracted from TTB PDF)

Each comparison slide contains two label versions side by side; we crop to the
relevant half as noted in the catalog below. Many slides also contain
explanatory text above or below the label; we strip that using a saturation-
based row-content detector (colorful label rows have HSV saturation > SAT_THRESHOLD;
grayscale text on white has saturation ~0).

Usage:
    uv run python tests/make_test_pdf.py
"""

import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).parent.parent
FORM_PDF = REPO / "reference" / "f510031.pdf"
SAMPLE_LABELS = REPO / "tests" / "sample_labels"
OUT_DIR = REPO / "tests" / "sample_applications"

ANCHOR_TEXT = "AFFIX COMPLETE SET OF LABELS BELOW"

# Default raster DPI for test PDFs that don't specify scan_dpi.
# Individual test cases may override this via "scan_dpi" to simulate
# scanners at different resolutions (200 DPI, 400 DPI, etc.).
#
# Scale normalization: PDF coordinates are in points (1 pt = 1/72 inch).
# To convert a point value to pixels at a given DPI:
#   pixel = point * (DPI / 72)
# This factor is recomputed per test case as `scale = scan_dpi / 72`
# and passed explicitly to avoid relying on a mutable global.
RENDER_DPI = 150
SCALE = RENDER_DPI / 72  # default; overridden per test case when scan_dpi differs

# Label affixing area in PDF points (from form's white-fill drawing rect)
LABEL_BOX_PTS = pymupdf.Rect(24.6, 681.2, 589.4, 979.1)
LABEL_INSET_PTS = 8

# Rows whose HSV saturation mean exceeds this are considered label content.
# Grayscale text on white background has mean saturation ~0; colorful labels > 40.
SAT_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Label catalog
#
# Each entry describes one extractable label from a source image:
#   image        — filename in SAMPLE_LABELS
#   crop         — "left" | "right" | None  (which half of comparison slide)
#   label_type   — "brand" | "back" | "combined"
#   product_type — "wine" | "distilled_spirits" | "malt_beverage" | None
#
# Images without an entry can be used directly (no crop, no text strip).
# ---------------------------------------------------------------------------

CATALOG: dict[str, dict] = {
    # --- Wine ---------------------------------------------------------------
    "wine-brand-01": {
        "image": "allowable-revision-comparison-awards-medals-01.jpg",
        "crop": "left",
        "label_type": "brand",
        "product_type": "wine",
    },
    "wine-back-01": {
        "image": "allowable-revision-comparison-bottle-deposit-info-01.jpg",
        "crop": "right",
        "label_type": "back",
        "product_type": "wine",
    },
    # From reference PDF (wine — "Guiding White" label set)
    "wine-brand-pdf": {
        "image": "allowable-revisions-pdf-brand-label-approved.jpeg",
        "crop": None,
        "label_type": "brand",
        "product_type": "wine",
    },
    "wine-back-pdf": {
        "image": "allowable-revisions-pdf-back-label-approved.jpeg",
        "crop": None,
        "label_type": "back",
        "product_type": "wine",
    },
    "wine-brand-pdf-changed": {
        "image": "allowable-revisions-pdf-brand-label-with-changes.jpeg",
        "crop": None,
        "label_type": "brand",
        "product_type": "wine",
    },
    "wine-back-pdf-changed": {
        "image": "allowable-revisions-pdf-back-label-with-changes.jpeg",
        "crop": None,
        "label_type": "back",
        "product_type": "wine",
    },
    # --- Distilled spirits --------------------------------------------------
    "ds-brand-01": {
        "image": "allowable-revision-comparison-awards-medals-02.jpg",
        "crop": "left",
        "label_type": "brand",
        "product_type": "distilled_spirits",
    },
    "ds-back-01": {
        "image": "allowable-revision-comparison-bottle-deposit-info-02.jpg",
        "crop": "left",
        "label_type": "back",
        "product_type": "distilled_spirits",
    },
    # --- Malt beverage ------------------------------------------------------
    "mb-brand-01": {
        "image": "allowable-revision-comparison-awards-medals-03.jpg",
        "crop": "left",
        "label_type": "brand",
        "product_type": "malt_beverage",
    },
    "mb-back-01": {
        "image": "allowable-revision-comparison-upc-and-2d-barcodes-03.jpg",
        "crop": "left",
        "label_type": "back",
        "product_type": "malt_beverage",
    },
    "mb-combined-01": {
        "image": "allowable-revision-comparison-bottle-deposit-info-03.jpg",
        "crop": "left",
        "label_type": "combined",
        "product_type": "malt_beverage",
    },
}


# ---------------------------------------------------------------------------
# Test case definitions
#
# Each entry produces one output PDF.
# "labels" — ordered list of catalog keys; placed left-to-right in the zone.
# "rotate_page1" — degrees CW to rotate the rasterized page 1.
# "skip_page1"   — if True, produce only instruction pages 2-5.
# ---------------------------------------------------------------------------

TESTS = [
    # --- Normal: wine -------------------------------------------------------
    {
        "output": "test-01-wine-brand-back-slides.pdf",
        "desc": "wine: brand + back (from comparison slides)",
        "labels": ["wine-brand-01", "wine-back-01"],
    },
    {
        "output": "test-02-wine-brand-back-pdf.pdf",
        "desc": "wine: brand + back (from allowable-revisions PDF)",
        "labels": ["wine-brand-pdf", "wine-back-pdf"],
    },
    # --- Normal: distilled spirits ------------------------------------------
    {
        "output": "test-03-ds-brand-back.pdf",
        "desc": "distilled spirits: brand + back",
        "labels": ["ds-brand-01", "ds-back-01"],
    },
    # --- Normal: malt beverage ----------------------------------------------
    {
        "output": "test-04-mb-combined.pdf",
        "desc": "malt beverage: single combined brand+back label",
        "labels": ["mb-combined-01"],
    },
    {
        "output": "test-05-mb-brand-back.pdf",
        "desc": "malt beverage: brand + back (separate labels)",
        "labels": ["mb-brand-01", "mb-back-01"],
    },
    {
        "output": "test-10-wine-brand-back-pdf-changed.pdf",
        "desc": "wine: brand + back (allowable-revisions PDF, post-change versions)",
        "labels": ["wine-brand-pdf-changed", "wine-back-pdf-changed"],
    },
    # --- Edge cases ---------------------------------------------------------
    {
        "output": "test-06-no-labels.pdf",
        "desc": "form with nothing affixed in label zone",
        "labels": [],
    },
    {
        "output": "test-07-bad-image.pdf",
        "desc": "label zone present but badly scanned (blurry, low contrast)",
        "labels": ["wine-brand-01"],
        "degrade": True,
    },
    {
        "output": "test-08-rotated-page1.pdf",
        "desc": "page 1 scanned sideways (90° CW rotation)",
        "labels": ["wine-brand-01", "wine-back-01"],
        "rotate_page1": 90,
    },
    {
        "output": "test-09-missing-page1.pdf",
        "desc": "only instruction pages present; page 1 absent",
        "skip_page1": True,
    },
    # --- Varying scan resolution --------------------------------------------
    {
        "output": "test-11-wine-200dpi-scan.pdf",
        "desc": "wine brand + back, simulated 200 DPI scan",
        "labels": ["wine-brand-01", "wine-back-01"],
        "scan_dpi": 200,
    },
    {
        "output": "test-12-wine-400dpi-scan.pdf",
        "desc": "wine brand + back, simulated 400 DPI scan",
        "labels": ["wine-brand-01", "wine-back-01"],
        "scan_dpi": 400,
    },
]


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------


def _sat_content_rows(img: Image.Image) -> np.ndarray:
    """Return boolean array (one entry per row) — True where mean HSV
    saturation exceeds SAT_THRESHOLD (i.e. the row contains label content)."""
    hsv = np.array(img.convert("HSV"))
    row_mean_sat = hsv[:, :, 1].mean(axis=1)
    return row_mean_sat > SAT_THRESHOLD


def trim_to_content(img: Image.Image) -> Image.Image:
    """Crop top and bottom white/text margins, keeping only rows that
    contain colorful label content."""
    content = _sat_content_rows(img)
    rows = np.where(content)[0]
    if len(rows) == 0:
        return img  # nothing to trim; return as-is
    top = max(0, rows[0] - 4)
    bot = min(img.height, rows[-1] + 5)
    return img.crop((0, top, img.width, bot))


def extract_label(key: str) -> Image.Image:
    """Load, crop, and trim a label image according to its catalog entry."""
    entry = CATALOG[key]
    img = Image.open(SAMPLE_LABELS / entry["image"]).convert("RGB")

    # Horizontal crop (left/right half of comparison slide)
    side = entry.get("crop")
    if side is not None:
        w, h = img.size
        half = w // 2
        img = img.crop((0, 0, half, h) if side == "left" else (half, 0, w, h))

    # Vertical trim: remove white borders and explanatory text bands
    img = trim_to_content(img)
    return img


def scale_to_fit(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Scale img (up or down) to fill as much of box_w x box_h as possible
    while preserving aspect ratio."""
    scale = min(box_w / img.width, box_h / img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def degrade(img: Image.Image) -> Image.Image:
    """Simulate a poor-quality scan: blur + reduce contrast + re-compress."""
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img = Image.blend(img, Image.new("RGB", img.size, (128, 128, 128)), alpha=0.35)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=8)
    buf.seek(0)
    return Image.open(buf).copy()


# Item 5 checkbox positions in PDF points — must match _CHECKBOX_PTS in pdf.py.
_CHECKBOX_PTS: dict[str, pymupdf.Rect] = {
    "wine": pymupdf.Rect(147.06, 170.10, 156.36, 179.40),
    "distilled_spirits": pymupdf.Rect(147.06, 181.14, 156.36, 190.44),
    "malt_beverage": pymupdf.Rect(147.06, 192.12, 156.36, 201.42),
}


def pts_to_px(val: float, scale: float) -> int:
    return round(val * scale)


def fill_checkbox(form_img: Image.Image, product_type: str | None, scale: float) -> Image.Image:
    """Fill the Item 5 checkbox for the given product type with a solid black square.

    The blank form has empty checkbox outlines.  Without filling one, all test
    PDFs would return product_type=None from _detect_product_type_checkbox.
    """
    if product_type is None:
        return form_img
    rect = _CHECKBOX_PTS.get(product_type)
    if rect is None:
        return form_img
    x0 = round(rect.x0 * scale)
    y0 = round(rect.y0 * scale)
    x1 = round(rect.x1 * scale)
    y1 = round(rect.y1 * scale)
    draw = ImageDraw.Draw(form_img)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    return form_img


def get_product_type(test: dict) -> str | None:
    """Infer product type from the first label in the test case."""
    labels = test.get("labels", [])
    if not labels:
        return None
    return CATALOG[labels[0]].get("product_type")


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------


def rasterize_page1(doc: pymupdf.Document, scale: float) -> Image.Image:
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def composite_labels(
    form_img: Image.Image,
    label_keys: list[str],
    scale: float,
    do_degrade: bool = False,
) -> Image.Image:
    if not label_keys:
        return form_img

    inset = LABEL_INSET_PTS
    bx0 = pts_to_px(LABEL_BOX_PTS.x0 + inset, scale)
    by0 = pts_to_px(LABEL_BOX_PTS.y0 + inset, scale)
    bx1 = pts_to_px(LABEL_BOX_PTS.x1 - inset, scale)
    by1 = pts_to_px(LABEL_BOX_PTS.y1 - inset, scale)
    box_w = bx1 - bx0
    box_h = by1 - by0

    n = len(label_keys)
    slot_w = box_w // n

    for i, key in enumerate(label_keys):
        lbl = extract_label(key)
        if do_degrade:
            lbl = degrade(lbl)
        lbl = scale_to_fit(lbl, slot_w, box_h)

        slot_x0 = bx0 + i * slot_w
        paste_x = slot_x0 + (slot_w - lbl.width) // 2
        paste_y = by0 + (box_h - lbl.height) // 2
        form_img.paste(lbl, (paste_x, paste_y))

    return form_img


def page1_to_pdf_page(
    out_doc: pymupdf.Document,
    img: Image.Image,
    orig_page: pymupdf.Page,
) -> None:
    out_page = out_doc.new_page(width=orig_page.rect.width, height=orig_page.rect.height)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    out_page.insert_image(out_page.rect, stream=buf.read())


def make_test_pdf(test: dict) -> None:
    output = OUT_DIR / test["output"]
    doc = pymupdf.open(str(FORM_PDF))

    if test.get("skip_page1"):
        out_doc = pymupdf.open()
        out_doc.insert_pdf(doc, from_page=1, to_page=len(doc) - 1)
        output.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(output))
        return

    if not doc[0].search_for(ANCHOR_TEXT):
        raise RuntimeError(f"Anchor text not found in {FORM_PDF}")

    scan_dpi = test.get("scan_dpi", RENDER_DPI)
    scale = scan_dpi / 72

    form_img = rasterize_page1(doc, scale)
    form_img = fill_checkbox(form_img, get_product_type(test), scale)
    form_img = composite_labels(
        form_img,
        test.get("labels", []),
        scale,
        do_degrade=test.get("degrade", False),
    )

    if test.get("rotate_page1"):
        form_img = form_img.rotate(-test["rotate_page1"], expand=True)

    out_doc = pymupdf.open()
    page1_to_pdf_page(out_doc, form_img, doc[0])
    out_doc.insert_pdf(doc, from_page=1, to_page=len(doc) - 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(output))


def main() -> None:
    print(f"Generating test PDFs in {OUT_DIR.relative_to(REPO)}/")
    for test in TESTS:
        try:
            make_test_pdf(test)
            print(f"  wrote {test['output']}  ({test.get('desc', '')})")
        except Exception as exc:
            print(f"  FAIL  {test['output']}: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
