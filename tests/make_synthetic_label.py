"""
Overlay compliance text onto AI-generated label background images, producing
PASS and FAIL variants for government warning font weight and capitalization
testing.

Output images are written to tests/sample_labels/ and referenced by the
CATALOG in make_test_pdf.py (with trim=False to suppress the saturation-based
content trimmer, which would strip the plain warning band).

Usage:
    uv run python tests/make_synthetic_label.py

Font requirement (Debian/Ubuntu):
    sudo apt-get install fonts-dejavu-core
Set PROOFREADER_LABEL_FONT_REGULAR and PROOFREADER_LABEL_FONT_BOLD to override.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).parent.parent
SAMPLE_LABELS = REPO / "tests" / "sample_labels"

# ---------------------------------------------------------------------------
# Government warning text
# ---------------------------------------------------------------------------

_WARNING_PREFIX_CORRECT = "GOVERNMENT WARNING:"
_WARNING_PREFIX_WRONG_CASE = "Government Warning:"
_WARNING_BODY = (
    "(1) According to the Surgeon General, women should not drink alcoholic "
    "beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a "
    "car or operate machinery, and may cause health problems."
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_BLACK = (0, 0, 0)
_DARK = (25, 25, 25)
_OFF_WHITE = (248, 246, 240)
_DIVIDER = (160, 155, 145)

# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_FONT_SEARCH: list[tuple[str, str]] = [
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ),
    # macOS
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
]


def _load_fonts(size: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Return (regular, bold) at *size* pt. Raises FileNotFoundError if not found."""
    reg_env = os.environ.get("PROOFREADER_LABEL_FONT_REGULAR")
    bold_env = os.environ.get("PROOFREADER_LABEL_FONT_BOLD")
    if reg_env and bold_env:
        return ImageFont.truetype(reg_env, size), ImageFont.truetype(bold_env, size)
    for reg_path, bold_path in _FONT_SEARCH:
        r, b = Path(reg_path), Path(bold_path)
        if r.exists() and b.exists():
            return ImageFont.truetype(str(r), size), ImageFont.truetype(str(b), size)
    raise FileNotFoundError(
        "No TrueType font pair found. Install fonts-dejavu-core:\n"
        "    sudo apt-get install fonts-dejavu-core\n"
        "or set PROOFREADER_LABEL_FONT_REGULAR and PROOFREADER_LABEL_FONT_BOLD."
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _line_h(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    max_width: int,
    fill: tuple[int, int, int] = _DARK,
    leading: int = 3,
) -> int:
    """Word-wrap *text* into *max_width* px. Returns y after the last line."""
    lh = _line_h(font)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if int(draw.textlength(candidate, font=font)) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += lh + leading
    return y


def _draw_government_warning(
    draw: ImageDraw.ImageDraw,
    warn_reg: ImageFont.FreeTypeFont,
    warn_bold: ImageFont.FreeTypeFont,
    x: int,
    y: int,
    max_width: int,
    variant: "WarningVariant",
) -> int:
    """
    Render the government warning statement.

    pass       — GOVERNMENT WARNING: bold + all-caps; body regular (correct)
    fail_case  — Government Warning: regular + title-case (wrong case, wrong weight)
    fail_bold  — GOVERNMENT WARNING: bold + all-caps; body also bold (wrong weight on body)
    """
    if variant == "pass":
        prefix, prefix_font, body_font = _WARNING_PREFIX_CORRECT, warn_bold, warn_reg
    elif variant == "fail_case":
        prefix, prefix_font, body_font = _WARNING_PREFIX_WRONG_CASE, warn_reg, warn_reg
    else:  # fail_bold
        prefix, prefix_font, body_font = _WARNING_PREFIX_CORRECT, warn_bold, warn_bold

    draw.text((x, y), prefix, font=prefix_font, fill=_DARK)
    y += _line_h(prefix_font) + 3
    y = _draw_wrapped(draw, _WARNING_BODY, body_font, x, y, max_width)
    return y


# ---------------------------------------------------------------------------
# Label specs
# ---------------------------------------------------------------------------

WarningVariant = Literal["pass", "fail_case", "fail_bold"]


@dataclass
class LabelSpec:
    background: str  # filename in SAMPLE_LABELS (AI-generated, no text)
    product_type: Literal["wine", "distilled_spirits", "malt_beverage"]
    brand: str
    class_type: str
    net_contents: str
    producer: str
    abv: str | None = None
    vintage: str | None = None
    warning_variants: list[WarningVariant] = field(default_factory=lambda: ["pass"])


SPECS: list[LabelSpec] = [
    # Distilled spirits — three warning variants to exercise font-weight detection
    LabelSpec(
        background="ai-generated-ds-brand1.jpg",
        product_type="distilled_spirits",
        brand="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
        producer="Bottled by Old Tom Distillery, Bardstown, KY 40004",
        warning_variants=["pass", "fail_case", "fail_bold"],
    ),
    LabelSpec(
        background="ai-generated-ds-brand2.jpg",
        product_type="distilled_spirits",
        brand="IRON RIDGE",
        class_type="Straight Rye Whiskey",
        abv="47.5% Alc./Vol. (95 Proof)",
        net_contents="750 mL",
        producer="Bottled by Iron Ridge Spirits Co., Louisville, KY 40202",
    ),
    LabelSpec(
        background="ai-generated-mb-brand.jpg",
        product_type="malt_beverage",
        brand="SUMMIT CREEK",
        class_type="Ale",
        # No ABV: malt beverage without added flavors — ABV optional per 27 CFR Part 7
        net_contents="12 fl oz (355 mL)",
        producer="Brewed and canned by Summit Creek Brewing Co., Denver, CO 80205",
    ),
    LabelSpec(
        background="ai-generated-wine-modern-brand.jpg",
        product_type="wine",
        brand="CRESTLINE",
        class_type="Napa Valley Cabernet Sauvignon",
        abv="14.2% Alc./Vol.",
        net_contents="750 mL",
        vintage="2021",
        producer="Produced and bottled by Crestline Vineyards, Napa, CA 94558",
    ),
    LabelSpec(
        background="ai-generated-wine-vintage-brand.jpg",
        product_type="wine",
        brand="MAISON DU SOLEIL",
        class_type="California Chardonnay",
        abv="13.5% Alc./Vol.",
        net_contents="750 mL",
        vintage="2022",
        producer="Produced and bottled by Maison du Soleil Winery, Sonoma, CA 95476",
    ),
]


def _output_name(spec: LabelSpec, variant: WarningVariant) -> str:
    stem = Path(spec.background).stem.removeprefix("ai-generated-")
    suffix = {
        "pass": "pass",
        "fail_case": "fail-warning-case",
        "fail_bold": "fail-warning-bold",
    }[variant]
    return f"synthetic-{stem}-{suffix}.jpg"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_label(spec: LabelSpec, variant: WarningVariant) -> Image.Image:
    """Composite compliance text onto the AI background for one variant."""
    bg = Image.open(SAMPLE_LABELS / spec.background).convert("RGB")
    W, H = bg.size

    margin = max(18, W // 28)
    usable_w = W - 2 * margin

    # Font sizes — proportional to image width, clamped to legible range
    brand_size = max(26, min(56, W // 18))
    body_size = max(14, min(28, W // 32))
    warn_size = max(10, min(16, W // 55))

    _, brand_bold = _load_fonts(brand_size)
    body_reg, _ = _load_fonts(body_size)
    warn_reg, warn_bold = _load_fonts(warn_size)

    # Warning band occupies bottom 27% of the image (solid off-white).
    # This matches the "plain area for text" from the generation prompts.
    warn_band_top = int(H * 0.73)
    warn_band_h = H - warn_band_top

    # Semi-transparent white info panel behind the main label text
    # (sits just above the warning band in the artwork area)
    info_panel_top = int(H * 0.32)
    info_panel_bot = warn_band_top - 6

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle(
        [(margin - 8, info_panel_top), (W - margin + 8, info_panel_bot)],
        fill=(255, 255, 255, 175),
    )
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(bg)

    # --- Warning band ---
    draw.rectangle([(0, warn_band_top), (W, H)], fill=_OFF_WHITE)
    draw.line(
        [(margin, warn_band_top + 1), (W - margin, warn_band_top + 1)],
        fill=_DIVIDER,
        width=1,
    )

    # --- Main label info ---
    y = info_panel_top + 12

    # Brand name (bold, centered)
    bw = int(draw.textlength(spec.brand, font=brand_bold))
    draw.text(((W - bw) // 2, y), spec.brand, font=brand_bold, fill=_DARK)
    y += _line_h(brand_bold) + 6

    # Class/type (centered)
    cw = int(draw.textlength(spec.class_type, font=body_reg))
    draw.text(((W - cw) // 2, y), spec.class_type, font=body_reg, fill=_DARK)
    y += _line_h(body_reg) + 4

    # Vintage (wine only)
    if spec.vintage:
        vw = int(draw.textlength(spec.vintage, font=body_reg))
        draw.text(((W - vw) // 2, y), spec.vintage, font=body_reg, fill=_DARK)
        y += _line_h(body_reg) + 4

    # ABV (if present)
    if spec.abv:
        aw = int(draw.textlength(spec.abv, font=body_reg))
        draw.text(((W - aw) // 2, y), spec.abv, font=body_reg, fill=_DARK)
        y += _line_h(body_reg) + 4

    # Net contents
    nw = int(draw.textlength(spec.net_contents, font=body_reg))
    draw.text(((W - nw) // 2, y), spec.net_contents, font=body_reg, fill=_DARK)
    y += _line_h(body_reg) + 6

    # Producer/bottler (left-aligned, wrapped)
    _draw_wrapped(draw, spec.producer, body_reg, margin, y, usable_w)

    # --- Government warning ---
    warn_y = warn_band_top + max(6, warn_band_h // 12)
    _draw_government_warning(draw, warn_reg, warn_bold, margin, warn_y, usable_w, variant)

    return bg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = SAMPLE_LABELS
    print(f"Writing synthetic label images to {out_dir.relative_to(REPO)}/")
    for spec in SPECS:
        bg_path = SAMPLE_LABELS / spec.background
        if not bg_path.exists():
            print(f"  SKIP  {spec.background} — not found")
            continue
        for variant in spec.warning_variants:
            out_name = _output_name(spec, variant)
            img = render_label(spec, variant)
            img.save(str(out_dir / out_name), "JPEG", quality=92)
            print(f"  wrote {out_name}")
    print("Done.")


if __name__ == "__main__":
    main()
