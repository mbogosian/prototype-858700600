"""
Unit tests for annotate.py — text normalization, matching, drawing, and full annotate().

All tests are fast: PaddleOCR is patched out via proofreader.annotate._run_ocr.
Image pixel comparisons use numpy to verify that drawing actually modified pixels.
"""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from proofreader.annotate import (
    _VERDICT_COLORS,
    _draw_dashed_rect,
    _draw_quad,
    _find_matching_quads,
    _normalize,
    annotate,
)
from proofreader.models import FieldFinding, LabelFindings, Verdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_image(w: int = 200, h: int = 200) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _blank_findings(**kwargs) -> LabelFindings:
    """LabelFindings with no fields unless overridden."""
    return LabelFindings(verdict=Verdict.PASS, fields=kwargs.get("fields", []))


def _ocr_hit(text: str, conf: float = 0.95, x=10, y=10, w=80, h=20):
    """Return a fake OCR result tuple: (quad, text, confidence)."""
    quad = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return (quad, text, conf)


def _pixels_changed(before: Image.Image, after: Image.Image) -> bool:
    """Return True if any pixel differs between the two images."""
    return bool(np.any(np.array(before) != np.array(after)))


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_lowercases() -> None:
    assert _normalize("HELLO") == "hello"


def test_normalize_strips_punctuation() -> None:
    assert _normalize("GOVERNMENT WARNING: (1)") == "government warning 1"


def test_normalize_collapses_whitespace() -> None:
    assert _normalize("  foo   bar  ") == "foo bar"


def test_normalize_strips_percent_and_dot() -> None:
    assert _normalize("Alc. 13.5% Vol.") == "alc 135 vol"


def test_normalize_empty_string() -> None:
    assert _normalize("") == ""


# ---------------------------------------------------------------------------
# _find_matching_quads
# ---------------------------------------------------------------------------


def test_find_matching_quads_exact_match() -> None:
    ocr = [_ocr_hit("Example Brand")]
    quads = _find_matching_quads("Example Brand", ocr)
    assert len(quads) == 1


def test_find_matching_quads_case_insensitive() -> None:
    ocr = [_ocr_hit("EXAMPLE BRAND")]
    quads = _find_matching_quads("Example Brand", ocr)
    assert len(quads) == 1


def test_find_matching_quads_substring_of_extracted() -> None:
    """OCR token is a substring of the extracted text."""
    ocr = [_ocr_hit("GOVERNMENT")]
    quads = _find_matching_quads("GOVERNMENT WARNING: ...", ocr)
    assert len(quads) == 1


def test_find_matching_quads_extracted_substring_of_ocr() -> None:
    """Extracted text is a substring of the OCR token."""
    ocr = [_ocr_hit("EXAMPLE BRAND WINERY")]
    quads = _find_matching_quads("Example Brand", ocr)
    assert len(quads) == 1


def test_find_matching_quads_no_match() -> None:
    ocr = [_ocr_hit("Totally Unrelated Text")]
    quads = _find_matching_quads("Example Brand", ocr)
    assert quads == []


def test_find_matching_quads_low_confidence_skipped() -> None:
    ocr = [_ocr_hit("Example Brand", conf=0.3)]
    quads = _find_matching_quads("Example Brand", ocr)
    assert quads == []


def test_find_matching_quads_short_token_skipped() -> None:
    """Short tokens (< 3 chars after normalization) are not matched."""
    ocr = [_ocr_hit("by", conf=0.99)]
    quads = _find_matching_quads("Bottled by Example Winery", ocr)
    assert quads == []


def test_find_matching_quads_multiple_hits() -> None:
    ocr = [
        _ocr_hit("GOVERNMENT", x=10, y=10),
        _ocr_hit("WARNING", x=100, y=10),
        _ocr_hit("Unrelated", x=10, y=50),
    ]
    quads = _find_matching_quads("GOVERNMENT WARNING: ...", ocr)
    assert len(quads) == 2


def test_find_matching_quads_empty_ocr() -> None:
    assert _find_matching_quads("Example Brand", []) == []


# ---------------------------------------------------------------------------
# _draw_quad
# ---------------------------------------------------------------------------


def test_draw_quad_modifies_pixels() -> None:
    img = _white_image()
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    quad = [[10, 10], [60, 10], [60, 40], [10, 40]]
    _draw_quad(draw, quad, color=(0, 200, 0))
    # At least some pixels should be green-ish after drawing.
    arr = np.array(img)
    # Check that the top edge has some green pixels (y=10, x in 10..60)
    top_row = arr[10, 10:61]
    assert any(px[1] > 150 and px[0] < 100 for px in top_row), "expected green pixels on top edge"


def test_draw_quad_does_not_fill_interior() -> None:
    """Outline only — interior pixels should remain white."""
    img = _white_image(100, 100)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    quad = [[10, 10], [90, 10], [90, 90], [10, 90]]
    _draw_quad(draw, quad, color=(200, 0, 0))
    arr = np.array(img)
    # Interior pixel should be unchanged (white).
    cx, cy = 50, 50
    assert list(arr[cy, cx]) == [255, 255, 255], "interior pixel should remain white"


# ---------------------------------------------------------------------------
# _draw_dashed_rect
# ---------------------------------------------------------------------------


def test_draw_dashed_rect_modifies_pixels() -> None:
    img = _white_image(200, 200)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    _draw_dashed_rect(draw, (10, 10, 190, 190), color=(255, 140, 0))
    arr = np.array(img)
    # Some pixels on the top edge should be orange-ish.
    top_row = arr[10, 10:190]
    has_orange = any(px[0] > 200 and px[1] > 100 and px[2] < 50 for px in top_row)
    assert has_orange, "expected orange pixels on top edge of dashed rect"


def test_draw_dashed_rect_does_not_fill_interior() -> None:
    img = _white_image(200, 200)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    _draw_dashed_rect(draw, (10, 10, 190, 190), color=(200, 0, 0))
    arr = np.array(img)
    assert list(arr[100, 100]) == [255, 255, 255], "interior pixel should remain white"


# ---------------------------------------------------------------------------
# annotate() — return value and image properties
# ---------------------------------------------------------------------------


def test_annotate_returns_image() -> None:
    result = annotate(_white_image(), _blank_findings())
    assert isinstance(result, Image.Image)


def test_annotate_preserves_dimensions() -> None:
    img = _white_image(320, 240)
    result = annotate(img, _blank_findings())
    assert result.size == (320, 240)


def test_annotate_returns_copy() -> None:
    img = _white_image()
    result = annotate(img, _blank_findings())
    assert result is not img


def test_annotate_does_not_modify_original() -> None:
    img = _white_image()
    original_data = np.array(img).copy()
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[_ocr_hit("Example Brand", x=10, y=10, w=100, h=20)],
    ):
        annotate(
            img,
            LabelFindings(
                verdict=Verdict.PASS,
                fields=[
                    FieldFinding(
                        field="brand_name", verdict=Verdict.PASS, extracted="Example Brand"
                    )
                ],
            ),
        )
    assert np.array_equal(np.array(img), original_data)


# ---------------------------------------------------------------------------
# annotate() — no OCR call when nothing to annotate
# ---------------------------------------------------------------------------


def test_annotate_no_ocr_call_when_no_annotatable_fields() -> None:
    """_run_ocr should not be called when all fields are ABSENT/INDETERMINATE/ERROR."""
    findings = LabelFindings(
        verdict=Verdict.ABSENT,
        fields=[
            FieldFinding(field="net_contents", verdict=Verdict.ABSENT),
            FieldFinding(field="brand_name", verdict=Verdict.INDETERMINATE),
            FieldFinding(field="__error__", verdict=Verdict.ERROR),
        ],
    )
    with patch("proofreader.annotate._run_ocr") as mock_ocr:
        annotate(_white_image(), findings)
    mock_ocr.assert_not_called()


def test_annotate_no_ocr_call_when_no_extracted_text() -> None:
    """_run_ocr should not be called when PASS/WARN/FAIL fields have no extracted text."""
    findings = LabelFindings(
        verdict=Verdict.FAIL,
        fields=[FieldFinding(field="brand_name", verdict=Verdict.FAIL)],
    )
    with patch("proofreader.annotate._run_ocr") as mock_ocr:
        annotate(_white_image(), findings)
    mock_ocr.assert_not_called()


# ---------------------------------------------------------------------------
# annotate() — OCR hit → solid polygon drawn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [Verdict.PASS, Verdict.WARN, Verdict.FAIL])
def test_annotate_draws_polygon_on_ocr_hit(verdict: Verdict) -> None:
    img = _white_image(200, 200)
    findings = LabelFindings(
        verdict=verdict,
        fields=[FieldFinding(field="brand_name", verdict=verdict, extracted="Example Brand")],
    )
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[_ocr_hit("Example Brand", x=20, y=20, w=100, h=25)],
    ):
        result = annotate(img, findings)

    assert _pixels_changed(img, result), "expected annotated pixels for OCR hit"


def test_annotate_pass_uses_green() -> None:
    img = _white_image(200, 200)
    findings = LabelFindings(
        verdict=Verdict.PASS,
        fields=[FieldFinding(field="brand_name", verdict=Verdict.PASS, extracted="Example Brand")],
    )
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[_ocr_hit("Example Brand", x=20, y=20, w=100, h=25)],
    ):
        result = annotate(img, findings)

    arr = np.array(result)
    green_color = _VERDICT_COLORS[Verdict.PASS]
    # Some pixel should be the exact PASS green.
    found = np.any(
        (arr[:, :, 0] == green_color[0])
        & (arr[:, :, 1] == green_color[1])
        & (arr[:, :, 2] == green_color[2])
    )
    assert found, f"expected pixel with PASS color {green_color}"


def test_annotate_fail_uses_red() -> None:
    img = _white_image(200, 200)
    findings = LabelFindings(
        verdict=Verdict.FAIL,
        fields=[
            FieldFinding(
                field="government_warning", verdict=Verdict.FAIL, extracted="Government Warning"
            )
        ],
    )
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[_ocr_hit("Government Warning", x=10, y=10, w=150, h=20)],
    ):
        result = annotate(img, findings)

    arr = np.array(result)
    red_color = _VERDICT_COLORS[Verdict.FAIL]
    found = np.any(
        (arr[:, :, 0] == red_color[0])
        & (arr[:, :, 1] == red_color[1])
        & (arr[:, :, 2] == red_color[2])
    )
    assert found, f"expected pixel with FAIL color {red_color}"


# ---------------------------------------------------------------------------
# annotate() — no OCR hit → dashed rect drawn (approximate location)
# ---------------------------------------------------------------------------


def test_annotate_draws_dashed_rect_on_ocr_miss() -> None:
    """When OCR can't locate extracted text, a dashed approximate outline is drawn."""
    img = _white_image(200, 200)
    findings = LabelFindings(
        verdict=Verdict.WARN,
        fields=[
            FieldFinding(field="brand_name", verdict=Verdict.WARN, extracted="Curved Brand Name")
        ],
    )
    with patch("proofreader.annotate._run_ocr", return_value=[]):
        result = annotate(img, findings)

    assert _pixels_changed(img, result), "expected dashed-rect pixels on OCR miss"


# ---------------------------------------------------------------------------
# annotate() — ABSENT/INDETERMINATE/ERROR fields are not annotated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [Verdict.ABSENT, Verdict.INDETERMINATE, Verdict.ERROR])
def test_annotate_skips_non_annotatable_verdicts(verdict: Verdict) -> None:
    img = _white_image()
    findings = LabelFindings(
        verdict=verdict,
        fields=[FieldFinding(field="net_contents", verdict=verdict, extracted="750 mL")],
    )
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[_ocr_hit("750 mL")],
    ) as mock_ocr:
        result = annotate(img, findings)

    # Image unchanged: no polygon should have been drawn.
    assert not _pixels_changed(img, result), f"expected no annotation for {verdict.name}"
    # OCR should not have been called since there are no annotatable fields.
    mock_ocr.assert_not_called()


# ---------------------------------------------------------------------------
# annotate() — mixed findings
# ---------------------------------------------------------------------------


def test_annotate_mixed_findings_only_draws_for_annotatable() -> None:
    """PASS and FAIL fields are drawn; ABSENT field is not."""
    img = _white_image(300, 300)
    findings = LabelFindings(
        verdict=Verdict.FAIL,
        fields=[
            FieldFinding(field="brand_name", verdict=Verdict.PASS, extracted="Example Brand"),
            FieldFinding(
                field="government_warning", verdict=Verdict.FAIL, extracted="Government Warning"
            ),
            FieldFinding(field="net_contents", verdict=Verdict.ABSENT),  # no annotation
        ],
    )
    with patch(
        "proofreader.annotate._run_ocr",
        return_value=[
            _ocr_hit("Example Brand", x=10, y=10, w=120, h=25),
            _ocr_hit("Government Warning", x=10, y=50, w=150, h=20),
        ],
    ):
        result = annotate(img, findings)

    # Both annotatable fields should have been drawn.
    arr = np.array(result)
    green = _VERDICT_COLORS[Verdict.PASS]
    red = _VERDICT_COLORS[Verdict.FAIL]
    assert np.any(
        (arr[:, :, 0] == green[0]) & (arr[:, :, 1] == green[1]) & (arr[:, :, 2] == green[2])
    ), "expected green pixels for PASS field"
    assert np.any((arr[:, :, 0] == red[0]) & (arr[:, :, 1] == red[1]) & (arr[:, :, 2] == red[2])), (
        "expected red pixels for FAIL field"
    )
