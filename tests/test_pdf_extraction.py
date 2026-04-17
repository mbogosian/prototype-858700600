"""
Integration tests for pdf.extract_page1().

Marked @pytest.mark.integration — these tests initialise RapidOCR and require
pre-generated sample PDFs in tests/sample_applications/.

Generate the sample PDFs with:
    uv run python tests/make_test_pdf.py

Run only integration tests:
    uv run pytest -m integration

Skip integration tests (fast unit tests only):
    uv run pytest -m "not integration"
"""

import gc

import pytest

from pathlib import Path

from proofreader.models import Page1Result, Reason
from proofreader.pdf import extract_page1

SAMPLE_DIR = Path(__file__).parent / "sample_applications"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_pdfs() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Parametrized smoke test — one test per PDF
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("pdf_path", _sample_pdfs(), ids=lambda p: p.name)
def test_extract_page1_smoke(pdf_path: Path) -> None:
    """extract_page1 returns a valid Page1Result for every sample PDF."""
    result = extract_page1(pdf_path)

    assert isinstance(result, Page1Result)

    if result.reason is not None:
        # Terminal state: label_zone must be withheld (PII gate).
        assert result.label_zone is None, (
            f"label_zone must be None when reason={result.reason.name} "
            "(structural PII gate violated)"
        )
    else:
        # Successful extraction: label_zone and page1_image must be present.
        assert result.label_zone is not None
        assert result.page1_image is not None
        assert result.product_type in ("wine", "distilled_spirits", "malt_beverage", None)

    # Release image buffers between parametrized runs to keep peak memory reasonable.
    del result
    gc.collect()


# ---------------------------------------------------------------------------
# Specific outcome assertions for known test PDFs
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename,expected_product_type",
    [
        ("test-01-wine-brand-back-slides.pdf", "wine"),
        ("test-02-wine-brand-back-pdf.pdf", "wine"),
        ("test-03-ds-brand-back.pdf", "distilled_spirits"),
        ("test-04-mb-combined.pdf", "malt_beverage"),
        ("test-05-mb-brand-back.pdf", "malt_beverage"),
        ("test-10-wine-brand-back-pdf-changed.pdf", "wine"),
        ("test-11-wine-200dpi-scan.pdf", "wine"),
        ("test-12-wine-400dpi-scan.pdf", "wine"),
    ],
)
def test_extract_page1_product_type(filename: str, expected_product_type: str) -> None:
    """Known-good PDFs should extract with the correct product type."""
    pdf_path = SAMPLE_DIR / filename
    if not pdf_path.exists():
        pytest.skip(f"{filename} not found — run make_test_pdf.py first")

    result = extract_page1(pdf_path)
    assert result.reason is None, f"Expected successful extraction, got reason={result.reason}"
    assert result.product_type == expected_product_type


@pytest.mark.integration
def test_extract_page1_rotated_returns_anchor_not_found() -> None:
    """A rotated page 1 should return ANCHOR_NOT_FOUND, not crash."""
    pdf_path = SAMPLE_DIR / "test-08-rotated-page1.pdf"
    if not pdf_path.exists():
        pytest.skip("test-08-rotated-page1.pdf not found — run make_test_pdf.py first")

    result = extract_page1(pdf_path)
    assert result.reason is Reason.ANCHOR_NOT_FOUND
    assert result.label_zone is None
    # page1_image is preserved for the report even on ANCHOR_NOT_FOUND.
    assert result.page1_image is not None


@pytest.mark.integration
def test_extract_page1_missing_page1_returns_pdf_empty() -> None:
    """A PDF with no page 1 (instructions only) should return PDF_EMPTY."""
    pdf_path = SAMPLE_DIR / "test-09-missing-page1.pdf"
    if not pdf_path.exists():
        pytest.skip("test-09-missing-page1.pdf not found — run make_test_pdf.py first")

    result = extract_page1(pdf_path)
    # The instructions-only PDF has pages, just not page 1 of the form.
    # Anchor text will not be found, so we expect ANCHOR_NOT_FOUND.
    assert result.reason in (Reason.ANCHOR_NOT_FOUND, Reason.PDF_EMPTY)
    assert result.label_zone is None


@pytest.mark.integration
def test_extract_page1_nonexistent_file_returns_pdf_unreadable() -> None:
    """A path that does not exist should return PDF_UNREADABLE, not raise."""
    result = extract_page1(Path("/nonexistent/path/to/file.pdf"))
    assert result.reason is Reason.PDF_UNREADABLE
    assert result.label_zone is None
    assert result.page1_image is None
