"""
Unit tests for the worker pipeline and report output.

All tests are fast (no PaddleOCR, no API calls). The pipeline stages that make
external calls (pdf.extract_page1, vision.read_labels) are patched with
controlled return values. Report output is written to tmp_path and verified.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from proofreader.models import FieldFinding, LabelFindings, Page1Result, Reason, Verdict
from proofreader.worker import _process, _set_job, _jobs, _jobs_lock
from tests.fake_readers import (
    FakeLabelReader,
    findings_with_fail,
    findings_with_warn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Relies on pytest's built-in tmp_path fixture:
# https://docs.pytest.org/en/stable/how-to/tmp_path.html#tmp-path


@pytest.fixture(autouse=True)
def clear_job_state():
    """Reset global job state between tests."""
    with _jobs_lock:
        _jobs.clear()
    yield
    with _jobs_lock:
        _jobs.clear()


def _blank_image(w: int = 100, h: int = 100) -> Image.Image:
    return Image.new("RGB", (w, h), color=(240, 240, 240))


def _page1_success(product_type: str = "wine") -> Page1Result:
    """A Page1Result representing successful extraction."""
    return Page1Result(
        reason=None,
        product_type=product_type,
        label_zone=_blank_image(400, 300),
        page1_image=_blank_image(595, 842),
    )


def _page1_terminal(reason: Reason) -> Page1Result:
    """A Page1Result representing a terminal extraction state."""
    page1_image = _blank_image() if reason is Reason.ANCHOR_NOT_FOUND else None
    return Page1Result(
        reason=reason,
        product_type=None,
        label_zone=None,
        page1_image=page1_image,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_process(
    tmp_path: Path,
    page1: Page1Result,
    findings: LabelFindings | None = None,
    job_id: str = "test001",
) -> None:
    """Run _process with pdf and vision stages patched."""
    fake_pdf = tmp_path / f"{job_id}.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    patch_pdf = patch("proofreader.worker.pdf.extract_page1", return_value=page1)
    patch_vision = patch(
        "proofreader.worker.vision.read_labels",
        return_value=findings or FakeLabelReader.passing()._findings,
    )
    # Patch annotate._run_ocr so tests don't require PaddleOCR/libGL at unit-test time.
    patch_ocr = patch("proofreader.annotate._run_ocr", return_value=[])
    with patch_pdf, patch_vision, patch_ocr:
        _process(fake_pdf, job_id, tmp_path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_process_happy_path_pass(tmp_path: Path) -> None:
    _set_job("test001", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success("wine"), job_id="test001")

    job = _jobs.get("test001")
    assert job is not None
    assert job["status"] == "complete"
    assert job["verdict"] == Verdict.PASS.name


def test_process_happy_path_fail(tmp_path: Path) -> None:
    _set_job("test002", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success("wine"), findings=findings_with_fail(), job_id="test002")

    job = _jobs.get("test002")
    assert job["verdict"] == Verdict.FAIL.name


def test_process_happy_path_warn(tmp_path: Path) -> None:
    _set_job("test003", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success("wine"), findings=findings_with_warn(), job_id="test003")

    job = _jobs.get("test003")
    assert job["verdict"] == Verdict.WARN.name


@pytest.mark.parametrize("product_type", ["wine", "distilled_spirits", "malt_beverage", None])
def test_process_product_type_passed_to_vision(tmp_path: Path, product_type: str | None) -> None:
    """vision.read_labels is called with the product_type from the Page1Result."""
    _set_job("test004", status="queued", original_filename="test.pdf")
    fake_pdf = tmp_path / "test004.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    captured = {}

    def capture_read_labels(label_zone, pt, reader=None):
        captured["product_type"] = pt
        return FakeLabelReader.passing()._findings

    with patch("proofreader.worker.pdf.extract_page1", return_value=_page1_success(product_type)):
        with patch("proofreader.worker.vision.read_labels", side_effect=capture_read_labels):
            with patch("proofreader.annotate._run_ocr", return_value=[]):
                _process(fake_pdf, "test004", tmp_path)

    assert captured["product_type"] == product_type


# ---------------------------------------------------------------------------
# Terminal states — vision is never called, label_zone never passed externally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason", [Reason.PDF_UNREADABLE, Reason.PDF_EMPTY, Reason.ANCHOR_NOT_FOUND]
)
def test_process_terminal_state_vision_not_called(tmp_path: Path, reason: Reason) -> None:
    """When extraction is terminal, vision.read_labels must not be called."""
    _set_job("test005", status="queued", original_filename="test.pdf")
    fake_pdf = tmp_path / "test005.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    mock_vision = MagicMock()
    with patch("proofreader.worker.pdf.extract_page1", return_value=_page1_terminal(reason)):
        with patch("proofreader.worker.vision.read_labels", mock_vision):
            _process(fake_pdf, "test005", tmp_path)

    mock_vision.assert_not_called()


@pytest.mark.parametrize(
    "reason", [Reason.PDF_UNREADABLE, Reason.PDF_EMPTY, Reason.ANCHOR_NOT_FOUND]
)
def test_process_terminal_state_job_marked_indeterminate(tmp_path: Path, reason: Reason) -> None:
    _set_job("test006", status="queued", original_filename="test.pdf")
    fake_pdf = tmp_path / "test006.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    with patch("proofreader.worker.pdf.extract_page1", return_value=_page1_terminal(reason)):
        _process(fake_pdf, "test006", tmp_path)

    job = _jobs.get("test006")
    assert job["status"] == "complete"
    assert job["verdict"] == Verdict.INDETERMINATE.name


# ---------------------------------------------------------------------------
# ERROR verdict — reader returns ERROR findings
# ---------------------------------------------------------------------------


def test_process_api_error_verdict_propagated(tmp_path: Path) -> None:
    """When the reader returns ERROR findings, the job is marked ERROR."""
    _set_job("test007", status="queued", original_filename="test.pdf")
    _run_process(
        tmp_path,
        _page1_success(),
        findings=FakeLabelReader.api_error()._findings,
        job_id="test007",
    )

    job = _jobs.get("test007")
    assert job["verdict"] == Verdict.ERROR.name


# ---------------------------------------------------------------------------
# Output files written to outbox
# ---------------------------------------------------------------------------


def test_process_writes_findings_json(tmp_path: Path) -> None:
    _set_job("test008", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success("distilled_spirits"), job_id="test008")

    findings_path = tmp_path / "test008" / "findings.json"
    assert findings_path.exists(), "findings.json not written"

    data = json.loads(findings_path.read_text())
    assert data["job_id"] == "test008"
    assert data["verdict"] == Verdict.PASS.name
    assert isinstance(data["fields"], list)


def test_process_writes_report_html(tmp_path: Path) -> None:
    _set_job("test009", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success(), job_id="test009")

    report_path = tmp_path / "test009" / "report.html"
    assert report_path.exists(), "report.html not written"
    assert "test009" in report_path.read_text()


def test_process_writes_thumbnail(tmp_path: Path) -> None:
    _set_job("test010", status="queued", original_filename="test.pdf")
    _run_process(tmp_path, _page1_success(), job_id="test010")

    assert (tmp_path / "test010" / "thumbnail.jpg").exists()


def test_process_terminal_writes_findings_json(tmp_path: Path) -> None:
    _set_job("test011", status="queued", original_filename="test.pdf")
    fake_pdf = tmp_path / "test011.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    with patch(
        "proofreader.worker.pdf.extract_page1",
        return_value=_page1_terminal(Reason.ANCHOR_NOT_FOUND),
    ):
        _process(fake_pdf, "test011", tmp_path)

    findings_path = tmp_path / "test011" / "findings.json"
    assert findings_path.exists()
    data = json.loads(findings_path.read_text())
    assert data["reason"] == Reason.ANCHOR_NOT_FOUND.name


# ---------------------------------------------------------------------------
# FakeLabelReader — call tracking
# ---------------------------------------------------------------------------


def test_fake_reader_call_count(tmp_path: Path) -> None:
    """FakeLabelReader records how many times read() was called."""
    reader = FakeLabelReader.passing()
    _set_job("test012", status="queued", original_filename="test.pdf")
    fake_pdf = tmp_path / "test012.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    with patch("proofreader.worker.pdf.extract_page1", return_value=_page1_success()):
        with patch("proofreader.worker.vision.read_labels", side_effect=reader.read):
            with patch("proofreader.annotate._run_ocr", return_value=[]):
                _process(fake_pdf, "test012", tmp_path)

    assert reader.call_count == 1
    assert reader.last_product_type == "wine"


def test_fake_reader_not_called_on_terminal(tmp_path: Path) -> None:
    reader = FakeLabelReader.passing()
    fake_pdf = tmp_path / "test013.pdf"
    fake_pdf.write_bytes(b"%PDF fake")
    _set_job("test013", status="queued", original_filename="test.pdf")

    with patch(
        "proofreader.worker.pdf.extract_page1",
        return_value=_page1_terminal(Reason.PDF_UNREADABLE),
    ):
        with patch("proofreader.worker.vision.read_labels", side_effect=reader.read):
            _process(fake_pdf, "test013", tmp_path)

    assert reader.call_count == 0
