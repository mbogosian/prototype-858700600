"""
Unit tests for the worker pipeline and report output.

All tests are fast (no PaddleOCR, no API calls). The pipeline stages that make
external calls (pdf.extract_page1, vision.read_labels) are patched with
controlled return values. Report output is written to tmp_path and verified.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from proofreader.models import FieldFinding, LabelFindings, Page1Result, Reason, Verdict
from proofreader import worker
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


# ---------------------------------------------------------------------------
# submitted_at passthrough — _process() passes it to report functions
# ---------------------------------------------------------------------------


def test_process_submitted_at_in_findings_json(tmp_path: Path) -> None:
    """submitted_at from job state is written into findings.json."""
    _set_job("test014", status="queued", original_filename="test.pdf", submitted_at=1_700_000_000.0)
    _run_process(tmp_path, _page1_success("wine"), job_id="test014")

    data = json.loads((tmp_path / "test014" / "findings.json").read_text())
    assert data["submitted_at"] == 1_700_000_000.0


def test_process_terminal_submitted_at_in_findings_json(tmp_path: Path) -> None:
    """submitted_at from job state is written into findings.json on the terminal path."""
    _set_job("test015", status="queued", original_filename="test.pdf", submitted_at=9_999_999.0)
    fake_pdf = tmp_path / "test015.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    with patch(
        "proofreader.worker.pdf.extract_page1",
        return_value=_page1_terminal(Reason.ANCHOR_NOT_FOUND),
    ):
        _process(fake_pdf, "test015", tmp_path)

    data = json.loads((tmp_path / "test015" / "findings.json").read_text())
    assert data["submitted_at"] == 9_999_999.0


def test_process_original_filename_in_findings_json(tmp_path: Path) -> None:
    """original_filename from job state is written into findings.json."""
    _set_job("test016", status="queued", original_filename="chardonnay_2024.pdf")
    _run_process(tmp_path, _page1_success("wine"), job_id="test016")

    data = json.loads((tmp_path / "test016" / "findings.json").read_text())
    assert data["original_filename"] == "chardonnay_2024.pdf"


# ---------------------------------------------------------------------------
# start() — outbox re-inflation
# ---------------------------------------------------------------------------


@pytest.fixture()
def worker_globals_reset():
    """Restore worker module globals modified by start() after each test."""
    yield
    with _jobs_lock:
        _jobs.clear()
    worker._executor = None
    worker._observer = None
    worker._inbox = None
    worker._outbox = None
    worker._log_loop = None


def _call_start(inbox: Path, outbox: Path) -> None:
    """Call worker.start() with mocked ThreadPoolExecutor and Observer."""
    loop = asyncio.new_event_loop()
    try:
        with patch("proofreader.worker.ThreadPoolExecutor"), \
             patch("proofreader.worker.Observer"):
            worker.start(inbox, outbox, 1, loop)
    finally:
        loop.close()


def _make_completed_job(outbox: Path, job_id: str, verdict: str = "PASS",
                         original_filename: str | None = "label.pdf",
                         submitted_at: float | None = 1_700_000_000.0) -> Path:
    """Write a minimal complete job directory to outbox; return the job dir."""
    job_dir = outbox / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "report.html").write_text("<!DOCTYPE html><html></html>")
    (job_dir / "findings.json").write_text(json.dumps({
        "job_id": job_id,
        "verdict": verdict,
        "original_filename": original_filename,
        "submitted_at": submitted_at,
        "fields": [],
    }))
    return job_dir


def test_start_reinflates_completed_job(tmp_path: Path, worker_globals_reset) -> None:
    """A complete outbox job (report.html + findings.json) is loaded into _jobs."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    _make_completed_job(outbox, "aabbcc", verdict="PASS")

    _call_start(inbox, outbox)

    job = worker.get_job("aabbcc")
    assert job is not None
    assert job["status"] == "complete"
    assert job["verdict"] == "PASS"


def test_start_reinflates_original_filename_and_submitted_at(tmp_path: Path, worker_globals_reset) -> None:
    """Re-inflated job carries original_filename and submitted_at from findings.json."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    _make_completed_job(outbox, "ddeeff", original_filename="chardonnay.pdf", submitted_at=42.5)

    _call_start(inbox, outbox)

    job = worker.get_job("ddeeff")
    assert job["original_filename"] == "chardonnay.pdf"
    assert job["submitted_at"] == 42.5


def test_start_skips_outbox_dir_without_report_html(tmp_path: Path, worker_globals_reset) -> None:
    """Outbox dir with findings.json but no report.html is not re-inflated (still processing)."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    job_dir = outbox / "incomplete"
    job_dir.mkdir(parents=True)
    (job_dir / "findings.json").write_text(json.dumps({"job_id": "incomplete", "verdict": "PASS", "fields": []}))

    _call_start(inbox, outbox)

    assert worker.get_job("incomplete") is None


def test_start_skips_outbox_dir_without_findings_json(tmp_path: Path, worker_globals_reset) -> None:
    """Outbox dir with report.html but no findings.json is not re-inflated."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    job_dir = outbox / "nofindings"
    job_dir.mkdir(parents=True)
    (job_dir / "report.html").write_text("<!DOCTYPE html><html></html>")

    _call_start(inbox, outbox)

    assert worker.get_job("nofindings") is None


def test_start_skips_malformed_findings_json(tmp_path: Path, worker_globals_reset) -> None:
    """Malformed findings.json is silently skipped; no exception raised."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    job_dir = outbox / "badjson"
    job_dir.mkdir(parents=True)
    (job_dir / "report.html").write_text("<!DOCTYPE html><html></html>")
    (job_dir / "findings.json").write_text("not valid json{{{")

    _call_start(inbox, outbox)  # must not raise

    assert worker.get_job("badjson") is None


def test_start_outbox_does_not_overwrite_inbox_job(tmp_path: Path, worker_globals_reset) -> None:
    """An outbox entry for a job already registered (e.g. from inbox re-queue) is skipped."""
    inbox = tmp_path / "inbox"
    outbox = tmp_path / "outbox"
    inbox.mkdir(parents=True)

    # Put a PDF in inbox so the inbox re-queue loop registers it.
    job_id = "overlap1"
    (inbox / f"{job_id}.pdf").write_bytes(b"%PDF fake")
    # Also put a completed dir in outbox for the same job_id.
    _make_completed_job(outbox, job_id, verdict="FAIL")

    _call_start(inbox, outbox)

    # The job should be registered but with status "queued" (from inbox), not "complete".
    job = worker.get_job(job_id)
    assert job is not None
    assert job["status"] == "queued"


# ---------------------------------------------------------------------------
# delete_job()
# ---------------------------------------------------------------------------


@pytest.fixture()
def set_outbox(tmp_path: Path):
    """Temporarily point worker._outbox at tmp_path; restore on teardown."""
    old = worker._outbox
    worker._outbox = tmp_path
    yield tmp_path
    worker._outbox = old


def test_delete_job_returns_true_for_complete(set_outbox: Path) -> None:
    _set_job("del001", status="complete", verdict="PASS")
    assert worker.delete_job("del001") is True


def test_delete_job_removes_from_memory(set_outbox: Path) -> None:
    _set_job("del002", status="complete", verdict="PASS")
    worker.delete_job("del002")
    assert worker.get_job("del002") is None


def test_delete_job_removes_outbox_directory(set_outbox: Path) -> None:
    _set_job("del003", status="complete", verdict="PASS")
    job_dir = set_outbox / "del003"
    job_dir.mkdir()
    (job_dir / "report.html").write_text("report")
    (job_dir / "findings.json").write_text("{}")

    worker.delete_job("del003")

    assert not job_dir.exists()


def test_delete_job_accepts_error_status(set_outbox: Path) -> None:
    _set_job("del004", status="error", verdict="ERROR")
    assert worker.delete_job("del004") is True
    assert worker.get_job("del004") is None


def test_delete_job_returns_false_for_unknown_job() -> None:
    assert worker.delete_job("doesnotexist") is False


def test_delete_job_rejects_queued_job() -> None:
    _set_job("del005", status="queued")
    assert worker.delete_job("del005") is False
    assert worker.get_job("del005") is not None   # still present


def test_delete_job_rejects_processing_job() -> None:
    _set_job("del006", status="processing")
    assert worker.delete_job("del006") is False
    assert worker.get_job("del006") is not None


def test_delete_job_handles_missing_outbox_dir(set_outbox: Path) -> None:
    """No outbox directory for the job — delete_job should not raise."""
    _set_job("del007", status="complete", verdict="PASS")
    # Intentionally no directory created for del007
    assert worker.delete_job("del007") is True
