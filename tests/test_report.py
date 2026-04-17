"""
Unit tests for report.py — Jinja2 rendering, base64 embedding, and JSON output.

All tests are fast: no PaddleOCR, no API calls. Images are synthesized with Pillow.
"""

import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from proofreader.models import FieldFinding, LabelFindings, Page1Result, Reason, Verdict
from proofreader.report import render, render_terminal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_image(w: int = 200, h: int = 150, color=(240, 240, 240)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def _page1_success(product_type: str | None = "wine") -> Page1Result:
    return Page1Result(
        reason=None,
        product_type=product_type,
        label_zone=_blank_image(),
        page1_image=_blank_image(595, 842),
    )


def _page1_terminal(reason: Reason = Reason.ANCHOR_NOT_FOUND) -> Page1Result:
    return Page1Result(
        reason=reason,
        product_type=None,
        label_zone=None,
        page1_image=_blank_image() if reason is Reason.ANCHOR_NOT_FOUND else None,
    )


def _passing_findings() -> LabelFindings:
    return LabelFindings(
        verdict=Verdict.PASS,
        fields=[
            FieldFinding(field="brand_name", verdict=Verdict.PASS, extracted="Example Brand"),
            FieldFinding(field="net_contents", verdict=Verdict.PASS, extracted="750 mL"),
        ],
    )


def _mixed_findings() -> LabelFindings:
    return LabelFindings(
        verdict=Verdict.FAIL,
        fields=[
            FieldFinding(field="brand_name", verdict=Verdict.PASS, extracted="Example Brand"),
            FieldFinding(
                field="government_warning",
                verdict=Verdict.FAIL,
                extracted="Government Warning: ...",
                note="Must be all-caps bold.",
            ),
            FieldFinding(field="net_contents", verdict=Verdict.ABSENT, note="[Item 15]"),
        ],
        import_indicators=True,
    )


# ---------------------------------------------------------------------------
# render_terminal — findings.json
# ---------------------------------------------------------------------------


def test_terminal_findings_json_written(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(Reason.PDF_UNREADABLE), tmp_path)
    assert (tmp_path / "findings.json").exists()


def test_terminal_findings_json_structure(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(Reason.ANCHOR_NOT_FOUND), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())

    assert data["job_id"] == "abc123"
    assert data["verdict"] == Verdict.INDETERMINATE.name
    assert data["reason"] == Reason.ANCHOR_NOT_FOUND.name
    assert isinstance(data["reason_description"], str) and data["reason_description"]
    assert data["fields"] == []


@pytest.mark.parametrize(
    "reason", [Reason.PDF_UNREADABLE, Reason.PDF_EMPTY, Reason.ANCHOR_NOT_FOUND]
)
def test_terminal_findings_json_reason_name(tmp_path: Path, reason: Reason) -> None:
    render_terminal("abc123", _page1_terminal(reason), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["reason"] == reason.name


# ---------------------------------------------------------------------------
# render_terminal — report.html
# ---------------------------------------------------------------------------


def test_terminal_report_html_written(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path)
    assert (tmp_path / "report.html").exists()


def test_terminal_report_html_contains_job_id(tmp_path: Path) -> None:
    render_terminal("myjobid", _page1_terminal(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "myjobid" in html


def test_terminal_report_html_contains_reason(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(Reason.PDF_EMPTY), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert Reason.PDF_EMPTY.name in html


def test_terminal_report_html_contains_indeterminate(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert Verdict.INDETERMINATE.name in html


def test_terminal_report_html_is_valid_html(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


# ---------------------------------------------------------------------------
# render_terminal — logs
# ---------------------------------------------------------------------------


def test_terminal_no_logs_section_when_empty(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path, logs=[])
    html = (tmp_path / "report.html").read_text()
    assert "<details>" not in html


def test_terminal_logs_section_present(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path, logs=["line one", "line two"])
    html = (tmp_path / "report.html").read_text()
    assert "<details>" in html
    assert "line one" in html
    assert "line two" in html


def test_terminal_logs_html_escaped(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path, logs=["<script>alert(1)</script>"])
    html = (tmp_path / "report.html").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# render — findings.json
# ---------------------------------------------------------------------------


def test_render_findings_json_written(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    assert (tmp_path / "findings.json").exists()


def test_render_findings_json_structure(tmp_path: Path) -> None:
    render("abc123", _page1_success("wine"), _blank_image(), _passing_findings(), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())

    assert data["job_id"] == "abc123"
    assert data["verdict"] == Verdict.PASS.name
    assert data["product_type"] == "wine"
    assert data["import_indicators"] is False
    assert len(data["fields"]) == 2
    assert data["fields"][0]["field"] == "brand_name"
    assert data["fields"][0]["verdict"] == Verdict.PASS.name


def test_render_findings_json_mixed_verdicts(tmp_path: Path) -> None:
    render("abc123", _page1_success("wine"), _blank_image(), _mixed_findings(), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["verdict"] == Verdict.FAIL.name
    assert data["import_indicators"] is True
    assert any(f["verdict"] == Verdict.ABSENT.name for f in data["fields"])


def test_render_findings_json_note_included(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _mixed_findings(), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())
    fail_field = next(f for f in data["fields"] if f["verdict"] == Verdict.FAIL.name)
    assert fail_field["note"] == "Must be all-caps bold."


# ---------------------------------------------------------------------------
# render — annotated.jpg
# ---------------------------------------------------------------------------


def test_render_saves_annotated_jpg(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    assert (tmp_path / "annotated.jpg").exists()


# ---------------------------------------------------------------------------
# render — report.html
# ---------------------------------------------------------------------------


def test_render_report_html_written(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    assert (tmp_path / "report.html").exists()


def test_render_report_html_contains_job_id(tmp_path: Path) -> None:
    render("myjobid", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "myjobid" in html


def test_render_report_html_contains_verdict(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert Verdict.PASS.name in html


def test_render_report_html_contains_field_names(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "brand_name" in html
    assert "net_contents" in html


def test_render_report_html_contains_extracted_text(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "Example Brand" in html
    assert "750 mL" in html


def test_render_report_html_contains_product_type(tmp_path: Path) -> None:
    render(
        "abc123", _page1_success("distilled_spirits"), _blank_image(), _passing_findings(), tmp_path
    )
    html = (tmp_path / "report.html").read_text()
    assert "distilled_spirits" in html


def test_render_report_html_escapes_field_content(tmp_path: Path) -> None:
    """Extracted text and notes with HTML special chars are properly escaped."""
    malicious = LabelFindings(
        verdict=Verdict.WARN,
        fields=[
            FieldFinding(
                field="brand_name",
                verdict=Verdict.WARN,
                extracted='<script>alert("xss")</script>',
                note="<em>note</em>",
            )
        ],
    )
    render("abc123", _page1_success(), _blank_image(), malicious, tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<em>note</em>" not in html


def test_render_report_html_is_valid_html(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html


# ---------------------------------------------------------------------------
# render — base64 image embedding
# ---------------------------------------------------------------------------


def test_render_report_html_embeds_image_as_base64(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "data:image/jpeg;base64," in html


def test_render_base64_image_is_decodable(tmp_path: Path) -> None:
    """The embedded base64 string decodes to a valid JPEG."""
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    html = (tmp_path / "report.html").read_text()

    # Extract the base64 payload from the img src attribute.
    marker = "data:image/jpeg;base64,"
    start = html.index(marker) + len(marker)
    # The value ends at the closing quote of the src attribute.
    end = html.index('"', start)
    b64_data = html[start:end]

    raw = base64.b64decode(b64_data)
    import io as _io

    img = Image.open(_io.BytesIO(raw))
    assert img.format == "JPEG"


# ---------------------------------------------------------------------------
# render — logs
# ---------------------------------------------------------------------------


def test_render_no_logs_section_when_empty(tmp_path: Path) -> None:
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path, logs=[])
    html = (tmp_path / "report.html").read_text()
    assert "<details>" not in html


def test_render_logs_section_present(tmp_path: Path) -> None:
    render(
        "abc123",
        _page1_success(),
        _blank_image(),
        _passing_findings(),
        tmp_path,
        logs=["stage start", "stage end"],
    )
    html = (tmp_path / "report.html").read_text()
    assert "<details>" in html
    assert "stage start" in html
    assert "stage end" in html


def test_render_logs_html_escaped(tmp_path: Path) -> None:
    render(
        "abc123",
        _page1_success(),
        _blank_image(),
        _passing_findings(),
        tmp_path,
        logs=["<b>bold log line</b>"],
    )
    html = (tmp_path / "report.html").read_text()
    assert "<b>bold log line</b>" not in html
    assert "&lt;b&gt;" in html


# ---------------------------------------------------------------------------
# render_terminal — original_filename and submitted_at in findings.json
# ---------------------------------------------------------------------------


def test_terminal_findings_json_includes_original_filename(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path, original_filename="wine_label.pdf")
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["original_filename"] == "wine_label.pdf"


def test_terminal_findings_json_includes_submitted_at(tmp_path: Path) -> None:
    render_terminal("abc123", _page1_terminal(), tmp_path, submitted_at=1_700_000_000.0)
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["submitted_at"] == 1_700_000_000.0


def test_terminal_findings_json_original_filename_none_by_default(tmp_path: Path) -> None:
    """original_filename key is present and null when not supplied."""
    render_terminal("abc123", _page1_terminal(), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())
    assert "original_filename" in data
    assert data["original_filename"] is None


# ---------------------------------------------------------------------------
# render — original_filename and submitted_at in findings.json
# ---------------------------------------------------------------------------


def test_render_findings_json_includes_original_filename(tmp_path: Path) -> None:
    render(
        "abc123",
        _page1_success(),
        _blank_image(),
        _passing_findings(),
        tmp_path,
        original_filename="wine_label.pdf",
    )
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["original_filename"] == "wine_label.pdf"


def test_render_findings_json_includes_submitted_at(tmp_path: Path) -> None:
    render(
        "abc123",
        _page1_success(),
        _blank_image(),
        _passing_findings(),
        tmp_path,
        submitted_at=1_700_000_000.0,
    )
    data = json.loads((tmp_path / "findings.json").read_text())
    assert data["submitted_at"] == 1_700_000_000.0


def test_render_findings_json_original_filename_none_by_default(tmp_path: Path) -> None:
    """original_filename key is present and null when not supplied."""
    render("abc123", _page1_success(), _blank_image(), _passing_findings(), tmp_path)
    data = json.loads((tmp_path / "findings.json").read_text())
    assert "original_filename" in data
    assert data["original_filename"] is None
