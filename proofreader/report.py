"""
Report generation for ProofReader.

Produces two output files per completed job:
  report.html   — self-contained HTML with base64-embedded annotated label image
  findings.json — structured compliance findings for downstream integration

Two entry points:
  render_terminal(job_id, page1, outbox_dir, logs)
      Called when pdf.extract_page1() returns a terminal reason code
      (PDF_UNREADABLE, PDF_EMPTY, ANCHOR_NOT_FOUND). No label analysis was
      possible; the report explains why and recommends resubmission.

  render(job_id, page1, annotated_zone, findings, outbox_dir, logs)
      Called after successful extraction and label analysis. Produces a full
      compliance report with the base64-embedded annotated label image and
      per-field findings.

Public API:
    render_terminal(job_id, page1, outbox_dir, logs)
    render(job_id, page1, annotated_zone, findings, outbox_dir, logs)
"""

import base64
import io
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from proofreader.models import LabelFindings, Page1Result, Verdict

_templates_dir = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(["html", "j2"]),
)


def _image_to_b64(image: Image.Image, quality: int = 90) -> str:
    """Encode a PIL Image as a base64 JPEG string for HTML embedding."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_terminal(
    job_id: str,
    page1: Page1Result,
    outbox_dir: Path,
    logs: list[str] | None = None,
    original_filename: str | None = None,
    submitted_at: float | None = None,
) -> None:
    """Write a minimal report for a terminal extraction state.

    The PDF could not be processed (unreadable, empty, or anchor not found).
    The report explains the failure and recommends resubmission.
    """
    reason_name = page1.reason.name if page1.reason else "UNKNOWN"
    reason_desc = page1.reason.description if page1.reason else ""

    findings_data = {
        "job_id": job_id,
        "verdict": Verdict.INDETERMINATE.name,
        "original_filename": original_filename,
        "submitted_at": submitted_at,
        "reason": reason_name,
        "reason_description": reason_desc,
        "fields": [],
    }
    (outbox_dir / "findings.json").write_text(json.dumps(findings_data, indent=2))

    tmpl = _env.get_template("terminal.html.j2")
    report_html = tmpl.render(
        job_id=job_id,
        original_filename=original_filename,
        verdict_description=Verdict.INDETERMINATE.description,
        reason_name=reason_name,
        reason_description=reason_desc,
        logs=logs or [],
    )
    (outbox_dir / "report.html").write_text(report_html)


def render(
    job_id: str,
    page1: Page1Result,
    annotated_zone: Image.Image,
    findings: LabelFindings,
    outbox_dir: Path,
    logs: list[str] | None = None,
    original_filename: str | None = None,
    submitted_at: float | None = None,
) -> None:
    """Write the full compliance report and findings JSON for a completed analysis."""
    findings_data = {
        "job_id": job_id,
        "verdict": findings.verdict.name,
        "original_filename": original_filename,
        "submitted_at": submitted_at,
        "product_type": page1.product_type,
        "import_indicators": findings.import_indicators,
        "fields": [
            {
                "field": f.field,
                "verdict": f.verdict.name,
                "extracted": f.extracted,
                "note": f.note,
            }
            for f in findings.fields
        ],
    }
    (outbox_dir / "findings.json").write_text(json.dumps(findings_data, indent=2))

    # Save annotated label zone for direct access via the static file server.
    annotated_zone.save(outbox_dir / "annotated.jpg", format="JPEG", quality=90)

    tmpl = _env.get_template("report.html.j2")
    report_html = tmpl.render(
        job_id=job_id,
        original_filename=original_filename,
        product_type=page1.product_type,
        import_indicators=findings.import_indicators,
        verdict=findings.verdict.name,
        verdict_description=findings.verdict.description,
        fields=findings.fields,
        annotated_b64=_image_to_b64(annotated_zone),
        logs=logs or [],
    )
    (outbox_dir / "report.html").write_text(report_html)
