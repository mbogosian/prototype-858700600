"""
Shared data types for the ProofReader pipeline.

Pipeline modules produce structured facts: reason codes describing terminal
extraction states, and field-level compliance findings from label analysis.
How those facts map to user-facing verdicts and messages is the responsibility
of the reporting layer (compare.py, report.py), keeping pipeline facts and
user-facing interpretation separate.

## Reason codes vs. verdicts

Page1Result.reason is a terminal pipeline state — when non-None, automation
stops. The reason code describes what happened factually; no verdict is implied.
The reporting layer decides whether Reason.ANCHOR_NOT_FOUND maps to
INDETERMINATE, ERROR, or something else, and what message to show the user.

FieldFinding.verdict and LabelFindings.verdict are compliance assessments
produced by the vision layer — factual outputs of what the AI found on the
label. The reporting layer may further interpret these (e.g. applying Item 15
excusal logic) before surfacing them to users.
"""

from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

from PIL import Image

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@total_ordering
class Verdict(Enum):
    """Compliance assessment for a single field or an overall label check.

    Each member packs an integer severity and a human-readable description.
    Severity enables direct comparison: Verdict.FAIL > Verdict.WARN.
    Description is the canonical user-facing explanation for this verdict.

    ERROR is only ever emitted by pipeline infrastructure (API timeout,
    parse failure, etc.) — never by the vision model itself.
    """

    description: str  # set by __new__; declared here for type checker visibility

    def __new__(cls, severity: int, description: str):
        obj = object.__new__(cls)
        obj._value_ = severity
        obj.description = description
        return obj

    def __lt__(self, other: "Verdict") -> bool:
        if isinstance(other, Verdict):
            return self.value < other.value
        return NotImplemented

    PASS = 0, "field present and compliant"
    EXEMPT = (
        1,
        "field absent but affirmatively not required for this product type or designation; no compliance issue",
    )
    WARN = 2, "field present but has a minor formatting or wording deviation; review advised"
    ABSENT = (
        3,
        "field not found on the label; may be excused (e.g. embossed container info listed in Item 15); flagged for review",
    )
    FAIL = 4, "field present but non-compliant; likely causes rejection"
    INDETERMINATE = (
        5,
        "image too degraded or field too ambiguous to assess; the pipeline ran but could not reach a conclusion",
    )
    ERROR = (
        6,
        'a component failure prevented analysis (API timeout, unexpected exception, parse failure, etc.); distinguishes "uncertain result" from "broken pipeline"',
    )


# ---------------------------------------------------------------------------
# Reason
# ---------------------------------------------------------------------------


class Reason(Enum):
    """Terminal extraction state emitted by pdf.extract_page1().

    When Page1Result.reason is non-None, automation stops and the submission
    should be flagged for human review. Each member packs a stable string
    code (the value, used for serialisation and logging) and a human-readable
    description. Verdict assignment is the reporting layer's responsibility.
    """

    description: str  # set by __new__; declared here for type checker visibility

    def __new__(cls, code: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = code
        obj.description = description
        return obj

    PDF_UNREADABLE = (
        "pdf_unreadable",
        "The file could not be opened (corrupt, wrong format, permission error, etc.).",
    )
    PDF_EMPTY = (
        "pdf_empty",
        "The PDF opened successfully but contains no pages.",
    )
    ANCHOR_NOT_FOUND = (
        "anchor_not_found",
        "Page 1 was rendered but the anchor text could not be located. "
        "The page may be rotated, upside-down, or not a Form 5100.31. "
        "label_zone is withheld to protect potential PII.",
    )


# ---------------------------------------------------------------------------
# Page 1 extraction result
# ---------------------------------------------------------------------------


@dataclass
class Page1Result:
    """Output of pdf.extract_page1(). Always returned; never None.

    reason: None when extraction succeeded and the anchor was confirmed —
        the result is safe to pass to the label analysis pipeline.
        Non-None is a terminal state: automation stops here and the
        submission should be flagged for human review. The reason describes
        what happened; verdict and messaging are the reporting layer's
        responsibility.

    product_type: "wine" | "distilled_spirits" | "malt_beverage" | None.
        None when Item 5 could not be determined from the checkbox. This
        is non-terminal: label analysis can still proceed with product
        type supplied by a human reviewer.

    label_zone: Cropped label affixing area. None whenever reason is
        non-None — withheld structurally so that potential PII cannot be
        sent to an external API when we cannot confirm this is a valid
        Form 5100.31 in the correct orientation.

    page1_image: Full render of page 1 at RENDER_DPI. Present when reason
        is None or Reason.ANCHOR_NOT_FOUND (useful for report display so
        reviewers can see what was submitted). None when the PDF could not
        be rendered (Reason.PDF_UNREADABLE, Reason.PDF_EMPTY).
    """

    reason: Reason | None
    product_type: str | None
    label_zone: Image.Image | None
    page1_image: Image.Image | None


# ---------------------------------------------------------------------------
# Label compliance findings
# ---------------------------------------------------------------------------


@dataclass
class FieldFinding:
    """Compliance result for a single TTB labeling requirement.

    field:     Snake-case identifier (e.g. "brand_name", "government_warning").
    verdict:   Compliance assessment for this field.
    extracted: Exact text found on the label, or None if absent or unreadable.
    note:      Human-readable explanation when verdict is not PASS, or None.
    """

    field: str
    verdict: Verdict
    extracted: str | None = None
    note: str | None = None


@dataclass
class LabelFindings:
    """Full compliance result for a label zone image.

    verdict:           Overall verdict (worst-case roll-up of all fields).
    fields:            Per-field findings.
    import_indicators: True when the label contains language suggesting an
        imported product ("Imported by ...", "Product of ...", etc.).
        Country-of-origin requirement is only triggered when True.
    raw_response:      Raw text returned by the backend, for debugging.
    """

    verdict: Verdict
    fields: list[FieldFinding]
    import_indicators: bool = False
    raw_response: str | None = None
