"""
Unit tests for proofreader.vision.

Tests _parse_response() directly — no network calls, no PaddleOCR.
ClaudeReader API-call tests use unittest.mock to patch anthropic.Anthropic.
"""

import json
from unittest.mock import MagicMock, patch

import anthropic

from proofreader.models import Verdict
from proofreader.vision import ClaudeReader, _parse_response

# ---------------------------------------------------------------------------
# _parse_response — valid inputs
# ---------------------------------------------------------------------------


def _make_response(fields: list[dict], import_indicators: bool = False) -> str:
    return json.dumps({"import_indicators": import_indicators, "fields": fields})


def test_parse_all_pass() -> None:
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "PASS", "extracted": "Château X", "note": None},
            {
                "field": "government_warning",
                "verdict": "PASS",
                "extracted": "GOVERNMENT WARNING: ...",
                "note": None,
            },
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.PASS
    assert len(result.fields) == 2
    assert all(f.verdict is Verdict.PASS for f in result.fields)


def test_parse_worst_case_rollup() -> None:
    """Overall verdict is the worst across all fields."""
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "PASS", "extracted": "X", "note": None},
            {
                "field": "class_type_designation",
                "verdict": "WARN",
                "extracted": "Y",
                "note": "minor deviation",
            },
            {
                "field": "government_warning",
                "verdict": "FAIL",
                "extracted": None,
                "note": "missing",
            },
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.FAIL


def test_parse_import_indicators_propagated() -> None:
    raw = _make_response(
        [{"field": "brand_name", "verdict": "PASS", "extracted": "X", "note": None}],
        import_indicators=True,
    )
    result = _parse_response(raw, "wine")
    assert result.import_indicators is True


def test_parse_import_indicators_false_by_default() -> None:
    raw = _make_response(
        [{"field": "brand_name", "verdict": "PASS", "extracted": "X", "note": None}]
    )
    result = _parse_response(raw, "wine")
    assert result.import_indicators is False


def test_parse_absent_verdict() -> None:
    raw = _make_response(
        [
            {
                "field": "net_contents",
                "verdict": "ABSENT",
                "extracted": None,
                "note": "not on label",
            },
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.ABSENT
    assert result.fields[0].verdict is Verdict.ABSENT


def test_parse_indeterminate_verdict() -> None:
    raw = _make_response(
        [
            {
                "field": "government_warning",
                "verdict": "INDETERMINATE",
                "extracted": None,
                "note": "too blurry",
            },
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.INDETERMINATE


# ---------------------------------------------------------------------------
# _parse_response — error / edge cases
# ---------------------------------------------------------------------------


def test_parse_model_emits_error_is_clamped_to_indeterminate() -> None:
    """The model must not emit ERROR; if it does, clamp to INDETERMINATE."""
    raw = _make_response(
        [
            {
                "field": "brand_name",
                "verdict": "ERROR",
                "extracted": None,
                "note": "should not happen",
            },
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.fields[0].verdict is Verdict.INDETERMINATE
    # Overall verdict is also INDETERMINATE (clamped from a single field).
    assert result.verdict is Verdict.INDETERMINATE
    # The note must report what the model actually returned.
    assert result.fields[0].note is not None
    assert "ERROR" in result.fields[0].note
    assert "should not happen" in result.fields[0].note  # original note preserved


def test_parse_unknown_verdict_name_clamped_to_indeterminate() -> None:
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "BOGUS", "extracted": None, "note": None},
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.fields[0].verdict is Verdict.INDETERMINATE
    assert result.fields[0].note is not None
    assert "BOGUS" in result.fields[0].note


def test_parse_malformed_json_returns_error() -> None:
    result = _parse_response("this is not json", "wine")
    assert result.verdict is Verdict.ERROR
    assert result.fields[0].field == "__parse_error__"


def test_parse_json_with_markdown_fences_recovered() -> None:
    """Accidental markdown code fences around the JSON are stripped and retried."""
    inner = _make_response(
        [
            {"field": "brand_name", "verdict": "PASS", "extracted": "X", "note": None},
        ]
    )
    raw = f"```json\n{inner}\n```"
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.PASS


def test_parse_empty_fields_list_returns_error() -> None:
    """Empty fields list is treated as a component failure (ERROR), not PASS."""
    raw = json.dumps({"import_indicators": False, "fields": []})
    result = _parse_response(raw, "wine")
    assert result.verdict is Verdict.ERROR
    assert result.fields[0].field == "__empty_response__"


def test_parse_null_extracted_and_note_preserved_as_none() -> None:
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "ABSENT", "extracted": None, "note": None},
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.fields[0].extracted is None
    assert result.fields[0].note is None


def test_parse_raw_response_stored() -> None:
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "PASS", "extracted": "X", "note": None},
        ]
    )
    result = _parse_response(raw, "wine")
    assert result.raw_response == raw


# ---------------------------------------------------------------------------
# ClaudeReader — API error handling (mocked)
# ---------------------------------------------------------------------------


def _make_mock_client(response_text: str) -> MagicMock:
    """
    Return a mock anthropic.Anthropic client that returns response_text.

    Ideally, we'd allow for passing an instance of the Anthropic API to our
    ClaudeReader for testing and avoid patching it in place, but this is good
    enough for a toy exercise.
    """
    mock_content = MagicMock(spec=anthropic.types.TextBlock)
    mock_content.text = response_text
    mock_response = MagicMock(spec=anthropic.types.Message)
    mock_response.content = [mock_content]
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = mock_response
    return mock_client


def _make_label_image():
    from PIL import Image

    return Image.new("RGB", (200, 200), color=(200, 200, 200))


def test_claude_reader_returns_findings_on_success() -> None:
    raw = _make_response(
        [
            {"field": "brand_name", "verdict": "PASS", "extracted": "Château X", "note": None},
        ]
    )
    mock_client = _make_mock_client(raw)

    with patch("anthropic.Anthropic", return_value=mock_client):
        reader = ClaudeReader()
        result = reader.read(_make_label_image(), "wine")

    assert result.verdict is Verdict.PASS


def test_claude_reader_timeout_returns_error_verdict() -> None:
    """ClaudeReader catches transport exceptions and returns ERROR, not raises."""
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = TimeoutError("connection timed out")

    with patch("anthropic.Anthropic", return_value=mock_client):
        reader = ClaudeReader()
        result = reader.read(_make_label_image(), "wine")

    assert result.verdict is Verdict.ERROR
    assert result.fields[0].field == "__api_error__"
    assert "timed out" in (result.fields[0].note or "")


def test_claude_reader_exception_does_not_propagate() -> None:
    """Any exception from the API call is swallowed and converted to ERROR."""
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = RuntimeError("unexpected failure")

    with patch("anthropic.Anthropic", return_value=mock_client):
        reader = ClaudeReader()
        # Must not raise.
        result = reader.read(_make_label_image(), "distilled_spirits")

    assert result.verdict is Verdict.ERROR
