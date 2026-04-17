"""
Unit tests for the comparison and excusal logic in compare.py.

All tests are pure Python — no PaddleOCR, no API calls, no file I/O.
"""

import pytest

from proofreader.compare import _append, assess
from proofreader.models import FieldFinding, LabelFindings, Verdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _findings(*fields: FieldFinding, import_indicators: bool = False) -> LabelFindings:
    """Build a LabelFindings with the overall verdict rolled up from fields."""
    overall = max((f.verdict for f in fields), key=lambda v: v.value) if fields else Verdict.PASS
    return LabelFindings(
        verdict=overall,
        fields=list(fields),
        import_indicators=import_indicators,
        raw_response="fake",
    )


# ---------------------------------------------------------------------------
# _append helper
# ---------------------------------------------------------------------------


def test_append_no_existing_note() -> None:
    assert _append(None, "added") == "[added]"


def test_append_existing_note() -> None:
    assert _append("original", "added") == "original [added]"


def test_append_empty_string_treated_as_falsy() -> None:
    # An empty string is falsy; treated the same as None.
    assert _append("", "added") == "[added]"


# ---------------------------------------------------------------------------
# assess — passthrough (no excusals apply)
# ---------------------------------------------------------------------------


def test_assess_pass_fields_unchanged() -> None:
    f = FieldFinding(field="brand_name", verdict=Verdict.PASS)
    result = assess(_findings(f), product_type="wine")
    assert result.fields[0].verdict is Verdict.PASS
    assert result.fields[0].note is None
    assert result.verdict is Verdict.PASS


def test_assess_empty_fields_preserves_verdict() -> None:
    """No fields: overall verdict is preserved from the input, not re-rolled to PASS."""
    empty = LabelFindings(verdict=Verdict.INDETERMINATE, fields=[], raw_response=None)
    result = assess(empty, product_type=None)
    assert result.fields == []
    assert result.verdict is Verdict.INDETERMINATE


def test_assess_preserves_import_indicators() -> None:
    f = FieldFinding(field="brand_name", verdict=Verdict.PASS)
    result = assess(_findings(f, import_indicators=True), product_type=None)
    assert result.import_indicators is True


def test_assess_preserves_raw_response() -> None:
    f = FieldFinding(field="brand_name", verdict=Verdict.PASS)
    findings = _findings(f)
    findings = LabelFindings(
        verdict=findings.verdict,
        fields=findings.fields,
        raw_response="original raw",
    )
    result = assess(findings, product_type=None)
    assert result.raw_response == "original raw"


# ---------------------------------------------------------------------------
# Item 15 excusal — net_contents and producer_bottler_name_address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["net_contents", "producer_bottler_name_address"])
def test_item15_fail_softened_to_absent(field_name: str) -> None:
    f = FieldFinding(field=field_name, verdict=Verdict.FAIL)
    result = assess(_findings(f), product_type=None)
    assert result.fields[0].verdict is Verdict.ABSENT


@pytest.mark.parametrize("field_name", ["net_contents", "producer_bottler_name_address"])
def test_item15_fail_note_added(field_name: str) -> None:
    f = FieldFinding(field=field_name, verdict=Verdict.FAIL)
    result = assess(_findings(f), product_type=None)
    note = result.fields[0].note
    assert note is not None
    assert "Item 15" in note
    assert "human review" in note


@pytest.mark.parametrize("field_name", ["net_contents", "producer_bottler_name_address"])
def test_item15_existing_note_preserved_and_appended(field_name: str) -> None:
    f = FieldFinding(field=field_name, verdict=Verdict.FAIL, note="prior note")
    result = assess(_findings(f), product_type=None)
    note = result.fields[0].note
    assert note is not None
    assert note.startswith("prior note")
    assert "Item 15" in note


@pytest.mark.parametrize("field_name", ["net_contents", "producer_bottler_name_address"])
def test_item15_absent_not_changed(field_name: str) -> None:
    """Fields already ABSENT are left alone — Item 15 only softens FAIL."""
    f = FieldFinding(field=field_name, verdict=Verdict.ABSENT, note="existing note")
    result = assess(_findings(f), product_type=None)
    assert result.fields[0].verdict is Verdict.ABSENT
    assert result.fields[0].note == "existing note"


def test_item15_does_not_apply_to_other_fields() -> None:
    """A FAIL on an unrelated field is not softened."""
    f = FieldFinding(field="brand_name", verdict=Verdict.FAIL)
    result = assess(_findings(f), product_type=None)
    assert result.fields[0].verdict is Verdict.FAIL


# ---------------------------------------------------------------------------
# Item 15 excusal — verdict re-roll
# ---------------------------------------------------------------------------


def test_item15_softening_raises_verdict_to_absent_not_fail() -> None:
    """After FAIL → ABSENT softening the overall verdict re-rolls to ABSENT, not FAIL."""
    f = FieldFinding(field="net_contents", verdict=Verdict.FAIL)
    result = assess(_findings(f), product_type=None)
    assert result.verdict is Verdict.ABSENT


def test_item15_softening_does_not_lower_overall_verdict() -> None:
    """If another field is already FAIL, the overall verdict stays FAIL after softening."""
    f1 = FieldFinding(field="net_contents", verdict=Verdict.FAIL)
    f2 = FieldFinding(field="brand_name", verdict=Verdict.FAIL)
    result = assess(_findings(f1, f2), product_type=None)
    assert result.fields[0].verdict is Verdict.ABSENT  # net_contents softened
    assert result.fields[1].verdict is Verdict.FAIL  # brand_name unchanged
    assert result.verdict is Verdict.FAIL  # worst-case still FAIL


# ---------------------------------------------------------------------------
# Wine ABV conditionality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", [Verdict.ABSENT, Verdict.FAIL])
def test_wine_abv_note_added(verdict: Verdict) -> None:
    f = FieldFinding(field="alcohol_content", verdict=verdict)
    result = assess(_findings(f), product_type="wine")
    note = result.fields[0].note
    assert note is not None
    assert "14%" in note
    assert "table wine" in note.lower() or "table wine" in note


@pytest.mark.parametrize("verdict", [Verdict.ABSENT, Verdict.FAIL])
def test_wine_abv_verdict_not_changed(verdict: Verdict) -> None:
    f = FieldFinding(field="alcohol_content", verdict=verdict)
    result = assess(_findings(f), product_type="wine")
    assert result.fields[0].verdict is verdict


def test_wine_abv_pass_not_annotated() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.PASS, extracted="Alc. 13.5% Vol.")
    result = assess(_findings(f), product_type="wine")
    assert result.fields[0].note is None


def test_wine_abv_existing_note_preserved_and_appended() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT, note="prior note")
    result = assess(_findings(f), product_type="wine")
    note = result.fields[0].note
    assert note is not None
    assert note.startswith("prior note")
    assert "14%" in note


def test_wine_abv_no_note_for_non_wine() -> None:
    """ABV conditionality note is only added for wine, not distilled_spirits."""
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT)
    result = assess(_findings(f), product_type="distilled_spirits")
    assert result.fields[0].note is None


def test_wine_abv_no_note_for_none_product_type() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT)
    result = assess(_findings(f), product_type=None)
    assert result.fields[0].note is None


# ---------------------------------------------------------------------------
# Malt beverage ABV conditionality
# ---------------------------------------------------------------------------


def test_malt_abv_absent_note_added() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT)
    result = assess(_findings(f), product_type="malt_beverage")
    note = result.fields[0].note
    assert note is not None
    assert "27 CFR 7.71" in note
    assert "non-beverage" in note


def test_malt_abv_absent_verdict_not_changed() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT)
    result = assess(_findings(f), product_type="malt_beverage")
    assert result.fields[0].verdict is Verdict.ABSENT


def test_malt_abv_fail_not_annotated() -> None:
    """Malt beverage ABV conditionality only applies to ABSENT, not FAIL."""
    f = FieldFinding(field="alcohol_content", verdict=Verdict.FAIL)
    result = assess(_findings(f), product_type="malt_beverage")
    assert result.fields[0].note is None


def test_malt_abv_pass_not_annotated() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.PASS, extracted="Alc. 5.0% Vol.")
    result = assess(_findings(f), product_type="malt_beverage")
    assert result.fields[0].note is None


def test_malt_abv_existing_note_preserved_and_appended() -> None:
    f = FieldFinding(field="alcohol_content", verdict=Verdict.ABSENT, note="prior note")
    result = assess(_findings(f), product_type="malt_beverage")
    note = result.fields[0].note
    assert note is not None
    assert note.startswith("prior note")
    assert "27 CFR 7.71" in note


# ---------------------------------------------------------------------------
# Verdict re-roll
# ---------------------------------------------------------------------------


def test_verdict_reroll_worst_case() -> None:
    """Overall verdict is the worst-case across all updated fields."""
    fields = [
        FieldFinding(field="brand_name", verdict=Verdict.PASS),
        FieldFinding(field="class_type_designation", verdict=Verdict.WARN),
        FieldFinding(field="net_contents", verdict=Verdict.FAIL),  # softened to ABSENT
    ]
    result = assess(_findings(*fields), product_type="wine")
    # net_contents FAIL → ABSENT; worst-case is WARN vs ABSENT → ABSENT (severity 2 vs 1)
    assert result.verdict is Verdict.ABSENT


def test_verdict_reroll_all_pass() -> None:
    fields = [
        FieldFinding(field="brand_name", verdict=Verdict.PASS),
        FieldFinding(field="net_contents", verdict=Verdict.PASS),
    ]
    result = assess(_findings(*fields), product_type=None)
    assert result.verdict is Verdict.PASS


def test_verdict_reroll_independent_of_input_verdict() -> None:
    """The input overall verdict is ignored; it is always recomputed from fields."""
    f = FieldFinding(field="brand_name", verdict=Verdict.PASS)
    # Manually set a stale overall verdict of FAIL even though the only field is PASS.
    stale = LabelFindings(verdict=Verdict.FAIL, fields=[f])
    result = assess(stale, product_type=None)
    assert result.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# Multiple excusals in a single findings set
# ---------------------------------------------------------------------------


def test_multiple_excusals_applied_independently() -> None:
    """Both Item 15 excusable fields can be softened in the same findings."""
    f1 = FieldFinding(field="net_contents", verdict=Verdict.FAIL)
    f2 = FieldFinding(field="producer_bottler_name_address", verdict=Verdict.FAIL)
    f3 = FieldFinding(field="brand_name", verdict=Verdict.PASS)
    result = assess(_findings(f1, f2, f3), product_type=None)

    assert result.fields[0].verdict is Verdict.ABSENT
    assert result.fields[1].verdict is Verdict.ABSENT
    assert result.fields[2].verdict is Verdict.PASS
    assert result.verdict is Verdict.ABSENT
