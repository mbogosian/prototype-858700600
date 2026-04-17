"""
Comparison logic and compliance assessment for ProofReader.

Receives the raw LabelFindings from the vision layer and applies
product-type-specific excusal logic before findings are passed to the
annotation and report stages.

## Excusals applied

**Item 15 (embossed/blown container info)**
    net_contents and producer_bottler_name_address may be printed on the
    container rather than affixed as a label. If the vision model returns
    FAIL for either (absent field treated as non-compliant), this is
    softened to ABSENT so human review can determine whether Item 15 covers
    the absence. Fields already ABSENT are left unchanged.

**Wine alcohol_content conditionality (27 CFR Part 4)**
    ABV is mandatory on wine labels only when the product exceeds 14% ABV.
    At 7-14% ABV with a "table wine" or "light wine" designation it is
    optional. If the vision model flags ABV as ABSENT or FAIL on a wine
    label, a note is appended to prompt the reviewer to confirm the actual
    ABV before treating the finding as non-compliant.

**Malt beverage alcohol_content conditionality (27 CFR 7.71)**
    ABV is mandatory on malt beverage labels only if the product contains
    added non-beverage alcohol or flavors. A note is appended to ABSENT
    findings to surface this conditionality for the reviewer.

**Verdict re-roll**
    The overall verdict is recalculated as the worst-case across all fields
    after excusals are applied.

Public API:
    assess(findings, product_type) -> LabelFindings
"""

from dataclasses import replace

from proofreader.models import FieldFinding, LabelFindings, Verdict

# Fields whose absence may be excused via Item 15 (embossed/blown container
# info). A FAIL verdict on these is softened to ABSENT; human review confirms
# whether Item 15 covers the absence. Fields already ABSENT are unchanged.
_ITEM15_EXCUSABLE: frozenset[str] = frozenset({
    "net_contents",
    "producer_bottler_name_address",
})


def assess(findings: LabelFindings, product_type: str | None) -> LabelFindings:
    """Apply product-type-specific excusal logic and return updated findings.

    Produces a new LabelFindings with excusals applied and the overall
    verdict re-rolled from the updated fields.
    """
    updated = [_apply_excusals(f, product_type) for f in findings.fields]
    overall = max((f.verdict for f in updated), key=lambda v: v.value) if updated else findings.verdict
    return LabelFindings(
        verdict=overall,
        fields=updated,
        import_indicators=findings.import_indicators,
        raw_response=findings.raw_response,
    )


def _apply_excusals(field: FieldFinding, product_type: str | None) -> FieldFinding:
    """Return a (possibly updated) copy of field with excusal logic applied."""

    # Item 15: absence of excusable fields marked FAIL → soften to ABSENT.
    if field.field in _ITEM15_EXCUSABLE and field.verdict is Verdict.FAIL:
        return replace(
            field,
            verdict=Verdict.ABSENT,
            note=_append(
                field.note,
                "FAIL softened to ABSENT: this field may be embossed or blown "
                "on the container and listed in Item 15; human review required",
            ),
        )

    # Wine ABV conditionality: mandatory only if >14%; optional at 7-14% with
    # "table wine" or "light wine" designation. A note is appended to prompt
    # the reviewer to verify before treating the absence as non-compliant.
    if (
        field.field == "alcohol_content"
        and product_type == "wine"
        and field.verdict in (Verdict.ABSENT, Verdict.FAIL)
    ):
        return replace(
            field,
            note=_append(
                field.note,
                "wine ABV is mandatory only if >14%; optional at 7-14% ABV with "
                "'table wine' or 'light wine' designation — verify actual ABV "
                "before treating this finding as non-compliant",
            ),
        )

    # Malt beverage ABV conditionality: mandatory only if the product contains
    # added non-beverage alcohol or flavors (27 CFR 7.71).
    if (
        field.field == "alcohol_content"
        and product_type == "malt_beverage"
        and field.verdict is Verdict.ABSENT
    ):
        return replace(
            field,
            note=_append(
                field.note,
                "malt beverage ABV is mandatory only if the product contains "
                "added non-beverage alcohol or flavors (27 CFR 7.71); verify "
                "ingredient statement before treating this finding as non-compliant",
            ),
        )

    return field


def _append(existing: str | None, addition: str) -> str:
    """Append addition to an existing note, bracketed."""
    return f"{existing} [{addition}]" if existing else f"[{addition}]"
