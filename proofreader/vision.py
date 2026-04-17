"""
Label compliance analysis via AI vision.

Defines the LabelReader protocol (the abstraction) and ClaudeReader
(backed by the Anthropic Claude API). Local model backends — e.g. Qwen2-VL
or Florence-2 via Ollama — would implement the same protocol and slot in here.

## PII gate

Callers MUST verify that Page1Result.reason is None before passing label_zone
to any LabelReader that transmits the image externally. When reason is non-None,
label_zone is None by construction — the structural absence enforces this gate.

## Verdict semantics

See models.py for the full Verdict type and semantics. In brief:
  PASS / WARN / FAIL / ABSENT / INDETERMINATE — compliance assessments from
    the vision model; produced per field and rolled up to an overall verdict
  ERROR — component failure (API timeout, parse error, etc.); only emitted
    by pipeline infrastructure, never by the vision model itself

Public API:
    LabelReader      — typing.Protocol defining the interface
    ClaudeReader     — Anthropic Claude API implementation
    read_labels()    — convenience function; calls reader.read()
"""

import base64
import io
import json
import logging
import os
from typing import Protocol, runtime_checkable

from PIL import Image

from proofreader.models import FieldFinding, LabelFindings, Verdict

logger = logging.getLogger(__name__)

# Verdict names the model is permitted to return. ERROR is reserved for
# pipeline infrastructure and is clamped to INDETERMINATE if the model emits it.
_MODEL_VERDICTS = {v for v in Verdict if v is not Verdict.ERROR}


def _worst(*verdicts: Verdict) -> Verdict:
    return max(verdicts, key=lambda v: v.value)


# ---------------------------------------------------------------------------
# LabelReader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LabelReader(Protocol):
    """Interface for a label compliance vision backend.

    Implementations receive a PIL image of the label zone and the product type
    (as detected from Item 5 of the form, or None if indeterminate) and return
    a LabelFindings object.
    """

    def read(
        self,
        label_zone: Image.Image,
        product_type: str | None,
    ) -> LabelFindings:
        """Analyze label_zone and return structured compliance findings.

        Args:
            label_zone:   Cropped label affixing area from pdf.extract_page1().
                          Must not contain PII (caller's responsibility).
            product_type: "wine" | "distilled_spirits" | "malt_beverage" | None
                          Determines which TTB requirement set is applied.

        Returns:
            LabelFindings with per-field verdicts.
        """
        ...


# ---------------------------------------------------------------------------
# TTB requirement sets
# ---------------------------------------------------------------------------

# Fields checked for every product type.
_COMMON_FIELDS = [
    "brand_name",
    "class_type_designation",
    "net_contents",
    "producer_bottler_name_address",
    "government_warning",
    # country_of_origin is conditional; handled dynamically based on import_indicators
]

_PRODUCT_FIELDS: dict[str, list[str]] = {
    "wine": [
        "alcohol_content",  # mandatory if >14%; optional 7-14% with table/light wine
        "sulfite_declaration",  # mandatory unless label explicitly states sulfite-free
        "appellation_of_origin",  # required when grape variety, vintage date, etc. are present
    ],
    "distilled_spirits": [
        "alcohol_content",  # mandatory
        "same_field_of_vision",  # brand + class/type + ABV must share a field of vision
        "age_statement",  # required if class/type implies it
    ],
    "malt_beverage": [
        "class_designation",  # mandatory (ale, lager, beer, malt liquor, etc.)
        "net_contents_us_units",  # US customary (fl oz or pints) is mandatory; metric is additive only
        "alcohol_content",  # mandatory only if beverage contains added alcohol flavors
    ],
}

# Human-readable descriptions injected into the AI prompt.
_FIELD_DESCRIPTIONS: dict[str, str] = {
    "brand_name": ("The brand name of the product. Must be present and clearly legible."),
    "class_type_designation": (
        "The class and/or type designation (e.g. 'Cabernet Sauvignon', "
        "'Bourbon Whiskey', 'Pale Ale'). Must be present."
    ),
    "net_contents": (
        "The net contents (volume). May be embossed on the container rather than "
        "printed on the label, in which case it may be absent from the label image. "
        "If present, note the stated volume."
    ),
    "producer_bottler_name_address": (
        "Name and address of the producer, bottler, or importer. Must be present. "
        "For domestic products: 'Bottled by' or 'Produced by' with city and state. "
        "For imports: importer name and US address is also required."
    ),
    "government_warning": (
        "The mandatory government warning statement. EXACT required text: "
        "'GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
        "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems.' "
        "Format requirements: 'GOVERNMENT WARNING:' must appear in ALL CAPITAL LETTERS "
        "and in visibly heavier (bold) type than the body of the warning statement. "
        "Use relative weight: if 'GOVERNMENT WARNING:' is clearly heavier than the surrounding "
        "warning text, that satisfies the requirement. If it is clearly the same weight or lighter, "
        "that is FAIL. If image resolution or compression makes the weight difference genuinely "
        "ambiguous, return WARN — do not default to FAIL. "
        "The statement must appear on a contrasting background, separate from other text."
    ),
    "country_of_origin": (
        "Country of origin statement. Required only for imported products. "
        "Must state the country where the product was produced."
    ),
    "alcohol_content": (
        "Alcohol content statement. IMPORTANT FORMAT RULE: the abbreviation 'ABV' alone "
        "is NOT acceptable under TTB regulations. Acceptable forms include: "
        "'Alc. X% by Vol.', 'Alcohol X% by volume', 'X% Alc./Vol.', 'Alc. X% By Vol.' "
        "For wine: mandatory only if >14%; optional for 7-14% with 'table wine' or 'light wine' on brand label. "
        "For distilled spirits: always mandatory. "
        "For malt beverages: mandatory only if the product contains added non-beverage alcohol or flavors "
        "(27 CFR §7.71); if absent, return ABSENT — not FAIL."
    ),
    "sulfite_declaration": (
        "Sulfite declaration ('Contains sulfites' or 'Contains [specific sulfite]'). "
        "Required for wine unless the label explicitly states it is sulfite-free. "
        "Note: assume sulfites are present unless the label explicitly claims otherwise."
    ),
    "appellation_of_origin": (
        "Appellation of origin (e.g. 'Napa Valley', 'California', 'American'). "
        "Required on the brand label when: a grape variety is used as the type "
        "designation, a vintage date appears, the label uses a semi-generic designation, "
        "or an 'estate bottled' claim is made. Note whether an appellation appears and "
        "whether it appears on what appears to be the brand (front) label."
    ),
    "same_field_of_vision": (
        "For distilled spirits: brand name, class/type designation, and alcohol content "
        "must ALL appear in the same field of vision on the label. Check whether these "
        "three elements can be seen together on a single face of the label."
    ),
    "age_statement": (
        "Age statement (e.g. 'Aged 12 years', '12 Year Old'). Required when the "
        "class/type designation implies aging (e.g. 'Straight Bourbon Whiskey', "
        "'12 Year Old Scotch', 'Aged Rum'). Optional otherwise; note if present."
    ),
    "class_designation": (
        "Malt beverage class designation. Mandatory. Acceptable classes: ale, lager, "
        "beer, malt liquor, stout, porter, malt beverage. Products under 0.5% ABV must "
        "use 'malt beverage', 'cereal beverage', or 'near beer' — NOT 'beer', 'ale', etc."
    ),
    "net_contents_us_units": (
        "Net contents in US customary units (fluid ounces or pints). For malt beverages, "
        "US customary units are mandatory; metric units may appear alongside but cannot "
        "replace them. Check that fluid ounces (fl oz) or pints appear somewhere."
    ),
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(product_type: str | None) -> str:
    """Construct the system + user prompt for the label compliance check."""

    if product_type == "wine":
        product_desc = "Wine (27 CFR Part 4)"
        extra_fields = _PRODUCT_FIELDS["wine"]
    elif product_type == "distilled_spirits":
        product_desc = "Distilled Spirits (27 CFR Part 5)"
        extra_fields = _PRODUCT_FIELDS["distilled_spirits"]
    elif product_type == "malt_beverage":
        product_desc = "Malt Beverage / Beer (27 CFR Part 7)"
        extra_fields = _PRODUCT_FIELDS["malt_beverage"]
    else:
        product_desc = "Unknown (product type could not be determined from form)"
        extra_fields = []

    all_fields = _COMMON_FIELDS + extra_fields

    # Build field-by-field instructions
    field_lines: list[str] = []
    for f in all_fields:
        desc = _FIELD_DESCRIPTIONS.get(f, f)
        field_lines.append(f'  - "{f}": {desc}')
    fields_block = "\n".join(field_lines)

    # Generate verdict list and definitions from the enum, excluding ERROR
    # (which is reserved for pipeline infrastructure and must not be emitted
    # by the model).
    verdict_names = " | ".join(v.name for v in _MODEL_VERDICTS)
    verdict_defs = "\n".join(f"  {v.name:<13} — {v.description}" for v in _MODEL_VERDICTS)

    prompt = f"""You are a TTB (Alcohol and Tobacco Tax and Trade Bureau) label compliance reviewer.

The image shows the label affixing area from a TTB Form F 5100.31 alcohol label application.
Product type (from Item 5 of the application form): **{product_desc}**

Your task is to evaluate the label artwork visible in the image for compliance with TTB
mandatory labeling requirements. Evaluate each of the following fields:

{fields_block}

Additionally, look for any language indicating this is an imported product
("Imported by …", "Product of [foreign country]", "Imported from …", or similar).

Return your findings as a single JSON object with this exact structure:

{{
  "import_indicators": <true | false>,
  "fields": [
    {{
      "field": "<field_name>",
      "verdict": "<{verdict_names}>",
      "extracted": "<exact text found on label, or null if genuinely absent>",
      "note": "<brief explanation if verdict is not PASS, or null>"
    }},
    ...
  ]
}}

Verdict definitions:
{verdict_defs}

Rules:
- Always populate "extracted" with the actual text you found on the label, even for WARN or FAIL
  verdicts. Only set "extracted" to null when the field is genuinely absent from the label.
  The extracted text is used to locate and highlight the field in the label image.
- If import_indicators is true, add a "country_of_origin" entry to the fields list.
- For the government_warning field: any deviation from the exact required text is FAIL.
  For bold formatting: FAIL only if "GOVERNMENT WARNING:" is clearly the same weight or lighter
  than the body of the warning; WARN if weight difference is ambiguous from the image quality.
  Do not default to FAIL when bold status is uncertain — prefer WARN.
- Use EXEMPT when a field is absent AND you can affirmatively determine from the label content
  that it is not required for this specific product or designation (e.g., age_statement for a
  product not claiming a designation that implies aging; alcohol_content for a malt beverage
  with no indication of added non-beverage alcohol or flavors). Do not guess — only use EXEMPT
  when the regulatory basis for non-requirement is clear. Use ABSENT when absence is noted but
  you cannot confidently determine whether the field is required.
- For alcohol_content: using "ABV" alone (without "Alc." and "Vol.") is FAIL.
- Respond ONLY with the JSON object — no explanation, no markdown fences.
"""
    return prompt


# ---------------------------------------------------------------------------
# Claude API implementation
# ---------------------------------------------------------------------------

# Default model for label analysis. Claude's vision capability is strong on
# small, stylized label text; Sonnet balances cost and accuracy.
_DEFAULT_MODEL = "claude-sonnet-4-6"

# JSON parse failure is a component failure — ERROR, not INDETERMINATE.
_PARSE_FAILURE_FINDING = FieldFinding(
    field="__parse_error__",
    verdict=Verdict.ERROR,
    note="Backend returned malformed JSON; could not extract structured findings.",
)


class ClaudeReader:
    """LabelReader implementation backed by the Anthropic Claude API.

    Initialisation is lightweight — the Anthropic client connects on demand.

    Args:
        api_key: Anthropic API key. Defaults to the ANTHROPIC_API_KEY
                 environment variable if not supplied.
        model:   Claude model ID. Defaults to _DEFAULT_MODEL.
        max_tokens: Maximum tokens for the response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._max_tokens = max_tokens
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def read(
        self,
        label_zone: Image.Image,
        product_type: str | None,
    ) -> LabelFindings:
        """Send label_zone to Claude and parse the JSON compliance findings.

        The image is JPEG-encoded and base64-transmitted. No PII is in this
        image (label_zone is None by construction when Page1Result.reason is set).

        On any API or parse error, returns a single ERROR finding rather than
        raising, so the worker pipeline can continue. ERROR distinguishes
        infrastructure failure from INDETERMINATE (ambiguous content).
        """
        try:
            return self._call_api(label_zone, product_type)
        except Exception as exc:
            logger.error("ClaudeReader API error: %s", exc, exc_info=True)
            return LabelFindings(
                verdict=Verdict.ERROR,
                fields=[
                    FieldFinding(
                        field="__api_error__",
                        verdict=Verdict.ERROR,
                        note=f"Vision API call failed: {exc}",
                    )
                ],
            )

    def _encode_image(self, img: Image.Image) -> str:
        """Return a base64-encoded JPEG of img."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")

    def _call_api(
        self,
        label_zone: Image.Image,
        product_type: str | None,
    ) -> LabelFindings:
        client = self._get_client()
        prompt = _build_prompt(product_type)
        image_data = self._encode_image(label_zone)

        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text if response.content else ""
        logger.debug("ClaudeReader raw response: %s", raw[:500])
        return _parse_response(raw, product_type)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_response(raw: str, product_type: str | None) -> LabelFindings:
    """Parse the JSON response from the vision model into LabelFindings.

    On any parse error, returns a single ERROR finding (component failure,
    not ambiguous content).
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Strip accidental markdown fences and retry once
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last fence lines
            inner = [ln for ln in lines if not ln.startswith("```")]
            cleaned = "\n".join(inner).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failure: %s — raw: %.200s", exc, raw)
            return LabelFindings(
                verdict=Verdict.ERROR,
                fields=[_PARSE_FAILURE_FINDING],
                raw_response=raw,
            )

    import_indicators = bool(data.get("import_indicators", False))

    fields: list[FieldFinding] = []
    for item in data.get("fields", []):
        verdict_str = item.get("verdict", "INDETERMINATE")
        # Clamp unrecognised names and any spoofed ERROR to INDETERMINATE.
        # ERROR is reserved for pipeline infrastructure; the model must not emit it.
        clamp_note: str | None = None
        try:
            verdict = Verdict[verdict_str]
            if verdict is Verdict.ERROR:
                verdict = Verdict.INDETERMINATE
                clamp_note = f"model returned {verdict_str!r} (reserved for pipeline failures; clamped to INDETERMINATE)"
        except KeyError:
            verdict = Verdict.INDETERMINATE
            clamp_note = (
                f"model returned unrecognised verdict {verdict_str!r} (clamped to INDETERMINATE)"
            )
        original_note = item.get("note") or None
        note = (
            f"{original_note} [{clamp_note}]"
            if original_note and clamp_note
            else clamp_note or original_note
        )
        fields.append(
            FieldFinding(
                field=str(item.get("field", "unknown")),
                verdict=verdict,
                extracted=item.get("extracted") or None,
                note=note,
            )
        )

    if not fields:
        # Empty fields list means the model returned nothing useful — treat as
        # a component failure (ERROR) since we cannot distinguish a silent API
        # success from a malformed response.
        fields = [
            FieldFinding(
                field="__empty_response__",
                verdict=Verdict.ERROR,
                note="Vision model returned no field findings.",
            )
        ]

    overall = _worst(*(f.verdict for f in fields))
    return LabelFindings(
        verdict=overall,
        fields=fields,
        import_indicators=import_indicators,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def read_labels(
    label_zone: Image.Image,
    product_type: str | None,
    reader: LabelReader | None = None,
) -> LabelFindings:
    """Analyze label_zone for TTB compliance and return structured findings.

    This is the primary entry point for the worker pipeline.

    IMPORTANT: caller must verify Page1Result.reason is None before calling
    this function. When reason is non-None, label_zone is None by
    construction. Passing it unchecked to any reader that transmits images
    externally may fail.

    Args:
        label_zone:   PIL image of the label affixing area.
        product_type: "wine" | "distilled_spirits" | "malt_beverage" | None.
        reader:       LabelReader implementation to use. If None, a
                      ClaudeReader is instantiated using the ANTHROPIC_API_KEY
                      environment variable.

    Returns:
        LabelFindings with per-field verdicts and an overall verdict.
    """
    if reader is None:
        reader = ClaudeReader()
    return reader.read(label_zone, product_type)
