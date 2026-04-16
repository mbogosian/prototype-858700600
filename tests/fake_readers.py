"""
Test doubles for the LabelReader protocol.

FakeLabelReader satisfies the LabelReader protocol without making any network
calls. Use it wherever a LabelReader is needed in tests — worker pipeline tests,
compare/annotate/report unit tests, etc.

Use the factory methods for common scenarios rather than constructing LabelFindings
by hand. For cases not covered by a factory, pass findings= or raises= directly.

## Simulating transport failures

Two distinct failure modes exist in the real pipeline:

  ClaudeReader catches its own errors
      ClaudeReader.read() wraps _call_api() in try/except and returns
      LabelFindings(verdict=ERROR) on any exception. Test this path with
      FakeLabelReader.api_error() or by patching anthropic.Anthropic directly.

  An unhandled exception escapes the reader
      If a reader implementation does not catch its errors, the exception
      propagates up through worker._process() and is caught by the executor
      done-callback (_on_done), which marks the job ERROR. Test this path with
      FakeLabelReader.raising(SomeException(...)).

Both paths result in a job marked ERROR but via different code routes, so both
are worth covering.
"""

from PIL import Image

from proofreader.models import FieldFinding, LabelFindings, Verdict

# ---------------------------------------------------------------------------
# FakeLabelReader
# ---------------------------------------------------------------------------


class FakeLabelReader:
    """Canned LabelReader for tests.

    Args:
        findings: Returned by read() when raises is None.
        raises:   Raised by read() instead of returning findings.
                  Simulates a reader that did not catch its own exception
                  (exercises the worker executor done-callback path).

    Use the factory methods below for common scenarios.
    """

    def __init__(
        self,
        findings: LabelFindings | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._findings = findings
        self._raises = raises
        # Track calls so tests can assert the reader was (or was not) invoked.
        self.call_count = 0
        self.last_product_type: str | None = None

    def read(self, label_zone: Image.Image, product_type: str | None) -> LabelFindings:
        self.call_count += 1
        self.last_product_type = product_type
        if self._raises is not None:
            raise self._raises
        if self._findings is not None:
            return self._findings
        return _all_pass()

    # ---------------------------------------------------------------------------
    # Factory methods
    # ---------------------------------------------------------------------------

    @classmethod
    def passing(cls) -> "FakeLabelReader":
        """All fields PASS; overall verdict PASS."""
        return cls(findings=_all_pass())

    @classmethod
    def with_verdict(cls, verdict: Verdict, note: str | None = None) -> "FakeLabelReader":
        """Single synthetic field carrying the given verdict.

        Useful when you only care about the overall verdict, not specific fields.
        """
        findings = LabelFindings(
            verdict=verdict,
            fields=[
                FieldFinding(
                    field="__synthetic__",
                    verdict=verdict,
                    note=note,
                )
            ],
        )
        return cls(findings=findings)

    @classmethod
    def with_fields(cls, fields: list[FieldFinding]) -> "FakeLabelReader":
        """Explicit field list; overall verdict rolled up from worst field."""
        overall = max((f.verdict for f in fields), key=lambda v: v.value)
        return cls(findings=LabelFindings(verdict=overall, fields=fields))

    @classmethod
    def api_error(cls, message: str = "simulated API error") -> "FakeLabelReader":
        """Simulates ClaudeReader catching a transport failure and returning ERROR.

        This is the normal return path — the reader caught its own exception
        and converted it to an ERROR LabelFindings. The worker pipeline receives
        a valid return value and handles it in the normal flow.
        """
        findings = LabelFindings(
            verdict=Verdict.ERROR,
            fields=[
                FieldFinding(
                    field="__api_error__",
                    verdict=Verdict.ERROR,
                    note=f"Vision API call failed: {message}",
                )
            ],
        )
        return cls(findings=findings)

    @classmethod
    def raising(cls, exc: BaseException | None = None) -> "FakeLabelReader":
        """Simulates a reader that does not catch its own exception.

        The exception escapes worker._process() and is caught by the executor
        done-callback, which marks the job ERROR. Use this to test that code
        path rather than the normal ERROR-verdict return path.

        If exc is None, raises RuntimeError("simulated reader failure").
        """
        if exc is None:
            exc = RuntimeError("simulated reader failure")
        return cls(raises=exc)

    @classmethod
    def timeout(cls) -> "FakeLabelReader":
        """Simulates an uncaught TimeoutError from a transport layer.

        Equivalent to raising(TimeoutError(...)) but more descriptive at the
        call site. Note: in ClaudeReader, timeouts are caught and converted to
        ERROR findings; this factory tests the uncaught path.
        """
        return cls(raises=TimeoutError("simulated connection timeout"))


# ---------------------------------------------------------------------------
# Canned LabelFindings helpers (used by factories; available for direct use)
# ---------------------------------------------------------------------------


def _all_pass() -> LabelFindings:
    return LabelFindings(
        verdict=Verdict.PASS,
        fields=[
            FieldFinding(field="brand_name", verdict=Verdict.PASS, extracted="Example Brand"),
            FieldFinding(
                field="class_type_designation", verdict=Verdict.PASS, extracted="Table Wine"
            ),
            FieldFinding(field="net_contents", verdict=Verdict.PASS, extracted="750 mL"),
            FieldFinding(
                field="producer_bottler_name_address",
                verdict=Verdict.PASS,
                extracted="Bottled by Example Winery, Napa, CA",
            ),
            FieldFinding(
                field="government_warning",
                verdict=Verdict.PASS,
                extracted="GOVERNMENT WARNING: ...",
            ),
        ],
    )


def findings_with_warn() -> LabelFindings:
    """One WARN field among otherwise PASSing fields; overall verdict WARN."""
    fields = _all_pass().fields.copy()
    fields[0] = FieldFinding(
        field="brand_name",
        verdict=Verdict.WARN,
        extracted="EXAMPLE BRAND",
        note="Brand name appears in all-caps; label uses different capitalisation than application.",
    )
    return LabelFindings(verdict=Verdict.WARN, fields=fields)


def findings_with_fail() -> LabelFindings:
    """One FAIL field (government warning non-compliant); overall verdict FAIL."""
    fields = _all_pass().fields.copy()
    fields[-1] = FieldFinding(
        field="government_warning",
        verdict=Verdict.FAIL,
        extracted="Government Warning: ...",
        note="'GOVERNMENT WARNING:' must appear in all-caps bold; found mixed-case.",
    )
    return LabelFindings(verdict=Verdict.FAIL, fields=fields)


def findings_with_absent() -> LabelFindings:
    """One ABSENT field (net_contents not found on label); overall verdict ABSENT."""
    fields = _all_pass().fields.copy()
    fields[2] = FieldFinding(
        field="net_contents",
        verdict=Verdict.ABSENT,
        note="Net contents not found on label; may be embossed on container (Item 15).",
    )
    return LabelFindings(verdict=Verdict.ABSENT, fields=fields)


def findings_indeterminate() -> LabelFindings:
    """Overall INDETERMINATE — label too degraded to assess."""
    return LabelFindings(
        verdict=Verdict.INDETERMINATE,
        fields=[
            FieldFinding(
                field="__image_quality__",
                verdict=Verdict.INDETERMINATE,
                note="Label image too degraded to assess required fields.",
            )
        ],
    )
