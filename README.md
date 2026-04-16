# ProofReader

AI-powered alcohol label verification for TTB (Alcohol and Tobacco Tax and Trade Bureau) label approval applications.

---

## Problem Summary

The TTB reviews approximately 150,000 label applications per year using a staff of 47 agents. Each application is submitted as TTB Form F 5100.31 — a one-page form with application data at the top and physical label artwork affixed at the bottom. Agents manually verify that the label artwork matches the declared application data and satisfies mandatory TTB labeling requirements.

Much of this work is mechanical: confirming that the government warning statement is present and correctly formatted, that required fields appear on the label, that alcohol content is stated in the correct form, etc. ProofReader reads the product type from the application form, applies the corresponding TTB requirement set, checks the affixed labels against those requirements, and produces a structured accept/reject recommendation with visual feedback — so agents can focus on edge cases and judgment calls rather than routine verification.

**Input:** A scanned raster PDF of a completed TTB Form F 5100.31 (single application).

**Output:** A web-based agent interface providing:
- Upload of single or batch PDFs; background processing with live status
- Browser notifications when submitted items complete
- Master list of all processed applications with form thumbnail, product type, verdict badge, and timestamp
- Per-application HTML recommendation report (downloadable), annotated label image, and JSON findings
- Streaming log window with filter

---

## Setup

```bash
# Install dependencies
uv sync

# Set API key and directory paths
export ANTHROPIC_API_KEY=your_key_here
export PROOFREADER_INBOX=./inbox
export PROOFREADER_OUTBOX=./outbox
export PROOFREADER_WORKERS=3

# Create directories if needed
mkdir -p inbox outbox

# Run development server (file watcher starts automatically on startup)
uv run uvicorn main:app --reload
```

The web UI is served at `http://localhost:8000`. Drop PDFs into the inbox directory or upload via the browser — both trigger the same worker pipeline.

---

## Architecture

### System overview

```
         Browser (agent)
              │
    ┌─────────▼──────────┐
    │   FastAPI web app   │  ← upload, SSE notifications, master list,
    │   (main.py)         │    log stream, report downloads
    └─────────┬──────────┘
              │ writes PDF to inbox/
              ▼
    ┌─────────────────────┐
    │   inbox/  (watched) │  ← prototype's work queue stand-in
    └─────────┬───────────┘
              │ watchdog detects new file
              ▼
    ┌─────────────────────┐
    │   Worker pool       │  ← ThreadPoolExecutor, N workers (configurable)
    │   (worker.py)       │    in-memory set tracks in-flight PDFs
    └─────────┬───────────┘
              │ per-PDF pipeline (see below)
              ▼
    ┌─────────────────────┐
    │   outbox/           │  ← completed items; web UI reads for master list
    │   {id}/             │
    │     original.pdf    │
    │     report.html     │
    │     findings.json   │
    │     thumbnail.jpg   │
    └─────────────────────┘
```

Uploads arrive via the web UI; agents never interact with the filesystem directly. The inbox/outbox directories are internal prototype infrastructure, standing in for a proper message queue. See Production Considerations for what replaces them.

### Threading model

The FastAPI app and all SSE generators run on a single asyncio event loop (uvicorn's main thread). PDF processing runs in a `ThreadPoolExecutor` so PaddleOCR and the Anthropic API call execute in parallel without blocking the event loop. A watchdog thread monitors the inbox for manually-dropped files.

The two concurrency domains are bridged in `worker.py` via `loop.call_soon_threadsafe()`: worker threads schedule log lines and job-completion events onto the event loop rather than touching asyncio primitives directly. Shared job state (`_jobs`) is protected by a plain `threading.Lock`; the critical sections are trivially short so event loop handlers blocking on the lock is not a practical concern.

### Per-PDF pipeline (inside each worker)

```
Input PDF
        │
        ▼
1. PyMuPDF — render page 1 at 300 DPI
   Save page 1 thumbnail (JPEG, small) for master list
        │
        ├──► Upper zone  →  PaddleOCR → read Item 5 checkbox (product type only)
        │                              No data transmitted externally
        │
        └──► Lower zone  →  crop label image area
             │
             ▼
2. LabelReader (abstracted interface)
   Current implementation: Anthropic Claude API
   Label images only — no PII transmitted
   Returns: per-field extracted values, government warning assessment,
            approximate element locations, import indicators
   Future slot: local vision model (Qwen2-VL, Florence-2 via Ollama), A/B testing
             │
             ▼
3. Comparison logic
   Verdict per field: PASS / WARN / FAIL / ABSENT / INDETERMINATE / ERROR
   Requirement set selected by Item 5 product type
   Country of origin: checked only if label contains import indicators
             │
             ▼
4. PaddleOCR on label zone — text + rotated quad bounding boxes
   Match LabelReader findings → pixel locations for annotation
   Angled text: draw native quad polygon
   Curved/unlocatable text: dashed approximate region
             │
             ▼
5. Pillow — annotate label image copy
   Green polygon: PASS     Orange polygon: WARN     Red polygon: FAIL
   Dashed outline: location approximate
   Absent/INDETERMINATE/ERROR: report text only, no box
             │
             ▼
6. Jinja2 — render report.html (base64-embedded annotated image)
   Write findings.json alongside
   Move original.pdf to outbox/{id}/
   Notify SSE subscribers → browser notification fires
```

**Stack:**
- Language: Python
- Package manager: uv
- Web framework: FastAPI (serves UI, handles uploads, SSE, report downloads)
- File watching: watchdog
- Worker pool: `concurrent.futures.ThreadPoolExecutor`
- PDF rendering: PyMuPDF (`pymupdf`)
- OCR: PaddleOCR (local — form field reading and label text localization)
- Vision/AI: `LabelReader` abstraction; current implementation uses Anthropic Claude API
- Image annotation: Pillow
- Report templating: Jinja2
- Deployment: Docker + Railway (see Hosting)

---

## Repository Structure (planned)

```
proofreader/
├── main.py                  # FastAPI app: routes, SSE, upload handler, startup
├── proofreader/
│   ├── __init__.py
│   ├── worker.py            # ThreadPoolExecutor pool, watchdog watcher, in-flight state
│   ├── pdf.py               # PDF rendering, zone extraction, thumbnail (PyMuPDF)
│   ├── ocr.py               # Item 5 extraction + label text localization (PaddleOCR)
│   ├── vision.py            # LabelReader abstraction + Claude API implementation
│   ├── requirements.py      # TTB requirement sets by product type
│   ├── compare.py           # Comparison logic and verdict generation
│   ├── annotate.py          # Polygon annotation (Pillow)
│   └── report.py            # HTML/JSON report generation (Jinja2)
├── reference/               # TTB regulatory reference materials (read-only)
├── static/                  # CSS, JS for web UI
├── templates/
│   ├── index.html           # Web UI: upload, master list, log window
│   └── report.html          # Per-application recommendation report
├── tests/
│   ├── sample_labels/       # TTB side-by-side comparison images (visual reference)
│   └── sample_applications/ # Test PDFs (scanned Form F 5100.31 submissions)
├── inbox/                   # Watched directory — gitignored
├── outbox/                  # Completed recommendations — gitignored
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Intent and Limitations

ProofReader produces **recommendations**, not decisions. The agent reviewing a report retains full authority to accept, reject, or override any finding. The goal is to substantially reduce the routine verification workload — not to eliminate agent judgment.

When the system cannot make a confident automated determination (e.g., a label image is too degraded to read, or the submitted PDF does not appear to be a valid Form F 5100.31), it returns an `INDETERMINATE` verdict rather than a false PASS or FAIL, and recommends the applicant resubmit with corrected materials. If a component failure prevents analysis entirely (API timeout, unexpected exception), it returns `ERROR` instead — distinguishing "the system tried but was uncertain" from "the infrastructure did not complete the attempt." Both require human review; `ERROR` additionally signals that an operator may need to investigate. Flagging for human review is always preferable to a spurious automated decision.

### PII protection

Form F 5100.31 contains applicant information in its upper zone (name, company, address, permit number, signature). ProofReader handles this data as follows:

- **Only the label affixing area** (the lower portion of page 1, below the "AFFIX COMPLETE SET OF LABELS BELOW" line) is extracted and sent to the AI vision API for label analysis. The upper zone containing applicant PII is never transmitted externally.
- **Before any external API call**, the system verifies that the submitted PDF is in fact a Form F 5100.31 in normal orientation by locating the anchor text that bounds the label zone. If this anchor cannot be confirmed, the label zone is not sent externally and the submission is flagged INDETERMINATE. This prevents an arbitrary document (which might contain PII in the label-zone position) from being submitted to the external API.
- **The full page render** is retained internally for annotating output reports but is not transmitted outside the system.
- In a production deployment, all processing should occur within the agency's existing cloud boundary (e.g., Azure), so label artwork sent to the AI API would remain within a controlled environment subject to applicable data governance requirements.

**Known gap — no authentication or authorization:** The prototype has no access controls. Any user with network access to the server can submit PDFs and retrieve all results, including reports for submissions they did not make. The structural protections above are therefore contingent on network-level isolation keeping untrusted users out entirely. This must be addressed before the system is used on live data. See Production Considerations.

**Known limitation — logs not available in real time for late-joining clients:** Pipeline log output is streamed live via SSE (`GET /logs`) and also captured in the completed `report.html` (in a collapsible section, up to 1000 lines). A user who connects after a job finishes can read the logs in the report. However, there is no way to follow a job's progress in real time if you did not connect before or during processing — there is no replay of the live stream.

---

## Scope

ProofReader evaluates **direct facial compliance with TTB Form F 5100.31** only. This means:

- We check that mandatory label fields (brand name, class/type, ABV, net contents, producer/bottler name and address, government warning) are present on the label and consistent with the application form.
- We check government warning text and formatting per 27 CFR Part 16.
- We check product-type-specific requirements (wine: sulfite declaration, appellation; distilled spirits: age statement if applicable; imports: country of origin).

We do **not** evaluate, flag, or opine on any of the following, even if visible on the label:
- Organic or natural claims
- Allergen labeling
- Caffeine or energy drink content disclosures
- Health or nutrient content claims
- Kosher, halal, or religious certifications
- Sustainability or environmental claims
- Any other specialty regulatory scheme not directly required on Form F 5100.31

If such content appears on a submitted label, it is outside our scope and is silently ignored. Agents reviewing applications that raise these issues should consult the applicable specialist guidance separately.

---

## Form Structure

Form F 5100.31 is a single page (pages 2–5 are instructions and are ignored). The page divides into two natural zones:

**Upper zone — application data (Part I):**
Only one field is extracted from the form:
- Item 5: Type of product (Wine / Distilled Spirits / Malt Beverages) — determines which TTB requirement set is applied to the label check

All other form fields are out of scope. ProofReader does not cross-reference label content against application-declared values (brand name, address, varietal, etc.). The label is checked against TTB requirements directly; the form is used only for routing.

**Lower zone — label artwork:**
Bounded by the text `AFFIX COMPLETE SET OF LABELS BELOW (See General Instructions 4 and 6)`. May contain one or more labels (front, back, neck, tax strip). Labels may be physical originals, photocopies, or printer's proofs. Labels may have been reduced to fit the space (see open questions).

---

## TTB Labeling Requirements by Product Type

Requirements are sourced from 27 CFR Parts 4 (Wine), 5 (Distilled Spirits), 7 (Malt Beverages), and 16 (Health Warning Statement). Full regulatory text is in `reference/`.

### All product types

| Requirement | Notes |
|---|---|
| Brand name | Must match Item 6 |
| Class/type designation | Must match Item 7 if stated |
| Net contents | May be on label or embossed on container (Item 15) |
| Name and address of producer/bottler | Must be present; importer required for imports |
| Government Warning Statement | See exact text and format requirements below |
| Country of origin | Required only if Item 3 = Imported |

**Alcohol content format (all products):** The abbreviation "ABV" is not permitted. Only "Alc." and "Vol." abbreviations are acceptable. Acceptable forms: `45% Alc./Vol.`, `Alcohol 45% by volume`, `45% Alc. By Vol.`

### Distilled Spirits (27 CFR Part 5)

- Alcohol content is **mandatory**
- Brand name, class/type, and alcohol content must appear in the **same field of vision** on the label
- Age statement required if class/type designation implies it (e.g., "Straight Bourbon Whiskey," "12 Year Old Scotch")
- Specialty products without a standard designation require a distinctive/fanciful name plus a statement of composition

### Wine (27 CFR Part 4)

- Brand name, class/type designation, and appellation of origin (when required) must all appear on the **brand label** specifically — not just anywhere in the label set
- Alcohol content is **mandatory if >14% ABV**; optional for 7–14% if "table wine" or "light wine" appears on the brand label
- Appellation of origin is required when the label includes: grape variety as type designation, vintage date, semi-generic designation, or "estate bottled" claim; must match Item 11
- Grape varietal(s) must match Item 10 if stated; if two or more varieties listed, percentages must appear and sum to 100%
- Sulfite declaration mandatory if wine contains ≥10 ppm sulfur dioxide (assume required unless label explicitly states sulfite-free)
- Vintage date, if present, requires appellation of origin on the brand label

### Malt Beverages / Beer (27 CFR Part 7)

- Alcohol content is **mandatory** only if the beverage contains added flavors or non-beverage ingredients containing alcohol (other than hops extract); otherwise optional
- Net contents must be stated in **US customary units** (fluid ounces or pints); metric units may appear alongside but cannot replace US units
- Class designation is mandatory; type is optional. Acceptable classes: ale, lager, beer, malt liquor, stout, porter, malt beverage
- Products under 0.5% ABV must use "malt beverage," "cereal beverage," or "near beer" — may not use "beer," "ale," "lager," etc.

### Government Warning Statement (27 CFR Part 16)

Exact required text:

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects. (2) Consumption of alcoholic beverages impairs your ability to drive a car or operate machinery, and may cause health problems.

Formatting requirements:
- `GOVERNMENT WARNING:` must appear in **all capital letters and bold type**
- The remainder of the statement must NOT appear in bold
- Must appear on a contrasting background
- Must be readily legible under ordinary conditions
- Must be separate and apart from all other information

Minimum font size by container volume (from net contents):

| Container size | Minimum type size | Max characters/inch |
|---|---|---|
| ≤237 mL (≤8 fl oz) | 1 mm | 40 |
| 237 mL–3 L (8–101 fl oz) | 2 mm | 25 |
| >3 L (>101 fl oz) | 3 mm | 12 |

---

## Prototyping Principles

**Simplicity first.** Where a simpler implementation is sufficient for the prototype, prefer it — even if a more robust solution is obviously possible. The architecture should make it easy to add augmentations later (e.g., auto-rotation of upside-down scans, multi-page ordering detection) without requiring structural changes. When a known limitation is accepted for simplicity, document it as a future enhancement rather than leaving it implicit.

**Flag rather than force.** When the system cannot make a confident determination, return `INDETERMINATE` and recommend human review. If a component failure prevents the attempt entirely, return `ERROR` — operators need to distinguish "uncertain result" from "broken pipeline." A missed edge case is far less harmful than a spurious automated decision. This applies at every level: individual fields, whole labels, and entire submissions.

---

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Primary language | Python | All key libraries (PyMuPDF, PaddleOCR, Pillow, Anthropic SDK, FastAPI) are Python-native |
| AI approach | Hybrid: local OCR + cloud vision | Form data (PII) stays local; label images are public artwork and can be transmitted |
| Vision model | Claude API | Substantially better than current 7B local models at reading small/stylized label text; label images carry no PII |
| Form field extraction | PaddleOCR (local) | No PII leaves the system; sufficient for extracting typed/printed text from structured form |
| Output format | Self-contained HTML + JSON | HTML is portable, printable, requires no tooling to open; JSON enables downstream integration |
| Matching strategy | Three-tier: exact / normalized / fail | Handles case like "STONE'S THROW" vs "Stone's Throw" without false rejections |
| Label zone extraction | Anchor text detection | "AFFIX COMPLETE SET OF LABELS BELOW" is a reliable split point on page 1 |
| Bounding box localization | Two-step: vision model identifies fields, PaddleOCR locates them in pixel space | Vision model understands content; OCR returns coordinates |
| Polygon annotation | Use `ImageDraw.polygon()` with PaddleOCR's native rotated quad output | More accurate than axis-aligned rectangles for angled text |

---

## Assumptions

1. Input is always a scanned raster PDF (not a native digital/fillable PDF). No text layer can be relied upon.
2. The application is always a single page (page 1 of Form F 5100.31). Pages 2–5 are instructions and are discarded.
3. Pages are right-side up and in the correct order.
4. The `AFFIX COMPLETE SET OF LABELS BELOW` text is legible in the scan and serves as a reliable zone boundary.
5. Only Item 5 (type of product) is read from the application form. No other form fields are extracted or cross-referenced.
6. Label images in the lower zone are treated as a set — TTB requirements are satisfied by the set collectively, not each label individually (e.g., government warning need not appear on every label, only somewhere in the set).
7. Label images are public artwork with no PII and may be transmitted to the `LabelReader` backend.
8. Country of origin is checked only if the label itself contains import indicators (e.g., "Imported by…", "Product of [country]"). If no such indicators are present, country of origin is not required and not checked.
9. Label text may appear at any orientation — horizontal, angled, vertical, or curved along a design element. The vision model handles this; the OCR-based localization step does not guarantee precise bounding boxes for non-horizontal text. Curved text in particular may only receive an approximate bounding region.

---

## Open Questions

_All open questions resolved. See Resolved Questions below._

### ~~1. Batch processing interface~~
- Directory of PDFs processed in sequence?
- API endpoint accepting a single PDF per request (caller handles batching)?
- Both, with a CLI entry point for batch use?


---

## Resolved Questions

### R1. Batch processing interface and agent-facing UI

**Question:** How should batch submission work, and what interface do agents use?

**Resolution: Web UI as the agent interface; inbox/outbox directories as the prototype's internal work queue.**

Agents interact exclusively through a browser-based web application:
- Upload single or multiple PDFs; processing happens in the background
- Master list shows all submitted applications with form thumbnail, product type, verdict badge, and timestamp
- SSE-based browser notifications fire when the agent's own uploads complete
- Per-application downloadable HTML report and JSON findings
- Streaming log window with filter

Internally, the web server writes uploaded PDFs to a watched `inbox/` directory. A watchdog process detects new arrivals and dispatches them to a configurable `ThreadPoolExecutor` worker pool. Completed output (original PDF, report, thumbnail) lands in `outbox/`. In-flight state is tracked in memory only; a restart simply re-discovers and re-processes anything remaining in inbox. This is the prototype simplification for what a production system would implement as a proper message queue (see Production Considerations).

**Trade-offs:** A native desktop app (Tkinter, PyQt) would avoid the web server dependency but is harder to build well and cannot satisfy the interview's "deployed URL" deliverable. A REST API with no UI would require agents to manage their own polling and file handling. A browser-based UI served by the same FastAPI process is the simplest path that meets both the interview requirement and the real workflow need. The inbox/outbox model is honest about being a simplification — it's a direct local analog of the production queue architecture and makes the production delta obvious.

### R3. Signature presence check

**Question:** Should we verify that a signature is present in the Part II applicant signature field?

**Resolution: Out of scope.**

**Trade-offs:** A missing signature is a form completeness issue, not a label compliance issue. ProofReader's mandate is label content verification; form completeness checks are a separate concern. Adding signature detection would require OCR or image analysis of a different form zone with no compliance benefit.

### R4. Local vs. cloud AI (security posture)

**Question:** Should we use the Claude API directly (Option A) or build an abstracted `LabelReader` interface that can be backed by different implementations (Option B)?

**Resolution: Option B — abstracted `LabelReader` interface. Current (and only) implementation uses the Claude API.**

**Trade-offs:** Option A is marginally simpler for a prototype but makes it harder to swap or compare backends later — which is a realistic need given the network access concern raised by Marcus (TTB's IT administrator). Option B requires defining a small interface contract but otherwise adds no complexity to the calling code. The abstraction point is also where parallel A/B testing of local vs. cloud implementations would be inserted. Local alternatives worth noting: Qwen2-VL 7B or Florence-2 via Ollama.

### R5. Handling unreadable labels

**Question:** If a label is too degraded to extract required fields (severe glare, obstruction, extreme reduction), should we return FAIL, INDETERMINATE, or attempt preprocessing?

**Resolution: Return `INDETERMINATE` with a resubmission recommendation. No preprocessing attempt.**

**Trade-offs:** Returning FAIL would penalize applicants for submission quality issues rather than compliance failures, and could produce incorrect downstream records. Attempting preprocessing (contrast enhancement, deskew) adds complexity and may still fail, while silently altering the evidence being evaluated. `INDETERMINATE` is the honest answer: we cannot determine compliance from this input. The report should include a human-readable explanation and suggest the applicant resubmit with a cleaner scan. This is consistent with the "flag rather than force" prototyping principle. Preprocessing remains a viable future enhancement.

### R6. Malformed or unexpected input PDFs

**Question:** How should we handle PDFs that don't appear to be a valid, correctly oriented Form F 5100.31?

**Resolution: Return `INDETERMINATE` with a resubmission recommendation. No auto-correction.**

**Trade-offs:** Auto-correcting (e.g., rotating an upside-down page) risks silently processing the wrong content and producing a confident but wrong result. The failure modes — upside-down pages, wrong form, out-of-order pages, empty label zone — are all detectable by checking for expected landmark text (`TTB F 5100.31`, `PART I - APPLICATION`, `AFFIX COMPLETE SET OF LABELS BELOW`). If landmarks are absent or ambiguous, we return `INDETERMINATE` and advise the applicant to resubmit as a standard single-page PDF with the form right-side up and labels affixed. Auto-rotation and multi-page ordering recovery remain viable future enhancements consistent with the simplicity-first principle.

### R7. Font size verification

**Question:** Should we verify that the government warning statement meets the minimum font size requirements (1–3 mm depending on container volume)?

**Resolution: Option B — skip font size checking in the prototype; document as a known limitation.**

**Trade-offs:** Option A (flag as unverifiable, report applicable size tier from net contents) provides more useful feedback to agents and would be a natural next step. However, font size cannot be reliably measured from a scanned image regardless of approach — labels may have been reduced per Instructions 4 and 6, and even unreduced scans lack a known scale reference. Building the plumbing to extract, compute, and surface the size tier for zero verified compliance value adds complexity without proportionate benefit at the prototype stage. The government warning font size table remains documented in this README as a reference; a future version could surface the applicable tier as an informational note even without measuring.

### R8. Reduced labels (Instructions 4 and 6 / Item 19)

**Question:** If Item 19 indicates labels have been reduced to fit the form, should we flag for human review of font size compliance?

**Resolution: Subsumed by R1 — no action needed in the prototype.**

**Trade-offs:** The question was specifically about font size compliance under reduction. Since font size checking is skipped entirely (R1), there is nothing to flag. The one remaining concern — that heavy reduction might degrade text legibility and impair the vision model's ability to read the label — is a special case of the general unreadable-label problem (open question #3) and does not warrant a separate treatment.

### R9. Multiple labels in the label zone
**Question:** Should we treat the label area as a single image (Option A) or attempt to segment individual labels and report per-label findings (Option B)?

**Resolution: Option A — treat the entire lower zone as one image.**

**Trade-offs:** Per-label reporting (B) would let us say "government warning found on back label" rather than just "government warning found." However, label segmentation is a non-trivial computer vision problem — labels vary in size, shape, and may overlap or touch each other. For TTB purposes, requirements are satisfied by the label set as a whole, so per-label attribution has limited compliance value. Option A is substantially simpler and deferring segmentation to a future enhancement is consistent with the prototyping principle.

### R10. Angled and curved text localization
**Question:** How do we handle bounding box annotation for text that is not strictly horizontal?

**Resolution: Use PaddleOCR's native rotated quad output (Option C) for angled text; draw approximate dashed region for curved text that cannot be precisely located (Option A). Both applied together.**

**Trade-offs:** PaddleOCR already returns four-point quadrilateral coordinates rather than axis-aligned rectangles, so using `ImageDraw.polygon()` costs nothing extra and is more accurate. Curved text (arcing along a logo border, etc.) genuinely cannot be captured by any single polygon without specialized curve-detection tooling — accepting an approximate enclosing region with a visual "approximate" indicator is the honest answer. The vision model reads all of this correctly regardless; the limitation is purely in the visual annotation layer.

---

## Reference Materials

`reference/` contains TTB regulatory materials used to define validation logic:
- `f510031.pdf` — TTB Form F 5100.31 (the application form)
- `Anatomy of a [Distilled Spirits|Wine|Malt Beverage] Label*.html` — TTB interactive label anatomy tools; each describes every element on a front and back label with mandatory/optional status and format requirements. Primary reference for validation logic and AI prompt design.
- `eCFR __ 27 CFR Part [4|5|7|16]*.html` — Full regulatory text for wine, distilled spirits, malt beverage labeling, and health warning statement
- `[Distilled Spirits|Wine|Malt Beverage] Labeling _ TTB*.html` — TTB summary pages for each product type
- `Allowable Changes Sample Label Generator*.html` — Source page for the comparison images in `tests/sample_labels/`

## Test Data

`tests/sample_labels/` contains 63 label comparison images downloaded from the TTB Allowable Changes Sample Label Generator. **Each image contains two labels side-by-side** (before and after an allowable revision) — they are not individual label samples. They are useful as visual references for what real TTB-compliant labels look like, but should not be used as single-label test inputs without first cropping one half.

Files are named `allowable-revision-comparison-{topic}-{NN}.{ext}`. Slides 7–9, 19–21, and 31–36 could not be matched to descriptions in the source HTML and are named `unclassified-{NN}`.

Actual test PDFs (scanned Form F 5100.31 submissions) should be placed in `tests/sample_applications/`.

---

## Production Considerations

The following are explicitly out of scope for the current prototype but would be required before production deployment.

**Work queue.** The prototype uses watched `inbox/outbox/` directories as a local stand-in for a message queue. In production, this becomes: upload → object storage (S3/Azure Blob) → queue event (SQS/Service Bus) → worker pool (ECS tasks, Lambda, or similar) → results back to object storage. The web UI and worker logic would change minimally; only the transport layer swaps.

```
[Agents / web UI]
      │  upload
      ▼
[Object storage: S3]  →  [Message queue: SQS]
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
        [Worker]           [Worker]           [Worker]
             └──────────────────┼──────────────────┘
                                │ results
                                ▼
                        [Object storage: S3]
                                │
                        [Dead-letter queue]  ← INDETERMINATE / ERROR
                                │
                        [Agent review queue UI]
```

**Decision tracking and agreement rate monitoring.** ProofReader produces recommendations; agents make final decisions. A production system should record whether each agent agreed with the automated recommendation (approved as recommended, rejected as recommended, or overrode). Agreement rate tracking over time enables calibration of confidence thresholds, identification of systematic errors, and measurement of actual workload reduction. This data collection is deliberately excluded from the prototype.

**Audit trail and record retention.** Federal compliance workflows require that decisions and their basis be retained. A deployed system would need to store the input PDF, structured findings, and the agent's decision consistent with TTB's document retention policies.

**Observability, alerting, and incident investigation.** The prototype emits unstructured log lines and exposes a `/health` endpoint used only to suppress Azure's scale-to-zero timer. A production deployment would need: structured logging with a correlation ID attached to each submission (so all log lines for one PDF are retrievable together); metrics on queue depth, per-stage latency, error rates, and Anthropic API quota consumption; alerting on elevated `ERROR` verdict rates, queue backlog growth, and API failures; and a readiness probe that verifies PaddleOCR can actually run, not just that the process is alive. Without these, a silent failure mode (e.g., the Claude API returning errors for every submission) would not surface until an agent noticed the review queue filling up.

**Authentication and authorization.** The prototype has no authentication or authorization controls. Any user with network access to the server can submit PDFs for processing and view all results — including reports for submissions they did not make. The broadcast log stream compounds this: any connected browser receives log lines for all in-flight jobs, including submission filenames and job IDs that belong to other users. Before deployment on live data involving PII, the system needs at minimum: authenticated access to the web UI, per-submission access control so users can only retrieve their own reports, per-session filtering of the log stream, and audit logging of who accessed what. The `LabelReader` abstraction and the outbox file layout are both natural enforcement points.

**PII handling and access control.** Form field data (applicant name, address, registry number) is processed locally and not persisted in the prototype. A production deployment would need appropriate access controls on submission, report viewing, and stored records, consistent with TTB's data governance requirements.

**Firewall and network access.** The prototype transmits label images to the Anthropic API. A production deployment in a restricted network environment would require either firewall allowlisting for the API endpoint or a local AI backend (the `LabelReader` abstraction is the insertion point).

**Human review queue.** Applications returning `INDETERMINATE`, `ERROR`, or agent-overridden recommendations should feed a dedicated review queue with assignment, status tracking, and resolution recording. `ERROR` submissions additionally warrant operator investigation before reprocessing. The prototype surfaces both statuses in the UI but has no queue workflow.

---

## Hosting

The prototype is deployed as a Docker container. **Azure Container Apps** is the target host for the interview demo.

### Why Azure Container Apps

- Scale-to-zero: minimum replica count of 0; container sleeps when idle, wakes automatically on HTTP request
- Thematically appropriate: TTB already operates on Azure
- Cost: free grant (180,000 vCPU-seconds/month) covers light demo use; pay-per-second beyond that; effectively $0 when idle
- Docker-native; standard container config applies

### Key configuration

```yaml
minReplicas: 0
maxReplicas: 1
resources:
  cpu: 1.0
  memory: 2Gi   # PaddleOCR requires ~1–1.5 GB
ingress:
  external: true
  targetPort: 8000
```

One-time setup: Azure resource group + Container Apps environment (3 CLI commands). Deploy via Azure Container Registry or Docker Hub.

### Cold start

PaddleOCR model loading adds 20–40 seconds on first request after idle. Acceptable for a demo; document as known limitation.

### Keeping alive during active work

When jobs are in-flight, a background thread pings the internal `/health` endpoint every 30 seconds. Azure sees continuous HTTP activity and does not spin down. The keepalive thread starts when the first job enters the in-flight set and stops when the set empties. This is a prototype-grade solution; production would use KEDA queue-depth scaling.

### Ephemeral storage

Container Apps uses ephemeral storage by default — the filesystem resets on container restart. For the prototype this is acceptable: a restart re-processes any PDFs remaining in `inbox/`. In production, mount an Azure Files share for persistence.

### Memory constraint

PaddleOCR requires approximately 1–1.5 GB RAM at inference time. The 2 GB allocation above provides headroom. This exceeds Railway's free-tier limit (512 MB), which is why Azure Container Apps is preferred.

### Local development

`docker compose up` with `inbox/` and `outbox/` as bind mounts is the standard local workflow. No Azure account required for development.
