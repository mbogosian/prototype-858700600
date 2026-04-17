"""
Worker pool, inbox watcher, and Server-Side Event (SSE) bridge for
ProofReader.

Responsibilities:
  - Assign job IDs; maintain per-job state
  - Run the per-PDF pipeline in a ThreadPoolExecutor
  - Watch the inbox directory for manually-dropped PDFs (watchdog)
  - Bridge worker-thread log output to async /logs SSE subscribers
  - Broadcast job completion/error events to async /events SSE subscribers

## Job ID assignment

Job IDs are assigned at submission time (12-char lowercase hex, e.g. "a3f9c2d17e8b").
Web uploads: ID is assigned by submit_upload() before the file is written to inbox,
so the watchdog handler can recognise the file as already-registered and skip it.
Manual inbox drops: the watchdog assigns a new ID on file creation.

## Log bridge

A logging.Handler (_SSELogHandler) runs on the worker threads. It formats log
records and puts them onto per-connection asyncio queues via loop.call_soon_threadsafe().
The /logs SSE endpoint manages subscriber queue membership. A logging.Filter
(JobIdFilter) injects the current job ID from a ContextVar into every log record
so all pipeline stages log under the same ID without threading the ID explicitly.

## Event bridge

_emit_event() fans out a JSON payload to all /events SSE subscribers in the same
way. The browser listens on /events for {job_id, verdict} notifications.

## Azure keepalive

A daemon thread (_keepalive_worker) pings GET /health every 20 seconds while any
jobs are queued or processing. Azure Container Apps scales a replica to zero when
it sees no incoming HTTP traffic for ~5 minutes; this prevents a scale-down event
from interrupting an in-flight pipeline run. The thread checks _jobs before each
ping and skips it when the queue is empty, so idle instances still scale down
normally. The target URL is http://127.0.0.1:{PROOFREADER_PORT}/health (default
port 8000; override via PROOFREADER_PORT env var).

Public API:
    start(inbox, outbox, n_workers, loop) — call from FastAPI startup
    stop()                                — call from FastAPI shutdown
    submit_upload(pdf_bytes, filename)    — web upload path; returns job_id
    get_jobs()                            — all known jobs (for master list)
    get_job(job_id)                       — single job state dict or None
    JobIdFilter                           — logging.Filter; install at app startup
    _SSELogHandler                        — logging.Handler; install at app startup
    _log_subscribers                      — set[Queue]; managed by /logs SSE endpoint
    _event_subscribers                    — set[Queue]; managed by /events SSE endpoint
"""

import asyncio
import json
import logging
import os
import secrets
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from proofreader import annotate, compare, pdf, report, vision
from proofreader.models import Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job ID context variable
# ---------------------------------------------------------------------------

_job_id: ContextVar[str] = ContextVar("job_id", default="-")


class JobIdFilter(logging.Filter):
    """Injects the current job_id ContextVar value into every log record.

    Install on each handler (not the root logger) at startup. Logger-level
    filters only run for records originating on that logger; records propagated
    from child loggers bypass parent-logger filters and go straight to handlers.
    Attaching this filter to each handler guarantees job_id is injected for
    every record those handlers receive, regardless of which logger emitted it.

    The ContextVar is set at the top of _process(), so every log line emitted
    by pdf.py, vision.py, compare.py, etc. carries the correct job ID without
    any of those modules knowing about it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# SSE bridges
# ---------------------------------------------------------------------------

_log_loop: asyncio.AbstractEventLoop | None = None

# Per-connection asyncio queues. SSE endpoints add/remove themselves.
_log_subscribers: set[asyncio.Queue[str]] = set()
_event_subscribers: set[asyncio.Queue[str]] = set()

# Maximum log lines retained per job. Older lines are dropped once the cap is
# reached so a runaway job cannot grow _jobs unboundedly.
_LOG_BUFFER_MAX = 1000


class _SSELogHandler(logging.Handler):
    """Forwards formatted log records to all connected /logs SSE subscribers
    and accumulates them in the per-job log buffer for inclusion in the report.

    Runs on worker threads; uses loop.call_soon_threadsafe() to safely enqueue
    onto asyncio queues owned by the event loop thread.
    """

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)

        # Accumulate into the per-job buffer so the completed report can include
        # the full log regardless of whether any SSE client was connected.
        job_id = getattr(record, "job_id", "-")
        if job_id != "-":
            with _jobs_lock:
                entry = _jobs.get(job_id)
                if entry is not None:
                    buf = entry.setdefault("logs", [])
                    if len(buf) < _LOG_BUFFER_MAX:
                        buf.append(line)

        # Fan out to SSE subscribers.
        if _log_loop is None or not _log_subscribers:
            return

        def _push() -> None:
            for q in list(_log_subscribers):
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    pass  # slow subscriber; drop rather than block

        _log_loop.call_soon_threadsafe(_push)


def _emit_event(payload: dict) -> None:
    """Broadcast a job status event to all /events SSE subscribers."""
    if _log_loop is None or not _event_subscribers:
        return
    data = json.dumps(payload)

    def _push() -> None:
        for q in list(_event_subscribers):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    _log_loop.call_soon_threadsafe(_push)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        # Always store job_id in the dict so get_jobs() can include it.
        entry = _jobs.setdefault(job_id, {"job_id": job_id})
        entry.update(fields)


def get_jobs() -> list[dict]:
    with _jobs_lock:
        return list(_jobs.values())


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def delete_job(job_id: str) -> bool:
    """Remove a finished job from memory and delete its outbox directory.

    Returns True if the job was found and removed, False if it does not exist
    or is not in a terminal state (complete/error). Only terminal jobs may be
    deleted; queued/processing jobs are rejected so an in-flight pipeline run
    is not interrupted mid-write.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.get("status") not in ("complete", "error"):
            return False
        del _jobs[job_id]

    if _outbox is not None:
        job_dir = _outbox / job_id
        if job_dir.is_dir():
            shutil.rmtree(job_dir)

    return True


def requeue_job(job_id: str) -> bool:
    """Move a finished job back to the inbox for re-processing.

    Reads original.pdf from the outbox, moves it back to the inbox,
    deletes the processed outputs (report.html, findings.json,
    thumbnail.jpg), resets in-memory job state to 'queued', and
    re-submits to the executor. The same job ID is reused so the UI
    card updates in place rather than being replaced.

    Returns True on success, False if the job does not exist, is not in
    a terminal state, or the original PDF is missing from the outbox.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.get("status") not in ("complete", "error"):
            return False
        original_filename = job.get("original_filename") or f"{job_id}.pdf"

    if _inbox is None or _outbox is None:
        return False

    original_pdf = _outbox / job_id / "original.pdf"
    if not original_pdf.exists():
        return False

    inbox_path = _inbox / f"{job_id}.pdf"
    shutil.move(str(original_pdf), inbox_path)

    for name in ("report.html", "findings.json", "thumbnail.jpg"):
        (_outbox / job_id / name).unlink(missing_ok=True)

    _write_sidecar(inbox_path, job_id, original_filename)

    _set_job(
        job_id,
        status="queued",
        original_filename=original_filename,
        verdict=None,
        submitted_at=time.time(),
        logs=[],
    )

    logger.info("Re-queued job %s (%r)", job_id, original_filename)
    _queue(inbox_path, job_id)
    return True


# ---------------------------------------------------------------------------
# Per-PDF pipeline
# ---------------------------------------------------------------------------


def _process(pdf_path: Path, job_id: str, outbox: Path) -> None:
    """Run the full ProofReader pipeline for one PDF. Executes in a worker thread."""
    _job_id.set(job_id)
    job_dir = outbox / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pipeline start: %s", pdf_path.name)
    _set_job(job_id, status="processing")

    # Stage 1: PDF extraction — always returns a result; reason is non-None on failure.
    page1 = pdf.extract_page1(pdf_path)
    _save_thumbnail(page1, job_dir)

    if page1.reason is not None:
        logger.warning("Terminal state %s: %s", page1.reason.name, page1.reason.description)
        with _jobs_lock:
            job_state = _jobs.get(job_id, {})
            logs = list(job_state.get("logs", []))
            original_filename = job_state.get("original_filename")
            submitted_at = job_state.get("submitted_at")
        report.render_terminal(job_id, page1, job_dir, logs, original_filename, submitted_at)
        shutil.move(str(pdf_path), job_dir / "original.pdf")
        pdf_path.with_suffix(".json").unlink(missing_ok=True)
        _set_job(job_id, status="complete", verdict=Verdict.INDETERMINATE.name)
        _emit_event({"job_id": job_id, "verdict": Verdict.INDETERMINATE.name})
        return

    logger.info("Extraction OK: product_type=%r", page1.product_type)

    # Stage 2: Label analysis via vision model.
    # label_zone is non-None when reason is None — structural guarantee from Page1Result.
    assert page1.label_zone is not None
    findings = vision.read_labels(page1.label_zone, page1.product_type)
    logger.info(
        "Label analysis complete: verdict=%s fields=%d",
        findings.verdict.name,
        len(findings.fields),
    )

    # Stage 3: Comparison and excusal logic.
    findings = compare.assess(findings, page1.product_type)

    # Stage 4: Annotate label zone with per-field verdict polygons.
    annotated_zone = annotate.annotate(page1.label_zone, findings)

    # Stage 5: Render report.html and findings.json; move original to outbox.
    # Log before snapshotting so "Pipeline complete" appears in the report.
    verdict = findings.verdict.name
    logger.info("Pipeline complete: verdict=%s", verdict)
    with _jobs_lock:
        job_state = _jobs.get(job_id, {})
        logs = list(job_state.get("logs", []))
        original_filename = job_state.get("original_filename")
        submitted_at = job_state.get("submitted_at")
    report.render(
        job_id, page1, annotated_zone, findings, job_dir, logs, original_filename, submitted_at
    )
    shutil.move(str(pdf_path), job_dir / "original.pdf")
    pdf_path.with_suffix(".json").unlink(missing_ok=True)
    _set_job(job_id, status="complete", verdict=verdict)
    _emit_event({"job_id": job_id, "verdict": verdict})


def _save_thumbnail(page1, job_dir: Path) -> None:
    """Save a small JPEG of page 1 for the master list. Silently skips if unavailable."""
    if page1.page1_image is None:
        return
    thumb = page1.page1_image.copy()
    thumb.thumbnail((600, 800))
    thumb.save(job_dir / "thumbnail.jpg", format="JPEG", quality=80)


# ---------------------------------------------------------------------------
# Azure keepalive
# ---------------------------------------------------------------------------

_KEEPALIVE_INTERVAL = 20  # seconds between pings
_KEEPALIVE_URL = f"http://127.0.0.1:{os.environ.get('PROOFREADER_PORT', '8000')}/health"


def _keepalive_worker() -> None:
    """Ping /health every 20 s while any jobs are queued or processing.

    Prevents Azure Container Apps from scaling the replica to zero mid-pipeline.
    Skips the ping when the job queue is empty so idle instances still scale down.
    Runs as a daemon thread for the lifetime of the process; no explicit stop needed.

    The first ping fires immediately (before the first sleep) so that a job
    submitted shortly after a cold start is protected before the first interval
    elapses.
    """
    first = True
    while True:
        if first:
            first = False
        else:
            time.sleep(_KEEPALIVE_INTERVAL)
        with _jobs_lock:
            active = any(j.get("status") in ("queued", "processing") for j in _jobs.values())
        if not active:
            continue
        try:
            urllib.request.urlopen(_KEEPALIVE_URL, timeout=5)
        except Exception:
            pass  # server may be briefly busy; safe to miss one ping


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_executor: ThreadPoolExecutor | None = None
_inbox: Path | None = None
_outbox: Path | None = None


def _queue(pdf_path: Path, job_id: str) -> None:
    """Submit one job to the executor; attach error-recovery done-callback."""
    assert _executor is not None and _outbox is not None

    def _on_done(fut) -> None:
        exc = fut.exception()
        if exc:
            logger.error("Unhandled pipeline error for job %s: %s", job_id, exc, exc_info=exc)
            _set_job(job_id, status="error", verdict=Verdict.ERROR.name)
            _emit_event({"job_id": job_id, "verdict": Verdict.ERROR.name})

    _executor.submit(_process, pdf_path, job_id, _outbox).add_done_callback(_on_done)


def _write_sidecar(pdf_path: Path, job_id: str, original_filename: str) -> None:
    """Write a JSON sidecar next to a PDF in the inbox.

    The sidecar survives server restarts and lets the re-queue loop recover
    the original filename and job ID for files that were in-flight when the
    server stopped. It is deleted once the job completes (findings.json in
    the outbox is the durable record thereafter).
    """
    sidecar = pdf_path.with_suffix(".json")
    sidecar.write_text(json.dumps({"job_id": job_id, "original_filename": original_filename}))


def submit_upload(pdf_bytes: bytes, original_filename: str) -> str:
    """Accept a web-uploaded PDF, write it to inbox, and queue processing.

    The job is registered before the file is written so the watchdog handler
    sees the file as already-known and does not issue a duplicate submission.

    Returns the job_id immediately (before processing starts).
    """
    assert _inbox is not None

    # Guard against the (astronomically unlikely) event of a job ID collision.
    while True:
        job_id = secrets.token_hex(6)
        dest = _inbox / f"{job_id}.pdf"
        with _jobs_lock:
            if job_id not in _jobs and not dest.exists():
                break

    _set_job(
        job_id,
        status="queued",
        original_filename=original_filename,
        verdict=None,
        submitted_at=time.time(),
    )
    dest.write_bytes(pdf_bytes)
    # Sidecar is written after the PDF, so a crash between the two leaves an
    # orphaned PDF with no sidecar. On restart the re-queue loop handles this
    # gracefully by falling back to path.stem/path.name. Acceptable gap.
    _write_sidecar(dest, job_id, original_filename)
    logger.info("Queued upload %r as job %s", original_filename, job_id)
    _queue(dest, job_id)
    return job_id


# ---------------------------------------------------------------------------
# Watchdog — inbox watcher for manually-dropped PDFs
# ---------------------------------------------------------------------------


class _InboxHandler(FileSystemEventHandler):
    """Picks up PDFs dropped directly into inbox (i.e., not via web upload)."""

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix.lower() != ".pdf":
            return

        # submit_upload() names its files {job_id}.pdf and pre-registers the job,
        # so if the stem is already a known job ID we can skip this file.
        with _jobs_lock:
            if path.stem in _jobs:
                return

        # This is a manually-dropped file with a user-chosen name, not a web
        # upload. Assign a fresh job ID; we cannot use the stem since it is
        # arbitrary and not guaranteed to be unique.
        job_id = secrets.token_hex(6)
        _set_job(
            job_id,
            status="queued",
            original_filename=path.name,
            verdict=None,
            submitted_at=time.time(),
        )
        _write_sidecar(path, job_id, path.name)
        logger.info("Watchdog picked up %s as job %s", path.name, job_id)
        _queue(path, job_id)


_observer = None  # watchdog.observers.Observer, or None before start

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start(
    inbox: Path,
    outbox: Path,
    n_workers: int,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Start the worker pool and inbox watcher. Call from FastAPI startup event."""
    global _executor, _inbox, _outbox, _observer, _log_loop

    _log_loop = loop
    _inbox = inbox
    _outbox = outbox
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)

    _executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="proofreader")

    # Re-queue any PDFs left in inbox from a previous run (e.g. after a crash).
    # This must run before the watchdog observer starts so that pre-existing
    # files are registered in _jobs first. If the observer started first it
    # could fire on_created for those files and assign them new hex job IDs,
    # while the re-queue loop would then see path.stem absent from _jobs and
    # queue the same file a second time — producing two competing jobs for one
    # file and a race on shutil.move.
    #
    # Read the sidecar (if present) to recover the original job ID and filename;
    # fall back to using the stem as the job ID for files without one.
    for path in sorted(inbox.glob("*.pdf")):
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                job_id = meta["job_id"]
                original_filename = meta.get("original_filename", path.name)
            except Exception:
                job_id = path.stem
                original_filename = path.name
        else:
            # No sidecar — either a pre-sidecar upload or a crash between the
            # two writes. For web uploads the stem is the job ID so processing
            # is correct, but original_filename is lost and will display as the
            # job-ID-based filename. Acceptable gap.
            job_id = path.stem
            original_filename = path.name

        with _jobs_lock:
            if job_id in _jobs:
                continue
        _set_job(
            job_id,
            status="queued",
            original_filename=original_filename,
            verdict=None,
            submitted_at=path.stat().st_mtime,
        )
        logger.info("Re-queuing leftover %s as job %s", path.name, job_id)
        _queue(path, job_id)

    # Re-inflate completed job state from the outbox. Any subdirectory that
    # contains both report.html and findings.json represents a finished job.
    # We skip jobs already registered by the inbox re-queue loop above (i.e.
    # a job that somehow has both an inbox PDF and an outbox directory, which
    # should not happen in normal operation but is safe to guard against).
    for job_dir in sorted(outbox.iterdir()):
        if not job_dir.is_dir():
            continue
        findings_path = job_dir / "findings.json"
        report_path = job_dir / "report.html"
        if not (findings_path.exists() and report_path.exists()):
            continue
        job_id = job_dir.name
        with _jobs_lock:
            if job_id in _jobs:
                continue
        try:
            meta = json.loads(findings_path.read_text())
        except Exception:
            logger.warning("Outbox re-inflation: could not read %s", findings_path)
            continue
        verdict = meta.get("verdict", Verdict.INDETERMINATE.name)
        original_filename = meta.get("original_filename")
        submitted_at = meta.get("submitted_at")
        _set_job(
            job_id,
            status="complete",
            verdict=verdict,
            original_filename=original_filename,
            submitted_at=submitted_at,
        )
        logger.debug("Re-inflated completed job %s (verdict=%s)", job_id, verdict)

    # Start the watchdog only after all pre-existing files are registered.
    # Files dropped into inbox between the glob above and observer.start() here
    # will be missed until the next server restart — acceptable startup gap.
    _observer = Observer()
    _observer.schedule(_InboxHandler(), str(inbox), recursive=False)
    _observer.start()

    threading.Thread(target=_keepalive_worker, daemon=True, name="proofreader-keepalive").start()

    logger.info("Worker pool started: n_workers=%d inbox=%s outbox=%s", n_workers, inbox, outbox)


def stop() -> None:
    """Graceful shutdown: drain the worker pool and stop the inbox watcher."""
    global _observer, _executor

    if _observer is not None:
        _observer.stop()
        _observer.join()
        _observer = None

    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None

    logger.info("Worker pool stopped")
