"""
ProofReader FastAPI application.

Configuration (environment variables):
  ANTHROPIC_API_KEY     — required; passed through to ClaudeReader
  PROOFREADER_INBOX     — directory watched for incoming PDFs (default: ./inbox)
  PROOFREADER_OUTBOX    — directory for completed results   (default: ./outbox)
  PROOFREADER_WORKERS   — ThreadPoolExecutor size           (default: 3)
  PROOFREADER_LOG_LEVEL        — root log level: DEBUG/INFO/WARNING/ERROR (default: INFO)
  PROOFREADER_REPORT_LOG_LEVEL — minimum level buffered into per-job reports and
                                  streamed via /logs SSE (default: INFO)

Routes:
  POST /upload                          — accept one or more PDFs; return job IDs
  GET  /results                         — master list of all jobs (JSON)
  GET  /results/{job_id}/report.html    — per-job HTML report
  GET  /results/{job_id}/findings.json  — per-job structured findings
  GET  /results/{job_id}/thumbnail.jpg  — page-1 thumbnail for master list
  GET  /results/{job_id}/annotated.jpg  — annotated label zone image
  GET  /events                          — SSE stream: job completion/error events
  GET  /logs                            — SSE stream: pipeline log output
  GET  /health                          — liveness probe
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from proofreader import worker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INBOX = Path(os.environ.get("PROOFREADER_INBOX", "./inbox"))
OUTBOX = Path(os.environ.get("PROOFREADER_OUTBOX", "./outbox"))
N_WORKERS = int(os.environ.get("PROOFREADER_WORKERS", "3"))
LOG_LEVEL = os.environ.get("PROOFREADER_LOG_LEVEL", "INFO").upper()
REPORT_LOG_LEVEL = os.environ.get("PROOFREADER_REPORT_LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(job_id)s] %(name)-30s %(levelname)-8s %(message)s"


def _configure_logging(loop: asyncio.AbstractEventLoop) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # JobIdFilter must be on the handlers, not the root logger.
    # Logger-level filters only run for records that originate on that logger;
    # records propagated from child loggers (e.g. watchdog, uvicorn) bypass the
    # parent logger's filters and go directly to its handlers. Attaching the
    # filter to each handler ensures job_id is injected regardless of origin.
    job_id_filter = worker.JobIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    console.addFilter(job_id_filter)
    root.addHandler(console)

    sse = worker._SSELogHandler()
    sse.setFormatter(logging.Formatter(_LOG_FORMAT))
    sse.setLevel(getattr(logging, REPORT_LOG_LEVEL, logging.INFO))
    sse.addFilter(job_id_filter)
    root.addHandler(sse)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    _configure_logging(loop)
    worker.start(INBOX, OUTBOX, N_WORKERS, loop)
    logger.info("ProofReader ready")
    yield
    worker.stop()


app = FastAPI(title="ProofReader", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload(files: list[UploadFile]) -> JSONResponse:
    """Accept one or more PDF files; return a list of {filename, job_id, status}."""
    results = []
    for f in files:
        filename = f.filename or "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            results.append({"filename": filename, "error": "not a PDF"})
            continue
        content = await f.read()
        job_id = worker.submit_upload(content, filename)
        results.append({"filename": filename, "job_id": job_id, "status": "queued"})
    return JSONResponse(results)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@app.get("/results")
async def results() -> JSONResponse:
    """Return the master list of all known jobs."""
    return JSONResponse(worker.get_jobs())


def _result_file(job_id: str, filename: str) -> Path:
    """Resolve a result file path; raise 404 if the job or file does not exist."""
    job = worker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = OUTBOX / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not available yet")
    return path


@app.get("/results/{job_id}/report.html")
async def report_html(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "report.html"), media_type="text/html")


@app.get("/results/{job_id}/findings.json")
async def findings_json(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "findings.json"), media_type="application/json")


@app.get("/results/{job_id}/thumbnail.jpg")
async def thumbnail(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "thumbnail.jpg"), media_type="image/jpeg")


@app.get("/results/{job_id}/annotated.jpg")
async def annotated(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "annotated.jpg"), media_type="image/jpeg")


@app.get("/results/{job_id}/original.pdf")
async def original_pdf(job_id: str) -> FileResponse:
    return FileResponse(_result_file(job_id, "original.pdf"), media_type="application/pdf")


@app.delete("/results/{job_id}")
async def delete_result(job_id: str) -> JSONResponse:
    """Delete a finished job: removes outbox files and in-memory state.

    Returns 404 if the job does not exist, 409 if the job is still queued or
    processing (deletion of in-flight jobs is not allowed).
    """
    job = worker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") not in ("complete", "error"):
        raise HTTPException(status_code=409, detail="job is not yet complete")
    worker.delete_job(job_id)
    return JSONResponse({"status": "deleted"})


# ---------------------------------------------------------------------------
# SSE: job events
# ---------------------------------------------------------------------------


@app.get("/events")
async def events(request: Request) -> EventSourceResponse:
    """SSE stream of job completion/error events.

    Each event data is a JSON object: {"job_id": "...", "verdict": "..."}.
    The browser can use this to fire a notification when the agent's own
    upload finishes.
    """
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    worker._event_subscribers.add(q)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"data": payload}
                except TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            worker._event_subscribers.discard(q)

    return EventSourceResponse(_gen())


# ---------------------------------------------------------------------------
# SSE: log stream
# ---------------------------------------------------------------------------


@app.get("/logs")
async def logs(request: Request) -> EventSourceResponse:
    """SSE stream of pipeline log output.

    Each event data is a formatted log line. Lines include a [job_id] field
    so the browser can filter by job. Note: all connected sessions receive
    all log lines (no per-session filtering). See README — Known Limitations.
    """
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
    worker._log_subscribers.add(q)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"data": line}
                except TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            worker._log_subscribers.discard(q)

    return EventSourceResponse(_gen())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe. Returns 200 when the server is running.

    Note: this confirms the process is alive but does not verify that
    PaddleOCR or the Anthropic API are reachable. See README — Known
    Limitations (observability gap).
    """
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Static UI — mounted last so all API routes above take precedence
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
