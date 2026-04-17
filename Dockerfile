FROM python:3.11-slim

# Runtime system libraries required by RapidOCR/ONNX Runtime and OpenCV headless.
#   libgl1       — libGL.so.1, needed by opencv-python-headless
#   libglib2.0-0 — GLib, needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Dependency layer ──────────────────────────────────────────────────────────
# Copy manifests only so this layer is reused when only application source
# changes. Installs all runtime dependencies but not the project package itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Application layer ─────────────────────────────────────────────────────────
# Copied last so code-only changes only rebuild this layer and the one above.
COPY . .
RUN uv sync --frozen --no-dev

RUN mkdir -p inbox outbox

ENV PYTHONUNBUFFERED=1 \
    PROOFREADER_INBOX=/app/inbox \
    PROOFREADER_OUTBOX=/app/outbox \
    PROOFREADER_WORKERS=3 \
    PROOFREADER_PORT=8000

EXPOSE 8000

# ANTHROPIC_API_KEY must be injected at runtime via -e or an env_file.
# Do not bake credentials into the image.
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
