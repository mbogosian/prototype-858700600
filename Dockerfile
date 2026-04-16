FROM python:3.11-slim

# Runtime system libraries required by PaddlePaddle (CPU) and OpenCV headless.
#   libgl1      — libGL.so.1, needed by opencv-python-headless
#   libglib2.0-0 — GLib, needed by OpenCV
#   libgomp1    — OpenMP runtime, needed by PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# ── Dependency layer ──────────────────────────────────────────────────────────
# Copy manifests only so this layer is reused when only application source
# changes. Installs all runtime dependencies but not the project package itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── PaddleOCR model layer ─────────────────────────────────────────────────────
# Pre-download PaddleOCR models (~200 MB) and bake them into the image so they
# are available immediately on container start.
#
# We use only Python stdlib (urllib.request + tarfile) — no PaddlePaddle native
# code is executed. This replicates paddleocr.ppocr.utils.network.maybe_download
# exactly: download the tar, extract only the .pdiparams / .pdiparams.info /
# .pdmodel members, rename each to inference.<ext>, then discard the tar.
#
# URLs and target paths are pinned to PaddleOCR 2.10.0 / PP-OCRv4, lang='en':
#   det_lang='en' → en_PP-OCRv3_det_infer  (PP-OCRv3 det is used even in v4)
#   lang='en'     → en_PP-OCRv4_rec_infer
#   cls           → ch_ppocr_mobile_v2.0_cls_infer (downloaded unconditionally
#                   regardless of use_angle_cls; omitting it causes a download
#                   on first container start)
RUN .venv/bin/python - <<'PYEOF'
import os, urllib.request, tarfile

MODELS = [
    (
        "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
        "/root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer",
    ),
    (
        "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
        "/root/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer",
    ),
    # Downloaded unconditionally by PaddleOCR regardless of use_angle_cls setting.
    (
        "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
        "/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
    ),
]
EXTS = (".pdiparams", ".pdiparams.info", ".pdmodel")

for url, dest in MODELS:
    os.makedirs(dest, exist_ok=True)
    tarpath = dest + ".tar"
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tarpath)
    with tarfile.open(tarpath) as tf:
        for member in tf.getmembers():
            for ext in EXTS:
                if member.name.endswith(ext):
                    fobj = tf.extractfile(member)
                    with open(os.path.join(dest, "inference" + ext), "wb") as out:
                        out.write(fobj.read())
                    break
    os.unlink(tarpath)
    print(f"  -> {dest}", flush=True)

print("PaddleOCR models ready.", flush=True)
PYEOF

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
