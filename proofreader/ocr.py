"""Shared RapidOCR singleton for ProofReader.

Both pdf.py and annotate.py import get_engine() and ocr_lock from here so only
one model instance is loaded per process regardless of which module triggers
first initialization.

RapidOCR bundles PP-OCRv4 models inside the wheel (no separate download step)
and runs on ONNX Runtime, which releases memory after inference rather than
holding a persistent allocator pool.

Thread safety: RapidOCR is not thread-safe. All inference callers must hold
ocr_lock for the duration of the call.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Serialises all inference calls across all callers. RapidOCR is not thread-safe.
ocr_lock = threading.Lock()

_engine = None
_init_lock = threading.Lock()


def get_engine():
    """Return (or lazily initialize) the shared RapidOCR engine."""
    global _engine
    if _engine is None:
        with _init_lock:
            if _engine is None:
                from rapidocr_onnxruntime import RapidOCR

                logger.info("Loading RapidOCR engine (first-use initialization)...")
                _engine = RapidOCR()
                logger.info("RapidOCR engine ready.")
    return _engine
