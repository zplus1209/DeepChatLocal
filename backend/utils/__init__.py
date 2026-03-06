from __future__ import annotations

import torch
import time
import logging
from typing import Optional
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("deepchat")

@contextmanager
def timer(label: str, extra: Optional[dict] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        parts = f"{label} ({elapsed:.2f}s)"
        if extra:
            parts += " | " + " | ".join(f"{k}={v}" for k, v in extra.items())
        log.info(parts)

def log_chat(question: str, elapsed: float, tokens_est: int, sources_n: int):
    log.info(
        f"CHAT | q_len={len(question)} | elapsed={elapsed:.2f}s | ~tokens={tokens_est} | sources={sources_n}"
    )

def log_ingest(filename: str, elapsed: float, chunks: int):
    log.info(f"INGEST | file={filename} | elapsed={elapsed:.2f}s | chunks={chunks}")

def log_folder_scan(path: str, found: int, skipped: int):
    log.info(f"FOLDER | path={path} | found={found} | skipped={skipped}")

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
