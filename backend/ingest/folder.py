from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import List, Tuple

from core import log, log_folder_scan, log_ingest, timer

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md",
    ".csv", ".xlsx", ".xls",
    ".html", ".htm", ".xml",
    ".pptx", ".ppt",
    ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs",
}


def scan_folder(folder_path: str, recursive: bool = True) -> Tuple[List[Path], List[Path]]:
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Đường dẫn không tồn tại: {folder_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Không phải thư mục: {folder_path}")

    glob = root.rglob("*") if recursive else root.glob("*")
    found, skipped = [], []
    for p in glob:
        if not p.is_file():
            continue
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(p)
        else:
            skipped.append(p)

    log_folder_scan(str(folder_path), len(found), len(skipped))
    return found, skipped


async def ingest_folder(
    folder_path: str,
    rag,
    parse_fn,
    chunk_fn,
    recursive: bool = True,
) -> List[dict]:
    files, skipped = scan_folder(folder_path, recursive=recursive)
    results = []

    for file_path in files:
        import time
        t0 = time.perf_counter()
        try:
            content = file_path.read_bytes()
            suffix = file_path.suffix.lower()
            filename = file_path.name

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                document, output_path = await parse_fn(tmp_path, suffix, filename)
                texts, metadatas = chunk_fn(document, filename, output_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            if texts:
                ids = rag.add_texts(texts, metadatas)
                elapsed = time.perf_counter() - t0
                log_ingest(filename, elapsed, len(ids))
                results.append({
                    "filename": filename,
                    "path": str(file_path),
                    "ids": ids,
                    "count": len(ids),
                    "status": "ok",
                })
            else:
                results.append({
                    "filename": filename,
                    "path": str(file_path),
                    "ids": [],
                    "count": 0,
                    "status": "empty",
                })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.warning(f"INGEST FAILED | file={file_path.name} | elapsed={elapsed:.2f}s | err={exc}")
            results.append({
                "filename": file_path.name,
                "path": str(file_path),
                "ids": [],
                "count": 0,
                "status": f"error: {exc}",
            })

    skipped_names = [p.name for p in skipped[:20]]
    return results, skipped_names