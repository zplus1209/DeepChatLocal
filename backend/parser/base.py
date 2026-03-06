from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from  utils import get_logger

# ─────────────────────────── Data models ────────────────────────────────────

class ChildItem(BaseModel):
    label: str
    content: Optional[str] = ""
    bbox: List[int]


class PageItem(BaseModel):
    label: str
    bbox: List[int]
    content: Optional[str] = None
    children: Optional[List[ChildItem]] = None


class Page(BaseModel):
    page: int
    width: int
    height: int
    items: List[PageItem]


class Document(BaseModel):
    source: str
    filename: str
    mimetype: str
    pages: List[Page]


TEXT_CLASSES = {"text", "title", "section-header", "list-item"}
CLASS_MAP = {
    "title": "title",
    "section-header": "section_header",
    "text": "paragraph",
    "list-item": "list_item",
    "picture": "image",
    "table": "table",
}


# ─────────────────────────── Base parser ────────────────────────────────────

class Parser:
    """
    Base parser với cả sync và async API, theo pattern của kreuzberg:

    Sync  : parse_document_sync()  / parse_pdf_sync()  / parse_image_sync()
    Async : parse_document()       / parse_pdf()        / parse_image()

    Async chạy sync version trong thread pool (asyncio.to_thread) để không
    block event loop — giống cách kreuzberg xử lý CPU-bound IO.
    """

    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS  = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS   = {".txt", ".md"}

    logger = get_logger(__name__)

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        output_path: Union[str, Path, None] = None,
        lang: str = "vi",
        provider: str = "pymupdf",
        ocr_backend: str = "lighton_ocr",
        **kwargs: Any,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file_stem   = self.file_path.stem.replace(" ", "_")
        self.lang        = lang
        self.provider    = provider
        self.ocr_backend = ocr_backend

        if output_path:
            path_hash = hashlib.md5(str(self.file_path.resolve()).encode()).hexdigest()[:8]
            self.output_path = (
                Path(output_path) / provider / f"{self.file_stem}_{path_hash}"
            )
        else:
            self.output_path = (
                self.file_path.parent / f"{provider}_output" / self.file_stem
            )

        self.output_path.mkdir(parents=True, exist_ok=True)

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_output_file(json_path: Union[str, Path]) -> Document:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Not found: {json_path}")
        return Document(**json.loads(json_path.read_text(encoding="utf-8")))

    @staticmethod
    def _render_md(md_content: str) -> str:
        from markdown_it import MarkdownIt
        from mdit_py_plugins.front_matter import front_matter_plugin
        from mdit_py_plugins.footnote import footnote_plugin

        md = (
            MarkdownIt('commonmark', {'breaks':True,'html':True})
            .use(front_matter_plugin)
            .use(footnote_plugin)
            .enable('table')
        )

        html_text = md.render(md_content)

        return html_text

    @classmethod
    def convert_office_to_pdf(
        cls, doc_path: Union[str, Path], output_path: Optional[str] = None
    ) -> Path:
        import platform
        import shutil
        import tempfile

        doc_path = Path(doc_path)
        if not doc_path.exists():
            raise FileNotFoundError(f"File not found: {doc_path}")

        out_dir = Path(output_path) if output_path else doc_path.parent / "libreoffice_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            for cmd in ["libreoffice", "soffice"]:
                try:
                    extra = (
                        {"creationflags": subprocess.CREATE_NO_WINDOW}
                        if platform.system() == "Windows"
                        else {}
                    )
                    result = subprocess.run(
                        [cmd, "--headless", "--convert-to", "pdf", "--outdir", tmp, str(doc_path)],
                        capture_output=True, text=True, timeout=60,
                        **extra,
                    )
                    if result.returncode == 0:
                        pdf_files = list(Path(tmp).glob("*.pdf"))
                        if pdf_files:
                            dest = out_dir / f"{doc_path.stem.replace(' ', '_')}.pdf"
                            shutil.copy2(pdf_files[0], dest)
                            return dest
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue

        raise RuntimeError(
            "LibreOffice conversion failed. Ensure LibreOffice is installed."
        )

    # ── Sync API ─────────────────────────────────────────────────────────────

    @abstractmethod
    def parse_document_sync(self, **kwargs: Any) -> Document:
        """Extract document (sync). Runs on the calling thread."""
        raise NotImplementedError

    @abstractmethod
    def parse_pdf_sync(self, **kwargs: Any) -> Document:
        raise NotImplementedError

    @abstractmethod
    def parse_image_sync(
        self,
        page_numbers: List[int],
        page_sizes: Dict[int, tuple],
        **kwargs: Any,
    ) -> List[Page]:
        raise NotImplementedError

    def to_markdown(
        self, save_md: bool = False, skip_labels: Optional[List[str]] = None
    ) -> str:
        raise NotImplementedError

    # ── Async API — wraps sync via thread pool, never blocks event loop ──────

    async def parse_document(self, **kwargs: Any) -> Document:
        """
        Async version of parse_document_sync().
        Uses asyncio.to_thread() so CPU/IO work runs in a thread pool
        without blocking the FastAPI event loop.
        (Same pattern as kreuzberg's extract_file() vs extract_file_sync())
        """
        return await asyncio.to_thread(self.parse_document_sync, **kwargs)

    async def parse_pdf(self, **kwargs: Any) -> Document:
        return await asyncio.to_thread(self.parse_pdf_sync, **kwargs)

    async def parse_image(
        self,
        page_numbers: List[int],
        page_sizes: Dict[int, tuple],
        **kwargs: Any,
    ) -> List[Page]:
        return await asyncio.to_thread(
            self.parse_image_sync, page_numbers, page_sizes, **kwargs
        )
