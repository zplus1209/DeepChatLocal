from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers.check_ocr import should_ocr_page

from .base import (
    CLASS_MAP, TEXT_CLASSES,
    ChildItem, Document, Page, PageItem, Parser,
)


class PyMuPDF4LLMParser(Parser):
    """
    Parser dùng pymupdf4llm với đầy đủ sync và async API.

    Sync  : parse_document_sync() / parse_pdf_sync() / parse_image_sync()
    Async : parse_document()      / parse_pdf()       / parse_image()
            ↳ tự động chạy sync version trong asyncio.to_thread (kế thừa từ Parser)
    """

    # ── Sync implementations ─────────────────────────────────────────────────

    def parse_document_sync(self, **kwargs: Any) -> Document:
        ext = self.file_path.suffix.lower()
        if ext in self.OFFICE_FORMATS:
            self.file_path = self.convert_office_to_pdf(self.file_path, self.output_path)
        return self.parse_pdf_sync(**kwargs)

    def parse_pdf_sync(self, **kwargs: Any) -> Document:
        json_file = self.output_path / f"{self.file_stem}.json"
        if json_file.exists():
            return self._read_output_file(json_file)

        doc = pymupdf.open(str(self.file_path))
        page_count   = doc.page_count
        page_sizes   = {i: (int(doc[i].rect.width), int(doc[i].rect.height)) for i in range(page_count)}
        ocr_pages    = [i for i in range(page_count) if should_ocr_page(doc[i])["should_ocr"]]
        normal_pages = [i for i in range(page_count) if i not in set(ocr_pages)]
        doc.close()

        all_pages: List[Page] = []

        if normal_pages:
            all_pages.extend(self._run_pymupdf4llm(page_sizes, only_pages=normal_pages, **kwargs))

        if ocr_pages:
            all_pages.extend(self.parse_image_sync(ocr_pages, page_sizes, **kwargs))

        all_pages.sort(key=lambda p: p.page)

        document = Document(
            source=str(self.file_path),
            filename=self.file_path.name,
            mimetype="application/pdf",
            pages=all_pages,
        )
        json_file.write_text(
            json.dumps(document.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return document

    def parse_image_sync(
        self,
        page_numbers: List[int],
        page_sizes: Dict[int, tuple],
        **kwargs: Any,
    ) -> List[Page]:
        """
        Xử lý các trang không có selectable text (scanned / image-only).
        Gọi exc_ocr: LayoutPredictor detect vùng → LightOnOCR → Page objects.
        """
        from .ocr import exc_ocr
        return exc_ocr(
            self.file_path,
            self.output_path,
            page_numbers,
            page_sizes,
            ocr_backend=self.ocr_backend,
            lang=self.lang,
        )

    # ── Markdown export ──────────────────────────────────────────────────────

    def to_markdown(
        self, save_md: bool = False, skip_labels: Optional[List[str]] = None
    ) -> str:
        skip: Set[str] = (
            set(skip_labels) if skip_labels else {"page-header", "page-footer", "image"}
        )
        document = self._read_output_file(self.output_path / f"{self.file_stem}.json")

        page_blocks: List[str] = []
        for page in document.pages:
            lines: List[str] = []
            for item in page.items:
                if item.label in skip:
                    continue
                if item.label == "text" and item.children:
                    lines.extend(
                        (c.content or "").strip()
                        for c in item.children
                        if (c.content or "").strip()
                    )
                else:
                    text = (item.content or "").strip()
                    if text:
                        lines.append(text)
            if lines:
                page_blocks.append("\n\n".join(lines))

        parts: List[str] = []
        for i, block in enumerate(page_blocks, 1):
            if i > 1:
                parts.append(f"\n\n<!-- Page {i} -->\n\n")
            parts.append(block)

        markdown = "".join(parts)
        if save_md:
            (self.output_path / "content.md").write_text(markdown, encoding="utf-8")
        return markdown

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _run_pymupdf4llm(
        self,
        page_sizes: Dict[int, tuple],
        only_pages: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[Page]:
        image_dir = self.output_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        md_kwargs: Dict[str, Any] = dict(
            write_images=True,
            image_path=str(image_dir),
            show_progress=False,
            page_chunks=True,
        )
        if only_pages is not None:
            md_kwargs["pages"] = only_pages
        # ocr_language is supported in newer pymupdf4llm; skip if not
        try:
            md_kwargs.update(kwargs)
            content_list = pymupdf4llm.to_markdown(str(self.file_path), **md_kwargs)
        except TypeError:
            md_kwargs.pop("ocr_language", None)
            content_list = pymupdf4llm.to_markdown(str(self.file_path), **md_kwargs)

        print(content_list)

        pages: List[Page] = []
        for i, item in enumerate(content_list, 1):
            pno        = item.get("metadata", {}).get("page_number", i) - 1
            page_text  = item.get("text", "")
            boxes      = sorted(item.get("page_boxes", []), key=lambda x: x["index"])
            width, height = page_sizes.get(pno, (0, 0))

            page_items: List[PageItem] = []
            for gtype, data in self._group_boxes(boxes):
                if gtype == "text_group":
                    page_items.append(self._build_text_group(data, page_text))
                else:
                    page_items.append(self._build_single(data, page_text))

            pages.append(Page(page=pno + 1, width=width, height=height, items=page_items))

        return pages

    def _group_boxes(self, boxes: List[dict]) -> List[tuple]:
        groups: List[tuple] = []
        current: List[dict] = []
        for box in boxes:
            if box["class"] in TEXT_CLASSES:
                current.append(box)
            else:
                if current:
                    groups.append(("text_group", current))
                    current = []
                groups.append(("single", box))
        if current:
            groups.append(("text_group", current))
        return groups

    def _build_text_group(self, boxes: List[dict], page_text: str) -> PageItem:
        children: List[ChildItem] = []
        contents: List[str] = []
        for b in boxes:
            s, e = b["pos"]
            text = page_text[s:e].strip()
            contents.append(text)
            children.append(ChildItem(
                label=CLASS_MAP.get(b["class"], "paragraph"),
                bbox=[int(x) for x in b["bbox"]],
                content=text,
            ))
        all_bb = [b["bbox"] for b in boxes]
        bbox = [
            int(min(bb[0] for bb in all_bb)), int(min(bb[1] for bb in all_bb)),
            int(max(bb[2] for bb in all_bb)), int(max(bb[3] for bb in all_bb)),
        ]
        return PageItem(
            label="text", bbox=bbox,
            content="\n".join(contents), children=children,
        )

    def _build_single(self, box: dict, page_text: str) -> PageItem:
        s, e = box["pos"]
        text = page_text[s:e].strip()
        return PageItem(
            label=CLASS_MAP.get(box["class"], box["class"]),
            bbox=[int(x) for x in box["bbox"]],
            content=text or None,
        )
