from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf

from ..base import ChildItem, Page, PageItem
from .base import OCRBackend
from .lighton_ocr import LightOnOCRBackend
from .layout_predictor import LayoutPredictor

logger = logging.getLogger(__name__)


def exc_ocr(
    pdf_path: Path,
    output_dir: Path,
    page_numbers: List[int],
    page_sizes: Dict[int, tuple],
    *,
    ocr_backend: str = "lighton_ocr",
    lang: str = "vi",
    dpi: int = 150,
) -> List[Page]:
    """
    Called by PyMuPDF4LLMParser for pages that have no selectable text
    (scanned pages, image-only pages).

    Flow per page:
      1. Render PDF page → PIL image at `dpi`
      2. LayoutPredictor detects regions (title, text, table, picture …)
         with reading-order sort
      3. Each region crop → OCRBackend.run() → text
      4. Assemble into Page / PageItem structure matching the normal parser output
    """
    backend: OCRBackend = _build_backend(ocr_backend)
    layout = LayoutPredictor()

    doc = pymupdf.open(str(pdf_path))
    pages: List[Page] = []

    for pno in page_numbers:
        page = doc[pno]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        from PIL import Image
        import numpy as np
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        bgr = np.array(img)[:, :, ::-1]

        regions = layout.predict(bgr)
        width, height = page_sizes.get(pno, (pix.width, pix.height))
        scale_x = width / pix.width
        scale_y = height / pix.height

        items: List[PageItem] = []
        for region in regions:
            # Skip non-text regions (pictures, etc.)
            if region.label in {"picture", "image"}:
                items.append(PageItem(
                    label="image",
                    bbox=[int(region.x1 * scale_x), int(region.y1 * scale_y),
                          int(region.x2 * scale_x), int(region.y2 * scale_y)],
                ))
                continue

            text = ""
            if region.image is not None:
                try:
                    text = backend.run(region.image).strip()
                except Exception as e:
                    logger.warning(f"OCR failed on region {region.label} p{pno}: {e}")

            items.append(PageItem(
                label=region.label,
                bbox=[int(region.x1 * scale_x), int(region.y1 * scale_y),
                      int(region.x2 * scale_x), int(region.y2 * scale_y)],
                content=text or None,
            ))

        pages.append(Page(page=pno + 1, width=width, height=height, items=items))

    doc.close()
    return pages


def _build_backend(name: str) -> OCRBackend:
    if name == "lighton_ocr":
        return LightOnOCRBackend()
    raise ValueError(f"Unknown OCR backend: {name}")


__all__ = ["exc_ocr", "OCRBackend", "LightOnOCRBackend", "LayoutPredictor"]
