from __future__ import annotations

import io
import json
import time
import tempfile
import traceback
from pathlib import Path
from collections import defaultdict
from typing import Annotated, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from api.deps import get_rag, get_reflection
from api.schemas import (
    ChatRequest, ChatResponse,
    DeleteRequest,
    FileIngestResult, IngestFilesResponse,
    HealthResponse,
    IngestRequest, IngestResponse,
    SearchRequest,
)
from core import log, timer, log_chat, log_ingest
from ingest import ingest_folder, scan_folder, SUPPORTED_EXTENSIONS
from ingest import storage
from rag import RAG, Reflection

router = APIRouter()
RagDep = Annotated[RAG, Depends(get_rag)]
RefDep = Annotated[Reflection, Depends(get_reflection)]


# ─── Chunk helpers ────────────────────────────────────────────────────────────

def _item_to_type(label: str) -> str:
    if label in {"image", "figure", "fig", "picture"}:
        return "image"
    if label == "table":
        return "table"
    return "text"


def _normalize_item_text(item) -> str:
    if item.label == "text" and item.children:
        joined = "\n".join((c.content or "").strip() for c in item.children if (c.content or "").strip())
        if joined.strip():
            return joined.strip()
    text = (item.content or "").strip()
    if text:
        return text
    t = _item_to_type(item.label)
    return "[IMAGE]" if t == "image" else "[TABLE]" if t == "table" else ""


def _merge_bbox(bboxes: List[List[int]]) -> Optional[List[int]]:
    """Hợp nhất nhiều bbox thành một bbox bao phủ tất cả."""
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return None
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def _build_structured_chunks(
    document,
    filename: str,
    output_path: Path,
) -> Tuple[List[str], List[dict]]:
    """
    Parse Document → danh sách chunk text + metadata.
    Metadata bao gồm bbox (merged), page_width, page_height để render về sau.
    """
    # Build page_dims lookup từ document.pages
    page_dims: Dict[int, Tuple[int, int]] = {
        p.page: (p.width, p.height) for p in document.pages
    }

    # Flatten tất cả items có content, sort theo reading order (top→bottom, left→right)
    pages: List[dict] = []
    for page in document.pages:
        sortable = sorted(page.items, key=lambda x: (x.bbox[1], x.bbox[0]))
        ordered = []
        for idx, item in enumerate(sortable, 1):
            content = _normalize_item_text(item)
            if not content:
                continue
            ordered.append({
                "page": page.page,
                "order_in_page": idx,
                "type": _item_to_type(item.label),
                "bbox": item.bbox,
                "content": content,
            })
        if ordered:
            pages.append({"page": page.page, "items": ordered})

    chunk_dir = output_path / "chunkings"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    metadatas: List[dict] = []
    mapping_records: List[dict] = []
    chunk_counter = 0

    for page_block in pages:
        page_no = page_block["page"]
        page_w, page_h = page_dims.get(page_no, (0, 0))

        groups: Dict[str, List[dict]] = defaultdict(list)
        for item in page_block["items"]:
            groups[item["type"]].append(item)

        for section_type in ("text", "image", "table"):
            if not groups[section_type]:
                continue

            lines: List[str] = []
            item_ranges: List[dict] = []
            line_cursor = 1

            for item in groups[section_type]:
                item_lines = item["content"].splitlines() or [item["content"]]
                start_line = line_cursor
                line_cursor += len(item_lines)
                end_line = line_cursor - 1
                lines.extend(item_lines)
                item_ranges.append({
                    "order_in_page": item["order_in_page"],
                    "bbox": item["bbox"],
                    "start_line": start_line,
                    "end_line": end_line,
                })

            chunk_text = "\n".join(lines).strip()
            if not chunk_text:
                continue

            chunk_counter += 1
            chunk_id = f"{Path(filename).stem}_p{page_no}_{section_type}_{chunk_counter:04d}"

            # ─── Merged bbox cho cả chunk ───────────────────────────────
            all_bboxes = [r["bbox"] for r in item_ranges if r.get("bbox")]
            merged_bbox = _merge_bbox(all_bboxes)

            mapping = {
                "chunk_id": chunk_id,
                "file": filename,
                "page": page_no,
                "page_width": page_w,
                "page_height": page_h,
                "section_type": section_type,
                "bbox": merged_bbox,
                "items": item_ranges,
                "text": chunk_text,
            }
            (chunk_dir / f"{chunk_id}.json").write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            texts.append(chunk_text)
            metadatas.append({
                "source": filename,
                "page": page_no,
                "page_width": page_w,
                "page_height": page_h,
                "bbox": merged_bbox,         # ← [x1,y1,x2,y2] trong không gian PDF
                "line_start": 1,
                "line_end": len(lines),
                "chunk_type": section_type,
                "chunk_id": chunk_id,
                "chunk_json": str((chunk_dir / f"{chunk_id}.json").as_posix()),
            })
            mapping_records.append(mapping)

    index_name = f"{Path(filename).stem}_chunk_index.json"
    (chunk_dir / index_name).write_text(
        json.dumps({"file": filename, "chunks": mapping_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return texts, metadatas


async def _parse_upload_document(content: bytes, suffix: str, filename: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        from parser import PyMuPDF4LLMParser
        parser = PyMuPDF4LLMParser(tmp_path)
        document = await parser.parse_document()
        return document, parser.output_path
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health(rag: RagDep):
    return HealthResponse(
        db_type=rag.db_type,
        retrieval_mode=rag.retrieval_mode,
        embedding_model=rag.embeddings._model.name,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RagDep, reflection: RefDep):
    t0 = time.perf_counter()
    history = [m.model_dump() for m in req.messages]
    question = history[-1]["content"] if history else ""

    answer, docs = rag.answer_with_docs(
        question,
        use_rag=req.use_rag,
        use_rerank=req.use_rerank,
        use_hybrid=req.use_hybrid,
        reflection=reflection if req.use_reflection else None,
        chat_history=history if req.use_reflection else None,
        neo4j_cypher=req.neo4j_cypher,
    )

    elapsed = time.perf_counter() - t0
    log_chat(question, elapsed, (len(question) + len(answer)) // 4, len(docs))

    return ChatResponse(
        answer=answer,
        # metadata giờ có bbox + page_width + page_height
        sources=[{"content": d.page_content, "metadata": d.metadata} for d in docs],
    )


@router.get("/page-image")
def page_image(
    filename: str = Query(...),
    page: int = Query(1, ge=1),
    dpi: int = Query(120, ge=72, le=300),
):
    """
    Render một trang PDF thành ảnh PNG.
    Frontend dùng endpoint này để hiển thị trang gốc với bbox highlight.
    """
    path = storage.find(filename)
    if not path:
        raise HTTPException(status_code=404, detail=f"File '{filename}' chưa được upload.")

    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        if page > doc.page_count:
            doc.close()
            raise HTTPException(status_code=404, detail=f"Trang {page} không tồn tại (tổng {doc.page_count} trang).")

        pg = doc[page - 1]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pg.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-Page-Width": str(pg.rect.width),
                "X-Page-Height": str(pg.rect.height),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploaded-files")
def list_uploaded():
    return {"files": storage.list_files()}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, rag: RagDep):
    with timer("ingest_texts", {"n": len(req.texts)}):
        ids = rag.add_texts(req.texts, req.metadatas)
    return IngestResponse(ids=ids, count=len(ids))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(rag: RagDep, file: UploadFile = File(...)):
    t0 = time.perf_counter()
    try:
        content = await file.read()
        suffix = Path(file.filename or "upload").suffix.lower() or ".pdf"
        filename = file.filename or "upload"

        # Lưu file gốc để dùng cho page-image endpoint
        storage.save(filename, content)

        document, output_path = await _parse_upload_document(content, suffix, filename)
        texts, metadatas = _build_structured_chunks(document, filename, output_path)
        if not texts:
            raise HTTPException(status_code=400, detail="Không trích xuất được nội dung")

        ids = rag.add_texts(texts, metadatas)
        log_ingest(filename, time.perf_counter() - t0, len(ids))
        return IngestResponse(ids=ids, count=len(ids))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/files", response_model=IngestFilesResponse)
async def ingest_files(rag: RagDep, files: List[UploadFile] = File(...)):
    results: List[FileIngestResult] = []
    total_ids = 0
    for file in files:
        t0 = time.perf_counter()
        try:
            content = await file.read()
            suffix = Path(file.filename or "upload").suffix.lower() or ".pdf"
            filename = file.filename or "upload"

            storage.save(filename, content)

            document, output_path = await _parse_upload_document(content, suffix, filename)
            texts, metadatas = _build_structured_chunks(document, filename, output_path)
            if not texts:
                results.append(FileIngestResult(filename=filename, ids=[], count=0))
                continue

            ids = rag.add_texts(texts, metadatas)
            log_ingest(filename, time.perf_counter() - t0, len(ids))
            results.append(FileIngestResult(filename=filename, ids=ids, count=len(ids)))
            total_ids += len(ids)
        except Exception as exc:
            log.warning(f"Skip {file.filename}: {exc}")
            results.append(FileIngestResult(filename=file.filename or "?", ids=[], count=0))

    return IngestFilesResponse(files=results, total_ids=total_ids, total_files=len(files))


class FolderIngestRequest(BaseModel):
    folder_path: str
    recursive: bool = True


class FolderIngestResponse(BaseModel):
    total_files: int
    total_chunks: int
    skipped_files: List[str]
    results: List[dict]
    supported_extensions: List[str]


@router.post("/ingest/folder", response_model=FolderIngestResponse)
async def ingest_folder_endpoint(req: FolderIngestRequest, rag: RagDep):
    """Quét và ingest toàn bộ thư mục — nhận bất kỳ đường dẫn nào."""
    t0 = time.perf_counter()
    try:
        results, skipped_names = await ingest_folder(
            folder_path=req.folder_path,
            rag=rag,
            parse_fn=_parse_upload_document,
            chunk_fn=_build_structured_chunks,
            recursive=req.recursive,
        )
        total_chunks = sum(r["count"] for r in results)
        log.info(f"FOLDER | path={req.folder_path} | files={len(results)} | chunks={total_chunks} | {time.perf_counter()-t0:.2f}s")
        return FolderIngestResponse(
            total_files=len(results),
            total_chunks=total_chunks,
            skipped_files=skipped_names,
            results=results,
            supported_extensions=sorted(SUPPORTED_EXTENSIONS),
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingest/folder/scan")
def scan_folder_endpoint(
    folder_path: str = Query(...),
    recursive: bool = Query(True),
):
    try:
        found, skipped = scan_folder(folder_path, recursive=recursive)
        return {"found": [str(p) for p in found], "total_found": len(found), "total_skipped": len(skipped)}
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search")
def search(req: SearchRequest, rag: RagDep):
    with timer("search", {"q": req.query[:40], "k": req.k}):
        if req.with_score:
            results = rag.retrieve_with_score(req.query, k=req.k)
            return [{"content": d.page_content, "metadata": d.metadata, "score": s} for d, s in results]
        docs = rag.retrieve(req.query, k=req.k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@router.delete("/documents")
def delete_documents(req: DeleteRequest, rag: RagDep):
    rag.delete_documents(req.ids)
    return {"deleted": len(req.ids)}