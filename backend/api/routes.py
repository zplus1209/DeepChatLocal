from __future__ import annotations

import json
import tempfile
from hashlib import md5
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import get_rag, get_reflection
from api.schemas import (
    ChatRequest, ChatResponse,
    DeleteRequest,
    FileIngestResult,
    HealthResponse,
    IngestFilesResponse,
    IngestRequest, IngestResponse,
    SearchRequest,
)
from rag import RAG, Reflection

router = APIRouter()

RagDep = Annotated[RAG, Depends(get_rag)]
RefDep = Annotated[Reflection, Depends(get_reflection)]


def _item_to_type(label: str) -> str:
    if label in {"image", "figure", "fig"}:
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
    if t == "image":
        return "[IMAGE]"
    if t == "table":
        return "[TABLE]"
    return ""


def _build_structured_chunks(document, filename: str, output_path: Path) -> Tuple[List[str], List[dict]]:
    """
    Build chunk list with source mapping into chunkings/*.json.
    Each record includes file/page/line/type/order/chunk_id.
    """
    pages = []
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
        groups: Dict[str, List[dict]] = defaultdict(list)
        for item in page_block["items"]:
            groups[item["type"]].append(item)

        for section_type in ("text", "image", "table"):
            if not groups[section_type]:
                continue

            # keep top-down order within each section
            lines = []
            item_ranges = []
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
            chunk_json_name = f"{chunk_id}.json"

            mapping = {
                "chunk_id": chunk_id,
                "file": filename,
                "page": page_no,
                "section_type": section_type,
                "line_start": 1,
                "line_end": len(lines),
                "items": item_ranges,
                "text": chunk_text,
            }
            (chunk_dir / chunk_json_name).write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            texts.append(chunk_text)
            metadatas.append({
                "source": filename,
                "page": page_no,
                "line_start": 1,
                "line_end": len(lines),
                "chunk_type": section_type,
                "chunk_id": chunk_id,
                "chunk_json": str((chunk_dir / chunk_json_name).as_posix()),
            })
            mapping_records.append(mapping)

    # file-level mapping index
    index_name = f"{Path(filename).stem}_chunk_index.json"
    (chunk_dir / index_name).write_text(
        json.dumps({"file": filename, "chunks": mapping_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return texts, metadatas


async def _parse_upload_document(file: UploadFile):
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    content = await file.read()

    # Text-first fast path for txt/markdown to avoid PDF parser errors.
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8", errors="ignore").strip()
        if not text:
            raise HTTPException(status_code=400, detail=f"File '{filename}' không có nội dung văn bản")

        key = md5(filename.encode("utf-8")).hexdigest()[:8]
        out_dir = Path(tempfile.gettempdir()) / "pymupdf_output" / f"{Path(filename).stem}_{key}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Fake a minimal one-page document compatible with _build_structured_chunks.
        from parser.base import Document, Page, PageItem

        doc = Document(
            source=filename,
            filename=filename,
            mimetype="text/plain",
            pages=[
                Page(
                    page=1,
                    width=0,
                    height=0,
                    items=[PageItem(label="text", bbox=[0, 0, 0, 0], content=text)],
                )
            ],
        )
        return doc, out_dir

    with tempfile.NamedTemporaryFile(suffix=suffix or ".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from parser import PyMuPDF4LLMParser

        parser = PyMuPDF4LLMParser(tmp_path)
        document = await parser.parse_document()
        return document, parser.output_path
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không thể parse file '{filename}'. Lỗi: {e}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)



@router.get("/health", response_model=HealthResponse)
def health(rag: RagDep):
    return HealthResponse(
        db_type=rag.db_type,
        retrieval_mode=rag.retrieval_mode,
        embedding_model=rag.embeddings._model.name,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RagDep, reflection: RefDep):
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

    sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    return ChatResponse(answer=answer, sources=sources)


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, rag: RagDep):
    ids = rag.add_texts(req.texts, req.metadatas)
    return IngestResponse(ids=ids, count=len(ids))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    rag: RagDep,
    file: UploadFile = File(...),
):
    try:
        document, output_path = await _parse_upload_document(file)
        texts, metadatas = _build_structured_chunks(document, file.filename or "unknown", output_path)
        if not texts:
            raise HTTPException(status_code=400, detail="Không trích xuất được nội dung từ file")
        ids = rag.add_texts(texts, metadatas)
        return IngestResponse(ids=ids, count=len(ids))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/files", response_model=IngestFilesResponse)
async def ingest_files(
    rag: RagDep,
    files: List[UploadFile] = File(...),
):
    results: List[FileIngestResult] = []
    total_ids = 0

    for file in files:
        try:
            document, output_path = await _parse_upload_document(file)
            texts, metadatas = _build_structured_chunks(document, file.filename or "unknown", output_path)
            if not texts:
                results.append(FileIngestResult(filename=file.filename or "unknown", ids=[], count=0))
                continue

            ids = rag.add_texts(texts, metadatas)
            results.append(FileIngestResult(filename=file.filename or "unknown", ids=ids, count=len(ids)))
            total_ids += len(ids)
        except Exception as e:
            results.append(FileIngestResult(filename=f"{file.filename or 'unknown'} (error: {e})", ids=[], count=0))

    return IngestFilesResponse(files=results, total_ids=total_ids, total_files=len(files))


@router.post("/search")
def search(req: SearchRequest, rag: RagDep):
    if req.with_score:
        results = rag.retrieve_with_score(req.query, k=req.k)
        return [
            {"content": d.page_content, "metadata": d.metadata, "score": s}
            for d, s in results
        ]
    docs = rag.retrieve(req.query, k=req.k)
    return [{"content": d.page_content, "metadata": d.metadata} for d in docs]


@router.delete("/documents")
def delete_documents(req: DeleteRequest, rag: RagDep):
    rag.delete_documents(req.ids)
    return {"deleted": len(req.ids)}
