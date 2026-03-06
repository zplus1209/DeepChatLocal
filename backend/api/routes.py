from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import get_rag, get_reflection, get_settings
from api.schemas import (
    ChatRequest, ChatResponse,
    DeleteRequest,
    HealthResponse,
    IngestRequest, IngestResponse,
    SearchRequest,
)
from rag import RAG, Reflection

router = APIRouter()

RagDep = Annotated[RAG, Depends(get_rag)]
RefDep = Annotated[Reflection, Depends(get_reflection)]


@router.get("/health", response_model=HealthResponse)
def health(rag: RagDep):
    return HealthResponse(
        db_type=rag.db_type,
        retrieval_mode=rag.retrieval_mode,
        # Bug fix: access .name via the underlying BaseEmbedding object
        embedding_model=rag.embeddings._model.name,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RagDep, reflection: RefDep):
    """Sync endpoint — FastAPI runs in thread pool automatically."""
    history  = [m.model_dump() for m in req.messages]
    question = history[-1]["content"] if history else ""

    answer = rag.ask(
        question,
        use_rerank=req.use_rerank,
        use_hybrid=req.use_hybrid,
        reflection=reflection if req.use_reflection else None,
        chat_history=history if req.use_reflection else None,
        neo4j_cypher=req.neo4j_cypher,
    )

    sources = []
    if req.use_rag:
        docs    = rag.retrieve(question)
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
    """
    Bug fix: parser.parse_document_sync() is CPU/IO-bound.
    We must NOT call it directly in an async function — that would block
    the event loop. Instead, use the async API: parser.parse_document()
    which internally calls asyncio.to_thread(parse_document_sync).
    (Same pattern as kreuzberg's extract_file() vs extract_file_sync())
    """
    import tempfile

    suffix = Path(file.filename or "upload").suffix or ".pdf"
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from parser import PyMuPDF4LLMParser

        parser = PyMuPDF4LLMParser(tmp_path)

        await parser.parse_document()
        markdown = parser.to_markdown()

        ids = rag.add_texts([markdown], [{"source": file.filename}])
        return IngestResponse(ids=ids, count=len(ids))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


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
