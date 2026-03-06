from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    use_rag: bool = True
    use_rerank: bool = False
    use_hybrid: bool = False
    use_reflection: bool = False
    neo4j_cypher: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class IngestResponse(BaseModel):
    ids: List[str]
    count: int


class DeleteRequest(BaseModel):
    ids: List[str]


class SearchRequest(BaseModel):
    query: str
    k: int = 4
    with_score: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    db_type: str
    retrieval_mode: str
    embedding_model: str


class FileIngestResult(BaseModel):
    filename: str
    ids: List[str]
    count: int


class IngestFilesResponse(BaseModel):
    files: List[FileIngestResult]
    total_ids: int
    total_files: int