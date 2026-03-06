from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import Settings
from llms import build_llm
from rag import RAG, Reflection, Route, SemanticRouter


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_rag() -> RAG:
    cfg = get_settings()

    llm = build_llm(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        temperature=cfg.llm_temperature,
    )

    # Build RAG first so we have a valid embeddings instance
    rag = RAG(
        llm=llm,
        db_type=cfg.db_type,
        retrieval_mode=cfg.retrieval_mode,
        embedding_name=cfg.embedding_name,
        embedding_backend=cfg.embedding_backend,
        embedding_dim=cfg.embedding_dim,
        collection_name=cfg.collection_name,
        top_k=cfg.top_k,
        reranker_model=cfg.reranker_model or None,
        # DB-specific kwargs
        qdrant_path=cfg.qdrant_path,
        milvus_uri=cfg.milvus_uri,
        mongodb_uri=cfg.mongodb_uri,
        mongodb_db=cfg.mongodb_db,
        neo4j_url=cfg.neo4j_url,
        neo4j_username=cfg.neo4j_username,
        neo4j_password=cfg.neo4j_password,
    )

    # Bug fix: SemanticRouter requires a valid embeddings object at init time.
    # Pass rag.embeddings (already built) — do NOT pass None and patch later.
    if cfg.enable_router:
        router = SemanticRouter(
            embeddings=rag.embeddings,
            routes=[
                Route(name="rag",      samples=["tìm kiếm", "tra cứu", "thông tin về", "cho biết"]),
                Route(name="chitchat", samples=["xin chào", "bạn khỏe không", "cảm ơn", "tạm biệt"]),
            ],
        )
        rag.router = router

    return rag


@lru_cache(maxsize=1)
def get_reflection() -> Optional[Reflection]:
    cfg = get_settings()
    if not cfg.enable_reflection:
        return None
    return Reflection(llm=get_rag().llm)
