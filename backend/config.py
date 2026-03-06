from __future__ import annotations

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_provider: Literal["ollama", "vllm", "huggingface", "onnx"] = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: Optional[str] = "http://localhost:11434"
    llm_temperature: float = 0.7

    # Embedding
    embedding_name: str = "Alibaba-NLP/gte-multilingual-base"
    embedding_backend: Literal["hf", "fastembed"] = "hf"
    embedding_dim: int = 768

    # Vector DB
    db_type: Literal["chromadb", "qdrant", "milvus", "mongodb", "neo4j"] = "qdrant"
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    collection_name: str = "deepchat"
    top_k: int = 4

    # DB paths / URIs
    qdrant_path: str = "./qdrant_db"
    milvus_uri: str = "./milvus_local.db"
    mongodb_uri: Optional[str] = None
    mongodb_db: Optional[str] = None
    neo4j_url: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None

    # RAG features
    reranker_model: Optional[str] = None
    enable_router: bool = False
    enable_reflection: bool = False
