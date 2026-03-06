from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


DBType = Literal["chromadb", "qdrant", "milvus", "mongodb", "neo4j"]


def build_vector_store(
    db_type: DBType,
    embeddings: Embeddings,
    collection_name: str,
    *,
    embedding_dim: int = 768,
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "dense",
    # Qdrant
    qdrant_path: Optional[str] = None,
    # Milvus
    milvus_uri: str = "./milvus_local.db",
    # MongoDB
    mongodb_uri: Optional[str] = None,
    mongodb_db: Optional[str] = None,
    # Neo4j
    neo4j_url: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    **kwargs: Any,
) -> VectorStore:

    if db_type == "chromadb":
        return _build_chromadb(embeddings, collection_name, **kwargs)

    if db_type == "qdrant":
        return _build_qdrant(
            embeddings, collection_name, embedding_dim,
            retrieval_mode, qdrant_path, **kwargs,
        )

    if db_type == "milvus":
        return _build_milvus(embeddings, collection_name, milvus_uri, **kwargs)

    if db_type == "mongodb":
        return _build_mongodb(embeddings, collection_name, mongodb_uri, mongodb_db, **kwargs)

    if db_type == "neo4j":
        return _build_neo4j(
            embeddings, collection_name,
            neo4j_url, neo4j_username, neo4j_password, **kwargs,
        )

    raise ValueError(f"Unsupported db_type: {db_type}")


def _build_chromadb(embeddings: Embeddings, collection_name: str, **kwargs) -> VectorStore:
    import chromadb
    from langchain_chroma import Chroma

    persist_dir = kwargs.pop("persist_directory", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        **kwargs,
    )


def _build_qdrant(
    embeddings: Embeddings,
    collection_name: str,
    embedding_dim: int,
    retrieval_mode: str,
    qdrant_path: Optional[str],
    **kwargs,
) -> VectorStore:
    from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

    qdrant_dir = Path(qdrant_path or "./qdrant_db").expanduser()
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        if retrieval_mode == "dense":
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )

    mode_map = {
        "dense": RetrievalMode.DENSE,
        "sparse": RetrievalMode.SPARSE,
        "hybrid": RetrievalMode.HYBRID,
    }
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")

    if retrieval_mode == "dense":
        return QdrantVectorStore(
            client=client, collection_name=collection_name,
            embedding=embeddings, retrieval_mode=mode_map["dense"],
        )
    if retrieval_mode == "sparse":
        return QdrantVectorStore(
            client=client, collection_name=collection_name,
            sparse_embedding=sparse_emb, retrieval_mode=mode_map["sparse"],
            sparse_vector_name="sparse",
        )
    return QdrantVectorStore(
        client=client, collection_name=collection_name,
        embedding=embeddings, sparse_embedding=sparse_emb,
        retrieval_mode=mode_map["hybrid"],
        vector_name="dense", sparse_vector_name="sparse",
    )


def _build_milvus(
    embeddings: Embeddings,
    collection_name: str,
    uri: str,
    **kwargs,
) -> VectorStore:
    from langchain_milvus import Milvus

    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": uri},
        auto_id=True,
        **kwargs,
    )


def _build_mongodb(
    embeddings: Embeddings,
    collection_name: str,
    uri: Optional[str],
    db_name: Optional[str],
    **kwargs,
) -> VectorStore:
    from pymongo import MongoClient
    from langchain_mongodb import MongoDBAtlasVectorSearch

    client = MongoClient(uri)
    collection = client[db_name][collection_name]
    index_name = kwargs.pop("index_name", "vector_index")

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name,
        **kwargs,
    )


def _build_neo4j(
    embeddings: Embeddings,
    collection_name: str,
    url: Optional[str],
    username: Optional[str],
    password: Optional[str],
    **kwargs,
) -> VectorStore:
    from langchain_neo4j import Neo4jVector

    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=url or "bolt://localhost:7687",
        username=username or "neo4j",
        password=password or "password",
        index_name=collection_name,
        **kwargs,
    )
