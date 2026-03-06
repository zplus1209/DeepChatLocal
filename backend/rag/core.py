from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from llms.embeddings import EmbeddingWrapper, build_embedding_wrapper
from rag.stores import DBType, build_vector_store
from rag.rerank import Reranker
from rag.reflection import Reflection
from rag.router import SemanticRouter


RAG_PROMPT = ChatPromptTemplate.from_template(
    "Bạn là trợ lý AI. Trả lời câu hỏi dựa trên ngữ cảnh.\n"
    "Nếu không có thông tin, hãy nói rõ.\n\n"
    "Ngữ cảnh:\n{context}\n\n"
    "Câu hỏi: {question}\n\nCâu trả lời:"
)


class RAG:
    """
    RAG pipeline hỗ trợ 5 vector DB và 3 chế độ tìm kiếm.

    Hybrid RAG hoạt động theo 2 cách:

    1. Qdrant native hybrid (retrieval_mode="hybrid"):
       - Dense search (embedding similarity) + Sparse search (BM25 via FastEmbedSparse)
       - Kết quả gộp bằng Reciprocal Rank Fusion (RRF) do Qdrant engine xử lý

    2. Manual hybrid (dùng hybrid_retrieve() với ChromaDB / Milvus / MongoDB):
       - Dense search lấy top-k*3 candidates
       - Keyword overlap scoring (sparse-like)
       - RRF merge thủ công: score(d) = Σ 1/(rank_i + 60)

    3. Neo4j hybrid (neo4j_hybrid_retrieve()):
       - Vector search tìm ngữ nghĩa
       - Cypher graph traversal tìm entity liên quan
       - Merge + dedup hai nguồn
    """

    def __init__(
        self,
        llm: BaseChatModel,
        db_type: DBType = "qdrant",
        retrieval_mode: Literal["dense", "sparse", "hybrid"] = "dense",
        embedding_name: str = "Alibaba-NLP/gte-multilingual-base",
        embedding_backend: Literal["hf", "fastembed"] = "hf",
        embedding_dim: int = 768,
        collection_name: Optional[str] = None,
        top_k: int = 4,
        reranker_model: Optional[str] = None,
        router: Optional[SemanticRouter] = None,
        **store_kwargs,
    ):
        self.llm = llm
        self.db_type = db_type
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.collection_name = collection_name or embedding_name.split("/")[-1]

        self.embeddings: EmbeddingWrapper = build_embedding_wrapper(
            name=embedding_name, backend=embedding_backend
        )
        self.vector_store: VectorStore = build_vector_store(
            db_type=db_type,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            embedding_dim=embedding_dim,
            retrieval_mode=retrieval_mode,
            **store_kwargs,
        )
        self.reranker = Reranker(reranker_model) if reranker_model else None
        self.router = router

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> List[str]:
        ids = [str(uuid4()) for _ in documents]
        self.vector_store.add_documents(documents, ids=ids)
        return ids

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        metadatas = metadatas or [{} for _ in texts]
        return self.add_documents(
            [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        )

    def delete_documents(self, ids: List[str]) -> None:
        self.vector_store.delete(ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Dense / Sparse / Qdrant-hybrid tùy retrieval_mode."""
        return self.vector_store.similarity_search(query, k=k or self.top_k)

    def retrieve_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k or self.top_k)

    def hybrid_retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Manual Hybrid RAG dùng cho ChromaDB / Milvus / MongoDB.
        Gộp dense results + keyword overlap bằng Reciprocal Rank Fusion.
        """
        k = k or self.top_k
        dense_results = self.vector_store.similarity_search_with_score(query, k=k * 3)

        query_tokens = set(query.lower().split())
        keyword_ranked = sorted(
            dense_results,
            key=lambda x: len(query_tokens & set(x[0].page_content.lower().split())) / (len(query_tokens) + 1e-10),
            reverse=True,
        )

        rrf: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, (doc, _) in enumerate(dense_results):
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (rank + 60)
            doc_map[doc.page_content] = doc
        for rank, (doc, _) in enumerate(keyword_ranked):
            rrf[doc.page_content] = rrf.get(doc.page_content, 0.0) + 1.0 / (rank + 60)
            doc_map[doc.page_content] = doc

        return [doc_map[key] for key in sorted(rrf, key=lambda x: rrf[x], reverse=True)][:k]

    def retrieve_and_rerank(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieval → CrossEncoder rerank."""
        k = k or self.top_k
        docs = self.retrieve(query, k=k * 2)
        if self.reranker is None or not docs:
            return docs[:k]
        _, ranked = self.reranker(query, [d.page_content for d in docs])
        doc_map = {d.page_content: d for d in docs}
        return [doc_map[p] for p in ranked if p in doc_map][:k]

    def neo4j_hybrid_retrieve(self, query: str, cypher: str, k: Optional[int] = None) -> List[Document]:
        """
        Vector search + Cypher graph traversal, merge + dedup.
        Cypher nhận param $query, trả về field 'text'.
        """
        if self.db_type != "neo4j":
            raise ValueError("Chỉ dùng được khi db_type='neo4j'")
        k = k or self.top_k
        vector_docs = self.retrieve(query, k=k)

        from langchain_neo4j import Neo4jGraph
        vs = self.vector_store
        graph = Neo4jGraph(url=vs.url, username=vs.username, password=vs.password)
        graph_docs = [
            Document(page_content=str(row.get("text", row)), metadata={"source": "graph"})
            for row in graph.query(cypher, params={"query": query})
        ]

        seen, merged = set(), []
        for doc in vector_docs + graph_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged[:k]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _resolve_docs(
        self,
        question: str,
        *,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        neo4j_cypher: Optional[str] = None,
    ) -> List[Document]:
        if neo4j_cypher:
            return self.neo4j_hybrid_retrieve(question, neo4j_cypher)
        if use_rerank:
            return self.retrieve_and_rerank(question)
        if use_hybrid and self.db_type != "qdrant":
            return self.hybrid_retrieve(question)
        return self.retrieve(question)

    def answer_with_docs(
        self,
        question: str,
        *,
        use_rag: bool = True,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        reflection: Optional[Reflection] = None,
        chat_history: Optional[list] = None,
        neo4j_cypher: Optional[str] = None,
    ) -> Tuple[str, List[Document]]:
        if reflection and chat_history:
            question = reflection.rewrite(chat_history)

        if self.router and self.router.guide(question) == "chitchat":
            return self.llm.invoke(question).content, []

        if not use_rag:
            return self.llm.invoke(question).content, []

        docs = self._resolve_docs(
            question,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            neo4j_cypher=neo4j_cypher,
        )

        context = "\n\n".join(d.page_content for d in docs)
        answer = (RAG_PROMPT | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )

        return answer, docs
    
    def ask(
        self,
        question: str,
        *,
        use_rag: bool = True,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        reflection: Optional[Reflection] = None,
        chat_history: Optional[list] = None,
        neo4j_cypher: Optional[str] = None,
    ) -> str:
        answer, _ = self.answer_with_docs(
            question,
            use_rag=use_rag,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            reflection=reflection,
            chat_history=chat_history,
            neo4j_cypher=neo4j_cypher,
        )
        return answer

    def as_retriever(self, **kwargs) -> BaseRetriever:
        return self.vector_store.as_retriever(**kwargs)
