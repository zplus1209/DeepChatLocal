from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings

from .base import BaseEmbedding
from .hf import HFEmbedding, HFEmbeddingConfig
from .fastEmbed import FastEmbedding


class EmbeddingWrapper(Embeddings):
    """Wraps any BaseEmbedding into the LangChain Embeddings interface."""

    def __init__(self, model: BaseEmbedding):
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._model.encode(texts)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def embed_query(self, text: str) -> List[float]:
        result = self._model.encode([text])
        if isinstance(result, np.ndarray):
            return result[0].tolist()
        return result[0]


def build_embedding_wrapper(
    name: str = "Alibaba-NLP/gte-multilingual-base",
    backend: str = "hf",
    **kwargs,
) -> EmbeddingWrapper:
    if backend == "hf":
        model = HFEmbedding(HFEmbeddingConfig(name=name, **kwargs))
    elif backend == "fastembed":
        model = FastEmbedding(name=name, **kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    return EmbeddingWrapper(model)
