from __future__ import annotations

from typing import List

from fastembed import TextEmbedding

from .base import BaseEmbedding


class FastEmbedding(BaseEmbedding):
    def __init__(self, name: str = "BAAI/bge-m3", max_length: int = 512):
        super().__init__(name)
        try:
            self._model = TextEmbedding(model_name=name, max_length=max_length)
        except Exception as e:
            raise ValueError(f"FastEmbed init failed: {e}") from e

    def encode(self, docs: List[str]) -> List[List[float]]:
        try:
            return [e.tolist() for e in self._model.embed(docs)]
        except Exception as e:
            raise ValueError(f"FastEmbed encode failed: {e}") from e
