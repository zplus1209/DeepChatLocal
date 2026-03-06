from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base"):
        self._model = CrossEncoder(model_name, trust_remote_code=True)

    def __call__(self, query: str, passages: List[str]) -> Tuple[List[float], List[str]]:
        pairs = [[query, p] for p in passages]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
        return [float(s) for s, _ in ranked], [p for _, p in ranked]
