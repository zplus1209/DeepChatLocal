from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from langchain_core.embeddings import Embeddings


@dataclass
class Route:
    name: str
    samples: List[str] = field(default_factory=list)


class SemanticRouter:
    def __init__(self, embeddings: Embeddings, routes: List[Route]):
        self._embeddings = embeddings
        self._routes = routes
        self._route_vecs = {
            r.name: np.array(embeddings.embed_documents(r.samples))
            for r in routes
        }

    def guide(self, query: str) -> str:
        q_vec = np.array(self._embeddings.embed_query(query))
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        best_score, best_name = -1.0, self._routes[0].name
        for name, vecs in self._route_vecs.items():
            norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            score = float(np.mean((vecs / norm) @ q_vec))
            if score > best_score:
                best_score, best_name = score, name

        return best_name
