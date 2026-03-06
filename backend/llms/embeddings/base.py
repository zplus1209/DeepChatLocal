from __future__ import annotations

import numpy as np
from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    name: str = Field(..., description="Model name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Model name must be a non-empty string")
        return v


class BaseEmbedding:
    def __init__(self, name: str):
        self.name = name

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError


class APIBaseEmbedding(BaseEmbedding):
    def __init__(self, name: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(name)
        self.base_url = base_url
        self.api_key = api_key
