from __future__ import annotations

import numpy as np
import torch
from functools import lru_cache
from typing import List, Optional, Union

from pydantic import Field
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbedding, EmbeddingConfig
from utils import get_device


@lru_cache(maxsize=4)
def _load_model(name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(name, device_map="auto", trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class HFEmbeddingConfig(EmbeddingConfig):
    device: Optional[str] = Field(default=None)
    trust_remote_code: bool = True


class HFEmbedding(BaseEmbedding):
    def __init__(self, config: HFEmbeddingConfig):
        super().__init__(config.name)
        self.device = config.device or get_device()
        self.model, self.tokenizer = _load_model(config.name, config.trust_remote_code)
        self.model = self.model.to(self.device).eval()

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        texts = [text] if isinstance(text, str) else text
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        emb = out.last_hidden_state.mean(dim=1)
        if emb.dtype == torch.bfloat16:
            emb = emb.to(torch.float32)
        return emb.detach().cpu().numpy()
