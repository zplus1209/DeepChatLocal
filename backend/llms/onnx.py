from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class ONNXChatModel(BaseChatModel):
    model_path: str
    local_dir: str = "./onnx_models"
    max_new_tokens: int = 2048
    _model: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = ORTModelForCausalLM.from_pretrained(
            self.model_path, export=False, provider="CUDAExecutionProvider"
        )

    def _msgs_to_text(self, messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            if isinstance(m, SystemMessage):
                parts.append(f"<|im_start|>system\n{m.content}<|im_end|>")
            elif isinstance(m, HumanMessage):
                parts.append(f"<|im_start|>user\n{m.content}<|im_end|>")
            elif isinstance(m, AIMessage):
                parts.append(f"<|im_start|>assistant\n{m.content}<|im_end|>")
        parts.append("<|im_start|>assistant")
        return "\n".join(parts)

    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs) -> ChatResult:
        prompt = self._msgs_to_text(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        out_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self._tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_clean(text)))])

    @property
    def _llm_type(self) -> str:
        return "onnx"
