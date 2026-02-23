import re
from typing import List, Dict

from .engines import HuggingFaceEngine, OllamaEngine, ONNXEngine, VLLMEngine

_ENGINE_MAP = {
    "ollama": OllamaEngine,
    "vllm": VLLMEngine,
    "onnx": ONNXEngine,
    "huggingface": HuggingFaceEngine,
}


class LocalLLMs:
    def __init__(self, engine: str, model_name: str, base_url: str = None, **kwargs):
        engine_cls = _ENGINE_MAP.get(engine)
        if not engine_cls:
            raise ValueError(f"Unsupported engine: {engine}. Supported: {list(_ENGINE_MAP.keys())}")
        init_kwargs = {"model_name": model_name, **kwargs}
        if base_url:
            init_kwargs["base_url"] = base_url
        self._engine = engine_cls(**init_kwargs)

    @staticmethod
    def remove_think_blocks(text: str) -> str:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return re.sub(r'\n\s*\n', '\n', cleaned).strip()

    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Sync inference."""
        result = self._engine.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            images=images,
        )
        return self.remove_think_blocks(result)
    
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Async inference."""
        result = await self._engine.ainvoke(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            images=images,
        )
        return self.remove_think_blocks(result)
