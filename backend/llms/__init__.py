from typing import List, Dict

from .localLLMs import LocalLLMs
from .onlineLLMs import OnlineLLMs

class LLMs:
    def __init__(
        self,
        type: str,
        model_name: str, 
        engine: str = None,
        api_key : str = None,
        base_url: str = None, 
        **kwargs
    ):
        self.type = type.lower()

        if type == "offline":
            self.llm = LocalLLMs(engine=engine, model_name=model_name, base_url=base_url, **kwargs)
        elif type == "online":
            self.llm = OnlineLLMs(engine=engine, model_name=model_name, api_key=api_key, base_url=base_url, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {type}")

    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Sync inference."""
        return self.llm.invoke(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            images=images,
        )
    
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Async inference."""
        return await self.llm.ainvoke(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            images=images,
        )