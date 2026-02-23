import base64
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

def load_image_base64(image_source: str) -> Tuple[str, str]:

    if Path(image_source).exists():
        suffix = Path(image_source).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        with open(image_source, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        return base64_data, media_type

    return image_source, "image/jpeg"


class BaseEngine(ABC):
    @abstractmethod
    def __init__(self, model_name: str, **kwargs): ...

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Sync inference."""
        ...

    @abstractmethod
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        """Async inference."""
        ...