from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        ...
    
    async def asplit_text(self, text: str) -> List[str]:
        ...
