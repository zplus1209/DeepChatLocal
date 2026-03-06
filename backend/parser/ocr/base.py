from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]


class OCRBackend(ABC):
    @abstractmethod
    def run(self, image: ImageInput) -> str:
        ...

    @classmethod
    def name(cls) -> str:
        return cls.__name__
