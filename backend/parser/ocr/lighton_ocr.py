from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import OCRBackend, ImageInput

logger = logging.getLogger(__name__)

_MODEL_ID = "lightonai/LightOnOCR-2-1B"


class LightOnOCRBackend(OCRBackend):
    """
    LightOn OCR 2-1B backend.

    The model is loaded lazily on the first call to `run` and cached for
    the lifetime of the backend instance. Create one instance and reuse it
    across regions / pages to avoid reloading weights.
    """

    def __init__(self, model_id: str = _MODEL_ID, max_new_tokens: int = 4096):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device: str = "cpu"
        self._dtype = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import (
            LightOnOcrForConditionalGeneration,
            LightOnOcrProcessor,
        )

        if torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float32
        elif torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.bfloat16
        else:
            self._device = "cpu"
            self._dtype = torch.bfloat16

        logger.info(f"Loading {self.model_id} on device={self._device} ...")
        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self._dtype
        ).to(self._device)
        self._processor = LightOnOcrProcessor.from_pretrained(self.model_id)
        logger.info("LightOnOCR model loaded and ready.")

    @staticmethod
    def _to_pil(image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def run(self, image: ImageInput) -> str:
        self._ensure_loaded()
        pil_image = self._to_pil(image)

        conversation = [
            {"role": "user", "content": [{"type": "image", "url": pil_image}]}
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: (
                v.to(device=self._device, dtype=self._dtype)
                if v.is_floating_point()
                else v.to(self._device)
            )
            for k, v in inputs.items()
        }

        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated_ids, skip_special_tokens=True)


def lighon_ocr(image: ImageInput) -> str:
    """Backwards-compatible convenience wrapper. Prefer LightOnOCRBackend for reuse."""
    return LightOnOCRBackend().run(image)