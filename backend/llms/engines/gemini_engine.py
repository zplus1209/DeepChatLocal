import asyncio
from typing import List, Dict
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine, load_image_base64


class GeminiEngine(BaseEngine):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        genai.configure(api_key=api_key)
        self._genai = genai

    def _build_model(self, system_prompt: str = None):
        return self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

    def _build_messages(
        self,
        prompt: str,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> List[Dict]:
        gemini_messages = []

        for msg in history_messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        parts = []
        for img in images:
            b64_data, media_type = load_image_base64(img)
            parts.append({"inline_data": {"mime_type": media_type, "data": b64_data}})
        parts.append(prompt)

        gemini_messages.append({"role": "user", "parts": parts})
        return gemini_messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        model = self._build_model(system_prompt)
        messages = self._build_messages(prompt, history_messages, images)
        response = model.generate_content(messages)
        try:
            return response.text
        except Exception:
            return response.candidates[0].content.parts[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        return await asyncio.to_thread(
            self.invoke, prompt, system_prompt, history_messages, images
        )
    