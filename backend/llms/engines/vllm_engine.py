import aiohttp
import requests
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine, load_image_base64


class VLLMEngine(BaseEngine):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000",
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self._sync_connect()

    def _sync_connect(self):
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            matched = next((m for m in models if m["id"] == self.model_name), None)
            if matched:
                self.max_tokens = matched.get("max_model_len", self.max_tokens)
            print(f"Kết nối vLLM thành công. max_tokens={self.max_tokens}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối vLLM tại {self.base_url}. Lỗi: {e}")

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> List[Dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)

        # vLLM dùng OpenAI format — content là list khi có ảnh
        if images:
            content = []
            for img in images:
                b64_data, media_type = load_image_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64_data}"}
                })
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

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
        messages = self._build_messages(prompt, system_prompt, history_messages, images)
        payload = {"model": self.model_name, "messages": messages}
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

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
        messages = self._build_messages(prompt, system_prompt, history_messages, images)
        payload = {"model": self.model_name, "messages": messages}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()
