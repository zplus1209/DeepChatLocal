import os
import ollama
import requests
from pathlib import Path
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine, load_image_base64


class OllamaEngine(BaseEngine):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = kwargs.get("timeout", None)
        self.api_key = kwargs.get("api_key") or os.getenv("OLLAMA_API_KEY")
        self._sync_connect_and_pull()

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sync_connect_and_pull(self):
        try:
            requests.get(self.base_url, timeout=5).raise_for_status()
            client = requests.Session()
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            if not any(self.model_name in m["name"] for m in models):
                print(f"Đang tải model '{self.model_name}'...")
                client.post(f"{self.base_url}/api/pull", json={"name": self.model_name}).raise_for_status()
                print(f"Tải model '{self.model_name}' thành công.")
            else:
                print(f"Model '{self.model_name}' đã có sẵn.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Không thể kết nối Ollama tại {self.base_url}. Lỗi: {e}")

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

        user_message: Dict = {"role": "user", "content": prompt}
        if images:
            encoded_images = []
            for img in images:
                b64_data, _ = load_image_base64(img)
                encoded_images.append(b64_data)
            user_message["images"] = encoded_images  

        messages.append(user_message)
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
        client = ollama.Client(
            host=self.base_url,
            timeout=self.timeout,
            headers=self._build_headers(),
        )
        response = client.chat(model=self.model_name, messages=messages)
        return response["message"]["content"].strip()
    
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
        ollama_client = ollama.AsyncClient(
            host=self.base_url,
            timeout=self.timeout,
            headers=self._build_headers(),
        )
        try:
            response = await ollama_client.chat(
                model=self.model_name,
                messages=messages,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise e
        finally:
            try:
                await ollama_client._client.aclose()
            except Exception:
                pass
