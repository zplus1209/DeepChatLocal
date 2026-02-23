import openai
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine, load_image_base64


class OpenAIEngine(BaseEngine):
    def __init__(self, model_name: str, api_key: str, base_url: str = None, **kwargs):
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

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
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content