import torch
import asyncio
from PIL import Image
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine


class HuggingFaceEngine(BaseEngine):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            self.processor = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def _load_pil_images(self, images: List[str]) -> List[Image.Image]:
        pil_images = []
        for img_source in images:
            from pathlib import Path
            import base64, io
            if Path(img_source).exists():
                pil_images.append(Image.open(img_source).convert("RGB"))
            else:
                img_data = base64.b64decode(img_source)
                pil_images.append(Image.open(io.BytesIO(img_data)).convert("RGB"))
        return pil_images
    
    def _build_inputs(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)

        if images and self.processor:
            content = [{"type": "image"} for _ in images]  # placeholder
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})

            pil_images = self._load_pil_images(images)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
            ).to(self.device)
        else:
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        return inputs

    def _run_generate(self, inputs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError,)),
    )
    def invoke(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
        images: List[str] = [],
    ) -> str:
        inputs = self._build_inputs(prompt, system_prompt, history_messages, images)
        return self._run_generate(inputs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError,)),
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