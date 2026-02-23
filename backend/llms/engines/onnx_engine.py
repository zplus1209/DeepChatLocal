import os
import time
import asyncio
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..base import BaseEngine


class ONNXModel:
    """ONNX model wrapper for efficient inference."""

    def __init__(self, model_version: str, local_dir: str = "./onnx_models"):
        self.model_version = model_version
        self.local_dir = local_dir
        self.onnx_session = None
        self.tokenizer = None
        self.input_names = []
        self.output_names = []
        self.num_layers = 28
        self.num_heads = 8
        self.head_dim = 128
        self._initialize_model()

    def _initialize_model(self):
        print(f"Downloading ONNX model: {self.model_version}")
        model_path = snapshot_download(
            repo_id=self.model_version,
            local_dir=self.local_dir,
        )
        print(f"Downloaded to: {model_path}")

        # Tìm file .onnx
        onnx_model_path = os.path.join(model_path, "onnx", "model.onnx")
        if not os.path.exists(onnx_model_path):
            candidates = [
                os.path.join(model_path, "model.onnx"),
                os.path.join(model_path, "onnx", "model_fp16.onnx"),
                os.path.join(model_path, "model_fp16.onnx"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    onnx_model_path = path
                    break
            else:
                raise FileNotFoundError(f"ONNX model file not found in {model_path}")

        print(f"Loading ONNX session from: {onnx_model_path}")
        self.onnx_session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
        )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]
        self._detect_model_architecture()

        print(f"ONNX Model loaded | layers={self.num_layers}, heads={self.num_heads}, head_dim={self.head_dim}")

    def _detect_model_architecture(self):
        try:
            config = AutoConfig.from_pretrained(self.model_version)
            self.num_layers = getattr(config, "num_hidden_layers",
                             getattr(config, "n_layer",
                             getattr(config, "num_layers", 28)))
            self.num_heads = getattr(config, "num_attention_heads",
                            getattr(config, "n_head",
                            getattr(config, "num_heads", 8)))
            hidden_size = getattr(config, "hidden_size",
                         getattr(config, "d_model", 1024))
            self.head_dim = hidden_size // self.num_heads
        except Exception as e:
            print(f"Could not auto-detect config: {e}. Using defaults (Qwen3-0.6B).")

    def set_architecture(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def prepare_inputs(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
        position_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        batch_size, seq_length = input_ids.shape
        inputs = {"input_ids": input_ids.astype(np.int64)}

        if attention_mask is None:
            attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
        inputs["attention_mask"] = attention_mask.astype(np.int64)

        if position_ids is None:
            if past_key_values is None:
                position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
                position_ids = np.repeat(position_ids, batch_size, axis=0)
            else:
                past_length = past_key_values["past_key_values.0.key"].shape[2]
                position_ids = np.array([[past_length]], dtype=np.int64)
                position_ids = np.repeat(position_ids, batch_size, axis=0)
        inputs["position_ids"] = position_ids

        if past_key_values is not None:
            past_seq_length = past_key_values["past_key_values.0.key"].shape[2]
            total_seq_length = past_seq_length + seq_length
            inputs["attention_mask"] = np.ones((batch_size, total_seq_length), dtype=np.int64)

        for layer_idx in range(self.num_layers):
            key_name = f"past_key_values.{layer_idx}.key"
            value_name = f"past_key_values.{layer_idx}.value"
            if past_key_values is None:
                cache_shape = (batch_size, self.num_heads, 0, self.head_dim)
                inputs[key_name] = np.zeros(cache_shape, dtype=np.float32)
                inputs[value_name] = np.zeros(cache_shape, dtype=np.float32)
            else:
                inputs[key_name] = past_key_values[key_name]
                inputs[value_name] = past_key_values[value_name]

        return inputs

    def extract_kv_cache(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        kv_cache = {}
        for i, output_name in enumerate(self.output_names[1:], 1):
            if output_name.startswith("present."):
                parts = output_name.split(".")
                past_name = f"past_key_values.{parts[1]}.{parts[2]}"
                kv_cache[past_name] = outputs[i]
        return kv_cache

    def generate_token(
        self,
        input_ids: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        inputs = self.prepare_inputs(input_ids, past_key_values=past_key_values)
        outputs = self.onnx_session.run(None, inputs)
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[0, -1, :]))
        new_kv_cache = self.extract_kv_cache(outputs)
        return next_token_id, new_kv_cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        prompt_tokens = input_ids.shape[1]
        generated_tokens = []
        kv_cache = None
        t0 = time.perf_counter()

        for step in range(max_new_tokens):
            current_input = input_ids if step == 0 else np.array([[generated_tokens[-1]]], dtype=np.int64)
            try:
                next_token_id, kv_cache = self.generate_token(current_input, past_key_values=kv_cache)
                generated_tokens.append(next_token_id)
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            except Exception as e:
                print(f"Error at step {step}: {e}")
                break

        elapsed = max(time.perf_counter() - t0, 1e-9)
        completion_tokens = len(generated_tokens)
        print(f"{completion_tokens} tokens in {elapsed:.3f}s "
              f"({completion_tokens / elapsed:.2f} tok/s) | prompt={prompt_tokens}")

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_model_info(self) -> Dict:
        return {
            "model_version": self.model_version,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.tokenizer.vocab_size,
            "input_names": self.input_names,
            "output_names": self.output_names,
        }
    

class ONNXEngine(BaseEngine):
    def __init__(self, model_name: str, local_dir: str = "./onnx_models", **kwargs):

        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 1.0)
        self._load_model(local_dir)

    def _load_model(self, local_dir: str):
        self._model = ONNXModel(
            model_version=self.model_name,
            local_dir=local_dir,
        )

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: str = None,
        history_messages: List[Dict] = [],
    ) -> str:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback: build thủ công theo ChatML format
        prompt_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
        return prompt_text

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
        images: List[str] = [],  # ONNX text model không hỗ trợ ảnh
    ) -> str:
        if images:
            raise NotImplementedError(
                "ONNXEngine hiện chỉ hỗ trợ text. "
                "Dùng HuggingFaceEngine cho multimodal."
            )

        prompt_text = self._build_prompt(prompt, system_prompt, history_messages)
        return self._model.generate(
            prompt=prompt_text,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
        )

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