from __future__ import annotations

import re
from typing import Any, Iterator, List, Literal, Optional

import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def build_llm(
    provider: Literal["ollama", "vllm", "huggingface", "onnx"],
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, **kwargs)

    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url=f"{base_url}/v1",
            api_key="none",
            **kwargs,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            max_new_tokens=kwargs.pop("max_new_tokens", 2048),
            do_sample=kwargs.pop("do_sample", True),
            temperature=kwargs.pop("temperature", 0.7),
            top_p=kwargs.pop("top_p", 0.9),
        )
        return HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=kwargs)

    if provider == "onnx":
        from .onnx import ONNXChatModel
        local_dir = kwargs.pop("local_dir", "./onnx_models")
        return ONNXChatModel(model_path=model, local_dir=local_dir, **kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
