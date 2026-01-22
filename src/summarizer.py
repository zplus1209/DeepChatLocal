import json
import asyncio
from pathlib import Path
from datetime import date
from tqdm.asyncio import tqdm
from typing import Literal, Dict, List, Callable, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

from loguru import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

from src.utils import resize_base64_image
from prompts.summary import PROMPTS


class EntityInfo(BaseModel):
    entity_name: str = Field(
        description="Descriptive name for this content."
    )

    entity_type: Literal["text", "table", "image", "equation"] = Field(
        description="Type of this content."
    )

    code_no: str | None = Field(
        default=None,
        description="Identifier of the Code No. this content belongs to. Null if unknown."
    )

    version: str | None = Field(
        default=None,
        description="Version of the SOP. Null if unknown."
    )

    valid_from: date | None = Field(
        default=None,
        description="Date from which this SOP version is valid (DD/MM/YYYY). Null if unknown."
    )

    summary: str = Field(
        description="Concise summary of the content's purpose and key points (max 100 words)."
    )
    
    @field_validator("valid_from", mode="before")
    @classmethod
    def normalize_valid_from(cls, v):
        """
        Accepts:
        - None
        - empty string
        - '0000-01-01'
        - invalid placeholders
        """
        if v in (None, "", "N/A", "NA"):
            return None

        if isinstance(v, str) and v.startswith("0000"):
            return None

        return v
    
    
class EnhancedDescription(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detailed_description: str = Field(
        description=(
            "Flexible, structured semantic analysis of the content. "
            "The structure is not fixed and may vary depending on content type "
            "(text, table, image, equation, etc.)."
        )
    )

    entity_info: EntityInfo = Field(
        description="Strict metadata describing the content entity."
    )
    

class ContentSummarizer:
    """Handles summarization of texts, tables, images and equations using LLM models"""
    
    def __init__(self, config = None):
        self.config = config
        
        self.llm = init_chat_model(
            model="qwen2.5:14b",
            model_provider="ollama",
            temperature=0.0
        )
        
        self.vison_llm = init_chat_model(
            model="qwen3-vl:8b",
            model_provider="ollama",
            temperature=0.0
        )
        
        self.callback = UsageMetadataCallbackHandler()

    async def asummarize_documents(
        self,
        text_elements: List[Dict],
        output_path: Union[str, Path] = None,
        detail: float = 1.0,
        llm = None,
        entity_name: str | None = None,
        additional_instructions: str | None = None,
        summarize_recursively: bool = False,
        verbose: bool = False,
    ):
        assert 0.0 <= detail <=1.0
        
        total_chunks = len(text_elements)
        if total_chunks == 0:
            return []
        
        num_chunks = max(1, int(detail * total_chunks))
        selected_elements = text_elements[:num_chunks]
        
        if verbose:
            logger.debug(
                f"Summarizing {len(selected_elements)} / {total_chunks} chunks"
            )
        
        results = await self.asummarize_text_elements(
            selected_elements,
            output_path=output_path,
            llm=llm,
            entity_name=entity_name,
            additional_instructions=additional_instructions,
            summarize_recursively=summarize_recursively,
            verbose=verbose
        )
        
        summary_document = "\n\n".join(
            s.get("entity_info", "").get("summary", "") for s in results
        )
        
        rewrite_messages = [
            SystemMessage(PROMPTS["REWRITE_SUMMARY"]),
            HumanMessage(f"Summary to improve:\n\n{summary_document}")
        ]
        
        final_summary = self.llm.invoke(rewrite_messages)
        
        if verbose:
            logger.info(f"Tokens usage of summary document: {final_summary.usage_metadata}")
        
        final_summary = final_summary.content
        
        return results, final_summary
    
    async def asummarize_text_elements(
        self,
        text_elements: List[Dict],
        output_path: Union[str, Path] = None,
        llm = None,
        entity_name: str | None = None,
        additional_instructions: str | None = None,
        summarize_recursively: bool = False,
        verbose: bool = False,
    ):
        
        if not isinstance(text_elements, List):
            text_elements = [text_elements]
            
        if llm is None:
            llm = self.llm

        chain = self._build_text_summary_chain(
            llm=llm, additional_instructions=additional_instructions
        )

        accumulated_summaries: list[str] = []
        results: list[EnhancedDescription] = []
        
        save_dir = None
        if output_path:
            save_dir = Path(output_path) / "text_summaries"
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, el in enumerate(tqdm(text_elements, desc="Summarizing text element...")):
            chunk = el.get("markdown", "")
            if not chunk.strip():
                continue

            summary_file = save_dir / f"{idx:05d}.json" if save_dir else None

            if summary_file and summary_file.exists():
                if verbose:
                    logger.debug(f"Loading summary from {summary_file.name}")

                with open(summary_file, "r", encoding="utf-8") as f:
                    result = json.load(f)

                results.append(result)

                if summarize_recursively:
                    summary_text = (
                        result
                        .get("entity_info", {})
                        .get("summary")
                    )
                    if summary_text:
                        accumulated_summaries.append(summary_text)

                continue
        
            if verbose:
                logger.debug(f"\n--- Summarizing chunk {idx} ---")
                logger.debug(el.get("metadata", {}))
            
            response: EnhancedDescription = await chain.ainvoke(
                {
                    "entity_name": entity_name or "descriptive name for this text",
                    "content": chunk,
                    "accumulated_summaries": (
                        "\n\n".join(accumulated_summaries)
                        if summarize_recursively else ""
                    ),
                }
            )              
            
            result = response.model_dump(mode="json")
            results.append(result)

            if summarize_recursively:
                accumulated_summaries.append(response.entity_info.summary)
                
            if summary_file:
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                
        return results
    
    async def asummarize_table_elements(
        self,
        table_elements: List[Dict],
        output_path: Union[str, Path] = None,
        llm = None,
        entity_name: str | None = None,
        additional_instructions: str | None = None,
        verbose: bool = False,
    ):
        if not isinstance(table_elements, List):
            table_elements = [table_elements]

        table_save_dir = None
        if output_path:
            table_save_dir = Path(output_path) / "table_summaries"
            table_save_dir.mkdir(parents=True, exist_ok=True)
                            
        results: list[EnhancedDescription] = []
        
        for idx, el in enumerate(tqdm(table_elements, desc="Summarizing table element...")):
            chunk = el.get("markdown", "")
            if not chunk.strip():
                continue
            
            grounding = el.get("grounding", {})
            image_path = grounding.get("image_path", "")
            table_caption = grounding.get("table_caption", "")
            table_footnote = grounding.get("table_footnote", "")
            
            table_summary_path = None
            if image_path:
                image_path = Path(image_path)
                table_summary_path = image_path.with_suffix(".json")
            elif table_save_dir:
                table_summary_path = table_save_dir / f"{idx:05d}.json"

            if table_summary_path and table_summary_path.exists():
                if verbose:
                    logger.debug(f"Loading table summary from {table_summary_path}")

                with open(table_summary_path, "r", encoding="utf-8") as f:
                    table_summary = json.load(f)

                results.append(table_summary)
                continue
        
            if llm is None:
                if image_path:
                    llm = self.vison_llm.with_structured_output(EnhancedDescription)
                else:
                    llm = self.llm.with_structured_output(EnhancedDescription)
                
            if verbose:
                logger.debug(f"\n--- Summarizing chunk {idx} ---")
                logger.debug(el.get("metadata", {}))

            messages = self._build_table_messages(
                table_data=chunk,
                entity_name=entity_name,
                table_caption=table_caption,
                table_footnote=table_footnote,
                table_image_path=image_path,
                additional_instructions=additional_instructions
            )

            async def _call_primary():
                return await llm.ainvoke(
                    messages, config={"callbacks": [self.callback]}
                )  
                
            try:
                response = await self.ainvoke_with_retry(
                    _call_primary,
                    timeout=90,
                    retries=3,
                    tag=f"TABLE_{idx}"
                )
            
            except Exception:
                logger.warning(f"Fallback vison model")
                response = await self.ainvoke_with_retry(
                    # _call_primary,
                    timeout=60,
                    retries=1,
                    tag=f"TABLE_FB_{idx}"
                )
            
            if verbose:
                logger.debug(f"Token usage: {self.callback.usage_metadata}")

            table_summary = response.model_dump(mode="json")
            
            if table_summary_path:
                with open(table_summary_path, "w", encoding="utf-8") as f:
                    json.dump(table_summary, f, ensure_ascii=False, indent=4)
            
            results.append(table_summary)
        
        return results
    
    async def asummarize_image_elements(
        self,
        image_elements: List[Dict],
        llm = None,
        entity_name: str | None = None,
        additional_instructions: str | None = None,
        verbose: bool = False,
    ):
        if not isinstance(image_elements, List):
            image_elements = [image_elements]
            
        if llm is None:
            llm = self.vison_llm.with_structured_output(EnhancedDescription)
              
        results = []
        
        for idx, el in enumerate(tqdm(image_elements, desc="Summarizing image element...")):
            chunk = el.get("markdown", "")
            if not chunk.strip():
                continue
            
            grounding = el.get("grounding", {})
            image_path = grounding.get("image_path", "")
            image_caption = grounding.get("image_caption", "")
            image_footnote = grounding.get("image_footnote", "")

            image_summary_path = Path(image_path).with_suffix(".json")
            if image_summary_path.exists():
                with open(image_summary_path, "r", encoding="utf-8") as f:
                    image_summary = json.load(f)
                    results.append(image_summary)
                    continue
            
            if verbose:
                logger.debug(f"\n--- Summarizing chunk {idx} ---")
                logger.debug(grounding)

            messages = self._build_image_messages(
                entity_name=entity_name,
                image_path=image_path,
                image_caption=image_caption,
                image_footnote=image_footnote,
                additional_instructions=additional_instructions
            )
            
            async def _call_primary():
                return await llm.ainvoke(
                    messages, config={"callbacks": [self.callback]}
                )   
                
            try:
                response = await self.ainvoke_with_retry(
                    _call_primary,
                    timeout=90,
                    retries=3,
                    tag=f"IMAGE_{idx}"
                )
            
            except Exception:
                logger.warning(f"Fallback vison model")
                response = await self.ainvoke_with_retry(
                    # _call_primary,
                    timeout=60,
                    retries=1,
                    tag=f"IMAGE_FB_{idx}"
                )

            if verbose:
                logger.debug(f"Token usage: {self.callback.usage_metadata}")
            
            image_summary = response.model_dump(mode="json")
            with open(image_summary_path, "w", encoding="utf-8") as f:
                json.dump(image_summary, f, ensure_ascii=False, indent=4)
            
            results.append(image_summary)
        
        return results

    async def ainvoke_with_retry(
        self,
        invoke_fn: Callable[[], Any],
        *,
        timeout: int = 60,
        retries: int = 2,
        retry_delay: float = 2.0,
        tag: str = ""
    ):
        last_err = None

        for attempt in range(1, retries + 2):
            try:
                return await asyncio.wait_for(
                    invoke_fn(),
                    timeout=timeout
                )
            except asyncio.TimeoutError as e:
                last_err = e
                logger.warning(
                    f"[TIMEOUT] {tag} attempt {attempt}/{retries+1} after {timeout}s"
                )
            except Exception as e:
                last_err = e
                logger.exception(
                    f"[ERROR] {tag} attempt {attempt}/{retries+1}"
                )

            if attempt <= retries:
                await asyncio.sleep(retry_delay * attempt)

        raise last_err
    
    def _build_text_summary_chain(
        self, 
        llm, 
        additional_instructions=None
    ):

        if additional_instructions:
            PROMPTS["SYSTEM_TEXT"] += f"\n\n{additional_instructions}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPTS["SYSTEM_TEXT"]),
                ("human", PROMPTS["TEXT_PROMPT"]),
            ]
        )

        return (
            {
                "entity_name": lambda x: x["entity_name"],
                "content": lambda x: x["content"],
                "accumulated_summaries": lambda x: x["accumulated_summaries"],
            }
            | prompt
            | llm.with_structured_output(EnhancedDescription)
        )
    
    def _build_table_messages(
        self, 
        table_data: str,
        table_caption: str | None = None,
        table_footnote: str | None = None,
        entity_name: str | None = None,
        table_image_path: str | None = None, 
        additional_instructions: str | None = None
    ):

        if additional_instructions:
            PROMPTS["SYSTEM_TABLE"] += f"\n\n{additional_instructions}"

        table_prompt = PROMPTS["TABLE_PROMPT"].format(
            entity_name=entity_name or "descriptive name for this table",
            table_caption=table_caption,
            table_footnote=table_footnote,
            table_image_path=table_image_path if table_image_path else "None",
            table_data=table_data
        )
        
        if table_image_path:
            human_message = [
                {"type": "text", "text": table_prompt},
                {
                    "type": "image",
                    "base64": resize_base64_image(image_path=table_image_path),
                    "mime_type": "image/jpeg",
                }  
            ]
        
        else:
            human_message = table_prompt
            
            
        messages = [
            SystemMessage(PROMPTS["SYSTEM_TABLE"]),
            HumanMessage(human_message)
        ]

        return messages

    def _build_image_messages(
        self, 
        image_path: str,
        image_caption: str | None = None,
        image_footnote: str | None = None,
        entity_name = None,
        additional_instructions: str | None = None
    ):  
        
        if not image_path:
            raise ValueError(f"No image to summary")

        if additional_instructions:
            PROMPTS["SYSTEM_IMAGE"] += f"\n\n{additional_instructions}"

        image_prompt = PROMPTS["IMAGE_PROMPT"].format(
            entity_name=entity_name or "descriptive name for this image",
            image_path=image_path,
            image_caption=image_caption if image_footnote else "None",
            image_footnote=image_footnote if image_footnote else "None",
        )
        
        human_message = [
            {"type": "text", "text": image_prompt},
            {
                "type": "image",
                "base64": resize_base64_image(image_path=image_path),
                "mime_type": "image/jpeg",
            }
        ]
        
        messages = [
            SystemMessage(PROMPTS["SYSTEM_IMAGE"]),
            HumanMessage(human_message)
        ]

        return messages

