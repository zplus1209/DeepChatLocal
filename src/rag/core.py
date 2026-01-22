from typing import List

from loguru import logger
from operator import itemgetter
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.multi_retriever import MultiRetriever
from src.utils import resize_base64_image
from prompts.qa import PROMPTS
from src.grader import grader_documents

class MultimodalRAG:
    
    def __init__(
        self,
        # config,
        retriever
    ):
        # self.config = config
        self.retriever = retriever
        
        self.chat_llm = init_chat_model(
            model="qwen3-vl:8b",
            model_provider="ollama",
            temperature=0.0
        )
        
        self.multimodal_rag = self._build_rag()
    
    def invoke(self, question: str):
        try:
            logger.info(f"Processing question: {question}")
                
            result = self.multimodal_rag.invoke({"input": question})
            
            return {**result}
        
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            
            return {
                "input": question,
                "answer": error_msg,
                "error": True
            }
            
    def batch(self, questions: List[str]):
        try:
            logger.info(f"Processing {len(questions)} questions in batch")

            inputs = [{"input": q} for q in questions]
            responses = self.multimodal_rag.batch(inputs)
            
            results = []
            for i, (question, response) in enumerate(zip(questions, responses)):
                results.append({
                    **response,
                    "batch_idx": i
                })
            
            return results
        
        except Exception as e:
            error_msg = f"Error batch processing: {str(e)}"
            logger.error(error_msg)
            
            return [{
                "input": question,
                "answer": error_msg,
                "error": True
            } for question in questions]
        
    def _split_image_text_types(self, docs):
        images = []
        texts = []

        for doc in docs:
            if not isinstance(doc, dict):
                continue

            doc_type = doc.get("type")
            markdown = doc.get("markdown", "")
            grounding = doc.get("grounding", {})
            image_path = grounding.get("image_path")
            
            if doc_type == "image":
                images.append(resize_base64_image(image_path=image_path))
                
            elif doc_type in ["table", "equation"]:
                if image_path:    
                    images.append(resize_base64_image(image_path=image_path))
                else:
                    texts.append(markdown)
                
            else:
                texts.append(markdown)
                
        return {"images": images, "texts": texts}
    
    def _multimodal_prompt_function(self, data_dict):
            
        texts = data_dict["context"].get("texts", [])
        images = data_dict["context"].get("images", [])
        question = data_dict["question"]
        
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        formatted_texts = "\n\n".join(texts)
        
        messages = []
        
        for image in images:
            messages.append({
                "type": "image",
                "base64": image,
                "mime_type": "image/jpeg",
            })
            
        messages.append({
            "type": "text",
            "text": PROMPTS["text_prompt"].format( 
                formatted_texts=formatted_texts,
                question=question,
            ),
        })
        
        final_messages = [
            SystemMessage(PROMPTS["QA_SYSTEM"]),
            HumanMessage(messages)
        ]                    
        
        return final_messages

    def _grader_document(self, docs, question: str, k: int = 4):
        logger.info(f"Using grader")
    
        retrieval_grader = grader_documents()
        results = []

        for doc in docs[:k]:
            res = retrieval_grader.invoke({
                "question": question,
                "document": doc.get("markdown")
            })
            if res.binary_score == "yes":
                results.append(doc)

        return results

    def _build_rag(self, grader: bool = True):
        
        multimodal_rag = (
            {
                "context": itemgetter('context'),
                "question": itemgetter('input'),
            }
            | RunnableLambda(self._multimodal_prompt_function)
            | self.chat_llm
            | StrOutputParser()
        )
        
        retrieve_docs = (
            itemgetter("input")
            | RunnableLambda(lambda q: self.retriever.invoke(q, grader=grader))
            | RunnableLambda(self._split_image_text_types)
        )
                
        multimodal_rag = RunnablePassthrough.assign(context=retrieve_docs) \
            .assign(answer=multimodal_rag)
        
        return multimodal_rag