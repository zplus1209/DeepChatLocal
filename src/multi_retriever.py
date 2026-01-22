import json
import uuid
from typing import Any, Union, List, Dict

from loguru import logger
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.documents import Document
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

from src.grader import grader_documents
from src.utils import init_embeddings, compute_mdhash_id

class MultiRetriever:
    
    def __init__(
        self, 
        config = None,
        embeddings = None,
        vectorstore = None,
        docstore = None,
        **kwargs
    ):
        
        self.config = config
        self.id_key = "doc_id"
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.docstore = docstore
        
        try:
            if self.embeddings is None:
                self.embeddings = init_embeddings(
                    model="",
                    model_provider="ollama",
                    **kwargs
                )
            
            if self.vectorstore is None:
                self.vectorstore = Chroma(
                    collection_name="multimodal_summaries",
                    embedding_function=self.embeddings,
                    persist_directory="./chroma_db"  # Persist to disk
                )
                
                # self.vectorstore = QdrantClient(
                #     self.embeddings,
                #     # url=None,
                #     # prefer_grpc=True,
                #     # api_key=None,
                #     path="/tmp/langchain_qdrant"
                #     collection_name="multimodal_summaries",
                # )
                
            if self.docstore is None:
                self.docstore = InMemoryStore()
                
            self.multi_retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                id_key=self.id_key,
                search_kwargs={"k": 4}
            )
            
            logger.info("Initialize MultiVectorRetriever")
        
        except Exception as e:
            logger.error(f"Failed to initialize MultiVectorRetriever: {str(e)}")
            raise
        
    @property
    def as_retriever(self):
        return self.multi_retriever
    
    def add_documents(
        self,
        summaries: Union[List[str], List[Dict]],
        original_contents: Union[List[str], List[Dict]],
        content_type: str
    ):
        
        if not summaries or not original_contents:
            logger.error(f"No {content_type} content to add")
            return
            
        if len(summaries) != len(original_contents):
            raise ValueError(f"Mismatch in {content_type} summaries and original content lengths")
        
        try:
            summary_docs = []
            doc_ids = []
            # doc_ids = [str(uuid.uuid4()) for _ in original_contents]
            
            for idx, (summary, original_content) in enumerate(zip(summaries, original_contents)):
                if isinstance(summary, Dict):
                    summary = summary.get("entity_info", {}).get("summary", "")
                    
                if isinstance(original_content, Dict):
                    original_content = original_content.get("markdown", "")
                    
                doc_id = compute_mdhash_id(summary, prefix=f"{content_type}-")
                doc_ids.append(doc_id)
                
                summary_doc = Document(
                    page_content=summary,
                    metadata = {
                        self.id_key: doc_id,
                        "content_type": content_type,
                        "index": idx,
                    }
                )
                summary_docs.append(summary_doc)
            
            if not summary_docs:
                logger.info()
                return
            
            self.vectorstore.add_documents(documents=summary_docs)
            
            doc_pairs = list(zip(doc_ids, original_contents))
            self.docstore.mset(doc_pairs)
            
            logger.info(
                f"Added {len(original_contents)}/{len(summaries)} "
                f"{content_type} documents to retriever"
            )
            
        except Exception as e:
            logger.error(f"Error adding {content_type} documents: {str(e)}")
            raise
        
    def add_all_documents(
        self,
        text_summaries: Union[List[str], List[Dict]], text_elements: Union[List[str], List[Dict]],
        table_summaries: Union[List[str], List[Dict]], table_elements: Union[List[str], List[Dict]],
        image_summaries: Union[List[str], List[Dict]], image_elements: Union[List[str], List[Dict]],
        # equation_summaries: Union[List[str], List[Dict]], equation_elements: Union[List[str], List[Dict]]
    ):
        logger.info("Adding all content to retriever...")
        
        if text_summaries and text_elements:
            self.add_documents(text_summaries, text_elements, "text")
            
        if table_summaries and table_elements:        
            self.add_documents(table_summaries, table_elements, "table")

        if image_summaries and image_elements:        
            self.add_documents(image_summaries, image_elements, "image")
            
        # if equation_summaries and equation_elements:        
        #     self.add_documents(equation_summaries, equation_elements, "equation")
        
        logger.info("All content added to retriever successfully")
        
    def invoke(
        self, query: str, grader: bool = True, k: int = 4
    ):
        try:
            results = []
            responses = self.multi_retriever.invoke(query)
            
            if grader:
                retrieval_grader = grader_documents()
            
                for response in responses[:k]:
                    res = retrieval_grader.invoke(
                        {"question": query, "document": response.get("markdown")}
                    )
                    if res.binary_score == "yes":
                        results.append(response)
                        
                return results
            else:
                return responses[:k]
        
        except Exception as e:
            logger.error(f"Error during invoke: {str(e)}")
            return [f"Invoke error: {str(e)}"]
        
    async def ainvoke(self, query: str, k: int = 4):
        try:
            results = await self.multi_retriever.ainvoke(query)
            
            return results[:k]
        
        except Exception as e:
            logger.error(f"Error during invoke: {str(e)}")
            return [f"Invoke error: {str(e)}"]

async def main():
    from parser.paddleocr_parser import PaddleOCRParser
    from src.utils import category_elements
    from rich import print
    from src.summarizer import ContentSummarizer

    ADDITIONAL_INSTRUCTIONS = """
    - Do NOT guess or invent code_no, version, or valid_from
    - If the information is not explicitly provided, set them to null
    - Never infer metadata from context
    """
    
    paddelocr_parser = PaddleOCRParser()
    
    output = paddelocr_parser.parse_document(
        "/media/mountHDD2/duong/git/DeepChatLocal/data/docs/HVNG_HuongDanSinhVienXemThoiKhoaBieu.docx",
        vl_rec_backend="vllm-server", 
        vl_rec_server_url="http://127.0.0.1:8118/v1",
        # method="ppstructurev3"
    )
    
    text_parts, text_elements, table_elements, image_elements, equation_elements = category_elements(output)
    
    summarizer = ContentSummarizer()
    multi_retriever = MultiRetriever()
    
    text_summaries, final_summary = await summarizer.asummarize_documents(
        text_elements=text_elements,
        detail=1,
        summarize_recursively=True,
        additional_instructions=ADDITIONAL_INSTRUCTIONS,
        verbose=True
    )

    table_summaries = await summarizer.asummarize_table_elements(
        table_elements, 
        additional_instructions=ADDITIONAL_INSTRUCTIONS,
        verbose=True
    )
    
    image_summaries = await summarizer.asummarize_image_elements(
        image_elements, 
        additional_instructions=ADDITIONAL_INSTRUCTIONS,
        verbose=True
    )
    
    multi_retriever.add_all_documents(
        text_summaries, text_elements,
        table_summaries, table_elements,
        image_summaries, image_elements
    )
    
    query = "Muốn xem thời khóa biểu cần truy cập vào đâu?"
    
    retrieved_docs = await multi_retriever.ainvoke(query)
    print(retrieved_docs)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())