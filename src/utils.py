import io
import os
import base64
import getpass
from PIL import Image
from hashlib import md5
from typing import List, Dict, Any

from loguru import logger
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def init_embeddings(
    model: str = "",
    model_provider: str = "ollama",
    **kwargs: Any
):
    if model_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=model, **kwargs
        )
        
    elif model_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        embeddings = OllamaEmbeddings(
            model=model, **kwargs
        )
        
    else:
        raise ValueError(f"Embeddings {model_provider.upper()} is not supported.")

def category_elements(raw_data: List[Dict]):
    markdown_document = raw_data.get("markdown", "")
    content_list = raw_data.get("chunks", [])
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
        
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=4000, chunk_overlap=800
    )
        
    splits = text_splitter.split_documents(md_header_splits)
    text_elements = [{
        "markdown": doc.page_content, "grounding": doc.metadata,
    } for doc in splits]
        
    text_parts, table_elements, image_elements, equation_elements = [], [], [], []
        
    for item in content_list:
        item_type = item.get("type")
            
        if item_type == "table":
            table_elements.append(item)
                
        elif item_type == "fomular":
            equation_elements.append(item)
                
        elif item_type in ["image", "seal", "chart"]:
            image_elements.append(item)
                
        else:
            text_parts.append(item)
                
    logger.info(f"📊 Categorized elements:")
    logger.info(f"  Text parts: {len(text_parts)}")
    logger.info(f"  Text elements: {len(text_elements)}")
    logger.info(f"  Table elements: {len(table_elements)}")
    logger.info(f"  Image elements: {len(image_elements)}")
    logger.info(f"  Equation elements: {len(equation_elements)}")
        
    return (
        text_parts, text_elements, 
        table_elements, image_elements, equation_elements
    )
    
def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def resize_base64_image(
    base64_string: str | None = None,
    image_path: str | None = None,
    size=(128, 128),
):
    """
    Resize an image encoded as a Base64 string
    """
    if image_path:
        base64_string = encode_image(image_path)

    if base64_string is None:
        raise ValueError("Either base64_string or image_path must be provided")
    
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling.

    Args:
        *args: Arguments to hash
    Returns:
        str: Hash string
    """
    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])

    # Use 'replace' error handling to safely encode problematic Unicode characters
    # This replaces invalid characters with Unicode replacement character (U+FFFD)
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)
