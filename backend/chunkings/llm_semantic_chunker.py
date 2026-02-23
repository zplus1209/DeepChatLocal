from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from abc import ABC, abstractmethod
from typing import List

from backend.llms import LocalLLMs


# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass


class LLMSemanticChunker(BaseChunker):

    def __init__(self, engine:str="ollama", api_key:str=None, model_name:str=None):

        if engine == "ollama":
            self.llm = LocalLLMs(
                engine=engine,
                model_version="qwen3-vl:30b",
            )

        else:
            raise ValueError("Invalid")
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
        )
    
    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk)+"." + (f"\nThe previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages
    
    def split_text(self, text):
        import re

        chunks = self.splitter.split_text(text)

        split_indices = []

        short_cut = len(split_indices) > 0

        from tqdm import tqdm

        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0

                chunked_input = ''

                for i in range(current_chunk, len(chunks)):
                    token_count += openai_token_count(chunks[i])
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.llm.generate_content(messages)
                    # Use regular expression to find all numbers in the string
                    split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                    numbers = re.findall(r'\d+', split_after_line)
                    # Convert the found numbers to integers
                    numbers = list(map(int, numbers))

                    # print(numbers)

                    # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                    if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                        break
                    else:
                        messages = self.get_prompt(chunked_input, current_chunk, numbers)
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)

                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs
    
if __name__ == "__main__":
    import os
    import sys
    
    txt_dir = "/media/mountHDD2/duong/git/eval/data/txts"

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]

    if not txt_files:
        print("Không tìm thấy file .txt nào trong thư mục.")
        sys.exit(1)

    chunker = LLMSemanticChunker(engine="ollama")

    for filename in txt_files:
        filepath = os.path.join(txt_dir, filename)
        print(f"\n{'='*60}")
        print(f"📄 Đang xử lý file: {filename}")
        print(f"{'='*60}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Độ dài văn bản: {len(text)} ký tự")

        chunks = chunker.split_text(text)

        print(f"\n✅ Tổng số chunk sau khi split: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)        
