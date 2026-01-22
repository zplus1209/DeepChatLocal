import sys
import time
import json
import uuid
import requests
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Tuple, Dict

from paddleocr import PPStructureV3, PaddleOCRVL

from parser.base_parser import BaseParser


class PaddleOCRParser(BaseParser):
    
    def __init__(self):
        super().__init__()
        pass
    
    def parse_document(
        self,
        file_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str = "paddleocrvl",
        lang: str | None = None,
        **kwargs: Any,
    ) -> Dict:
    
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self.parse_pdf(
                file_path, output_path=output_path, method=method, lang=lang, **kwargs
            )
        
        elif ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(
                file_path, output_path=output_path, method=method, lang=lang, **kwargs
            )
            
        elif ext in self.IMAGE_FORMATS:
            return self.parse_pdf(
                file_path, output_path=output_path, method=method, lang=lang, **kwargs
            )
            
        else:
            self.logger.warning(
                f"Warning: Unsupported file extension '{ext}', "
                f"attempting to parse as PDF"
            )
            return self.parse_pdf(
                file_path, output_path=output_path, method=method, lang=lang, **kwargs
            )
        
    def parse_pdf(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str = "paddleocrvl",
        lang: str | None = None,
        **kwargs: Any,
    ) -> Dict:
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"File pdf does not exist: {str(pdf_path)}")

            file_stem = pdf_path.stem
            
            if output_path:
                base_output_path = Path(output_path)
            else:
                base_output_path = Path("./output") / method / file_stem
                
            base_output_path.mkdir(parents=True, exist_ok=True)            
            
            md_file = base_output_path / f"{file_stem}.md"
            json_file = base_output_path / f"{file_stem}_content_list.json"
        
            if not md_file.exists() or not json_file.exists():
                self.logger.debug(f"Analysing document...")
                self._run_paddle_ocr(
                    str(pdf_path),
                    output_path=base_output_path,
                    method=method,
                    lang=lang,
                    **kwargs
                )
            
            output = self._read_output_files(
                base_output_path, file_stem, method
            )
            
            return output
        
        except Exception as e:
            self.logger.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_office_doc(
        self,
        doc_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str = "paddleocrvl",
        lang: str | None = None,
        **kwargs: Any,
    ) -> Dict:
        try:
            pdf_path = self.convert_office_doc_2_pdf(doc_path, output_path)
            
            return self.parse_pdf(
                pdf_path,
                output_path=output_path,
                method=method,
                lang=lang,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error in parse_office_doc: {str(e)}")
            raise
        
    def _read_output_files(
        self,
        output_path: str | Path,
        file_stem: str ,
        method: str = "paddleocrvl",
    ) -> Dict:
        
        md_file = Path(output_path) / f"{file_stem}.md"
        json_file = Path(output_path) / f"{file_stem}_content_list.json"
    
        file_stem_subdir = Path(output_path) / method / file_stem
        if file_stem_subdir.exists():
            md_file = Path(file_stem_subdir) / f"{file_stem}.md"
            json_file = Path(file_stem_subdir) / f"{file_stem}_content_list.json"
        
        if not md_file.exists() and json_file.exists():
            raise ValueError(f"Do not exist markdown file and json file")
        
        md_content = ""
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()
                self.logger.info(f"Read from markdown file: {str(md_file)}")
        except Exception as e:
            self.logger.error(f"Could not read markdown file {str(md_file)}: {str(e)}")  
               
        content_list = []
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content_list = json.load(f)    
                self.logger.info(f"Read from json file: {str(json_file)}")
        except Exception as e:
            self.logger.error(f"Could not read json file {str(json_file)}: {str(e)}")  
            
        return {
            "markdown": md_content,
            "chunks": content_list
        }   
    
    def _run_paddle_ocr(
        self,
        input_file: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str | None = "paddleocrvl",
        lang: str | None = None,
        **kwargs: Any,
    ) -> None:
        
        file_stem = Path(input_file).stem
        
        if output_path:
            base_output_path = Path(output_path)
        else:
            base_output_path = Path("./output") / method / file_stem
            
        base_output_path.mkdir(parents=True, exist_ok=True)            
        
        if method == "paddleocrvl":
            server_process = None
            
            if kwargs.get("vl_rec_backend") and kwargs.get("vl_rec_server_url"):
                vl_rec_backend = kwargs.get("vl_rec_backend")
                if vl_rec_backend == "vllm-server":
                    backend = "vllm"
                elif vl_rec_backend == "sglang-server":
                    backend = "sglang"
                    
                server_process = self._inference_with_vllm(
                    kwargs.get("vl_rec_server_url"), backend
                )
            
            try:
                pipeline = PaddleOCRVL(**kwargs)
                self.logger.info(f"Initializing parsing {method.upper()}")
            except Exception as e:
                self.logger.error(f"Error parsing: {str(e)}")
                
            results = pipeline.predict(input=input_file)

            if server_process:
                server_process.terminate()
            
        elif method == "ppstructurev3":
            if kwargs.get("vl_rec_backend") and kwargs.get("vl_rec_server_url"):
                kwargs.pop("vl_rec_backend")
                kwargs.pop("vl_rec_server_url")
            
            if kwargs.get("text_recognition_model_name", None):
                lang = None
            
            try:
                pipeline = PPStructureV3(lang=lang, **kwargs)
                self.logger.info(f"Initializing parsing {method.upper()}")
            except Exception as e:
                self.logger.error(f"Error parsing: {str(e)}")
                
            results = pipeline.predict(input=input_file)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        markdown_list = []
        json_list = []
        bbox_to_image_path = {}
        
        for result in results:
            md_info = result.markdown
            markdown_list.append(md_info)
            
            # Save image path
            markdown_images = md_info.get("markdown_images", {})
            for path, image in markdown_images.items():
                bbox_number = path.rstrip(".jpg").split("_")[-4:]
                bbox_str = "_".join(bbox_number)
                
                image_file_path = base_output_path / path
                image_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                absolute_image_file_path = str(image_file_path.resolve())
                image.save(absolute_image_file_path)
                bbox_to_image_path[bbox_str] = absolute_image_file_path
            
            # Content list    
            json_info = result.json.get("res")
            
            page_index = json_info.get("page_index", None)
            parsing_res_list = json_info.get("parsing_res_list", [])
            for item in parsing_res_list:
                item_type = item.get("block_label", "")
                content = item.get("block_content", "")
                bbox = item.get("block_bbox", "")
                bbox_str = "_".join(map(str, bbox))
        
                id, text = self._md_with_anchor(content)
                
                json_list.append({
                    "markdown": text,
                    "type": item_type,
                    "id": id,
                    "grounding": {
                        "bbox": {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
                        "image_path": bbox_to_image_path.get(bbox_str, ""),
                        "page": page_index
                    }
                })
        
        markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
        mkd_file_path = base_output_path / f"{file_stem}.md"
        mkd_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            markdown_texts.save_to_markdown(str(mkd_file_path))
        except Exception as e:
            self.logger.warning(f"Warning save mardown file: {str(e)}")
            with open(mkd_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_texts)
        self.logger.info(f"Save markdown file: {mkd_file_path}")

        json_file_path = base_output_path / f"{file_stem}_content_list.json"
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Save json file: {mkd_file_path}")    
        
    def _inference_with_vllm(self, server_url: str, backend: str):
        parsed = urlparse(server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port

        cmd = [
            "paddleocr", "genai_server",
            "--model_name", "PaddleOCR-VL-0.9B",
            "--backend", backend,
            "--port", str(port),
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            raise ValueError(f"Error launching {backend.upper()} inference service: {str(e)}")

        try:
            self._wait_vllm_ready(host, port)
        except Exception as e:
            process.terminate()
            process.wait(timeout=5)
            raise RuntimeError(
                f"{backend.upper()} server failed to become ready: {str(e)}"
            )

        return process

    @staticmethod
    def _wait_vllm_ready(
        host: str, port: int, timeout: int = 200
    ) -> None:
        if host == "0.0.0.0":
            host = "localhost"

        url = f"http://{host}:{port}/health"
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    return
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            time.sleep(1)

        raise TimeoutError(f"GenAI server not ready at {url} after {timeout}s")
    
if __name__ == "__main__":
    paddelocr_parser = PaddleOCRParser()
    
    output = paddelocr_parser.parse_document(
        "/media/mountHDD2/duong/git/DeepChatLocal/data/pdfs/TLQT_Nhap hang duong thuy.pdf",
        vl_rec_backend="vllm-server", 
        vl_rec_server_url="http://127.0.0.1:8118/v1",
        method="ppstructurev3"
    )
    
    print(output)