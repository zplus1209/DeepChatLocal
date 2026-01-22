import shutil
import tempfile
import platform
import subprocess
from typing import Any
from pathlib import Path

from loguru import logger

from parser.utils import md_with_anchor

class BaseParser:
    """
    Docstring for BaseParser
    """
    
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}
    
    logger = logger
    
    def parse_pdf(
        self,
        pdf_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str | None = None,
        lang: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Docstring for parse_pdf
        
        :param self: Description
        :param pdf_path: Description
        :type pdf_path: str | Path
        :param output_path: Description
        :type output_path: str | Path | None
        :param method: Description
        :type method: str | None
        :param lang: Description
        :type lang: str | None
        :param kwargs: Description
        :type kwargs: Any
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_office_doc(
        self,
        doc_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str | None = None,
        lang: str | None = None,
        **kwargs: Any,
    ):
        """
        Docstring for parse_office_doc
        
        :param self: Description
        :param doc_path: Description
        :type doc_path: str | Path
        :param output_path: Description
        :type output_path: str | Path | None
        :param method: Description
        :type method: str | None
        :param lang: Description
        :type lang: str | None
        :param kwargs: Description
        :type kwargs: Any
        """
        return NotImplementedError("parse_office_doc must be implemented by subclasses")

    def parse_image(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
        method: str | None = None,
        lang: str | None = None,
        **kwargs: Any,
    ):
        """
        Docstring for parse_image
        
        :param self: Description
        :param image_path: Description
        :type image_path: str | Path
        :param output_path: Description
        :type output_path: str | Path | None
        :param method: Description
        :type method: str | None
        :param lang: Description
        :type lang: str | None
        :param kwargs: Description
        :type kwargs: Any
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")
    
    @classmethod
    def convert_office_doc_2_pdf(
        cls, doc_path: str | Path, output_path: str | Path | None = None,
    ) -> Path:
        """
        Convert Office document (.doc, .docx, .ppt, .pptx, .xls, .xlsx) to PDF.
        Requires LibreOffice to be installed.
        
        Args:
            doc_path: Path to the Office document file
            output_path: Output directory for the PDF file
            
        Returns:
            Path to the generated PDF file
            
        Raises:
            FileNotFoundError: If the input document doesn't exist
            RuntimeError: If LibreOffice conversion fails
        """
        try:
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")
            
            if output_path:
                base_output_path = Path(output_path)
            else:
                base_output_path = Path("./data/pdfs") / "libreoffice_output"
            
            base_output_path.mkdir(parents=True, exist_ok=True)    
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                cls.logger.info(
                    f"Converting {doc_path.name} to PDF using LibreOffice..."
                )
                
                commands_to_try = ["libreoffice", "soffice"]
                
                conversion_successful = False
                for cmd in commands_to_try:
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]
                        
                        convert_subprocess_kwargs = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }
                        
                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                            
                        result = subprocess.run(convert_cmd, **convert_subprocess_kwargs)
                        
                        if result.returncode == 0:
                            conversion_successful = True
                            cls.logger.info(
                                f"Successfully converted {doc_path.name} to PDF using {cmd}"
                            )
                            break
                        else:
                            cls.logger.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except FileNotFoundError:
                        cls.logger.warning(f"LibreOffice command '{cmd}' not found")
                    except subprocess.TimeoutExpired:
                        cls.logger.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        cls.logger.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )
                
                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}. "
                        f"Please ensure LibreOffice is installed:\n"
                        "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                        "- macOS: brew install --cask libreoffice\n"
                        "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                        "- CentOS/RHEL: sudo yum install libreoffice\n"
                        "Alternatively, convert the document to PDF manually."
                    )
                    
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        "Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                
                if pdf_path.stat().st_size < 100:
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )
                    
                final_pdf_path = base_output_path / f"{doc_path.stem}.pdf"
                shutil.copy2(pdf_path, final_pdf_path)
                
                return final_pdf_path
            
        except Exception as e:
            cls.logger.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise
        
    @staticmethod
    def _md_with_anchor(text: str):  
        return md_with_anchor(text)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BaseParser")
    parser.add_argument("doc_path", type=str, help="Input doc path")
    args = parser.parse_args()
    
    BaseParser.convert_office_doc_2_pdf(
        args.doc_path
    )