import io
from pathlib import Path
from fastapi import HTTPException, status

from app.core.logging import setup_logger

logger = setup_logger(__name__)


class FileParserService:
    """
    Handles parsing of uploaded documents (PDF, DOCX, TXT)
    and returns clean plain text content.
    """

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower().lstrip(".")

        parsers = {
            "pdf": FileParserService._parse_pdf,
            "docx": FileParserService._parse_docx,
            "txt": FileParserService._parse_txt,
        }

        if ext not in parsers:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: '.{ext}'. Allowed: pdf, docx, txt",
            )

        logger.info(f"Parsing file: {filename} | type: {ext}")
        text = parsers[ext](file_bytes)

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The document appears to be empty or contains no readable text.",
            )

        return text.strip()

    # ──────────────────────────────────────────────
    # Private Parsers
    # ──────────────────────────────────────────────

    @staticmethod
    def _parse_pdf(file_bytes: bytes) -> str:
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n".join(text_parts)

        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF parsing library (pdfplumber) is not installed.",
            )
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse PDF: {str(e)}",
            )

    @staticmethod
    def _parse_docx(file_bytes: bytes) -> str:
        try:
            from docx import Document

            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(paragraphs)

        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DOCX parsing library (python-docx) is not installed.",
            )
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse DOCX: {str(e)}",
            )

    @staticmethod
    def _parse_txt(file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                logger.error(f"TXT decoding failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to decode text file. Ensure it is UTF-8 or Latin-1 encoded.",
                )
