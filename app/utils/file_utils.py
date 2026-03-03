import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


async def save_upload_file(upload_file: UploadFile) -> tuple[str, bytes]:
    """
    Validates and reads an uploaded file.
    Returns (filename, file_bytes).
    """
    # Validate extension
    ext = Path(upload_file.filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '.{ext}' is not allowed. Accepted: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Read bytes
    file_bytes = await upload_file.read()

    # Validate size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit.",
        )

    logger.info(f"File received: {upload_file.filename} | size: {len(file_bytes)} bytes")
    return upload_file.filename, file_bytes


def ensure_upload_dir() -> str:
    """
    Ensures the upload directory exists and returns its path.
    """
    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    return str(upload_path)


def chunk_text(text: str, max_tokens: int = 1024) -> list[str]:
    """
    Splits long text into manageable chunks for model inference.
    Uses word-level splitting to stay under token limits.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > max_tokens * 4:  # ~4 chars per token
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
