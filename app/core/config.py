from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Intelligent Document Summarizer"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "AI-powered API to summarize documents, extract keywords, and answer questions."
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"

    # File Upload
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt"]
    UPLOAD_DIR: str = "uploads"

    # Summarization Model (lightweight & fast)
    SUMMARIZER_MODEL: str = "sshleifer/distilbart-cnn-6-6"
    SUMMARIZER_MIN_LENGTH: int = 50
    SUMMARIZER_MAX_LENGTH: int = 300

    # QA Model (lightweight & fast)
    QA_MODEL: str = "deepset/minilm-uncased-squad2"

    # Keyword Extraction
    KEYWORD_TOP_N: int = 10

    # Groq (LLaMA 3.3 70B — free, fast, high quality)
    GROQ_API_KEY: str = ""             # Set this in .env to enable Groq
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_TOKENS: int = 1500
    GROQ_TEMPERATURE: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
