from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    detailed = "detailed"


class SupportedLanguage(str, Enum):
    english = "en"
    french = "fr"
    german = "de"
    spanish = "es"


# ──────────────────────────────────────────────
# Request Schemas
# ──────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    summary_length: SummaryLength = Field(
        default=SummaryLength.medium,
        description="Desired summary length: short, medium, or detailed",
    )
    language: SupportedLanguage = Field(
        default=SupportedLanguage.english,
        description="Language of the document",
    )


class QARequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Question to answer based on the document content",
        example="What is the main topic of this document?",
    )


# ──────────────────────────────────────────────
# Response Schemas
# ──────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    filename: str               = Field(..., description="Name of the uploaded file")
    file_type: str              = Field(..., description="Type of the uploaded file (pdf/docx/txt)")
    summary_length: str         = Field(..., description="Requested summary length mode")
    engine: str                 = Field(..., description="Summarization engine used: 'groq' or 'ensemble'")

    # Content
    summary: str                = Field(..., description="AI-generated summary")
    key_points: List[str]       = Field(..., description="Top key bullet points from the document")

    # Word counts
    original_word_count: int    = Field(..., description="Word count of the original document")
    summary_word_count: int     = Field(..., description="Word count of the generated summary")
    compression_ratio: float    = Field(..., description="Ratio of summary to original")

    # Document analytics
    sentence_count: int         = Field(..., description="Total sentences in the document")
    paragraph_count: int        = Field(..., description="Total paragraphs in the document")
    reading_time_minutes: int   = Field(..., description="Estimated reading time in minutes")
    reading_level: str          = Field(..., description="Flesch reading level label")
    flesch_score: float         = Field(..., description="Flesch Reading Ease score (0-100)")


class KeywordResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    keywords: List[str] = Field(..., description="Top extracted keywords from the document")
    keyword_scores: List[dict] = Field(..., description="Keywords with their relevance scores")


class QAResponse(BaseModel):
    filename: str            = Field(..., description="Name of the uploaded file")
    question: str            = Field(..., description="The question asked")
    answer: str              = Field(..., description="Clean multi-sentence answer from the document")
    confidence: float        = Field(..., description="Confidence score of the answer (0 to 1)")
    context_snippet: str     = Field(..., description="Best matching sentence used as primary evidence")
    source_sentences: List[str] = Field(default=[], description="Top supporting sentences from the document")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    app_name: str
    version: str


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Internal error code")
