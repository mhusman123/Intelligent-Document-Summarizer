from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    SummarizeResponse,
    KeywordResponse,
    QAResponse,
    SummaryLength,
    QARequest,
)
from app.services.file_parser import FileParserService
from app.services.summarizer import SummarizerService
from app.services.keyword_extractor import KeywordService
from app.services.qa_service import QAService
from app.utils.file_utils import save_upload_file
from app.core.logging import setup_logger

logger = setup_logger(__name__)
router = APIRouter(tags=["Document Summarizer"])


# ──────────────────────────────────────────────
# POST /summarize
# ──────────────────────────────────────────────

@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize a Document",
    description=(
        "Upload a PDF, DOCX, or TXT file and receive an AI-generated summary. "
        "You can control the summary length (short / medium / detailed)."
    ),
)
async def summarize_document(
    file: UploadFile = File(..., description="Document to summarize (PDF, DOCX, TXT)"),
    summary_length: SummaryLength = Form(
        default=SummaryLength.medium,
        description="Desired summary length",
    ),
):
    filename, file_bytes = await save_upload_file(file)
    original_text = FileParserService.parse(file_bytes, filename)
    result = await SummarizerService.summarize(original_text, summary_length)

    original_wc = len(original_text.split())
    summary_wc  = len(result["summary"].split())
    compression = round(summary_wc / original_wc, 4) if original_wc > 0 else 0.0

    logger.info(
        f"Summary generated | original: {original_wc} words | "
        f"summary: {summary_wc} words | key_points: {len(result['key_points'])}"
    )

    return SummarizeResponse(
        filename=filename,
        file_type=filename.rsplit(".", 1)[-1].lower(),
        original_word_count=original_wc,
        summary_word_count=summary_wc,
        summary_length=summary_length.value,
        engine=result["engine"],
        summary=result["summary"],
        key_points=result["key_points"],
        compression_ratio=compression,
        sentence_count=result["sentence_count"],
        paragraph_count=result["paragraph_count"],
        reading_time_minutes=result["reading_time_minutes"],
        reading_level=result["reading_level"],
        flesch_score=result["flesch_score"],
    )


# ──────────────────────────────────────────────
# POST /keywords
# ──────────────────────────────────────────────

@router.post(
    "/keywords",
    response_model=KeywordResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Keywords from a Document",
    description=(
        "Upload a PDF, DOCX, or TXT file and receive the top relevant "
        "keywords and keyphrases extracted using KeyBERT."
    ),
)
async def extract_keywords(
    file: UploadFile = File(..., description="Document to extract keywords from"),
):
    filename, file_bytes = await save_upload_file(file)
    original_text = FileParserService.parse(file_bytes, filename)
    keyword_data = await KeywordService.extract_keywords(original_text)

    keywords_list = [item["keyword"] for item in keyword_data]

    logger.info(f"Keywords extracted from: {filename}")

    return KeywordResponse(
        filename=filename,
        keywords=keywords_list,
        keyword_scores=keyword_data,
    )


# ──────────────────────────────────────────────
# POST /qa
# ──────────────────────────────────────────────

@router.post(
    "/qa",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a Question about a Document",
    description=(
        "Upload a PDF, DOCX, or TXT file and ask a natural language question. "
        "The API will find and return the most relevant answer from the document."
    ),
)
async def question_answering(
    file: UploadFile = File(..., description="Document to query"),
    question: str = Form(
        ...,
        min_length=5,
        max_length=500,
        description="Your question about the document",
        example="What is the main conclusion of this document?",
    ),
):
    filename, file_bytes = await save_upload_file(file)
    original_text = FileParserService.parse(file_bytes, filename)
    result = await QAService.answer(original_text, question)

    logger.info(f"Q&A completed for: {filename} | question: '{question}'")

    return QAResponse(
        filename=filename,
        question=question,
        answer=result["answer"],
        confidence=result["confidence"],
        context_snippet=result["context_snippet"],
        source_sentences=result.get("source_sentences", []),
    )
