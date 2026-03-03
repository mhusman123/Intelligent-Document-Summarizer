from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


def load_keyword_model():
    """
    YAKE requires no model downloads.
    This is a no-op kept for startup consistency.
    """
    logger.info("Keyword extractor ready (YAKE — no model download required).")


def _extract_sync(text: str, top_n: int) -> list[dict]:
    """
    Runs YAKE keyword extraction synchronously.
    """
    import yake

    extractor = yake.KeywordExtractor(
        lan="en",
        n=2,           # max ngram size
        dedupLim=0.7,  # deduplication threshold
        top=top_n,
        features=None,
    )
    keywords = extractor.extract_keywords(text)
    # YAKE scores: lower = more relevant, so invert for display
    max_score = max((s for _, s in keywords), default=1.0)
    return [
        {"keyword": kw, "score": round(1.0 - (score / (max_score + 1e-9)), 4)}
        for kw, score in keywords
    ]


class KeywordService:
    """
    Extracts the most relevant keywords from a document using YAKE.
    No model downloads required — runs fully offline and instantly.
    """

    @staticmethod
    async def extract_keywords(text: str, top_n: int = None) -> list[dict]:
        if top_n is None:
            top_n = settings.KEYWORD_TOP_N

        try:
            results = await run_in_threadpool(_extract_sync, text, top_n)
            logger.info(f"Extracted {len(results)} keywords.")
            return results
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Keyword extraction failed: {str(e)}",
            )
