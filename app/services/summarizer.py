from __future__ import annotations
from collections import defaultdict
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.logging import setup_logger
from app.models.schemas import SummaryLength

logger = setup_logger(__name__)


# ──────────────────────────────────────────────
# Sentence count per summary length
# ──────────────────────────────────────────────

LENGTH_SENTENCES = {
    SummaryLength.short:    3,
    SummaryLength.medium:   7,
    SummaryLength.detailed: 15,
}

KEY_POINTS_COUNT = {
    SummaryLength.short:    3,
    SummaryLength.medium:   5,
    SummaryLength.detailed: 8,
}


def load_summarizer():
    """
    Downloads NLTK data (tiny, <2MB, one-time).
    Called once at application startup.
    """
    import nltk
    logger.info("Initializing ensemble summarizer (no large model downloads).")
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)
    logger.info("Summarizer ready (Ensemble: LSA + LexRank + Luhn + TextRank).")


# ──────────────────────────────────────────────
# Ensemble core
# ──────────────────────────────────────────────

def _run_algorithm(algo_cls, parser, stemmer, count: int) -> list[str]:
    """Run a single sumy algorithm and return its selected sentence strings."""
    from sumy.utils import get_stop_words
    try:
        s = algo_cls(stemmer)
        s.stop_words = get_stop_words("english")
        return [str(sent) for sent in s(parser.document, count)]
    except Exception as e:
        logger.warning(f"Algorithm {algo_cls.__name__} failed: {e}")
        return []


def _jaccard_similarity(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity to detect near-duplicate sentences."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(sentences: list[str], threshold: float = 0.55) -> list[str]:
    """Remove sentences that are too similar to already-kept ones."""
    kept = []
    for sent in sentences:
        if all(_jaccard_similarity(sent, k) < threshold for k in kept):
            kept.append(sent)
    return kept


def _ensemble_summarize(text: str, sentence_count: int) -> dict:
    """
    Ensemble summarization:
      1. Run LSA, LexRank, Luhn, TextRank independently.
      2. Score each unique sentence by how many algorithms selected it
         + a position bonus (earlier sentences slightly preferred).
      3. Pick top-N by score (deduplicated).
      4. Also derive key_points: top distinct bullet sentences.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")

    # Collect all sentences in original order
    all_sentences = [str(s) for s in parser.document.sentences]
    total = len(all_sentences)
    if total == 0:
        return {"summary": "", "key_points": []}

    # Request more candidates than needed so we have room to pick
    candidates = min(total, max(sentence_count + 4, int(total * 0.6)))

    # Run all 4 algorithms
    algos = [LsaSummarizer, LexRankSummarizer, LuhnSummarizer, TextRankSummarizer]
    scores: dict[str, float] = defaultdict(float)

    for algo in algos:
        selected = _run_algorithm(algo, parser, stemmer, candidates)
        for sent in selected:
            scores[sent] += 1.0   # +1 vote per algorithm

    # Position bonus: sentences in first 20% and last 10% of the doc get a boost
    for i, sent in enumerate(all_sentences):
        position_ratio = i / total
        if position_ratio <= 0.20:                    # opening sentences
            scores[sent] += 0.6
        elif position_ratio >= 0.90:                  # closing sentences
            scores[sent] += 0.3

    # Length bonus: prefer medium-length sentences (not too short, not too long)
    for sent in scores:
        word_count = len(sent.split())
        if 10 <= word_count <= 40:
            scores[sent] += 0.4
        elif word_count < 6:
            scores[sent] -= 1.0   # penalise very short sentences

    # Sort by score descending, preserving original document order as tiebreaker
    order = {s: i for i, s in enumerate(all_sentences)}
    ranked = sorted(scores.keys(), key=lambda s: (-scores[s], order.get(s, 9999)))

    # Deduplicate, then take top-N
    deduplicated = _deduplicate(ranked, threshold=0.50)
    top_sentences = deduplicated[:sentence_count]

    # Re-order selected sentences to match original document flow
    top_sentences.sort(key=lambda s: order.get(s, 9999))

    summary = " ".join(top_sentences)

    # Key points: additional unique sentences NOT already in summary (≥3 words)
    key_point_pool = [
        s for s in deduplicated
        if s not in top_sentences and len(s.split()) >= 6
    ]
    key_points = key_point_pool[:KEY_POINTS_COUNT.get(SummaryLength.medium, 5)]

    return {"summary": summary, "key_points": key_points}


# ──────────────────────────────────────────────
# Document analytics helper
# ──────────────────────────────────────────────

def _compute_analytics(text: str) -> dict:
    """
    Computes document-level statistics:
    - sentence count, paragraph count
    - estimated reading time
    - Flesch reading ease score + label
    """
    import re

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences  = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences  = [s for s in sentences if s.strip()]
    words      = text.split()

    word_count      = len(words)
    sentence_count  = max(len(sentences), 1)
    paragraph_count = max(len(paragraphs), 1)
    reading_time    = max(1, round(word_count / 200))  # ~200 wpm average reader

    # Flesch Reading Ease (approximation without syllable counter)
    avg_sentence_len = word_count / sentence_count
    if avg_sentence_len < 10:
        flesch, level = 90.0, "Very Easy"
    elif avg_sentence_len < 14:
        flesch, level = 75.0, "Easy"
    elif avg_sentence_len < 18:
        flesch, level = 60.0, "Standard"
    elif avg_sentence_len < 24:
        flesch, level = 45.0, "Fairly Difficult"
    else:
        flesch, level = 30.0, "Difficult"

    return {
        "sentence_count":  sentence_count,
        "paragraph_count": paragraph_count,
        "reading_time_minutes": reading_time,
        "reading_level": level,
        "flesch_score": flesch,
    }


# ──────────────────────────────────────────────
# Public service
# ──────────────────────────────────────────────

class SummarizerService:
    """
    Summarization with automatic engine selection (priority: Groq → Ensemble):
      • If GROQ_API_KEY is set   → LLaMA 3.3 70B via Groq (best free quality)
      • Otherwise                → Offline ensemble (LSA + LexRank + Luhn + TextRank)
    Any API failure falls back silently so the API never goes down.
    """

    @staticmethod
    async def summarize(text: str, summary_length: SummaryLength = SummaryLength.medium) -> dict:
        from app.core.config import get_settings
        settings = get_settings()

        sentence_count = LENGTH_SENTENCES[summary_length]
        result = None
        engine = "ensemble"

        # ── Try Groq first ─────────────────────────────────────────────────
        use_groq = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
        if use_groq:
            logger.info(f"Groq summarizing | length={summary_length} | model={settings.GROQ_MODEL}")
            try:
                from app.services.groq_summarizer import _groq_summarize_sync
                result = await run_in_threadpool(_groq_summarize_sync, text, summary_length)
                engine = "groq"
            except Exception as e:
                logger.warning(f"Groq failed ({e}) — falling back to ensemble.")
                result = None

        # ── Ensemble fallback (or primary when no key) ─────────────────────────
        if result is None:
            logger.info(f"Ensemble summarizing | length={summary_length} | target={sentence_count} sentences")
            try:
                result = await run_in_threadpool(_ensemble_summarize, text, sentence_count)
                engine = "ensemble"
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Summarization failed: {str(e)}",
                )

        if not result.get("summary", "").strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document is too short or contains no readable text to summarize.",
            )

        analytics = await run_in_threadpool(_compute_analytics, text)

        logger.info(
            f"Summarization complete | engine={engine} | key_points={len(result['key_points'])} | "
            f"reading_time={analytics['reading_time_minutes']}min"
        )
        return {**result, **analytics, "engine": engine}
