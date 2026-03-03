from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.core.logging import setup_logger

logger = setup_logger(__name__)

# ── Stop-words for keyword extraction ──────────────────────────────────────────
_STOP = {
    "what","is","the","a","an","of","in","on","at","to","for","by","with","this","that",
    "it","its","as","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","about","how","why","who",
    "when","where","which","not","can","if","and","or","but","so","yet","because","since",
    "after","before","during","from","into","through","over","under","between","among",
    "i","me","my","we","our","you","your","he","she","they","their","us","am","any","all",
    "more","some","such","than","then","these","those","there","here","just","also","only",
    "very","too","no","yes","up","out","use","used","using","each","give","gives","given",
    "describe","explain","tell","mention","state","list","define","identify","please","can",
}


def load_qa_pipeline():
    """No-op — TF-IDF QA requires no model downloads."""
    logger.info("QA service ready (TF-IDF cosine similarity — no model download required).")


def _extract_keywords(text: str) -> set:
    """Extract meaningful content keywords from text."""
    import re
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\-\']{2,}\b', text.lower())
    return {w for w in words if w not in _STOP}


def _jaccard(s1: str, s2: str) -> float:
    """Jaccard similarity between two strings (word-level)."""
    a, b = set(s1.lower().split()), set(s2.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _answer_sync(text: str, question: str) -> dict:
    """
    Multi-signal QA engine:
      1. TF-IDF cosine similarity with bigrams (semantic match)
      2. Question keyword overlap (lexical match)
      3. Position bonus (introductory/conclusive sentences ranked slightly higher)
      4. Combined score → clean multi-sentence answer + normalized confidence
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # ── 1. Sentence segmentation ───────────────────────────────────────────────
    raw = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    sentences = [s.strip() for s in raw if len(s.split()) >= 6]
    if not sentences:
        return None

    n = len(sentences)

    # ── 2. TF-IDF cosine similarity (bigrams, sub-linear TF) ──────────────────
    corpus = [question] + sentences
    try:
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, max_features=10000, min_df=1,
        )
        mat = vec.fit_transform(corpus)
    except ValueError:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        mat = vec.fit_transform(corpus)

    cosine_scores = cosine_similarity(mat[0:1], mat[1:]).flatten()

    # ── 3. Keyword overlap (fraction of question keywords found in sentence) ───
    q_kw = _extract_keywords(question)
    overlap_scores = []
    for sent in sentences:
        s_kw = _extract_keywords(sent)
        overlap = len(q_kw & s_kw) / max(len(q_kw), 1)
        overlap_scores.append(overlap)

    # ── 4. Position bonus (first 15 % and last 10 % of doc score higher) ──────
    pos_scores = []
    for i in range(n):
        frac = i / n
        if frac <= 0.15 or frac >= 0.90:
            pos_scores.append(1.0)
        else:
            pos_scores.append(0.80)

    # ── 5. Combined score ──────────────────────────────────────────────────────
    combined = [
        0.55 * cosine_scores[i]
        + 0.35 * overlap_scores[i]
        + 0.10 * pos_scores[i]
        for i in range(n)
    ]

    top_indices = sorted(range(n), key=lambda i: combined[i], reverse=True)[:5]
    best_idx   = top_indices[0]
    best_score = combined[best_idx]

    # ── 6. Reject if document contains no relevant content ────────────────────
    if overlap_scores[best_idx] < 0.12 and cosine_scores[best_idx] < 0.08:
        return {
            "answer": None,
            "confidence": 0.0,
            "context_snippet": "",
            "source_sentences": [],
        }

    # ── 7. Build clean multi-sentence answer (document order, deduped) ────────
    candidate_idxs = sorted(top_indices[:4], key=lambda i: i)
    chosen = [candidate_idxs[0]]
    for idx in candidate_idxs[1:]:
        if _jaccard(sentences[idx], sentences[chosen[-1]]) < 0.50:
            chosen.append(idx)
        if len(chosen) == 3:
            break

    answer = " ".join(sentences[i] for i in chosen)
    # Clean whitespace and ensure sentence ends properly
    answer = re.sub(r'\s+', ' ', answer).strip()
    if answer and answer[-1] not in '.!?':
        answer += '.'

    # ── 8. Normalize confidence to intuitive human-readable scale ─────────────
    # A strong keyword match (overlap > 0.5) anchors confidence at 88 %+
    # Semantic-only matches (cosine > 0.2) anchor at 75 %+
    # Everything is boosted by the raw cosine score quality
    ov   = overlap_scores[best_idx]
    cos  = cosine_scores[best_idx]

    if ov >= 0.60:
        base = 0.90
    elif ov >= 0.40:
        base = 0.84
    elif ov >= 0.25:
        base = 0.78
    elif cos >= 0.25:
        base = 0.72
    elif cos >= 0.12:
        base = 0.62
    else:
        base = 0.50

    confidence = round(min(base + cos * 0.12, 0.99), 4)

    # Source evidence = best matching single sentence
    source_sentences = [sentences[i] for i in sorted(top_indices[:3], key=lambda i: i)]

    logger.info(
        f"QA | best_cos={cos:.3f} overlap={ov:.3f} combined={best_score:.3f} → conf={confidence:.2f}"
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "context_snippet": sentences[best_idx],
        "source_sentences": source_sentences,
    }


class QAService:
    """
    Answers questions using a multi-signal TF-IDF engine.
    Fully offline — no model downloads required.
    """

    @staticmethod
    async def answer(text: str, question: str) -> dict:
        logger.info(f"Running multi-signal QA for question: '{question}'")

        try:
            result = await run_in_threadpool(_answer_sync, text, question)
        except Exception as e:
            logger.error(f"QA failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"QA processing failed: {str(e)}",
            )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The document could not be processed for Q&A.",
            )

        if result["answer"] is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="The document does not contain enough relevant information to answer this question.",
            )

        logger.info(f"Answer found with confidence: {result['confidence']}")
        return result
