"""
Groq LLaMA 3.3 70B — Abstractive Summarization Service
--------------------------------------------------------
Used when GROQ_API_KEY is set in .env.
Falls back to the offline ensemble if the key is missing or the call fails.
"""
from __future__ import annotations

import json
import re

from app.core.logging import setup_logger
from app.models.schemas import SummaryLength

logger = setup_logger(__name__)

# ── Prompt config per summary length ──────────────────────────────────────────
LENGTH_SPEC = {
    SummaryLength.short:    ("3–4 sentences",  3),
    SummaryLength.medium:   ("6–8 sentences",  5),
    SummaryLength.detailed: ("12–15 sentences", 8),
}

# Cap text at ~12 000 words — well within LLaMA 3.3's 128K context window
_MAX_WORDS = 12_000


def _truncate(text: str, max_words: int = _MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    logger.info(f"Document truncated to {max_words} words for Groq context window.")
    return " ".join(words[:max_words]) + "\n\n[Document truncated for summarization]"


def _build_prompt(text: str, summary_length: SummaryLength) -> str:
    length_spec, kp_count = LENGTH_SPEC[summary_length]
    return f"""You are an expert document analyst and technical writer.
Carefully read the document below and produce:

1. A **clear, fluent summary** written in {length_spec}. The summary must:
   - Be in plain prose (no bullet points)
   - Capture the main purpose, key findings, and conclusions
   - Be informative enough to replace reading the full document

2. Exactly {kp_count} **key points** — the most important individual facts,
   conclusions, or takeaways. Each key point must be:
   - A single complete sentence
   - Specific (not vague like "the document discusses AI")
   - Different from each other — no repetition

DOCUMENT:
---
{text}
---

Respond ONLY with valid JSON in this exact format — no extra text, no markdown fences:
{{
  "summary": "your full summary here",
  "key_points": [
    "Key point sentence 1.",
    "Key point sentence 2."
  ]
}}"""


def _groq_summarize_sync(text: str, summary_length: SummaryLength) -> dict:
    """
    Calls Groq (LLaMA 3.3 70B) and returns {summary, key_points}.
    Raises on failure so the caller falls back to the ensemble.
    """
    from groq import Groq
    from app.core.config import get_settings

    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)

    prompt = _build_prompt(_truncate(text), summary_length)

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document summarization assistant. "
                    "Always respond with valid JSON only — no markdown, no extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings.GROQ_TEMPERATURE,
        max_tokens=settings.GROQ_MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences just in case
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    summary    = data.get("summary", "").strip()
    key_points = [p.strip() for p in data.get("key_points", []) if str(p).strip()]

    if not summary:
        raise ValueError("Groq returned an empty summary.")

    logger.info(
        f"Groq summarization complete | model={settings.GROQ_MODEL} | "
        f"summary={len(summary.split())} words | key_points={len(key_points)}"
    )
    return {"summary": summary, "key_points": key_points}
