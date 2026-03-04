---
title: Intelligent Document Summarizer
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Intelligent Document Summarizer API

> An AI-powered REST API built with **FastAPI** that accepts PDF, DOCX, and TXT documents and returns summaries, extracted keywords, document analytics, and natural language question answering — all through clean, versioned HTTP endpoints.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Focus — FastAPI & Endpoints](#core-focus--fastapi--endpoints)
3. [Tech Stack & Tools Explained](#tech-stack--tools-explained)
4. [Folder Structure](#folder-structure)
5. [API Endpoints (v1)](#api-endpoints-v1)
6. [Request & Response Schemas](#request--response-schemas)
7. [Engine Priority — How Summarization Works](#engine-priority--how-summarization-works)
8. [Configuration & Environment Variables](#configuration--environment-variables)
9. [Getting Started](#getting-started)
10. [Future Work](#future-work)

---

## Project Overview

The **Intelligent Document Summarizer** is a production-structured FastAPI project that demonstrates how to build real-world NLP-powered APIs. Upload any document and the API will:

- Generate a fluent **AI summary** (short / medium / detailed)
- Extract **top keywords and keyphrases** with relevance scores
- Answer **natural language questions** about the document content
- Return **document analytics** — word count, reading time, Flesch readability score, sentences, paragraphs, compression ratio
- Automatically select the **best available AI engine** (Groq LLaMA 3.3 70B when a key is set, otherwise offline ensemble)

The entire project also ships with a **dark-themed single-page web UI** served directly by FastAPI's static file middleware, so you can test everything from a browser without Postman.

---

## Core Focus — FastAPI & Endpoints

This project was intentionally built to demonstrate **FastAPI best practices** for endpoint design:

| Practice | How it's applied here |
|---|---|
| **Versioned routing** | All endpoints are under `/api/v1/` via an `APIRouter` prefix |
| **Pydantic validation** | Every request form field and JSON response is typed with Pydantic v2 models |
| **Async handlers** | All route functions are `async def`, CPU-heavy NLP work is offloaded with `run_in_threadpool` |
| **File upload handling** | `UploadFile` + `python-multipart` for streaming multipart uploads |
| **OpenAPI docs** | Swagger UI at `/docs`, ReDoc at `/redoc`, machine-readable spec at `/openapi.json` |
| **Lifespan events** | Startup/shutdown hooks via `@asynccontextmanager lifespan` — all services warm up once at boot |
| **CORS middleware** | Configured for cross-origin access so the frontend SPA can call the API |
| **Structured error responses** | `ErrorResponse` Pydantic model + `JSONResponse` with proper HTTP status codes |
| **Settings management** | `pydantic-settings` reads `.env` file — zero hardcoded secrets |
| **Separation of concerns** | Routes only orchestrate; business logic lives in `services/`; I/O lives in `utils/` |

---

## Tech Stack & Tools Explained

### FastAPI `0.115.5`
The web framework at the heart of the project. FastAPI is built on top of **Starlette** (ASGI) and **Pydantic**. It generates interactive API documentation automatically from Python type hints. Chosen because it is the industry standard for high-performance Python APIs and has native async support.

### Uvicorn `0.32.1`
ASGI server that runs the FastAPI application. The `[standard]` extras add `httptools` and `uvloop` for maximum throughput. Started with `--reload` during development to hot-restart on file changes.

### Pydantic v2 + pydantic-settings `2.10.3 / 2.6.1`
Pydantic handles all data validation — request bodies, response shapes, and settings. The `Settings` class in `config.py` reads from the `.env` file automatically using `pydantic-settings`. The `@lru_cache` decorator ensures settings are parsed only once for the lifetime of the process.

### python-multipart `0.0.12`
Required dependency for FastAPI to parse `multipart/form-data` requests — this is how file uploads (`UploadFile`) work. Without it, FastAPI cannot read uploaded files.

### Groq SDK + LLaMA 3.3 70B
[Groq](https://console.groq.com) is an inference provider offering extremely fast LLM inference via a hardware accelerator called the LPU. The `groq` Python package wraps their REST API. When `GROQ_API_KEY` is set in `.env`, every summarization request is sent to **LLaMA 3.3 70B Versatile** — Meta's open-source model capable of producing fluent, abstractive summaries that read like human-written text. Groq's free tier is sufficient for development and testing.

### sumy
Pure-Python extractive summarization library. Implements four algorithms used in the offline ensemble:
- **LSA (Latent Semantic Analysis)** — finds sentences that together cover the most unique concepts using matrix decomposition
- **LexRank** — graph-based algorithm that scores sentences by their similarity to all other sentences (like PageRank for text)
- **Luhn** — scores sentences based on the frequency and proximity of high-value words
- **TextRank** — another graph-based method; sentences that link to important sentences score higher

The four results are merged and deduplicated using sentence-level Jaccard similarity scoring to produce a single ensemble summary.

### YAKE (Yet Another Keyword Extractor)
Statistical, language-model-free keyword extraction. YAKE scores candidate keyphrases using features like term frequency, position in the document, co-occurrence, and casing. It works fully offline without any model downloads, making it fast and reliable.

### scikit-learn
Used in the Q&A service for **TF-IDF vectorization** with bigrams. Every sentence in the document is transformed into a TF-IDF vector, and the user's question is compared against all sentences using **cosine similarity**. The best-scoring sentences are combined into a multi-sentence answer. Also includes position bonus (sentences near the start of the document score higher) and a keyword overlap signal to improve accuracy.

### NLTK
Natural Language Toolkit — used for sentence tokenization (`punkt` tokenizer) that feeds into sumy's extractive algorithms. Initialized at startup in the lifespan hook.

### pdfplumber `0.11.4`
Extracts text from PDF files by parsing the PDF content stream directly. Handles multi-page documents and preserves paragraph structure better than PyPDF2 for typical documents.

### python-docx `1.1.2`
Reads `.docx` Microsoft Word files by parsing the OpenXML format. Text is extracted paragraph by paragraph, preserving the logical document structure.

### aiofiles `24.1.0`
Async file I/O library used when saving uploaded files to disk, so the file write doesn't block the event loop.

### httpx `0.28.1`
Modern async HTTP client. Available as a utility dependency for making outbound HTTP calls (e.g., health checks or future webhook integrations).

---

## Folder Structure

```
F:\FastAPI\
│
├── .env                          # Secret keys and environment overrides (git-ignored)
├── .env.example                  # Template showing all available env vars
├── .gitignore
├── requirements.txt              # All Python dependencies
├── README.md
│
├── uploads/                      # Uploaded files are saved here temporarily
│
└── app/                          # Application package root
    │
    ├── main.py                   # FastAPI app instance, middleware, lifespan hooks, static files
    ├── __init__.py
    │
    ├── api/                      # HTTP layer — only routing lives here
    │   └── v1/
    │       ├── router.py         # Aggregates all v1 route modules into one APIRouter
    │       ├── __init__.py
    │       └── routes/
    │           ├── health.py     # GET  /api/v1/health
    │           ├── summarizer.py # POST /api/v1/summarize
    │           │                 # POST /api/v1/keywords
    │           │                 # POST /api/v1/qa
    │           └── __init__.py
    │
    ├── core/                     # Cross-cutting concerns
    │   ├── config.py             # Pydantic Settings class — reads .env, exposes typed config
    │   ├── logging.py            # Structured logger factory (colorised console output)
    │   └── __init__.py
    │
    ├── models/                   # Pydantic request/response schemas
    │   ├── schemas.py            # SummarizeRequest, SummarizeResponse, QARequest, QAResponse …
    │   └── __init__.py
    │
    ├── services/                 # Business logic — no HTTP code here
    │   ├── summarizer.py         # Orchestrator: Groq → Ensemble fallback + analytics
    │   ├── groq_summarizer.py    # Groq API wrapper (LLaMA 3.3 70B sync call)
    │   ├── file_parser.py        # PDF / DOCX / TXT text extraction
    │   ├── keyword_extractor.py  # YAKE keyword extraction with relevance scoring
    │   ├── qa_service.py         # TF-IDF + bigram multi-signal Q&A engine
    │   └── __init__.py
    │
    ├── utils/                    # Utility helpers
    │   ├── file_utils.py         # Upload dir creation, file save, extension validation
    │   └── __init__.py
    │
    └── static/
        └── index.html            # Single-page dark UI (HTML + Vanilla JS, served by FastAPI)
```

---

## API Endpoints (v1)

Base URL: `http://localhost:8000`

### `GET /api/v1/health`
Returns the running status of the API. Useful for load balancers and uptime monitors.

**Response**
```json
{
  "status": "ok",
  "app_name": "Intelligent Document Summarizer",
  "version": "1.0.0"
}
```

---

### `POST /api/v1/summarize`
Upload a document and receive an AI-generated summary with full document analytics.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF, DOCX, or TXT — max 10 MB |
| `summary_length` | string | No | `short` \| `medium` \| `detailed` (default: `medium`) |

**Response** — `200 OK`
```json
{
  "filename": "report.pdf",
  "file_type": "pdf",
  "summary_length": "medium",
  "engine": "groq",
  "summary": "Machine learning is transforming industries...",
  "key_points": [
    "Python is the leading language for AI development",
    "Neural networks require large training datasets",
    "Cloud platforms accelerate model deployment"
  ],
  "original_word_count": 1240,
  "summary_word_count": 98,
  "compression_ratio": 0.0790,
  "sentence_count": 62,
  "paragraph_count": 14,
  "reading_time_minutes": 5,
  "reading_level": "Standard",
  "flesch_score": 58.4
}
```

---

### `POST /api/v1/keywords`
Extract the top keywords and keyphrases from a document using YAKE.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF, DOCX, or TXT — max 10 MB |

**Response** — `200 OK`
```json
{
  "filename": "report.pdf",
  "keywords": ["machine learning", "neural networks", "Python", "TensorFlow"],
  "keyword_scores": [
    { "keyword": "machine learning", "score": 0.032 },
    { "keyword": "neural networks",  "score": 0.041 }
  ]
}
```

---

### `POST /api/v1/qa`
Ask a natural language question; the API finds the best answer from the document.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | PDF, DOCX, or TXT — max 10 MB |
| `question` | string | Yes | Free-text question (5–500 characters) |

**Response** — `200 OK`
```json
{
  "filename": "report.pdf",
  "question": "What programming language is most used for AI?",
  "answer": "Python is the top language for AI development. It is widely used with frameworks like TensorFlow and PyTorch.",
  "confidence": 0.96,
  "context_snippet": "Python is the top language for AI development.",
  "source_sentences": [
    "Python is the top language for AI development.",
    "Key skills include TensorFlow, PyTorch, and cloud platforms."
  ]
}
```

---

### `GET /docs`
Interactive Swagger UI — test every endpoint directly in the browser. Auto-generated from Pydantic schemas and route docstrings.

### `GET /redoc`
ReDoc-style API reference documentation — clean two-panel layout.

### `GET /openapi.json`
Raw OpenAPI 3.1 specification — import into Postman, Insomnia, or any API client.

---

## Request & Response Schemas

All schemas are defined with **Pydantic v2** in `app/models/schemas.py`.

### Enums

```python
SummaryLength   →  "short" | "medium" | "detailed"
SupportedLanguage → "en" | "fr" | "de" | "es"
```

### Key Response Fields — SummarizeResponse

| Field | Type | Description |
|---|---|---|
| `engine` | `str` | Which engine produced the summary (`groq` or `ensemble`) |
| `compression_ratio` | `float` | Fraction of original words kept in summary |
| `flesch_score` | `float` | 0–100 readability score (higher = easier to read) |
| `reading_level` | `str` | Human label: Very Easy / Easy / Standard / Difficult / Very Difficult |
| `reading_time_minutes` | `int` | Estimated at 200 words per minute |

### Key Response Fields — QAResponse

| Field | Type | Description |
|---|---|---|
| `confidence` | `float` | Normalized score 0.50–0.99 from multi-signal TF-IDF ranking |
| `source_sentences` | `List[str]` | Top supporting sentences from the document |
| `context_snippet` | `str` | Single best-match sentence used as primary evidence |

---

## Engine Priority — How Summarization Works

```
Upload Request
      │
      ▼
 GROQ_API_KEY set?
      │
    Yes ──► Groq API (LLaMA 3.3 70B)  ──► Success ──► Return summary
      │              │
      │          API error?
      │              │
      No ◄───────────┘
      │
      ▼
 Offline Ensemble
  ├── LSA          ┐
  ├── LexRank      ├── score + merge + deduplicate (Jaccard)
  ├── Luhn         │
  └── TextRank     ┘
      │
      ▼
 Return summary (engine: "ensemble")
```

Key points:
- If Groq fails for any reason (network, rate limit) the API **silently falls back** to ensemble — the endpoint never returns a 500 error due to an AI engine issue
- The `engine` field in the response tells you exactly which path was taken
- The ensemble is entirely **offline** — no API keys, no internet required

---

## Configuration & Environment Variables

All settings are managed through `.env`. Copy `.env.example` to `.env` and fill in your values.

```env
# -- Required for Groq LLM summarization --
GROQ_API_KEY=gsk_your_key_here   # Get free key at https://console.groq.com

# -- Optional overrides (defaults shown) --
APP_NAME=Intelligent Document Summarizer
APP_VERSION=1.0.0
DEBUG=False
MAX_FILE_SIZE_MB=10
UPLOAD_DIR=uploads
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=1500
GROQ_TEMPERATURE=0.3
KEYWORD_TOP_N=10
```

If `GROQ_API_KEY` is left blank, the API works completely offline using the ensemble.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com) (optional but recommended)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd FastAPI

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Run the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points

| URL | Description |
|---|---|
| `http://localhost:8000` | Dark-theme Web UI |
| `http://localhost:8000/docs` | Swagger interactive docs |
| `http://localhost:8000/redoc` | ReDoc API reference |
| `http://localhost:8000/api/v1/health` | Health check |

### Quick Test with curl

```bash
# Summarize a document
curl -X POST http://localhost:8000/api/v1/summarize \
  -F "file=@your_document.pdf" \
  -F "summary_length=short"

# Extract keywords
curl -X POST http://localhost:8000/api/v1/keywords \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/qa \
  -F "file=@your_document.pdf" \
  -F "question=What is the main conclusion?"
```

---

## Future Work

The project has a solid, extensible foundation. The following features are planned as natural next steps:

### Endpoint Enhancements

- **`POST /api/v1/translate`** — Translate a document summary into French, German, Spanish, or Arabic using a lightweight translation model or DeepL API
- **`POST /api/v1/compare`** — Accept two documents and return a similarity score + key differences
- **`POST /api/v1/classify`** — Classify a document into predefined categories (legal, medical, finance, technical)
- **`POST /api/v1/sentiment`** — Return overall sentiment (positive / negative / neutral) and per-paragraph breakdown
- **`GET /api/v1/history`** — Return paginated list of previously processed documents with summaries (requires DB)
- **`DELETE /api/v1/document/{id}`** — Delete a stored document and its metadata

### Authentication & Security

- **API key authentication** — `X-API-Key` header middleware protecting all `/api/v1/` routes
- **JWT Bearer token auth** — `OAuth2PasswordBearer` flow with `/auth/login` and `/auth/refresh` endpoints
- **Rate limiting** — per-IP and per-API-key request throttling using `slowapi` middleware
- **File content validation** — virus scan hook and MIME-type verification beyond extension check

### Database & Persistence

- **SQLite with SQLAlchemy + Alembic** — store document metadata, summaries, and Q&A history; manage schema migrations
- **Async DB access** — swap to `asyncpg` (PostgreSQL) or `aiosqlite` for non-blocking queries
- **Redis caching** — cache summarization results by document hash so re-uploading the same file is instant

### Document Support

- **`PPTX` support** — extract text from PowerPoint slides using `python-pptx`
- **Image-based PDF (OCR)** — run Tesseract OCR on scanned PDFs via `pytesseract`
- **URL ingestion** — accept a URL instead of a file, scrape the page content with `httpx` + `BeautifulSoup`
- **Batch processing** — accept a ZIP file containing multiple documents and return an array of results

### AI & NLP Improvements

- **Streaming responses** — use `StreamingResponse` + Groq's streaming API to return the summary token-by-token (SSE)
- **Multi-turn Q&A** — maintain conversation context across multiple questions about the same document
- **Named Entity Recognition (NER)** — extract people, organizations, dates, and locations using `spaCy`
- **Automatic language detection** — detect the document language with `langdetect` and switch tokenizers accordingly
- **Local LLM fallback** — integrate `llama-cpp-python` to run a quantized GGUF model locally as a second fallback when Groq is unavailable

### DevOps & Deployment

- **Docker Compose setup** — containerize the app with multi-stage Dockerfile (already scaffolded)
- **GitHub Actions CI** — run `pytest` on every pull request, enforce `ruff` linting
- **Unit and integration tests** — `pytest` + `httpx.AsyncClient` + `TestClient` covering all endpoints
- **Environment-specific config** — separate `dev`, `staging`, `prod` settings classes
- **Health check expansion** — deep health endpoint reporting DB connectivity, disk space, and AI engine availability

---

*Built with FastAPI · Python 3.10 · LLaMA 3.3 70B via Groq · sumy · YAKE · scikit-learn*
