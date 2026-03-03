from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logger
from app.api.v1.router import api_router
from app.utils.file_utils import ensure_upload_dir

logger = setup_logger(__name__)
settings = get_settings()


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    ensure_upload_dir()
    logger.info(f"Upload directory ready: '{settings.UPLOAD_DIR}'")

    # Initialize all services at startup (no large downloads — all offline)
    logger.info("Initializing all services...")
    from app.services.summarizer import load_summarizer
    from app.services.qa_service import load_qa_pipeline
    from app.services.keyword_extractor import load_keyword_model

    await run_in_threadpool(load_summarizer)
    load_qa_pipeline()
    load_keyword_model()

    logger.info("All services ready. API is accepting requests.")
    yield
    # ── Shutdown ──
    logger.info(f"Shutting down {settings.APP_NAME}")


# ──────────────────────────────────────────────
# FastAPI App Instance
# ──────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Global Exception Handlers
# ──────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_code": "INTERNAL_SERVER_ERROR"},
    )


# ──────────────────────────────────────────────
# Include Routers
# ──────────────────────────────────────────────

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# ──────────────────────────────────────────────
# Mount Static Files
# ──────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ──────────────────────────────────────────────
# Root — Serve UI
# ──────────────────────────────────────────────

@app.get("/", tags=["Root"], summary="Web UI", include_in_schema=False)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
