"""
Dublaj — Video Dubbing Pipeline.

Entry point: creates and launches the FastAPI application with all dependencies wired.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config.settings import get_settings
from infrastructure.api.dubbing_router import init_router, router as dubbing_router
from infrastructure.api.jobs_router import init_jobs_router, router as jobs_router
from providers.service_factory import ServiceFactory

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dublaj")


# ── Application Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.

    On startup: create all services via the factory and inject into the router.
    On shutdown: cleanup resources if needed.
    """
    settings = get_settings()
    logger.info("Starting Dublaj — Video Dubbing Pipeline")
    logger.info("Whisper service: %s", settings.whisper_url)
    logger.info("Translation model: %s (%s)", settings.translation_model, settings.translation_provider)
    if settings.translation_provider == "minimax":
        logger.info("MiniMax model: %s (%s)", settings.minimax_model, settings.minimax_base_url)
    logger.info("TTS service: OmniVoice (%s, voice: %s)", settings.omnivoice_url, settings.omnivoice_voice)
    logger.info("Audio separator: %s (enabled: %s)", settings.audio_separator_url, settings.use_voice_separation)

    # Wire dependencies
    factory = ServiceFactory(settings)
    orchestrator = factory.create_orchestrator()
    init_router(orchestrator)
    init_jobs_router(orchestrator)

    logger.info("All services initialized — ready to accept requests")
    yield
    logger.info("Shutting down Dublaj")


# ── FastAPI Application ──────────────────────────────────────────────────────
app = FastAPI(
    title="Dublaj — Video Dubbing API",
    description=(
        "Automated video dubbing pipeline: "
        "Whisper transcription → AI translation → Voice-cloned TTS → FFmpeg merge"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(dubbing_router)
app.include_router(jobs_router)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
