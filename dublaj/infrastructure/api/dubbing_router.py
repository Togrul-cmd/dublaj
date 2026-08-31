"""
Dubbing API Router — FastAPI endpoints for the dubbing pipeline.

This is a thin adapter layer. It receives HTTP requests, delegates to
the DubbingOrchestrator (business logic), and returns HTTP responses.

Available pipelines (Speaker Diarization only):
  - Seed-VC V2 + Speaker Diarization V2 (БЕЗ F5-TTS, Seed-VC V1)
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.domain.entities import DubbingJob, JobStatus
from app.services.dubbing_orchestrator import DubbingOrchestrator
from config.settings import get_settings
from infrastructure.service_manager import ServiceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dubbing", tags=["dubbing"])

# Will be injected by the application lifespan
_orchestrator: DubbingOrchestrator | None = None
_service_mgr: ServiceManager | None = None


def init_router(orchestrator: DubbingOrchestrator) -> None:
    """Inject the orchestrator dependency into this router module."""
    global _orchestrator, _service_mgr  # noqa: WPS420
    _orchestrator = orchestrator
    try:
        _service_mgr = ServiceManager()
        logger.info("ServiceManager initialized (passive health-check)")
    except Exception as exc:
        logger.warning("ServiceManager not available: %s", exc)
        _service_mgr = None


def _get_orchestrator() -> DubbingOrchestrator:
    """Retrieve the orchestrator, raising if not initialized."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _orchestrator


async def _ensure_services(pipeline_name: str) -> None:
    """Дождаться готовности нужных Docker сервисов (контейнеры НЕ запускаются)."""
    if _service_mgr is None:
        raise HTTPException(status_code=503, detail="ServiceManager not initialized")
    try:
        await _service_mgr.ensure_services(pipeline_name)
    except Exception as exc:
        logger.exception("ServiceManager failed for '%s'", pipeline_name)
        raise HTTPException(status_code=503, detail="Services not ready") from exc


def _save_upload(video: UploadFile, target_language: str):
    """Save uploaded video into a per-job temp dir and return (job, input_path)."""
    settings = get_settings()
    job = DubbingJob(target_language=target_language)
    job_temp = settings.temp_dir / job.id
    job_temp.mkdir(parents=True, exist_ok=True)

    # Безопасное имя файла — убираем пути и спецсимволы
    safe_name = Path(video.filename).name if video.filename else "video.mp4"
    input_path = job_temp / f"input_{safe_name}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    job.input_video_path = input_path
    return job, input_path


# ── Seed-VC V2 + Speaker Diarization V2 (БЕЗ F5-TTS) ───────────────────────


@router.post("/intonation/seedvc-v2/speaker-diarization-v2")
async def seedvc_v2_speaker_diarization_v2(
    video: UploadFile = File(..., description="Video file to dub"),
    target_language: str = Form(..., description="Target language code (e.g. 'ru', 'tr', 'az')"),
):
    """
    Seed-VC V2 + Speaker Diarization V2 — БЕЗ F5-TTS!

    PyAnnote определяет спикеров в аудио (SPEAKER_00, SPEAKER_01, ...).
    Для каждого спикера:
      1. Извлекается sample голоса (30 сек) из clean_vocals
      2. OmniVoice DEFAULT генерирует перевод с правильным произношением
      3. Seed-VC V1 накладывает голос ЭТОГО спикера (ref из clean_vocals)

    Результат: каждый спикер говорит переводом СВОИМ голосом!

    Требует Docker сервисы:
      - PyAnnote (8500)
      - Audio Separator (8300)
      - Whisper (8100)
      - OmniVoice (8200)
      - Seed-VC V1 (8700)
    """
    orchestrator = _get_orchestrator()
    job, _ = _save_upload(video, target_language)

    logger.info(
        "New SEEDVC V2 SPEAKER DIARIZATION V2 job %s — file=%s, target=%s",
        job.id, video.filename, target_language,
    )

    await _ensure_services("seedvc_v2_speaker_diarization_v2")
    result = await orchestrator.process_seedvc_v2_speaker_diarization_v2(job)

    if result.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Dubbing failed: {result.error_message}",
        )

    return {
        "job_id": result.id,
        "status": result.status.value,
        "mode": "seedvc_v2_speaker_diarization_v2",
        "num_speakers": len(set(seg["speaker"] for seg in result.diarization_segments)) if hasattr(result, "diarization_segments") and result.diarization_segments else 0,
        "source_language": result.transcription.language if result.transcription else None,
        "target_language": result.target_language,
        "output_video": str(result.output_video_path),
    }


@router.post("/intonation/seedvc-v2/speaker-diarization-v2/download")
async def seedvc_v2_speaker_diarization_v2_download(
    video: UploadFile = File(..., description="Video file to dub"),
    target_language: str = Form(..., description="Target language code (e.g. 'ru', 'tr', 'az')"),
):
    """Seed-VC V2 + Speaker Diarization V2 — без F5-TTS + download."""
    orchestrator = _get_orchestrator()
    job, _ = _save_upload(video, target_language)

    await _ensure_services("seedvc_v2_speaker_diarization_v2")
    result = await orchestrator.process_seedvc_v2_speaker_diarization_v2(job)

    if result.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Dubbing failed: {result.error_message}",
        )

    safe_name = Path(video.filename).name if video.filename else "video.mp4"
    return FileResponse(
        path=str(result.output_video_path),
        media_type="video/mp4",
        filename=f"seedvc_v2_speakers_{safe_name}",
    )


# ── Health ─────────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """Check if the dubbing service and Docker containers are operational."""
    if _service_mgr is None:
        return {"status": "degraded", "service": "dublaj", "detail": "ServiceManager not initialized"}

    from infrastructure.service_manager import PIPELINE_SERVICES, HEALTH_URLS
    import httpx

    services_status = {}
    all_ok = True
    for svc_name in PIPELINE_SERVICES.get("seedvc_v2_speaker_diarization_v2", []):
        url = HEALTH_URLS.get(svc_name)
        if not url:
            continue
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                services_status[svc_name] = "ok" if resp.status_code == 200 else f"error:{resp.status_code}"
                if resp.status_code != 200:
                    all_ok = False
        except Exception as exc:
            services_status[svc_name] = f"error:{exc}"
            all_ok = False

    return {
        "status": "ok" if all_ok else "degraded",
        "service": "dublaj",
        "services": services_status,
    }
