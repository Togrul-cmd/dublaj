"""
Job-based Async API — неблокирующий вариант дубляжа.

Три эндпоинта вместо одного долгого запроса:
  POST /jobs                  — принять видео, сразу вернуть job_id (202 Accepted)
  GET  /jobs/{job_id}/status  — queued / processing / completed / failed + этап
  GET  /jobs/{job_id}/result  — стриминг готового файла (FileResponse)

Обработка идёт в фоне (BackgroundTasks). Хранилище задач — в памяти процесса:
один инстанс приложения, при рестарте список задач теряется.
Лимит одновременных задач — settings.max_concurrent_jobs (остальные ждут
в статусе queued).
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.domain.entities import DubbingJob, JobStatus
from app.services.dubbing_orchestrator import DubbingOrchestrator
from config.settings import get_settings
from infrastructure.service_manager import ServiceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Единственный активный пайплайн проекта
PIPELINE_NAME = "seedvc_v2_speaker_diarization_v2"

# Внедряется из lifespan приложения (main.py)
_orchestrator: DubbingOrchestrator | None = None
_service_mgr: ServiceManager | None = None
_concurrency: asyncio.Semaphore | None = None


@dataclass
class _JobRecord:
    """Запись в хранилище задач: сама задача + метаданные запроса."""
    job: DubbingJob
    source_filename: str
    created_at: str


# Хранилище задач в памяти: {job_id: _JobRecord}
_JOBS: dict[str, _JobRecord] = {}


def init_jobs_router(orchestrator: DubbingOrchestrator) -> None:
    """Внедрить зависимости (вызывается из lifespan в main.py)."""
    global _orchestrator, _service_mgr, _concurrency
    _orchestrator = orchestrator
    settings = get_settings()
    _concurrency = asyncio.Semaphore(settings.max_concurrent_jobs)
    try:
        _service_mgr = ServiceManager()
        logger.info(
            "JobsRouter initialized — max_concurrent_jobs=%d", settings.max_concurrent_jobs,
        )
    except Exception as exc:
        logger.warning("ServiceManager not available: %s", exc)
        _service_mgr = None


def _get_orchestrator() -> DubbingOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _orchestrator


def _external_status(job: DubbingJob) -> str:
    """Внутренний статус пайплайна → внешний статус для клиента."""
    if job.status == JobStatus.PENDING:
        return "queued"
    if job.status == JobStatus.COMPLETED:
        return "completed"
    if job.status == JobStatus.FAILED:
        return "failed"
    return "processing"


def _save_upload(video: UploadFile, target_language: str) -> DubbingJob:
    """Сохранить загруженное видео в per-job temp dir и вернуть job."""
    settings = get_settings()
    job = DubbingJob(target_language=target_language)
    job_temp = settings.temp_dir / job.id
    job_temp.mkdir(parents=True, exist_ok=True)

    safe_name = Path(video.filename).name if video.filename else "video.mp4"
    input_path = job_temp / f"input_{safe_name}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    job.input_video_path = input_path
    return job


async def _run_job(job_id: str) -> None:
    """Фоновая обработка задачи: проверка сервисов → пайплайн → статусы."""
    record = _JOBS.get(job_id)
    if record is None or _orchestrator is None:
        return
    job = record.job

    # Задачи сверх лимита ждут здесь — их статус остаётся queued
    async with _concurrency:
        logger.info("Job %s — started (queue slot acquired)", job_id)
        try:
            if _service_mgr is not None:
                await _service_mgr.ensure_services(PIPELINE_NAME)

            result = await _orchestrator.process_seedvc_v2_speaker_diarization_v2(job)
            if result.status == JobStatus.COMPLETED:
                logger.info("Job %s — completed: %s", job_id, result.output_video_path)
            else:
                logger.error("Job %s — failed: %s", job_id, result.error_message)
        except Exception as exc:
            # Любой сбой (сервис не ответил, GPU занят и т.д.) → failed с текстом
            logger.exception("Job %s — unexpected error", job_id)
            job.fail(f"{type(exc).__name__}: {exc}")


# ── POST /jobs ───────────────────────────────────────────────────────────────


@router.post("", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to dub"),
    target_language: str = Form(..., description="Target language code (e.g. 'ru', 'tr', 'az')"),
):
    """
    Принять видео в дубляж и сразу ответить — обработка идёт в фоне.

    Возвращает 202 Accepted + job_id.
    Дальше опрашивать GET /jobs/{job_id}/status и скачать результат
    через GET /jobs/{job_id}/result.
    """
    _get_orchestrator()

    job = _save_upload(video, target_language)
    safe_name = Path(video.filename).name if video.filename else "video.mp4"
    _JOBS[job.id] = _JobRecord(
        job=job,
        source_filename=safe_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    background_tasks.add_task(_run_job, job.id)

    logger.info(
        "New async job %s — file=%s, target=%s", job.id, safe_name, target_language,
    )
    return {
        "job_id": job.id,
        "status": "queued",
        "status_url": f"/jobs/{job.id}/status",
        "result_url": f"/jobs/{job.id}/result",
    }


# ── GET /jobs/{job_id}/status ────────────────────────────────────────────────


@router.get("/{job_id}/status")
async def job_status(job_id: str):
    """Статус задачи: queued / processing / completed / failed + текущий этап."""
    record = _JOBS.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = record.job

    response = {
        "job_id": job.id,
        "status": _external_status(job),
        "stage": job.status.value,  # extracting_audio / diarizing / transcribing / ...
        "target_language": job.target_language,
        "source_filename": record.source_filename,
        "created_at": record.created_at,
        "error": job.error_message,
    }

    # Дополнительная информация по мере появления
    if job.diarization_segments:
        response["num_speakers"] = len(
            set(seg["speaker"] for seg in job.diarization_segments)
        )
    if job.transcription:
        response["source_language"] = job.transcription.language

    if job.status == JobStatus.COMPLETED:
        response["result_url"] = f"/jobs/{job.id}/result"
        if job.output_video_path:
            response["output_video"] = job.output_video_path.name

    return response


# ── GET /jobs/{job_id}/result ────────────────────────────────────────────────


@router.get("/{job_id}/result")
async def job_result(job_id: str):
    """Скачать готовое видео (стриминг через FileResponse)."""
    record = _JOBS.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    job = record.job

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail={"message": "Job failed", "error": job.error_message},
        )
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Job not finished yet",
                "status": _external_status(job),
                "stage": job.status.value,
            },
        )

    output = job.output_video_path
    if output is None or not Path(output).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result file missing for job '{job_id}'",
        )

    return FileResponse(
        path=str(output),
        media_type="video/mp4",
        filename=output.name,  # dubbed_{job_id}.mp4
    )
