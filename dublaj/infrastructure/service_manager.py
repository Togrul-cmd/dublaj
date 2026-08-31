"""
Service Manager — пассивная проверка готовности Docker-контейнеров.

Модели НЕ запускаются и НЕ останавливаются из кода приложения.
Все контейнеры запускаются ОДИН РАЗ вручную (docker compose up -d)
и работают постоянно. Пользователь может использовать все модели
одновременно — VRAM на сервере достаточно.

Задача этого класса: только убедиться, что нужные сервисы живы
и отвечают на health-check перед запуском пайплайна.
"""

import asyncio
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

# Хост для health-check. В Docker используй SERVICES_HOST=host.docker.internal
_SERVICES_HOST = os.getenv("SERVICES_HOST", "localhost")

# ── Пайплайн → нужные Docker сервисы (только для проверки) ─────────
PIPELINE_SERVICES: dict[str, list[str]] = {
    # Seed-VC V2 + Speaker Diarization V2 (БЕЗ F5-TTS, Seed-VC V1)
    "seedvc_v2_speaker_diarization_v2": [
        "dublaj-windowed-roformer", "dublaj-pyannote-diarization",
        "dublaj-whisper", "dublaj-omnivoice", "dublaj-seedvc",
    ],
}

# Health-check URLs
HEALTH_URLS: dict[str, str] = {
    "dublaj-whisper": f"http://{_SERVICES_HOST}:8100/health",
    "dublaj-omnivoice": f"http://{_SERVICES_HOST}:8200/health",
    "dublaj-windowed-roformer": f"http://{_SERVICES_HOST}:8310/health",
    "dublaj-audio-separator": f"http://{_SERVICES_HOST}:8300/health",
    "dublaj-pyannote-diarization": f"http://{_SERVICES_HOST}:8500/health",
    "dublaj-seedvc": f"http://{_SERVICES_HOST}:8700/health",
}


class ServiceManager:
    """
    Пассивный менеджер сервисов.

    Не запускает и не останавливает Docker-контейнеры — они работают
    постоянно (запущены вручную через docker compose up -d).

    Единственная задача — дождаться готовности всех нужных сервисов
    перед запуском пайплайна.
    """

    async def ensure_services(self, pipeline_name: str) -> None:
        """
        Дождаться пока все нужные сервисы для пайплайна ответят на health-check.

        Контейнеры НЕ запускаются и НЕ останавливаются — только проверка.
        Проверка идёт ПАРАЛЛЕЛЬНО для всех сервисов.
        """
        needed = PIPELINE_SERVICES.get(pipeline_name)
        if not needed:
            logger.warning(
                "ServiceManager: unknown pipeline '%s', skipping readiness check",
                pipeline_name,
            )
            return

        logger.info(
            "ServiceManager: waiting for %d services to be ready for '%s'",
            len(needed), pipeline_name,
        )

        # Проверяем все сервисы параллельно
        tasks = []
        for name in needed:
            url = HEALTH_URLS.get(name)
            if not url:
                logger.warning(
                    "ServiceManager: no health URL for '%s', skipping check", name,
                )
                continue
            tasks.append(self._wait_healthy(name, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed = [name for name, result in zip(needed, results) if isinstance(result, Exception)]
        if failed:
            raise RuntimeError(
                f"Services not ready for '{pipeline_name}': {', '.join(failed)}"
            )

        logger.info(
            "ServiceManager: ✓ all %d services ready for '%s'",
            len(needed), pipeline_name,
        )

    async def _wait_healthy(
        self,
        name: str,
        url: str,
        timeout: int = 300,
        interval: float = 5.0,
    ) -> None:
        """Ждём пока сервис не ответит 200 на health-check."""
        t0 = time.monotonic()
        # Один клиент на весь цикл проверки (не создаём на каждую попытку)
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() - t0 < timeout:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        elapsed = time.monotonic() - t0
                        logger.info("    ✓ %s ready (%.0fs)", name, elapsed)
                        return
                except Exception as exc:
                    logger.debug("    ⏳ %s health-check error: %s", name, exc)
                await asyncio.sleep(interval)

        raise RuntimeError(
            f"Service {name} not ready after {timeout}s — проверьте что контейнер запущен"
        )
