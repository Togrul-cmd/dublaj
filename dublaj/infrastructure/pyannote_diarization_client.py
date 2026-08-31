"""
PyAnnote Speaker Diarization Client.

HTTP-клиент к серверу PyAnnote (localhost:8400).
Определяет КТО КОГДА говорит в аудиофайле.
"""

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class PyAnnoteDiarizationClient:
    """
    Клиент к PyAnnote Diarization серверу.

    Определяет спикеров в аудио:
    - Возвращает список сегментов {start, end, speaker}
    - Например: SPEAKER_00, SPEAKER_01, ...
    """

    def __init__(self, pyannote_url: str = "http://localhost:8500", timeout: float = 600.0):
        self._url = pyannote_url.rstrip("/")
        self._timeout = timeout

    async def diarize(self, audio_path: Path) -> list[dict]:
        """
        Выполняет диаризацию спикеров в аудиофайле.

        Returns:
            список сегментов:
                [
                    {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"},
                    {"start": 5.2, "end": 12.1, "speaker": "SPEAKER_01"},
                    ...
                ]
        """
        logger.info("PyAnnote: diarizing %s", audio_path.name)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{self._url}/diarize",
                    files={"file": (audio_path.name, f, "audio/wav")},
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"PyAnnote diarization error: {response.status_code} - {response.text}"
                )

            result = response.json()
            segments = result.get("segments", [])
            num_speakers = result.get("num_speakers", 0)

            logger.info(
                "PyAnnote: %d speakers, %d segments",
                num_speakers, len(segments),
            )
            return segments

    async def health_check(self) -> bool:
        """Проверяет, работает ли сервер PyAnnote."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning("PyAnnote health check failed: %s", e)
            return False