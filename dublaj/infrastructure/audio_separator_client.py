"""
Audio Separator Client — HTTP-клиент к сервису разделения голос/музыка.

Реализует IVoiceSeparator через вызов HTTP API windowed-roformer
(Mel-Band RoFormer + WSA), работающего в Docker-контейнере (порт 8310).
Контракт /separate_json совместим со старым audio-separator (8300).
"""

import base64
import logging
from pathlib import Path

import httpx

from app.domain.interfaces import IVoiceSeparator

logger = logging.getLogger(__name__)


class AudioSeparatorClient(IVoiceSeparator):
    """
    Отправляет аудио на HTTP-сервис audio-separator и получает чистый вокал.

    Сервис использует модели UVR (MDX23C) для извлечения вокала из смешанного аудио.
    """

    def __init__(self, separator_url: str = "http://localhost:8310", timeout: float = 7200.0) -> None:
        self._separator_url = separator_url.rstrip("/")
        self._timeout = timeout

    async def separate_vocals(self, audio_path: Path, output_path: Path) -> tuple[Path, Path | None]:
        """
        Отправляет аудио в сервис разделения, получает чистый вокал + инструментал.

        Args:
            audio_path: Путь к смешанному аудио (голос + музыка/шум).
            output_path: Куда сохранить изолированный вокал.

        Returns:
            Кортеж (путь_к_вокалу, путь_к_инструменталу).
            instrumental_path может быть None, если сервер не поддерживает.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Separating vocals: %s → %s (via %s)",
            audio_path.name,
            output_path.name,
            self._separator_url,
        )

        # Читаем аудио и кодируем в base64
        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._separator_url}/separate_json",
                json={
                    "audio_base64": audio_base64,
                    "filename": audio_path.name,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Audio separator error: {response.status_code} - {response.text}"
                )

            result = response.json()
            vocals_base64 = result["vocals_base64"]
            vocals_bytes = base64.b64decode(vocals_base64)

            output_path.write_bytes(vocals_bytes)

            # Сохраняем инструментал, если есть
            instrumental_path: Path | None = None
            instrumental_base64 = result.get("instrumental_base64")
            if instrumental_base64:
                instrumental_path = output_path.parent / (output_path.stem + "_instrumental.wav")
                instrumental_path.write_bytes(base64.b64decode(instrumental_base64))
                logger.info("Instrumental saved: %s", instrumental_path)

        logger.info("Vocals separated: %s (%d bytes)", output_path, len(vocals_bytes))
        return output_path, instrumental_path

    async def health_check(self) -> bool:
        """Проверяет, работает ли сервис audio-separator."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._separator_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning("Audio separator health check failed: %s", e)
            return False
