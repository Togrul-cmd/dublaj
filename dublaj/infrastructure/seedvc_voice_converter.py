"""
Seed-VC Voice Converter Client (F0 Base Singing Model).

HTTP клиент к Seed-VC серверу для zero-shot voice conversion.

Используется модель: seed-uvit-whisper-base (200M, F0 conditioning)
- 44100 Hz sample rate
- F0 (pitch) conditioning для переноса высоты голоса ← ВСЕГДА ВКЛЮЧЕНО
- RMVPE pitch extractor
- Auto F0 adjust ← ВСЕГДА ВКЛЮЧЕНО

Параметры по умолчанию оптимизированы для МАКСИМАЛЬНОГО сходства голоса:
- diffusion_steps=100 (макс качество)
- inference_cfg_rate=0.8 (выше чем default 0.7)
- auto_f0_adjust=True (ВСЕГДА - автоподстройка высоты)
- pitch_shift=0 (без сдвига)
- f0_condition=True (ВСЕГДА - F0 модель активирована)

API: POST /voice_convert
  - source: аудио для преобразования
  - reference: reference audio для клонирования голоса
"""

import logging
from pathlib import Path

import httpx

from app.domain.interfaces import IVoiceConverter

logger = logging.getLogger(__name__)


class SeedVCVoiceConverter(IVoiceConverter):
    """
    HTTP клиент к Seed-VC серверу (localhost:8700).

    Использует F0 Base (200M) модель:
    - F0 conditioning ВСЕГДА ВКЛЮЧЕН (эта модель работает ТОЛЬКО с F0)
    - Auto F0 adjust ВСЕГДА ВКЛЮЧЕН для лучшего сходства
    - 44100 Hz sample rate
    - Подходит как для речи, так и для пения

    Основной метод: voice_convert()
      source_audio: аудио с текстом/произношением (что говорить)
      reference_audio: аудио с целевым голосом (чьим голосом)
      Результат: контент source_audio в голосе reference_audio

    Параметры по умолчанию (максимальное сходство):
    - diffusion_steps=100 (макс качество)
    - inference_cfg_rate=0.8
    - auto_f0_adjust=True ← ВСЕГДА
    - f0_condition=True ← ВСЕГДА (F0 модель)
    """

    # Параметры по умолчанию (МАКСИМАЛЬНОЕ СХОДСТВО)
    DEFAULT_DIFFUSION_STEPS = 100
    DEFAULT_INFERENCE_CFG_RATE = 0.8
    DEFAULT_AUTO_F0_ADJUST = True    # ← ВСЕГДА ВКЛЮЧЕНО
    DEFAULT_F0_CONDITION = True      # ← ВСЕГДА ВКЛЮЧЕНО (эта модель F0)
    DEFAULT_PITCH_SHIFT = 0

    def __init__(self, seedvc_url: str = "http://localhost:8700", timeout: float = 1800.0):
        self._url = seedvc_url.rstrip("/")
        self._timeout = timeout  # 30 минут для HQ

    async def health_check(self) -> bool:
        """Проверка доступности Seed-VC сервера."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._url}/health")
                return resp.status_code == 200
        except Exception as e:
            logger.warning("Seed-VC health check failed: %s", e)
            return False

    async def voice_convert(
        self,
        source_wav_path: Path,
        reference_wav_path: Path,
        output_path: Path,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = DEFAULT_INFERENCE_CFG_RATE,
        auto_f0_adjust: bool = DEFAULT_AUTO_F0_ADJUST,  # ← ВСЕГДА TRUE
        f0_condition: bool = DEFAULT_F0_CONDITION,      # ← ВСЕГДА TRUE
        pitch_shift: int = DEFAULT_PITCH_SHIFT,
    ) -> Path:
        """
        Voice Conversion с помощью Seed-VC F0 Base.

        Seed-VC: Zero-shot voice conversion с F0 conditioning
        source_audio → аудио с нужным текстом/произношением
        reference_audio → аудио с целевым голосом (клонируемый голос)

        Результат: текст из source_audio в голосе reference_audio + перенос высоты.

        Параметры по умолчанию (F0 модель, макс сходство):
        - diffusion_steps=100
        - inference_cfg_rate=0.8
        - auto_f0_adjust=True ← ВСЕГДА
        - f0_condition=True ← ВСЕГДА (F0 модель whisper-base)

        Args:
            source_wav_path: Аудио с текстом/произношением (что говорить).
            reference_wav_path: Аудио с целевым голосом (чьим голосом говорить).
            output_path: Куда сохранить результат.
            diffusion_steps: Качество (100 default).
            length_adjust: Скорость (<1.0 быстрее, >1.0 медленнее).
            inference_cfg_rate: CFG rate (0.8 default).
            auto_f0_adjust: ВСЕГДА True (автоподстройка высоты).
            f0_condition: ВСЕГДА True (использует F0 модель).
            pitch_shift: Сдвиг высоты в полутонах (-24 до +24, default 0).

        Returns:
            Path к аудио с voice conversion результатом.
        """
        # Принудительно включаем F0 (эта модель работает ТОЛЬКО с F0)
        f0_condition = True
        auto_f0_adjust = True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        source_bytes = source_wav_path.read_bytes()
        reference_bytes = reference_wav_path.read_bytes()

        logger.info(
            "Seed-VC F0 voice convert: source=%s (%d bytes), ref=%s (%d bytes), "
            "steps=%d, cfg=%.2f, f0_condition=ALWAYS, f0_adjust=ALWAYS, pitch_shift=%d",
            source_wav_path.name, len(source_bytes),
            reference_wav_path.name, len(reference_bytes),
            diffusion_steps, inference_cfg_rate, pitch_shift,
        )

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            files = {
                "source": ("source.wav", source_bytes, "audio/wav"),
                "reference": ("reference.wav", reference_bytes, "audio/wav"),
            }
            data = {
                "diffusion_steps": str(diffusion_steps),
                "length_adjust": str(length_adjust),
                "inference_cfg_rate": str(inference_cfg_rate),
                "auto_f0_adjust": "true",       # ← ВСЕГДА TRUE
                "f0_condition": "true",          # ← ВСЕГДА TRUE
                "pitch_shift": str(pitch_shift),
            }

            response = await client.post(
                f"{self._url}/voice_convert",
                files=files,
                data=data,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Seed-VC voice convert error: {response.status_code} - {response.text}"
                )

            output_path.write_bytes(response.content)

        logger.info("Seed-VC F0 voice convert done: %s", output_path.name)
        return output_path