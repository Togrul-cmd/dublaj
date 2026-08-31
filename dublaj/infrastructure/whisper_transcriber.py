"""
Whisper Transcriber — HTTP-клиент к локальному сервису faster-whisper.

Реализует ITranscriber через OpenAI-совместимый эндпоинт /v1/audio/transcriptions,
который открывает Docker-контейнер faster-whisper-server.

Длинное аудио (> CHUNK_THRESHOLD_SECONDS) автоматически режется на куски
по CHUNK_SECONDS и отправляется ПО КУСКАМ — один запрос на кусок.
Таймкоды сегментов склеиваются со сдвигом на начало куска.
Причина: один гигантский запрос на 5+ минут валил контейнер Whisper
(смерть процесса / память), кусками — стабильно.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

from app.domain.entities import TranscriptionResult, TranscriptionSegment
from app.domain.interfaces import ITranscriber

logger = logging.getLogger(__name__)


class WhisperTranscriber(ITranscriber):
    """
    Отправляет аудио на локальный HTTP-сервис faster-whisper и парсит ответ.

    Аудио дольше 2 минут → нарезка на 2-минутные куски → последовательные
    запросы → склейка сегментов с корректными таймкодами.
    """

    # Аудио длиннее этого порога режется на куски (сек)
    CHUNK_THRESHOLD_SECONDS = 120.0
    # Длина одного куска (сек)
    CHUNK_SECONDS = 120.0

    def __init__(self, whisper_url: str, timeout: float = 1200.0) -> None:
        self._whisper_url = whisper_url
        self._timeout = timeout

    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Транскрипция: короткое аудио — одним запросом, длинное — по кускам."""
        duration = await self._get_duration(audio_path)

        if duration <= self.CHUNK_THRESHOLD_SECONDS:
            logger.info(
                "Sending %s to Whisper at %s (%.1fs, single request)",
                audio_path.name, self._whisper_url, duration,
            )
            return await self._transcribe_once(audio_path)

        # Длинное аудио → по кускам
        return await self._transcribe_chunked(audio_path, duration)

    # ── Чанковая транскрипция ───────────────────────────────────────────

    async def _transcribe_chunked(
        self, audio_path: Path, duration: float
    ) -> TranscriptionResult:
        """Режем аудио на куски, транскрибируем по одному, склеиваем."""
        chunk_files = await self._split_audio(audio_path)
        total_chunks = len(chunk_files)

        logger.info(
            "Whisper CHUNKED mode: %.1fs audio → %d chunks of ≤%.0fs",
            duration, total_chunks, self.CHUNK_SECONDS,
        )

        all_segments: list[TranscriptionSegment] = []
        detected_language = "unknown"
        chunk_texts: list[str] = []

        try:
            for i, (chunk_path, offset) in enumerate(chunk_files):
                logger.info(
                    "Whisper chunk %d/%d: [%.0fs..%.0fs]",
                    i + 1, total_chunks, offset,
                    min(offset + self.CHUNK_SECONDS, duration),
                )
                result = await self._transcribe_once(chunk_path)

                if i == 0:
                    detected_language = result.language

                # Сдвигаем таймкоды на начало куска в полном аудио
                for seg in result.segments:
                    all_segments.append(
                        TranscriptionSegment(
                            start=seg.start + offset,
                            end=seg.end + offset,
                            text=seg.text,
                        )
                    )
                if result.full_text:
                    chunk_texts.append(result.full_text)
        finally:
            # Чистим временные куски
            for chunk_path, _ in chunk_files:
                chunk_path.unlink(missing_ok=True)

        # Убираем петли галлюцинаций по всей склейке
        segments = self._deduplicate_segments(all_segments)
        full_text = " ".join(chunk_texts).strip()
        if not full_text and segments:
            full_text = " ".join(seg.text for seg in segments)

        logger.info(
            "Whisper chunked transcription complete — language=%s, segments=%d "
            "in %d chunks (raw=%d, removed %d duplicates)",
            detected_language, len(segments), total_chunks,
            len(all_segments), len(all_segments) - len(segments),
        )

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            full_text=full_text,
        )

    async def _split_audio(self, audio_path: Path) -> list[tuple[Path, float]]:
        """Разрезает WAV на куски по CHUNK_SECONDS. Возвращает [(путь, offset)]."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))
        pattern = str(tmp_dir / "chunk_%03d.wav")

        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(int(self.CHUNK_SECONDS)),
            "-c", "copy",
            pattern,
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg split error: {stderr.decode(errors='ignore')[:300]}"
            )

        chunks = sorted(tmp_dir.glob("chunk_*.wav"))
        return [(path, i * self.CHUNK_SECONDS) for i, path in enumerate(chunks)]

    async def _get_duration(self, audio_path: Path) -> float:
        """Длительность аудио через ffprobe (сек)."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        try:
            return float(stdout.decode().strip())
        except ValueError:
            # Не смогли узнать длительность — шлём целиком (как раньше)
            logger.warning("Could not get audio duration — sending whole file")
            return 0.0

    # ── Один запрос к Whisper ───────────────────────────────────────────

    async def _transcribe_once(self, audio_path: Path) -> TranscriptionResult:
        """Один запрос транскрипции (аудио ≤ CHUNK_SECONDS)."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            with open(audio_path, "rb") as audio_file:
                files = {"file": (audio_path.name, audio_file, "audio/wav")}
                data = {
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                    "max_chars": 200,  # Меньше символов, длиннее сегменты
                    "segment_size": 10,  # Делаем сегменты длиннее
                    "max_line_width": 0,  # Без автоматического разбиения
                    "temperature": 0.0,  # Greedy — убирает рандом, который вызывает петли
                    "compression_ratio_threshold": 2.4,  # Пропускает сегменты с повторами
                    "logprob_threshold": -1.0,  # Пропускает сегменты с низкой уверенностью
                    "no_speech_threshold": 0.6,  # Пропускает тишину
                    "condition_on_previous_text": False,  # НЕ зацикливаться на предыдущем тексте
                }

                response = await client.post(
                    self._whisper_url,
                    files=files,
                    data=data,
                )
                response.raise_for_status()

        result = response.json()

        # Парсим сегменты из развёрнутого JSON-ответа
        segments = [
            TranscriptionSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            )
            for seg in result.get("segments", [])
        ]

        full_text = result.get("text", "").strip()
        if not full_text and segments:
            full_text = " ".join(seg.text for seg in segments)

        detected_language = result.get("language", "unknown")

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            full_text=full_text,
        )

    @staticmethod
    def _deduplicate_segments(
        segments: list[TranscriptionSegment],
        similarity_threshold: float = 0.8,
    ) -> list[TranscriptionSegment]:
        """
        Убирает петли галлюцинаций — подряд идущие сегменты с одинаковым
        или очень похожим текстом.

        Пример петли галлюцинаций:
          [144.0-144.5] "Они дают картинку видимым и инфраструктурам."
          [144.5-145.0] "Они дают картинку видимым и инфраструктурам."
          [145.0-145.5] "Они дают картинку видимым и инфраструктурам."
          → оставляет только ПЕРВОЕ вхождение.
        """
        if not segments:
            return segments

        cleaned: list[TranscriptionSegment] = [segments[0]]
        duplicate_count = 0

        for seg in segments[1:]:
            prev = cleaned[-1]
            # Проверяем, идентичен ли текст
            if seg.text == prev.text:
                duplicate_count += 1
                continue

            # Проверяем схожесть (нормализованную)
            if len(seg.text) > 10 and len(prev.text) > 10:
                shorter = min(len(seg.text), len(prev.text))
                longer = max(len(seg.text), len(prev.text))
                # Если один текст является подстрокой другого → дубликат
                if seg.text in prev.text or prev.text in seg.text:
                    duplicate_count += 1
                    continue
                # Если длины похожи и текст сильно совпадает → дубликат
                if shorter / longer >= similarity_threshold:
                    # Простое посимвольное сравнение
                    matches = sum(a == b for a, b in zip(seg.text, prev.text))
                    if matches / longer >= similarity_threshold:
                        duplicate_count += 1
                        continue

            cleaned.append(seg)

        if duplicate_count > 0:
            logger.info(
                "Whisper: removed %d hallucinated duplicates (raw=%d → clean=%d)",
                duplicate_count,
                len(segments),
                len(cleaned),
            )

        return cleaned
