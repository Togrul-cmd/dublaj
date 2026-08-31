"""
Output Saver — копирует промежуточные артефакты пайплайна в output/{job_id}/.

Сохраняет всё, что генерирует пайплайн, чтобы пользователь мог видеть
и использовать файлы после завершения:
  1. Оригинальное аудио (из видео)
  2. Вокал после Audio Separator
  3. Whisper транскрипция (txt + json с таймкодами)
  4. Перевод (txt + json с таймкодами)
  5. Синтезированное TTS аудио
  6. Финальное видео
  7. README со сводкой
"""

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.entities import TranscriptionResult, TranslationResult

logger = logging.getLogger(__name__)


class OutputSaver:
    """
    Копирует артефакты пайплайна из job_temp в output/{job_id}/ с человеко-читаемыми именами.

    Все методы безопасны: если исходный файл не существует — пропускают без ошибки.
    """

    def __init__(self, output_dir: Path, job_id: str) -> None:
        self._output_dir = output_dir
        self._job_id = job_id
        # Финальная папка для всех артефактов задачи
        self._job_output_dir = output_dir / job_id
        self._job_output_dir.mkdir(parents=True, exist_ok=True)
        # Сводка по сохранённым файлам для README
        self._summary: list[str] = []

    @property
    def job_output_dir(self) -> Path:
        """Папка с сохранёнными артефактами."""
        return self._job_output_dir

    def _copy(self, src: Path | None, dest_name: str) -> Path | None:
        """Скопировать файл в output/{job_id}/, если src существует. Вернуть путь или None."""
        if src is None or not Path(src).exists():
            return None
        dest = self._job_output_dir / dest_name
        try:
            shutil.copy2(src, dest)
            size_kb = dest.stat().st_size / 1024
            self._summary.append(f"  • {dest_name}  ({size_kb:.1f} KB)")
            logger.info("  💾 Saved: %s  (%.1f KB)", dest_name, size_kb)
            return dest
        except Exception as exc:
            logger.warning("  ⚠ Failed to save %s: %s", dest_name, exc)
            return None

    def save_original_audio(self, audio_path: Path | None) -> None:
        """Шаг 1: Оригинальное аудио из видео."""
        self._copy(audio_path, "1_original_audio.wav")

    def save_clean_vocals(self, vocals_path: Path | None) -> None:
        """Шаг 2: Чистый вокал после Audio Separator."""
        self._copy(vocals_path, "2_clean_vocals.wav")

    def save_whisper_transcript(
        self,
        transcription: "TranscriptionResult | None",
    ) -> None:
        """
        Шаг 3: Whisper транскрипция — текст + JSON с таймкодами.

        Сохраняет:
          - 3_whisper_transcript.txt  (полный текст)
          - 3_whisper_transcript.json (сегменты с таймкодами + язык)
        """
        if transcription is None:
            return

        # TXT — plain text
        txt_path = self._job_output_dir / "3_whisper_transcript.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Whisper Transcription ===\n")
                f.write(f"Language: {transcription.language}\n")
                f.write(f"Segments: {len(transcription.segments)}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(transcription.full_text)
            self._summary.append(f"  • 3_whisper_transcript.txt  ({len(transcription.full_text)} chars)")
            logger.info("  💾 Saved: 3_whisper_transcript.txt")
        except Exception as exc:
            logger.warning("  ⚠ Failed to save whisper txt: %s", exc)

        # JSON — сегменты с таймкодами
        json_path = self._job_output_dir / "3_whisper_transcript.json"
        try:
            data = {
                "language": transcription.language,
                "full_text": transcription.full_text,
                "segments": [
                    {
                        "index": i,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for i, seg in enumerate(transcription.segments)
                ],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._summary.append(
                f"  • 3_whisper_transcript.json  ({len(transcription.segments)} segments)"
            )
            logger.info("  💾 Saved: 3_whisper_transcript.json  (%d segments)",
                        len(transcription.segments))
        except Exception as exc:
            logger.warning("  ⚠ Failed to save whisper json: %s", exc)

    def save_translation(
        self,
        translation: "TranslationResult | None",
    ) -> None:
        """
        Шаг 4: Перевод — текст.

        Сохраняет:
          - 4_translation.txt  (plain text)
        """
        if translation is None:
            return

        txt_path = self._job_output_dir / "4_translation.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Translation ===\n")
                f.write(f"Source: {translation.source_language}\n")
                f.write(f"Target: {translation.target_language}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"--- ORIGINAL ({translation.source_language}) ---\n")
                f.write(translation.original_text)
                f.write(f"\n\n--- TRANSLATED ({translation.target_language}) ---\n")
                f.write(translation.translated_text)
            self._summary.append(
                f"  • 4_translation.txt  "
                f"({len(translation.original_text)} → {len(translation.translated_text)} chars)"
            )
            logger.info("  💾 Saved: 4_translation.txt")
        except Exception as exc:
            logger.warning("  ⚠ Failed to save translation txt: %s", exc)

    def save_translated_segments(
        self,
        translated_segments: list | None,
        source_language: str = "",
        target_language: str = "",
    ) -> None:
        """
        Шаг 4b: Перевод по сегментам (с таймкодами Whisper).

        Сохраняет:
          - 4_segments.json  (массив с таймкодами)
          - 4_segments.txt   (человеко-читаемый: [время] оригинал → перевод)
        """
        if not translated_segments:
            return

        # JSON — массив сегментов
        json_path = self._job_output_dir / "4_segments.json"
        try:
            data = {
                "source_language": source_language,
                "target_language": target_language,
                "segments": [
                    {
                        "index": seg.index,
                        "start": seg.start,
                        "end": seg.end,
                        "original_text": seg.original_text,
                        "translated_text": seg.translated_text,
                    }
                    for seg in translated_segments
                ],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._summary.append(
                f"  • 4_segments.json  ({len(translated_segments)} segments)"
            )
            logger.info("  💾 Saved: 4_segments.json  (%d segments)",
                        len(translated_segments))
        except Exception as exc:
            logger.warning("  ⚠ Failed to save segments json: %s", exc)

        # TXT — side-by-side с таймкодами
        txt_path = self._job_output_dir / "4_segments.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"=== Translation (per segment) ===\n")
                f.write(f"Source: {source_language} → Target: {target_language}\n")
                f.write(f"Segments: {len(translated_segments)}\n")
                f.write(f"{'=' * 70}\n\n")
                for seg in translated_segments:
                    f.write(
                        f"[{seg.index:3d}] "
                        f"[{seg.start:6.2f} - {seg.end:6.2f}s]\n"
                        f"  {source_language}: {seg.original_text}\n"
                        f"  {target_language}: {seg.translated_text}\n\n"
                    )
            self._summary.append(
                f"  • 4_segments.txt  ({len(translated_segments)} segments)"
            )
            logger.info("  💾 Saved: 4_segments.txt")
        except Exception as exc:
            logger.warning("  ⚠ Failed to save segments txt: %s", exc)

    def save_synthesized_audio(self, audio_path: Path | None) -> None:
        """Шаг 5: Финальное TTS аудио (клонированный голос)."""
        self._copy(audio_path, "5_synthesized_audio.wav")

    def save_final_video(self, video_path: Path | None) -> None:
        """Шаг 6: Финальное видео с дубляжом."""
        self._copy(video_path, "6_final_video.mp4")

    def write_readme(self, pipeline_name: str, job_id: str) -> None:
        """
        Записать README.md со сводкой всех сохранённых файлов.
        Вызывается в конце пайплайна.
        """
        readme = self._job_output_dir / "README.md"
        try:
            with open(readme, "w", encoding="utf-8") as f:
                f.write(f"# Dubbing Job: {job_id}\n\n")
                f.write(f"**Pipeline:** `{pipeline_name}`\n\n")
                f.write(f"## Saved Artifacts\n\n")
                if self._summary:
                    for line in self._summary:
                        f.write(line + "\n")
                else:
                    f.write("_No files were saved._\n")
                f.write(f"\n## File Descriptions\n\n")
                f.write(
                    "1. **1_original_audio.wav** — оригинальное аудио, извлечённое из видео.\n"
                    "2. **2_clean_vocals.wav** — вокал, отделённый от фоновой музыки "
                    "(через Audio Separator / UVR).\n"
                    "3. **3_whisper_transcript.{txt,json}** — исходная транскрипция от Whisper. "
                    "JSON содержит таймкоды.\n"
                    "4. **4_translation.txt** — перевод полным текстом. "
                    "**4_segments.{txt,json}** — перевод по сегментам с таймкодами.\n"
                    "5. **5_synthesized_audio.wav** — синтезированный TTS с клонированным голосом.\n"
                    "6. **6_final_video.mp4** — финальное видео с новой озвучкой.\n"
                )
            logger.info("  💾 Saved: README.md")
        except Exception as exc:
            logger.warning("  ⚠ Failed to save README: %s", exc)

    def log_summary(self) -> None:
        """Красиво напечатать в лог что куда сохранилось."""
        logger.info("=" * 60)
        logger.info("📁 All artifacts saved to: %s", self._job_output_dir)
        logger.info("=" * 60)
        for line in self._summary:
            logger.info(line)
        logger.info("=" * 60)
