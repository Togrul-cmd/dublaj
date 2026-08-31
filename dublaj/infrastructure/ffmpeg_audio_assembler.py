# FFmpeg Audio Assembler — размещает аудиосегменты с таймкодами на таймлайне.
# Берёт аудиофайлы каждого сегмента с временем начала/конца, заполняет промежутки тишиной
# и склеивает всё в одну непрерывную аудиодорожку, совпадающую с длительностью оригинального видео.

import asyncio
import logging
from pathlib import Path

from app.domain.interfaces import IAudioAssembler

logger = logging.getLogger(__name__)


# Собирает аудиосегменты с таймкодами через ffmpeg concat demuxer
class FFmpegAudioAssembler(IAudioAssembler):

    # Конструктор — принимает путь к ffmpeg и частоту дискретизации
    def __init__(self, ffmpeg_path: str = "ffmpeg", sample_rate: int = 24000) -> None:
        """Инициализирует assembler с путём к ffmpeg и частотой дискретизации."""
        self._ffmpeg = ffmpeg_path
        self._sample_rate = sample_rate

    # Собирает сегменты в одно аудио, заполняя промежутки тишиной
    async def assemble_segments(
        self,
        segment_audios: list[tuple[Path, float, float]],
        total_duration: float,
        output_path: Path,
    ) -> Path:
        """Собирает аудиосегменты в одно непрерывное аудио, заполняя промежутки тишиной."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not segment_audios:
            logger.warning("No segments — generating silent audio of %.1fs", total_duration)
            await self._generate_silence(total_duration, output_path)
            return output_path

        work_dir = output_path.parent / "assemble_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        concat_entries: list[Path] = []
        current_time = 0.0

        for i, (audio_path, start, end) in enumerate(segment_audios):
            seg_duration = end - start
            if seg_duration <= 0:
                continue

            # Нормализуем и обрезаем сегмент до точной длительности
            norm_path = work_dir / f"seg_{i:03d}.wav"
            await self._normalize_and_trim(audio_path, seg_duration, norm_path)

            # Тишина перед этим сегментом (зазор)
            if start > current_time + 0.05:
                gap = start - current_time
                silence_path = work_dir / f"silence_{i:03d}.wav"
                await self._generate_silence(gap, silence_path)
                concat_entries.append(silence_path)

            concat_entries.append(norm_path)
            current_time = end

        # Тишина в конце, чтобы совпадало с общей длительностью
        if current_time < total_duration - 0.05:
            gap = total_duration - current_time
            silence_path = work_dir / "silence_final.wav"
            await self._generate_silence(gap, silence_path)
            concat_entries.append(silence_path)

        # Записываем список файлов с абсолютными путями для склейки
        concat_list = work_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for entry in concat_entries:
                f.write(f"file '{entry.resolve()}'\n")

        # Склеиваем все файлы в одно аудио
        cmd = [
            self._ffmpeg,
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:a", "pcm_s16le",
            "-ar", str(self._sample_rate),
            "-ac", "1",
            "-y", str(output_path),
        ]
        await self._run(cmd)

        # Удаляем временные файлы
        for p in concat_entries:
            p.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)
        work_dir.rmdir()

        logger.info("Assembled %d segments → %s (%.1fs)", len(segment_audios), output_path.name, total_duration)
        return output_path

    # Перекодирует аудио в единый PCM формат
    async def _normalize_audio(self, src: Path, dst: Path) -> None:
        """Перекодирует аудио в единый PCM формат (WAV 16-bit)."""
        cmd = [
            self._ffmpeg, "-y", "-i", str(src),
            "-ar", str(self._sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
            str(dst),
        ]
        await self._run(cmd)

    # Перекодирует аудио и выравнивает до точной длительности окна
    async def _normalize_and_trim(self, src: Path, duration: float, dst: Path) -> None:
        """
        Перекодирует аудио в единый формат и приводит к точной длительности:
        apad дополняет тишиной, если аудио короче окна (пауза после речи),
        -t обрезает, если длиннее. Без этого таймлайн уезжает при склейке.
        """
        cmd = [
            self._ffmpeg, "-y", "-i", str(src),
            # apad: дополняем тишиной до окна; afade: 20мс fade-in гасит
            # стартовый щелчок Seed-VC ("к" в начале куска) — неслышимо для речи
            "-af", "apad,afade=t=in:d=0.02",
            "-t", f"{duration:.3f}",
            "-ar", str(self._sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
            str(dst),
        ]
        await self._run(cmd)

    # Генерирует тихий WAV-файл заданной длительности
    async def _generate_silence(self, duration: float, dst: Path) -> None:
        """Создаёт WAV-файл с тишиной указанной длительности."""
        cmd = [
            self._ffmpeg,
            "-f", "lavfi", "-i", f"anullsrc=r={self._sample_rate}:cl=mono",
            "-t", f"{duration:.3f}",
            "-c:a", "pcm_s16le", "-ar", str(self._sample_rate), "-ac", "1",
            "-y", str(dst),
        ]
        await self._run(cmd)

    # Асинхронно выполняет команду ffmpeg
    async def _run(self, cmd: list[str]) -> None:
        """Запускает команду ffmpeg и обрабатывает ошибки."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error (code {process.returncode}): {stderr.decode().strip()}")
