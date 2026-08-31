# FFmpeg Audio Mixer — извлекает аудио из видео и накладывает новое аудио обратно.
# Реализует IAudioMixer через вызовы ffmpeg в subprocess.
# Путь к ffmpeg настраивается через Settings.

import asyncio
import logging
from pathlib import Path

from app.domain.interfaces import IAudioMixer

logger = logging.getLogger(__name__)


class FFmpegAudioMixer(IAudioMixer):
    """
    Использует ffmpeg для извлечения и склейки аудиодорожек в видеофайлах.
    Все пути и сам ffmpeg настраиваются — ничего не захардкожено.
    """

    # Конструктор — принимает путь к ffmpeg
    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg_path

    # Извлекает аудио из видеофайла
    async def extract_audio(self, video_path: Path, output_audio_path: Path) -> Path:
        # Извлекает аудио из видео
        logger.info("Extracting audio: %s → %s", video_path, output_audio_path)

        cmd = [
            self._ffmpeg,
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output_audio_path),
        ]

        await self._run_ffmpeg(cmd)
        logger.info("Audio extraction complete: %s", output_audio_path)
        return output_audio_path

    async def extract_audio_segment(
        self,
        video_path: Path,
        output_audio_path: Path,
        max_seconds: float,
    ) -> Path:
        """
        Извлекает только первые `max_seconds` секунд аудио из видео.
        Используется как оптимизация скорости для длинных видео:
        для voice cloning нужен только короткий референсный клип,
        поэтому можно не обрабатывать весь файл целиком.
        """
        logger.info(
            "Extracting first %.1fs of audio: %s → %s",
            max_seconds, video_path, output_audio_path,
        )

        cmd = [
            self._ffmpeg,
            "-i", str(video_path),
            "-t", str(max_seconds),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output_audio_path),
        ]

        await self._run_ffmpeg(cmd)
        logger.info("Segmented audio extraction complete: %s", output_audio_path)
        return output_audio_path

    async def extract_audio_from_to(
        self,
        video_path: Path,
        output_audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        """
        Извлекает аудио из конкретного диапазона времени (от и до).
        Используется для извлечения sample голоса конкретного спикера
        по timestamp из diarization.
        """
        duration = end_seconds - start_seconds
        logger.info(
            "Extracting audio [%.1f-%.1fs] (%.1fs): %s → %s",
            start_seconds, end_seconds, duration, video_path, output_audio_path,
        )

        cmd = [
            self._ffmpeg,
            "-i", str(video_path),
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output_audio_path),
        ]

        await self._run_ffmpeg(cmd)
        logger.info("Audio segment extraction complete: %s", output_audio_path)
        return output_audio_path

    async def concat_audio_segments(
        self,
        video_path: Path,
        segments: list[dict],
        output_path: Path,
        max_duration: float = 30.0,
    ) -> Path:
        """
        Склеивает несколько аудиосегментов из видео в один файл.
        Если общая длительность > max_duration, обрезает до max_duration.

        Args:
            video_path: Исходный видеофайл
            segments: Список словарей {"start": float, "end": float}
            output_path: Куда сохранить склеенное аудио
            max_duration: Максимальная длительность в секундах (по умолчанию 30.0)

        Returns:
            Путь к склеенному аудиофайлу
        """
        import tempfile

        # Сортируем сегменты по времени начала (хронологически)
        sorted_segs = sorted(segments, key=lambda s: s["start"])

        # Извлекаем каждый сегмент во временные файлы
        temp_files = []
        total_duration = 0.0
        for i, seg in enumerate(sorted_segs):
            seg_duration = seg["end"] - seg["start"]
            if seg_duration <= 0:
                continue
            # Проверяем, не превышает ли добавление этого сегмента max_duration
            if total_duration + seg_duration > max_duration:
                # Берём только оставшееся время
                remaining = max_duration - total_duration
                if remaining < 0.5:  # Меньше 0.5с — пропускаем
                    break
                seg_duration = remaining

            temp_path = Path(tempfile.mktemp(suffix=f"_seg_{i:04d}.wav"))
            await self.extract_audio_from_to(
                video_path=video_path,
                output_audio_path=temp_path,
                start_seconds=seg["start"],
                end_seconds=seg["start"] + seg_duration,
            )
            temp_files.append(temp_path)
            total_duration += seg_duration

            if total_duration >= max_duration:
                break

        if not temp_files:
            raise ValueError("No valid segments to concatenate")

        # Склеиваем все сегменты через ffmpeg concat demuxer
        concat_list = Path(tempfile.mktemp(suffix="_concat.txt"))
        with open(concat_list, "w") as f:
            for p in temp_files:
                f.write(f"file '{p.resolve()}'\n")

        cmd = [
            self._ffmpeg,
            "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-t", str(max_duration),
            str(output_path),
        ]

        await self._run_ffmpeg(cmd)

        # Удаляем временные файлы
        for p in temp_files:
            p.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

        logger.info(
            "Concatenated %d segments → %s (%.1fs, max=%.1fs)",
            len(temp_files), output_path.name, total_duration, max_duration,
        )
        return output_path

    async def get_video_duration(self, video_path: Path) -> float:
        # Возвращает длительность видео в секундах через ffprobe
        return await self.get_duration(video_path)

    async def create_mute_video(self, video_path: Path, output_video_path: Path) -> Path:
        # Создаёт видео без звука
        logger.info("Creating mute video: %s → %s", video_path, output_video_path)

        cmd = [
            self._ffmpeg,
            "-i", str(video_path),
            "-c:v", "copy",
            "-an",
            "-y",
            str(output_video_path),
        ]

        await self._run_ffmpeg(cmd)
        logger.info("Mute video created: %s", output_video_path)
        return output_video_path

    async def get_duration(self, audio_path: Path) -> float:
        # Возвращает длительность аудио/видео в секундах через ffprobe
        cmd = [
            self._ffmpeg.replace("ffmpeg", "ffprobe"),
            "-v", "error",
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
        return float(stdout.decode().strip())

    async def adjust_audio_speed(self, audio_path: Path, target_duration: float, output_path: Path) -> Path:
        # Убирает тишину, затем плавно подстраивает скорость под целевую длительность
        current_duration = await self.get_duration(audio_path)
        if current_duration <= 0 or target_duration <= 0:
            return audio_path

        # Шаг 1: Убираем тишину (паузы > 0.5с сжимаются до 0.15с)
        silenced = output_path.parent / "silenced.wav"
        cmd_silence = [
            self._ffmpeg,
            "-i", str(audio_path),
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.4:stop_threshold=-30dB",
            "-y",
            str(silenced),
        ]
        await self._run_ffmpeg(cmd_silence)

        silenced_duration = await self.get_duration(silenced)
        ratio = silenced_duration / target_duration

        # Шаг 2: Если после удаления тишины уже достаточно близко — используем как есть
        if abs(ratio - 1.0) < 0.03:
            logger.info("Audio OK after silence removal: %.1fs → %.1fs", current_duration, silenced_duration)
            silenced.rename(output_path)
            return output_path

        # Шаг 3: Плавная подстройка скорости (макс 1.3x для естественного звучания)
        atempo = min(max(ratio, 0.7), 1.3)
        logger.info(
            "Adjusting: %.1fs → silence removed → %.1fs → speed x%.2f → %.1fs",
            current_duration, silenced_duration, atempo, target_duration,
        )
        cmd_speed = [
            self._ffmpeg,
            "-i", str(silenced),
            "-filter:a", f"atempo={atempo}",
            "-y",
            str(output_path),
        ]
        await self._run_ffmpeg(cmd_speed)
        silenced.unlink(missing_ok=True)
        logger.info("Audio adjusted: %s", output_path)
        return output_path

    async def remove_silence(self, audio_path: Path, output_path: Path) -> Path:
        # Убирает тишину из аудио без изменения скорости
        cmd = [
            self._ffmpeg,
            "-i", str(audio_path),
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.4:stop_threshold=-30dB",
            "-y", str(output_path),
        ]
        await self._run_ffmpeg(cmd)
        return output_path

    async def apply_atempo(self, audio_path: Path, atempo: float, output_path: Path) -> Path:
        """
        Применяет коэффициент скорости к аудио.

        Args:
            audio_path: Путь к исходному аудиофайлу.
            atempo: Коэффициент скорости (1.0 = без изменений, 1.5 = ускорить в 1.5 раза).
            output_path: Куда сохранить результат.

        Returns:
            Путь к обработанному аудиофайлу.
        """
        if abs(atempo - 1.0) < 0.02:
            return audio_path
        cmd = [
            self._ffmpeg,
            "-i", str(audio_path),
            "-filter:a", f"atempo={atempo:.3f}",
            "-y", str(output_path),
        ]
        await self._run_ffmpeg(cmd)
        return output_path

    async def merge_audio_into_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_video_path: Path,
    ) -> Path:
        # Накладывает новую аудиодорожку на видео
        logger.info(
            "Merging audio into video: %s + %s → %s",
            video_path,
            audio_path,
            output_video_path,
        )

        cmd = [
            self._ffmpeg,
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-y",
            str(output_video_path),
        ]

        await self._run_ffmpeg(cmd)
        logger.info("Audio merge complete: %s", output_video_path)
        return output_video_path

    async def _run_ffmpeg(self, cmd: list[str]) -> None:
        # Асинхронно выполняет команду ffmpeg и обрабатывает ошибки
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error("FFmpeg failed (code %d): %s", process.returncode, error_msg)
            raise RuntimeError(f"FFmpeg error (code {process.returncode}): {error_msg}")

    async def extract_segment(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> Path:
        # Вырезает сегмент из аудиофайла по таймкодам
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid segment duration: {duration}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(duration),
            "-ac", "1",
            "-ar", "24000",
            str(output_path),
        ]

        logger.info(
            "Extracting segment [%.1f-%.1fs] from %s",
            start_time, end_time, audio_path.name,
        )
        await self._run_ffmpeg(cmd)

        logger.info("Segment extraction complete: %s", output_path)
        return output_path

    async def split_audio(
        self,
        audio_path: Path,
        chunk_seconds: float,
        output_dir: Path,
    ) -> list[tuple[Path, float, float]]:
        """
        Разбивает аудиофайл на равные части.

        Каждая часть извлекается через ffmpeg с точным позиционированием.
        Последний кусок может быть короче, если аудио делится не нацело.

        Args:
            audio_path: Путь к аудиофайлу для разбивки.
            chunk_seconds: Длительность каждой части в секундах.
            output_dir: Куда сохранить файлы частей.

        Returns:
            Список кортежей (путь_к_части, время_начала, время_конца).
        """
        total_duration = await self.get_duration(audio_path)
        num_chunks = int(total_duration / chunk_seconds) + (1 if total_duration % chunk_seconds > 0 else 0)

        logger.info(
            "Splitting %s (%.1fs) into %d chunks of %.1fs each",
            audio_path.name, total_duration, num_chunks, chunk_seconds,
        )

        chunks: list[tuple[Path, float, float]] = []
        for i in range(num_chunks):
            start = i * chunk_seconds
            end = min((i + 1) * chunk_seconds, total_duration)
            chunk_path = output_dir / f"vocals_chunk_{i:04d}.wav"

            await self.extract_segment(audio_path, start, end, chunk_path)
            chunks.append((chunk_path, start, end))
            logger.info(
                "  Chunk %d/%d: %.1f-%.1fs (%.1fs) → %s",
                i + 1, num_chunks, start, end, end - start, chunk_path.name,
            )

        logger.info("Split complete: %d chunks from %.1fs audio", len(chunks), total_duration)
        return chunks

    async def mix_audios(
        self,
        background_path: Path,
        foreground_path: Path,
        output_path: Path,
        foreground_volume: float = 1.0,
        background_volume: float = 0.5,
    ) -> Path:
        """
        Смешивает два аудиофайла: фоновый (музыка/шум) + основной (TTS).

        Args:
            background_path: Инструментал/фоновое аудио.
            foreground_path: TTS аудио (голос).
            output_path: Куда сохранить смешанное аудио.
            foreground_volume: Громкость основного (TTS). По умолчанию 1.0.
            background_volume: Громкость фона. По умолчанию 0.5 (тише).

        Returns:
            Путь к смешанному аудиофайлу.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._ffmpeg, "-y",
            "-i", str(background_path),
            "-i", str(foreground_path),
            "-filter_complex",
            f"[0:a]volume={background_volume}[bg];"
            f"[1:a]volume={foreground_volume}[fg];"
            f"[bg][fg]amix=inputs=2:duration=longest:dropout_transition=2[out]",
            "-map", "[out]",
            str(output_path),
        ]

        logger.info(
            "Mixing audio: bg=%s (vol=%.1f) + fg=%s (vol=%.1f) → %s",
            background_path.name, background_volume,
            foreground_path.name, foreground_volume,
            output_path.name,
        )
        await self._run_ffmpeg(cmd)
        logger.info("Audio mix complete: %s", output_path)
        return output_path
