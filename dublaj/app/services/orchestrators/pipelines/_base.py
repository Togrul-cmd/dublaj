"""
Base Pipeline — shared setup logic for all pipeline strategies.

Provides common step helpers (extract audio, separate vocals,
transcribe, translate, speed-adjust, merge) so each concrete pipeline
only declares its own unique steps.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.entities import DubbingJob, JobStatus
from app.domain.interfaces import (
    IAudioAssembler,
    IAudioMixer,
    IPyAnnoteDiarization,
    ITranscriber,
    ITranslator,
    ITTSSynthesizer,
    IVoiceConverter,
    IVoiceSeparator,
)
from app.services.orchestrators.output_saver import OutputSaver

logger = logging.getLogger(__name__)


class PipelineContext:
    """
    Shared context passed through every pipeline step.
    Holds working directories and intermediate file paths.
    """

    def __init__(self, job: DubbingJob, temp_dir: Path, output_dir: Path) -> None:
        self.job = job
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.job_temp = temp_dir / job.id

        # Populated during pipeline execution
        self.extracted_audio_path: Path | None = None
        self.mute_video_path: Path | None = None
        self.ref_audio_path: Path | None = None
        # Full clean vocals — for Whisper transcription (entire audio).
        self.clean_vocals_path: Path | None = None
        # Instrumental (no vocals) — background music/noise for mixing with TTS.
        self.instrumental_path: Path | None = None
        # Audio used as input for Audio Separator / OmniVoice.
        # For long videos this is a trimmed sample of `extracted_audio_path`.
        self.separation_audio_path: Path | None = None
        # Speaker Diarization сегменты [{start, end, speaker}, ...]
        self.diarization_segments: list[dict] | None = None
        # Ref audio для каждого спикера: {speaker_id: path}
        self.speaker_refs: dict[str, Path] | None = None
        # Савер для сохранения всех артефактов в output/{job_id}/
        self.output_saver: OutputSaver | None = None

    def ensure_dirs(self) -> None:
        """Create all working directories."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.job_temp.mkdir(parents=True, exist_ok=True)
        # Инициализируем савер сразу — он создаст output/{job_id}/
        self.output_saver = OutputSaver(self.output_dir, self.job.id)


class BasePipeline(ABC):
    """
    Abstract base for all dubbing pipeline strategies.

    Subclasses implement `run()` with their unique pipeline steps.
    """

    def __init__(
        self,
        transcriber: ITranscriber,
        translator: ITranslator,
        tts: ITTSSynthesizer,
        mixer: IAudioMixer,
        voice_separator: IVoiceSeparator,
        assembler: IAudioAssembler,
        seedvc_converter: IVoiceConverter | None = None,
        pyannote: IPyAnnoteDiarization | None = None,
        *,
        use_voice_separation: bool = False,
        temp_dir: Path = Path("./tmp"),
        output_dir: Path = Path("./output"),
    ) -> None:
        self._transcriber = transcriber
        self._translator = translator
        self._tts = tts
        self._mixer = mixer
        self._voice_separator = voice_separator
        self._assembler = assembler
        self._seedvc_converter = seedvc_converter
        self._pyannote = pyannote
        self._use_voice_separation = use_voice_separation
        self._temp_dir = temp_dir
        self._output_dir = output_dir

    # ── Pipeline entry point ──────────────────────────────────────────

    async def execute(self, job: DubbingJob) -> DubbingJob:
        """
        Run the pipeline with error handling.
        Calls `run()` which subclasses implement.
        """
        ctx = PipelineContext(job, self._temp_dir, self._output_dir)
        try:
            ctx.ensure_dirs()
            await self.run(ctx)
        except Exception as exc:
            logger.exception("Job %s — pipeline %s failed", job.id, self.__class__.__name__)
            job.fail(str(exc))
        return job

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> None:
        """Implement the pipeline steps. Mutate ctx.job in-place."""
        ...

    # ── Shared step helpers ───────────────────────────────────────────

    async def step_extract_audio(self, ctx: PipelineContext) -> None:
        """Extract audio from video and create mute video."""
        ctx.job.status = JobStatus.EXTRACTING_AUDIO
        logger.info("Job %s — extracting audio from video", ctx.job.id)

        extracted = ctx.job_temp / "original_audio.wav"
        ctx.extracted_audio_path = await self._mixer.extract_audio(
            video_path=ctx.job.input_video_path,
            output_audio_path=extracted,
        )

        mute = ctx.job_temp / "mute_video.mp4"
        ctx.mute_video_path = await self._mixer.create_mute_video(
            video_path=ctx.job.input_video_path,
            output_video_path=mute,
        )

        # Сохраняем оригинальное аудио в output/{job_id}/
        if ctx.output_saver:
            ctx.output_saver.save_original_audio(ctx.extracted_audio_path)

    async def step_separate_vocals(self, ctx: PipelineContext) -> None:
        """Separate vocals from FULL audio (for Whisper).

        Logic:
        1. Separate vocals from the FULL extracted audio → clean_vocals_full.wav (for Whisper)
        2. ctx.ref_audio_path = clean_vocals_full (same file)
        """
        if self._use_voice_separation:
            ctx.job.status = JobStatus.SEPARATING_VOCALES
            logger.info("Job %s — separating vocals from background", ctx.job.id)

            # Step A: Separate vocals from FULL audio (for Whisper)
            clean_vocals_full = ctx.job_temp / "clean_vocals_full.wav"
            vocals_path, instrumental_path = await self._voice_separator.separate_vocals(
                audio_path=ctx.extracted_audio_path,
                output_path=clean_vocals_full,
            )
            ctx.clean_vocals_path = vocals_path
            ctx.instrumental_path = instrumental_path
            logger.info(
                "Job %s — full clean vocals saved: %s (for Whisper)",
                ctx.job.id, ctx.clean_vocals_path.name,
            )
            if instrumental_path:
                logger.info(
                    "Job %s — instrumental saved: %s (for background mix)",
                    ctx.job.id, instrumental_path.name,
                )

            # Set ref_audio_path for OmniVoice
            if ctx.separation_audio_path and ctx.separation_audio_path != ctx.extracted_audio_path:
                # Long video: separate vocals from trimmed piece
                clean_vocals_trimmed = ctx.job_temp / "clean_vocals_trimmed.wav"
                vocals_trimmed, _ = await self._voice_separator.separate_vocals(
                    audio_path=ctx.separation_audio_path,
                    output_path=clean_vocals_trimmed,
                )
                ctx.ref_audio_path = vocals_trimmed
                logger.info(
                    "Job %s — trimmed clean vocals: %s",
                    ctx.job.id, ctx.ref_audio_path.name,
                )
            else:
                # Short video: same file for both
                ctx.ref_audio_path = ctx.clean_vocals_path

            # Сохраняем чистый вокал в output/{job_id}/
            if ctx.output_saver:
                ctx.output_saver.save_clean_vocals(ctx.clean_vocals_path)
        else:
            ctx.ref_audio_path = ctx.separation_audio_path or ctx.extracted_audio_path

    async def step_trim_for_voice_clone(
        self,
        ctx: PipelineContext,
        threshold_seconds: float = 300.0,
        sample_seconds: float = 60.0,
    ) -> None:
        """
        If the source video is longer than `threshold_seconds`, set
        `ctx.separation_audio_path` to the first `sample_seconds` of audio.

        Voice cloning (OmniVoice) only needs a short reference clip,
        so we can skip processing the rest of the audio for long videos.
        """
        if ctx.extracted_audio_path is None:
            return

        try:
            duration = await self._mixer.get_duration(ctx.extracted_audio_path)
        except Exception as exc:
            logger.warning("Job %s — could not get audio duration: %s", ctx.job.id, exc)
            return

        if duration <= threshold_seconds:
            logger.info(
                "Job %s — audio is %.1fs (≤%ss threshold), no trim needed",
                ctx.job.id, duration, threshold_seconds,
            )
            ctx.separation_audio_path = ctx.extracted_audio_path
            return

        trimmed = ctx.job_temp / f"voice_clone_sample_{int(sample_seconds)}s.wav"
        await self._mixer.extract_audio_segment(
            video_path=ctx.job.input_video_path,
            output_audio_path=trimmed,
            max_seconds=sample_seconds,
        )
        logger.info(
            "Job %s — video is %.1fs long, trimmed first %ss for voice clone ref",
            ctx.job.id, duration, sample_seconds,
        )
        ctx.separation_audio_path = trimmed

    async def step_transcribe(self, ctx: PipelineContext) -> None:
        """Transcribe audio via Whisper."""
        ctx.job.status = JobStatus.TRANSCRIBING
        logger.info("Job %s — transcribing audio", ctx.job.id)

        # Приоритет: clean_vocals_full (весь вокал без музыки) > оригинал
        audio_for_whisper = ctx.clean_vocals_path or ctx.extracted_audio_path
        logger.info("Job %s — Whisper using: %s", ctx.job.id, audio_for_whisper.name)

        ctx.job.transcription = await self._transcriber.transcribe(
            audio_path=audio_for_whisper,
        )
        logger.info(
            "Job %s — detected language: %s, segments: %d",
            ctx.job.id,
            ctx.job.transcription.language,
            len(ctx.job.transcription.segments),
        )
        # Сохраняем Whisper транскрипцию в output/{job_id}/
        if ctx.output_saver:
            ctx.output_saver.save_whisper_transcript(ctx.job.transcription)

    async def step_translate(self, ctx: PipelineContext) -> None:
        """Translate full transcription text."""
        ctx.job.status = JobStatus.TRANSLATING
        logger.info("Job %s — translating to %s", ctx.job.id, ctx.job.target_language)

        ctx.job.translation = await self._translator.translate(
            text=ctx.job.transcription.full_text,
            source_language=ctx.job.transcription.language,
            target_language=ctx.job.target_language,
        )
        # Сохраняем перевод в output/{job_id}/
        if ctx.output_saver:
            ctx.output_saver.save_translation(ctx.job.translation)

    async def step_translate_segments(self, ctx: PipelineContext):
        """Translate individual segments. Returns list of TranslatedSegment."""
        ctx.job.status = JobStatus.TRANSLATING
        logger.info("Job %s — translating %d segments",
                    ctx.job.id, len(ctx.job.transcription.segments))

        translated = await self._translator.translate_segments(
            segments=ctx.job.transcription.segments,
            source_language=ctx.job.transcription.language,
            target_language=ctx.job.target_language,
        )
        # Сохраняем перевод по сегментам в output/{job_id}/
        if ctx.output_saver:
            ctx.output_saver.save_translated_segments(
                translated,
                source_language=ctx.job.transcription.language,
                target_language=ctx.job.target_language,
            )
        return translated

    async def step_adjust_speed(
        self,
        ctx: PipelineContext,
        audio_path: Path,
        reference_path: Path | None = None,
        output_name: str = "adjusted.wav",
    ) -> Path:
        """
        Speed-adjust audio to match original duration.
        Returns adjusted path or original if no adjustment needed.
        """
        ref = reference_path or ctx.extracted_audio_path
        original_duration = await self._mixer.get_duration(ref)
        audio_duration = await self._mixer.get_duration(audio_path)

        logger.info(
            "Job %s — durations: original=%.1fs, audio=%.1fs",
            ctx.job.id, original_duration, audio_duration,
        )

        if audio_duration > 0 and original_duration > 0:
            ratio = audio_duration / original_duration
            if ratio > 1.5 or ratio < 0.67:
                atempo = min(max(ratio, 0.5), 1.5)
                logger.info(
                    "Job %s — adjusting speed: ratio=%.2f, atempo=%.2f",
                    ctx.job.id, ratio, atempo,
                )
                adjusted = ctx.job_temp / output_name
                return await self._mixer.apply_atempo(audio_path, atempo, adjusted)

        return audio_path

    async def step_merge_video(
        self,
        ctx: PipelineContext,
        audio_path: Path,
        filename_prefix: str = "dubbed",
    ) -> None:
        """Merge audio into mute video and set output path on job."""
        ctx.job.status = JobStatus.MERGING
        logger.info("Job %s — merging audio into video", ctx.job.id)

        output_video = self._output_dir / f"{filename_prefix}_{ctx.job.id}.mp4"
        ctx.job.output_video_path = await self._mixer.merge_audio_into_video(
            video_path=ctx.mute_video_path,
            audio_path=audio_path,
            output_video_path=output_video,
        )

    def finalize(self, ctx: PipelineContext, label: str = "") -> None:
        """Mark job as completed and save final artifacts + README."""
        ctx.job.status = JobStatus.COMPLETED
        # Сохраняем финальное TTS аудио и видео, пишем README
        if ctx.output_saver:
            ctx.output_saver.save_synthesized_audio(ctx.job.synthesized_audio_path)
            ctx.output_saver.save_final_video(ctx.job.output_video_path)
            ctx.output_saver.write_readme(pipeline_name=label, job_id=ctx.job.id)
            ctx.output_saver.log_summary()
        logger.info("Job %s — %s completed: %s", ctx.job.id, label, ctx.job.output_video_path)

    def get_ref_text(self, ctx: PipelineContext) -> str:
        """Get ref_text for TTS."""
        return ctx.job.transcription.full_text
