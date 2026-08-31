"""
Seed-VC Speaker Diarization V2 Pipeline — БЕЗ F5-TTS, с Seed-VC V1!

Pipeline:
  Extract → Separate Vocals → PyAnnote Diarization (определяем спикеров)
  → Для КАЖДОГО спикера: извлечь sample из clean_vocals (30 сек)
  → Whisper per-segment (с пометкой speaker)
  → Translate per-segment
  → OmniVoice DEFAULT per-segment (per-speaker ref_audio)
  → Seed-VC V1 per-segment: source=default, reference=clean_vocals_sample
  → Assemble → Mix instrumental → Merge video
"""

import asyncio
import logging
from bisect import bisect_right
from pathlib import Path

from app.domain.entities import DubbingJob, JobStatus, TranslatedSegment
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
from app.services.orchestrators.pipelines._base import BasePipeline, PipelineContext

logger = logging.getLogger(__name__)


class SeedVCV2SpeakerDiarizationV2Pipeline(BasePipeline):
    """
    Multi-speaker Seed-VC V1 Intonation (без F5-TTS).
    SeedVC V1 берёт ref_audio напрямую из clean_vocals.
    """

    SPEAKER_SAMPLE_SECONDS = 30.0
    # ЭКСПЕРИМЕНТ: True = один запрос Seed-VC на весь таймлайн (только 1 спикер).
    # False = обычный режим: по блокам, до 3 параллельных запросов (semaphore)
    WHOLE_AUDIO_VC = False
    # Блоки TTS: микросегменты Whisper (0.2–1 сек) склеиваются в блоки
    # 1.5–4 сек. Верхняя граница важна для lips sync: OmniVoice растягивает
    # речь на всю длительность окна (duration), и в длинных окнах темп плывёт.
    # Нижняя — чтобы не плодить микрокуски.
    MIN_BLOCK_SECONDS = 1.5
    MAX_BLOCK_SECONDS = 4.0
    # Пауза между фразами больше этого → новый блок: пауза сохранится
    # реальной тишиной между блоками (Whisper-синхрон), а не "зашивается"
    # внутрь непрерывной речи
    MAX_GAP_IN_BLOCK = 0.8

    def __init__(
        self,
        transcriber: ITranscriber,
        translator: ITranslator,
        tts: ITTSSynthesizer,
        mixer: IAudioMixer,
        voice_separator: IVoiceSeparator,
        assembler: IAudioAssembler,
        seedvc_converter: IVoiceConverter | None = None,
        pyannote: IPyAnnoteDiarization = None,
        *,
        use_voice_separation: bool = True,
        seedvc_max_parallel: int = 3,
        temp_dir: Path = Path("./tmp"),
        output_dir: Path = Path("./output"),
    ) -> None:
        super().__init__(
            transcriber=transcriber,
            translator=translator,
            tts=tts,
            mixer=mixer,
            voice_separator=voice_separator,
            assembler=assembler,
            seedvc_converter=seedvc_converter,
            pyannote=pyannote,
            use_voice_separation=use_voice_separation,
            temp_dir=temp_dir,
            output_dir=output_dir,
        )
        # Class-level semaphore — разделяется между всеми вызовами
        self._seedvc_semaphore = asyncio.Semaphore(seedvc_max_parallel)

    async def run(self, ctx: PipelineContext) -> None:
        """Execute multi-speaker Seed-VC V1 intonation pipeline (без F5-TTS)."""
        if self._pyannote is None:
            raise RuntimeError("PyAnnote client not provided")
        if self._seedvc_converter is None:
            raise RuntimeError("Seed-VC V1 converter not provided")

        # === STEP 1-3: Extract + Separate + Diarization ===
        await self.step_extract_audio(ctx)
        await self.step_separate_vocals(ctx)

        # PyAnnote: определить спикеров
        ctx.job.status = JobStatus.DIARIZING
        logger.info("Job %s — PyAnnote: speaker diarization", ctx.job.id)
        ctx.diarization_segments = await self._pyannote.diarize(ctx.clean_vocals_path)
        speakers = sorted(set(seg["speaker"] for seg in ctx.diarization_segments))
        logger.info("Job %s — found %d speakers: %s",
                    ctx.job.id, len(speakers), speakers)

        # === STEP 4: Извлечь sample для КАЖДОГО спикера ===
        await self._extract_speaker_samples_only(ctx, speakers)

        # === STEP 5: Whisper per-segment ===
        await self.step_transcribe(ctx)

        # === STEP 6: Translate per-segment ===
        translated_segments = await self.step_translate_segments(ctx)

        # === STEP 7: Привязать каждый сегмент к спикеру по timestamp ===
        segments_with_speaker = self._assign_speakers_to_segments(
            translated_segments, ctx.diarization_segments
        )

        # === STEP 8: OmniVoice DEFAULT по блокам ~4 сек (per-speaker ref_audio) ===
        ctx.job.status = JobStatus.SYNTHESIZING

        default_paths, blocks = await self._synthesize_default_per_speaker(
            ctx, segments_with_speaker
        )

        # Экспериментальный режим: ОДИН запрос Seed-VC на весь таймлайн.
        # Работает только с 1 спикером (иначе один голос на всех).
        if self.WHOLE_AUDIO_VC and len(speakers) == 1:
            await self._whole_audio_vc_flow(ctx, default_paths, blocks)
            return

        # === STEP 9: Seed-VC V1 по блокам с per-speaker reference ===
        logger.info("Job %s — Seed-VC V1 voice conversion over %d blocks",
                    ctx.job.id, len(blocks))
        vc_paths = await self._seedvc_v1_per_speaker(
            ctx, default_paths, blocks
        )

        # === STEP 10: Assemble + Mix + Merge (только FFmpeg, без GPU) ===
        await self._finalize_output(ctx, vc_paths, blocks)

    async def _whole_audio_vc_flow(
        self,
        ctx: PipelineContext,
        default_paths: list[Path],
        blocks: list[dict],
    ) -> None:
        """
        Режим проверки: один запрос Seed-VC на всё аудио.

        1. Собираем все TTS-блоки в ОДИН файл (с тишиной по окнам) — assembler
        2. Один voice_convert на весь файл (reference = speaker sample)
        3. Готовое аудио → mix → merge в видео
        """
        logger.info("Job %s — WHOLE-AUDIO VC mode: %d blocks → 1 request",
                    ctx.job.id, len(blocks))

        # 1. Merge всех TTS-блоков в один таймлайн
        assembled_tts = ctx.job_temp / "assembled_tts.wav"
        total_duration = await self._mixer.get_duration(ctx.extracted_audio_path)
        segments_for_assembly = [
            (path, block["start"], block["end"])
            for path, block in zip(default_paths, blocks)
        ]
        await self._assembler.assemble_segments(
            segment_audios=segments_for_assembly,
            total_duration=total_duration,
            output_path=assembled_tts,
        )
        logger.info("Job %s — TTS assembled into one file: %.1fs",
                    ctx.job.id, total_duration)

        # 2. ОДИН запрос Seed-VC на весь файл
        ref_audio = ctx.speaker_refs.get(
            blocks[0]["speaker"], ctx.clean_vocals_path
        ) if blocks else ctx.clean_vocals_path
        vc_whole = ctx.job_temp / "vc_whole.wav"
        await self._seedvc_converter.voice_convert(
            source_wav_path=assembled_tts,
            reference_wav_path=ref_audio,
            output_path=vc_whole,
        )
        logger.info("Job %s — whole-audio VC done: %s", ctx.job.id, vc_whole.name)

        # 3. Mix с instrumental + merge в видео
        if ctx.instrumental_path:
            mixed_path = ctx.job_temp / "mixed_whole.wav"
            final_audio = await self._mixer.mix_audios(
                background_path=ctx.instrumental_path,
                foreground_path=vc_whole,
                output_path=mixed_path,
            )
        else:
            final_audio = vc_whole

        ctx.job.synthesized_audio_path = final_audio
        await self.step_merge_video(ctx, final_audio)
        self.finalize(ctx, label="seedvc_v2_speaker_diarization_v2_whole")

    async def _extract_speaker_samples_only(
        self, ctx: PipelineContext, speakers: list[str]
    ) -> None:
        """Объединить ВСЕ сегменты спикера в один файл."""
        ctx.speaker_refs = {}
        for speaker in speakers:
            speaker_segments = [
                s for s in ctx.diarization_segments if s["speaker"] == speaker
            ]
            if not speaker_segments:
                continue

            sample_path = ctx.job_temp / f"speaker_{speaker}_sample.wav"
            await self._mixer.concat_audio_segments(
                video_path=ctx.clean_vocals_path,
                segments=speaker_segments,
                output_path=sample_path,
                max_duration=self.SPEAKER_SAMPLE_SECONDS,
            )
            ctx.speaker_refs[speaker] = sample_path
            logger.info("Job %s — speaker %s sample: %d segments, max=%.1fs",
                        ctx.job.id, speaker, len(speaker_segments), self.SPEAKER_SAMPLE_SECONDS)

    def _assign_speakers_to_segments(
        self,
        segments: list[TranslatedSegment],
        diarization: list[dict],
    ) -> list[tuple[TranslatedSegment, str]]:
        """Для каждого сегмента найти спикер по timestamp — O(n log m)."""
        if not diarization:
            return [(seg, "SPEAKER_00") for seg in segments]

        # Сортируем диаризацию по start time и строим массив для bisect
        sorted_dia = sorted(diarization, key=lambda d: d["start"])
        starts = [d["start"] for d in sorted_dia]

        result = []
        for seg in segments:
            seg_mid = (seg.start + seg.end) / 2
            # Бинарный поиск: находим последний диаризационный сегмент, который начинается <= seg_mid
            idx = bisect_right(starts, seg_mid) - 1
            if idx >= 0 and sorted_dia[idx]["start"] <= seg_mid <= sorted_dia[idx]["end"]:
                speaker = sorted_dia[idx]["speaker"]
            else:
                speaker = "SPEAKER_00"
            result.append((seg, speaker))
        return result

    def _group_segments_into_blocks(
        self, segments_with_speaker: list[tuple[TranslatedSegment, str]]
    ) -> list[dict]:
        """
        Склеить сегменты в блоки MIN–MAX_BLOCK_SECONDS (1.5–4с).

        Whisper часто выдаёт микросегменты (0.2–1 сек) — если каждый отдавать
        OmniVoice отдельно, получается рваная речь из крошечных кусков.
        Здесь соседние сегменты одного спикера объединяются в блок ~4 сек
        (текст склеивается пробелом), блок несёт своё окно [start, end].
        Спикер внутри блока не меняется — Seed-VC нужен один референс на кусок.
        """
        blocks: list[dict] = []
        cur_speaker: str | None = None
        cur_texts: list[str] = []
        cur_start: float | None = None
        cur_end: float | None = None
        cur_speech = 0.0  # чистое время речи (без пауз между фразами)

        def _flush() -> None:
            nonlocal cur_speaker, cur_texts, cur_start, cur_end, cur_speech
            if cur_texts:
                blocks.append({
                    "speaker": cur_speaker,
                    "text": " ".join(cur_texts),
                    "start": cur_start,   # начало первой фразы
                    "end": cur_end,       # конец последней фразы (окно с паузами)
                    # duration для TTS = чистая речь: OmniVoice не растягивает
                    # речь на паузы → естественный темп; паузы доберёт тишиной
                    # сборщик (assembler) до полного окна [start, end]
                    "duration": cur_speech,
                })
            cur_speaker = None
            cur_texts = []
            cur_start = None
            cur_end = None
            cur_speech = 0.0

        for seg, speaker in segments_with_speaker:
            seg_dur = seg.end - seg.start
            changed_speaker = cur_speaker is not None and speaker != cur_speaker
            # Пауза между фразами больше MAX_GAP_IN_BLOCK → блок закрываем:
            # пауза сохранится тишиной между блоками (Whisper-синхрон)
            big_gap = cur_end is not None and (seg.start - cur_end) > self.MAX_GAP_IN_BLOCK
            # Правило (1.5, 4]: добавляем фразу, если блок остаётся ≤4с.
            # Если уже не влезает, но блок ещё <1.5с — добавляем принудительно
            # (переброс лучше микрокуска). Одиночная фраза длиннее 4с остаётся
            # целиком — предложение не рвём.
            fits = cur_speech + seg_dur <= self.MAX_BLOCK_SECONDS
            too_small = cur_speech < self.MIN_BLOCK_SECONDS
            if cur_texts and (changed_speaker or big_gap or (not fits and not too_small)):
                _flush()

            if cur_speaker is None:
                cur_speaker = speaker
            cur_texts.append(seg.translated_text)
            cur_speech += seg_dur
            if cur_start is None:
                cur_start = seg.start
            cur_end = seg.end

        _flush()
        return blocks

    async def _synthesize_default_per_speaker(
        self, ctx: PipelineContext, segments_with_speaker: list[tuple[TranslatedSegment, str]]
    ) -> tuple[list[Path], list[dict]]:
        """OmniVoice DEFAULT по блокам ~4 сек — все блоки спикера одним batch."""
        blocks = self._group_segments_into_blocks(segments_with_speaker)
        logger.info(
            "Job %s — grouped %d segments into %d blocks (%.1f–%.1fs each)",
            ctx.job.id, len(segments_with_speaker), len(blocks),
            self.MIN_BLOCK_SECONDS, self.MAX_BLOCK_SECONDS,
        )

        output_dir = ctx.job_temp / "default_blocks"
        output_dir.mkdir(exist_ok=True)
        ref_text = self.get_ref_text(ctx)

        # Группируем блоки по спикерам → один batch-запрос на спикера
        speaker_groups: dict[str, list[int]] = {}
        for bi, block in enumerate(blocks):
            speaker_groups.setdefault(block["speaker"], []).append(bi)

        all_paths = []
        for speaker, indices in speaker_groups.items():
            ref_audio = ctx.speaker_refs.get(speaker, ctx.clean_vocals_path)
            texts = [blocks[i]["text"] for i in indices]
            durations = [blocks[i]["duration"] for i in indices]

            logger.info(
                "Job %s — OmniVoice batch for %s: %d blocks, windows=%s",
                ctx.job.id, speaker, len(indices),
                [round(d, 1) for d in durations],
            )
            paths = await self._tts.synthesize_batch(
                texts=texts,
                durations=durations,
                reference_audio_path=ref_audio,
                reference_text=ref_text,
                output_dir=output_dir / speaker,
                target_language=ctx.job.target_language,
            )
            all_paths.extend(zip(indices, paths))

        # Восстанавливаем исходный порядок блоков
        all_paths.sort(key=lambda x: x[0])
        return [path for _, path in all_paths], blocks

    async def _seedvc_v1_per_speaker(
        self,
        ctx: PipelineContext,
        default_paths: list[Path],
        blocks: list[dict],
    ) -> list[Path]:
        """Seed-VC V1 по блокам с per-speaker reference — ПАРАЛЛЕЛЬНО."""
        vc_dir = ctx.job_temp / "vc_blocks"
        vc_dir.mkdir(exist_ok=True)

        async def _convert_one(i: int, default_path: Path, speaker: str | None) -> Path:
            async with self._seedvc_semaphore:
                vc_path = vc_dir / f"vc_{i:04d}.wav"
                ref_for_speaker = ctx.speaker_refs.get(speaker, ctx.clean_vocals_path)
                await self._seedvc_converter.voice_convert(
                    source_wav_path=default_path,
                    reference_wav_path=ref_for_speaker,
                    output_path=vc_path,
                )
                return vc_path

        # Запускаем ВСЕ конвертации параллельно
        tasks = [
            _convert_one(i, default_path, block["speaker"])
            for i, (default_path, block) in enumerate(zip(default_paths, blocks))
        ]
        vc_paths = await asyncio.gather(*tasks)

        logger.info("Job %s — Seed-VC V1: %d blocks converted in parallel",
                    ctx.job.id, len(vc_paths))
        return list(vc_paths)

    async def _finalize_output(
        self,
        ctx: PipelineContext,
        vc_paths: list[Path],
        blocks: list[dict],
    ) -> None:
        """Assemble → Mix → Merge → Finalize."""
        assembled_path = ctx.job_temp / "assembled_speakers.wav"
        total_duration = await self._mixer.get_duration(ctx.extracted_audio_path)

        segments_for_assembly = [
            (vc_path, block["start"], block["end"])
            for vc_path, block in zip(vc_paths, blocks)
        ]
        final_tts = await self._assembler.assemble_segments(
            segment_audios=segments_for_assembly,
            total_duration=total_duration,
            output_path=assembled_path,
        )

        # Mix с instrumental
        if ctx.instrumental_path:
            mixed_path = ctx.job_temp / "mixed_speakers.wav"
            final_audio = await self._mixer.mix_audios(
                background_path=ctx.instrumental_path,
                foreground_path=final_tts,
                output_path=mixed_path,
            )
        else:
            final_audio = final_tts

        ctx.job.synthesized_audio_path = final_audio
        await self.step_merge_video(ctx, final_audio)
        self.finalize(ctx, label="seedvc_v2_speaker_diarization_v2")
