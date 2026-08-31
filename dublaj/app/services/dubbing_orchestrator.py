"""
Dubbing Orchestrator — thin dispatcher that delegates to pipeline strategies.

Each pipeline lives in its own file under
`app/services/orchestrators/pipelines/`.  This class only creates the right
pipeline and calls `execute()`.

Available pipelines:
  - Seed-VC V2 + Speaker Diarization V2 (Seed-VC V1, без F5-TTS)

All concrete dependencies are injected via interfaces (DIP).
"""

import logging
from pathlib import Path

from app.domain.entities import DubbingJob
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
from app.services.orchestrators.pipelines.seedvc_v2_speaker_diarization_v2_pipeline import SeedVCV2SpeakerDiarizationV2Pipeline

logger = logging.getLogger(__name__)


class DubbingOrchestrator:
    """
    Dispatches dubbing jobs to the appropriate pipeline strategy.

    Single Responsibility: route to the right pipeline.
    Open/Closed: add new pipelines without changing this class.
    """

    def __init__(
        self,
        transcriber: ITranscriber,
        translator: ITranslator,
        tts_synthesizer: ITTSSynthesizer,
        audio_mixer: IAudioMixer,
        voice_separator: IVoiceSeparator,
        audio_assembler: IAudioAssembler,
        seedvc_converter: IVoiceConverter | None = None,
        pyannote: IPyAnnoteDiarization | None = None,
        use_voice_separation: bool = False,
        seedvc_max_parallel: int = 3,
        temp_dir: Path = Path("./tmp"),
        output_dir: Path = Path("./output"),
    ) -> None:
        self._transcriber = transcriber
        self._translator = translator
        self._tts = tts_synthesizer
        self._mixer = audio_mixer
        self._voice_separator = voice_separator
        self._assembler = audio_assembler
        self._seedvc_converter = seedvc_converter
        self._pyannote = pyannote
        self._use_voice_separation = use_voice_separation
        self._seedvc_max_parallel = seedvc_max_parallel
        self._temp_dir = temp_dir
        self._output_dir = output_dir

    # ── Internal: build a pipeline with shared config ─────────────────

    def _kwargs(self) -> dict:
        """Common kwargs passed to every pipeline constructor."""
        return dict(
            transcriber=self._transcriber,
            translator=self._translator,
            tts=self._tts,
            mixer=self._mixer,
            voice_separator=self._voice_separator,
            assembler=self._assembler,
            seedvc_converter=self._seedvc_converter,
            pyannote=self._pyannote,
            use_voice_separation=self._use_voice_separation,
            seedvc_max_parallel=self._seedvc_max_parallel,
            temp_dir=self._temp_dir,
            output_dir=self._output_dir,
        )

    # ── Public API ────────────────────────────────────────────────────

    async def process_seedvc_v2_speaker_diarization_v2(self, job: DubbingJob) -> DubbingJob:
        """
        Multi-speaker Seed-VC V2 Intonation V2 — БЕЗ F5-TTS!

        PyAnnote определяет спикеров → для КАЖДОГО спикера:
        - Извлечь sample голоса из clean_vocals (30 сек)
        - OmniVoice DEFAULT synthesis per-segment
        - Seed-VC V1 voice conversion (source=default, reference=clean_vocals_sample)

        Результат: каждый спикер говорит своим голосом!
        """
        kwargs = self._kwargs()
        return await SeedVCV2SpeakerDiarizationV2Pipeline(**kwargs).execute(job)
