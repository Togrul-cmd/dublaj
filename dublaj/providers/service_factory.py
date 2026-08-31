"""
Service Factory — провайдер Dependency Injection.

Создаёт и подключает все конкретные реализации к их интерфейсам.
Это ЕДИНСТВЕННОЕ место, где создаются экземпляры конкретных классов.
Если нужно заменить инфраструктурный компонент — меняете только ЭТОТ файл.
"""

from app.domain.interfaces import IAudioMixer, IAudioAssembler, IPyAnnoteDiarization, ITranscriber, ITranslator, ITTSSynthesizer, IVoiceSeparator
from app.services.dubbing_orchestrator import DubbingOrchestrator
from config.settings import Settings
from infrastructure.audio_separator_client import AudioSeparatorClient
from infrastructure.ffmpeg_audio_mixer import FFmpegAudioMixer
from infrastructure.ffmpeg_audio_assembler import FFmpegAudioAssembler
from infrastructure.llm_translator import LLMTranslator
from infrastructure.llamacpp_translator import LlamaCppTranslator
from infrastructure.minimax_translator import MiniMaxTranslator
from infrastructure.mimo_translator import MiMoTranslator
from infrastructure.omnivoice_tts_synthesizer import OmniVoiceTTSSynthesizer
from infrastructure.pyannote_diarization_client import PyAnnoteDiarizationClient
from infrastructure.whisper_transcriber import WhisperTranscriber
from infrastructure.seedvc_voice_converter import SeedVCVoiceConverter


class ServiceFactory:
    """
    Composition Root — собирает полный граф зависимостей.

    Следует принципу Dependency Inversion:
    - Бизнес-логика зависит от интерфейсов (domain слой)
    - Эта фабрика подставляет вместо интерфейсов конкретные реализации
    - Для смены провайдера достаточно отредактировать только этот файл
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_transcriber(self) -> ITranscriber:
        """Создаёт транскрибатор speech-to-text."""
        return WhisperTranscriber(
            whisper_url=self._settings.whisper_url,
        )

    def create_translator(self) -> ITranslator:
        """Создаёт переводчик текста на основе настройки TRANSLATION_PROVIDER."""
        provider = self._settings.translation_provider.lower()

        if provider == "minimax":
            return MiniMaxTranslator(
                api_key=self._settings.minimax_api_key,
                base_url=self._settings.minimax_base_url,
                model_name=self._settings.minimax_model,
            )

        if provider == "llamacpp":
            return LlamaCppTranslator(
                base_url=self._settings.llamacpp_base_url,
                model_name=self._settings.llamacpp_model,
                thinking_enabled=self._settings.thinking_enabled,
            )

        if provider == "mimo":
            return MiMoTranslator(
                api_key=self._settings.mimo_api_key,
                model_name=self._settings.mimo_model,
                base_url=self._settings.mimo_base_url,
            )

        # По умолчанию: Gemini
        return LLMTranslator(
            api_key=self._settings.gemini_api_key,
            model_name=self._settings.translation_model,
        )

    def create_tts_synthesizer(self) -> ITTSSynthesizer:
        """Создаёт TTS-синтезатор OmniVoice."""
        return OmniVoiceTTSSynthesizer(
            omnivoice_url=self._settings.omnivoice_url,
            default_voice=self._settings.omnivoice_voice,
        )

    def create_audio_mixer(self) -> IAudioMixer:
        """Создаёт сервис извлечения / склейки аудио."""
        return FFmpegAudioMixer(
            ffmpeg_path=self._settings.ffmpeg_path,
        )

    def create_voice_separator(self) -> IVoiceSeparator:
        """Создаёт сервис разделения голоса и музыки."""
        return AudioSeparatorClient(
            separator_url=self._settings.audio_separator_url,
        )

    def create_audio_assembler(self) -> IAudioAssembler:
        """Создаёт сборщик аудиосегментов."""
        return FFmpegAudioAssembler(
            ffmpeg_path=self._settings.ffmpeg_path,
        )

    def create_seedvc_converter(self) -> SeedVCVoiceConverter:
        """Создаёт клиент конверсии голоса Seed-VC V1."""
        return SeedVCVoiceConverter(
            seedvc_url=self._settings.seedvc_url,
        )

    def create_pyannote_diarization(self) -> IPyAnnoteDiarization:
        """Создаёт клиент диаризации спикеров PyAnnote."""
        return PyAnnoteDiarizationClient(
            pyannote_url=self._settings.pyannote_url,
        )

    def create_orchestrator(self) -> DubbingOrchestrator:
        """Создаёт полностью собранный оркестратор дубляжа."""
        return DubbingOrchestrator(
            transcriber=self.create_transcriber(),
            translator=self.create_translator(),
            tts_synthesizer=self.create_tts_synthesizer(),
            audio_mixer=self.create_audio_mixer(),
            voice_separator=self.create_voice_separator(),
            audio_assembler=self.create_audio_assembler(),
            seedvc_converter=self.create_seedvc_converter(),
            pyannote=self.create_pyannote_diarization(),
            use_voice_separation=self._settings.use_voice_separation,
            seedvc_max_parallel=self._settings.seedvc_max_parallel,
            temp_dir=self._settings.temp_dir,
            output_dir=self._settings.output_dir,
        )
