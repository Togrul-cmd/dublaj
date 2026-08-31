# Интерфейсы домена — абстрактные контракты, которые инфраструктура должна реализовать.
# Бизнес-логика зависит ТОЛЬКО от этих интерфейсов, никогда от конкретных реализаций.

from abc import ABC, abstractmethod
from pathlib import Path

from app.domain.entities import TranscriptionResult, TranscriptionSegment, TranslationResult, TranslatedSegment


# Контракт для сервисов распознавания речи (Speech-to-Text)
class ITranscriber(ABC):

    @abstractmethod
    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Распознаёт аудиофайл и возвращает текст с таймкодами.

        Args:
            audio_path: Путь к аудиофайлу (WAV/MP3).

        Returns:
            TranscriptionResult с определённым языком и сегментами с таймкодами.
        """
        ...


# Контракт для сервисов перевода текста
class ITranslator(ABC):

    @abstractmethod
    async def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        """
        Переводит текст с исходного языка на целевой.

        Args:
            text: Текст для перевода.
            source_language: Код исходного языка (например "en").
            target_language: Код целевого языка (например "ru").

        Returns:
            TranslationResult с оригинальным и переведённым текстом.
        """
        ...

    async def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        source_language: str,
        target_language: str,
    ) -> list[TranslatedSegment]:
        """
        Переводит сегменты с таймкодами, сохраняя границы сегментов.

        По умолчанию переводит каждый сегмент отдельно.
        Можно переопределить для пакетного перевода с сохранением контекста.
        """
        results = []
        for i, seg in enumerate(segments):
            result = await self.translate(seg.text, source_language, target_language)
            results.append(TranslatedSegment(
                index=i,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text=result.translated_text,
            ))
        return results


# Контракт для синтеза речи (Text-to-Speech) с voice cloning
class ITTSSynthesizer(ABC):

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        reference_audio_path: Path,
        reference_text: str,
        target_language: str,
        output_path: Path,
        duration: float = None,
    ) -> Path:
        """
        Синтезирует речь из текста, копируя голос из референсного аудио.

        Args:
            text: Текст для озвучки.
            reference_audio_path: Аудио для клонирования голоса.
            reference_text: Транскрипция референсного аудио.
            target_language: Язык выходной речи.
            output_path: Куда сохранить синтезированное аудио.
            duration: Фиксированная длительность в секундах (модель подстраивает скорость).

        Returns:
            Путь к сгенерированному аудиофайлу.
        """
        ...

    @abstractmethod
    async def synthesize_batch(
        self,
        texts: list[str],
        durations: list[float],
        reference_audio_path: Path,
        reference_text: str,
        output_dir: Path,
        target_language: str = None,
    ) -> list[Path]:
        """
        Пакетный синтез голосом по умолчанию — несколько текстов с индивидуальными длительностями за один запрос.

        Args:
            texts: Список текстов для синтеза.
            durations: Длительность для каждого текста (секунды). Должна совпадать с длиной texts.
            reference_audio_path: Аудио для клонирования (игнорируется в Voice Design режиме).
            reference_text: Транскрипция (игнорируется в Voice Design режиме).
            output_dir: Куда сохранить сгенерированные файлы.
            target_language: Код целевого языка.

        Returns:
            Список путей к сгенерированным аудиофайлам.
        """
        ...

    @abstractmethod
    async def synthesize_with_clone(
        self,
        text: str,
        ref_audio_path: Path,
        ref_text: str,
        output_path: Path,
        target_language: str = None,
        duration: float = None,
    ) -> Path:
        """
        Синтез речи с КЛОНИРОВАНИЕМ ГОЛОСА из референсного аудио.

        Сгенерированный голос будет совпадать с говорящим из референса.

        Args:
            text: Текст для синтеза (переведённый текст).
            ref_audio_path: Путь к референсному аудио для клонирования.
            ref_text: Транскрипция от Whisper (что сказал человек).
            output_path: Куда сохранить синтезированное аудио.
            target_language: Код целевого языка (например "ru") — помогает модели
                генерировать правильное произношение без акцента источника.
            duration: Фиксированная длительность в секундах. Модель подстраивает
                скорость речи, чтобы текст поместился ровно в эту длительность.

        Returns:
            Путь к сгенерированному аудиофайлу.
        """
        ...

    async def synthesize_clone_batch(
        self,
        texts: list[str],
        durations: list[float],
        ref_audio_path: Path,
        ref_text: str,
        output_dir: Path,
        target_language: str = None,
    ) -> list[Path]:
        """
        Пакетный синтез с voice cloning — несколько текстов за один запрос с клонированием голоса.

        По умолчанию вызывает synthesize_with_clone для каждого текста.
        Переопределить для настоящей пакетной обработки.

        Returns:
            Список путей к сгенерированным аудиофайлам.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, (text, dur) in enumerate(zip(texts, durations)):
            path = output_dir / f"seg_{i:04d}.wav"
            await self.synthesize(
                text=text,
                reference_audio_path=ref_audio_path,
                reference_text=ref_text,
                target_language=target_language if target_language else "auto",
                duration=dur,
                output_path=path,
            )
            paths.append(path)
        return paths


# Контракт для разделения голоса и музыки
class IVoiceSeparator(ABC):

    @abstractmethod
    async def separate_vocals(self, audio_path: Path, output_path: Path) -> tuple[Path, Path | None]:
        """
        Извлекает чистый вокал из смешанного аудио (убирает музыку/шум).

        Args:
            audio_path: Путь к смешанному аудиофайлу.
            output_path: Куда сохранить изолированный вокал.

        Returns:
            Кортеж (путь_к_вокалу, путь_к_инструменталу).
            Инструментал может быть None, если сервер не поддерживает.
        """
        ...


# Контракт для извлечения аудио и склейки видео/аудио (FFmpeg)
class IAudioMixer(ABC):

    @abstractmethod
    async def extract_audio(self, video_path: Path, output_audio_path: Path) -> Path:
        # Извлекает аудиодорожку из видеофайла
        ...

    @abstractmethod
    async def create_mute_video(self, video_path: Path, output_video_path: Path) -> Path:
        # Создаёт видео без звука
        ...

    @abstractmethod
    async def merge_audio_into_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_video_path: Path,
    ) -> Path:
        """
        Заменяет/накладывает аудиодорожку на видеофайл.

        Args:
            video_path: Путь к оригинальному видео.
            audio_path: Путь к новой аудиодорожке.
            output_video_path: Куда сохранить выходное видео.

        Returns:
            Путь к финальному видеофайлу.
        """
        ...

    @abstractmethod
    async def get_duration(self, audio_path: Path) -> float:
        # Возвращает длительность аудио/видео в секундах
        ...

    @abstractmethod
    async def remove_silence(self, audio_path: Path, output_path: Path) -> Path:
        # Убирает тишину из аудио без изменения скорости
        ...

    @abstractmethod
    async def apply_atempo(self, audio_path: Path, atempo: float, output_path: Path) -> Path:
        # Применяет коэффициент скорости к аудио
        ...

    @abstractmethod
    async def extract_segment(self, audio_path: Path, start_time: float, end_time: float, output_path: Path) -> Path:
        # Вырезает сегмент из аудиофайла по таймкодам
        ...

    @abstractmethod
    async def extract_audio_segment(
        self,
        video_path: Path,
        output_audio_path: Path,
        max_seconds: float,
    ) -> Path:
        # Извлекает только первые `max_seconds` секунд аудио из видео
        ...

    @abstractmethod
    async def extract_audio_from_to(
        self,
        video_path: Path,
        output_audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        # Извлекает аудио из конкретного диапазона времени (от и до)
        ...

    @abstractmethod
    async def concat_audio_segments(
        self,
        video_path: Path,
        segments: list[dict],
        output_path: Path,
        max_duration: float = 30.0,
    ) -> Path:
        # Склеивает несколько сегментов из видео, обрезая до max_duration
        ...

    @abstractmethod
    async def get_video_duration(self, video_path: Path) -> float:
        # Возвращает длительность видео в секундах
        ...

    @abstractmethod
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
        ...


# Контракт для диаризации спикеров (определение кто говорит когда)
class IPyAnnoteDiarization(ABC):

    @abstractmethod
    async def diarize(self, audio_path: Path) -> list[dict]:
        """
        Определяет спикеров в аудио.

        Returns:
            Список сегментов:
                [
                    {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00"},
                    {"start": 5.2, "end": 12.1, "speaker": "SPEAKER_01"},
                    ...
                ]
        """
        ...


# Контракт для конвертации голоса (Voice Conversion)
class IVoiceConverter(ABC):

    @abstractmethod
    async def voice_convert(
        self,
        source_wav_path: Path,
        reference_wav_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Конвертирует голос: накладывает тембр reference на source.

        Args:
            source_wav_path: Аудио с нужным текстом/произношением.
            reference_wav_path: Аудио с целевым голосом (клонируемый голос).
            output_path: Куда сохранить конвертированное аудио.

        Returns:
            Путь к конвертированному аудиофайлу.
        """
        ...


# Контракт для сборки нескольких аудиосегментов на таймлайне
class IAudioAssembler(ABC):

    @abstractmethod
    async def assemble_segments(
        self,
        segment_audios: list[tuple[Path, float, float]],
        total_duration: float,
        output_path: Path,
    ) -> Path:
        """
        Собирает сегменты с таймкодами в одно непрерывное аудио.

        Каждый сегмент размещается на своём месте. Промежутки между
        сегментами заполняются тишиной.

        Args:
            segment_audios: Список кортежей (путь_к_аудио, время_начала, время_конца).
            total_duration: Общая длительность выходного аудио в секундах.
            output_path: Куда сохранить собранное аудио.

        Returns:
            Путь к собранному аудиофайлу.
        """
        ...
