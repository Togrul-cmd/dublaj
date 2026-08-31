# Сущности домена — чистые структуры данных без зависимостей от инфраструктуры.
# Представляют основные бизнес дубляжного пайплайна.

from __future__ import annotations  # чтобы list[str] работало в старых Python
import uuid                          # генерация уникальных ID
from dataclasses import dataclass, field  # создание классов-структур
from enum import Enum                # перечисления (статусы)
from pathlib import Path             # пути к файлам
from typing import Optional          # поля которые могут быть None


# Статусы жизненного цикла дубляжной задачи
class JobStatus(str, Enum):
    PENDING = "pending"                        # Ожидает обработки
    EXTRACTING_AUDIO = "extracting_audio"      # Извлекает аудио из видео
    TRANSCRIBING = "transcribing"              # Распознаёт речь (Whisper)
    DIARIZING = "diarizing"                    # Определяет спикеров (PyAnnote)
    TRANSLATING = "translating"                # Переводит текст (LLM)
    SEPARATING_VOCALES = "separating_vocals"   # Отделяет голос от музыки
    SYNTHESIZING = "synthesizing"              # Синтезирует речь (TTS)
    MERGING = "merging"                        # Собирает финальное видео
    COMPLETED = "completed"                    # Готово
    FAILED = "failed"                          # Ошибка


# Один сегмент транскрипции с таймкодами (начало, конец, текст)
@dataclass(frozen=True)
class TranscriptionSegment:
    start: float  # секунды
    end: float    # секунды
    text: str


# Полный результат транскрипции от Whisper (все сегменты + язык + текст)
@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[TranscriptionSegment]
    language: str          # исходный язык (например "en")
    full_text: str         # весь текст целиком


# Результат перевода от LLM (исходный язык, целевой язык, оригинальный и переведённый текст)
@dataclass(frozen=True)
class TranslationResult:
    source_language: str
    target_language: str
    original_text: str
    translated_text: str


# Один переведённый сегмент с таймкодами из транскрипции
@dataclass(frozen=True)
class TranslatedSegment:
    index: int
    start: float  # секунды (из Whisper)
    end: float    # секунды (из Whisper)
    original_text: str
    translated_text: str


# Главная сущность — отслеживает весь жизненный цикл одной дубляжной задачи
@dataclass
class DubbingJob:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)  # уникальный ID задачи
    status: JobStatus = JobStatus.PENDING                      # текущий статус
    target_language: str = ""                                  # целевой язык перевода

    # Пути к файлам
    input_video_path: Optional[Path] = None          # исходное видео
    extracted_audio_path: Optional[Path] = None      # извлечённое аудио
    synthesized_audio_path: Optional[Path] = None    # синтезированная речь
    output_video_path: Optional[Path] = None         # финальное видео

    # Промежуточные результаты
    transcription: Optional[TranscriptionResult] = None       # результат транскрипции
    translation: Optional[TranslationResult] = None           # результат перевода
    diarization_segments: Optional[list[dict]] = None         # сегменты диаризации [{start, end, speaker}]

    # Информация об ошибке
    error_message: Optional[str] = None

    # Пометить задачу как_failed с описанием ошибки
    def fail(self, message: str) -> None:
        self.status = JobStatus.FAILED
        self.error_message = message
