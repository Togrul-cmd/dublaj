"""
Централизованные настройки приложения.
Все значения загружаются из файла .env — ничего не захардкожено.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Конфигурация приложения (неизменяемая).
    Pydantic-settings автоматически читает переменные из .env файла.
    Каждое поле = переменная окружения. Например, поле `app_host`
    читается из переменной окружения `APP_HOST`.
    """

    # --- Общие настройки ---
    # Хост и порт, на котором запускается FastAPI сервер
    app_host: str = "0.0.0.0"        # Слушать на всех интерфейсах
    app_port: int = 8000              # Порт API сервера
    temp_dir: Path = Path("./tmp")    # Временные файлы (аудио, промежуточные результаты)
    output_dir: Path = Path("./output")  # Финальные результаты (видео с дубляжем)
    # Сколько дубляж-задач обрабатывать одновременно (остальные ждут в статусе queued)
    max_concurrent_jobs: int = 2

    # --- Whisper (распознавание речи, Speech-to-Text) ---
    # Whisper — это AI модель, которая превращает аудио в текст с таймкодами.
    # Работает в Docker-контейнере на порту 8100.
    whisper_base_url: str = "http://localhost:8100"
    whisper_endpoint: str = "/v1/audio/transcriptions"

    # --- Переводчик (выбор провайдера) ---
    # translation_provider определяет, какой сервис перевода использовать.
    # Варианты: "gemini", "minimax", "mimo", "llamacpp"
    translation_provider: str = "gemini"

    # Google Gemini — облачный переводчик от Google (быстрый, качественный)
    gemini_api_key: str = ""                    # API ключ (получить на ai.google.dev)
    translation_model: str = "gemini-2.0-flash" # Модель Gemini

    # MiniMax — облачный переводчик (OpenAI-совместимый API)
    minimax_api_key: str = ""                              # API ключ MiniMax
    minimax_base_url: str = "https://api.minimax.io/v1"    # Базовый URL
    minimax_model: str = "MiniMax-M2.7-highspeed"          # Модель MiniMax

    # llama.cpp — локальный перевод через Gemma 4 26B A4B (в Docker на порту 8600)
    # Работает на GPU, не требует интернета, но медленнее облачных
    llamacpp_base_url: str = "http://localhost:8600"
    llamacpp_model: str = "gemma-4-26b-a4b"
    thinking_enabled: bool = False  # True = модель "думает" перед ответом (chain-of-thought), False = сразу отвечает

    # MiMo — облачный перевод от Xiaomi (дешёвый, хороший для азиатских языков)
    mimo_api_key: str = ""
    mimo_base_url: str = "https://token-plan-sgp.xiaomimimo.com/v1"
    mimo_model: str = "mimo-v2.5-pro"

    # --- TTS (синтез речи, Text-to-Speech) ---
    # TTS превращает переведённый текст обратно в аудио (речь).
    # tts_provider — какой TTS-движок использовать (пока только omnivoice)
    tts_provider: str = "omnivoice"
    omnivoice_url: str = "http://localhost:8200"  # OmniVoice в Docker на порту 8200
    omnivoice_voice: str = "male"                  # Голос по умолчанию: "male" или "female"

    # --- Разделитель голоса и музыки (Audio Separator) ---
    # Отделяет вокал от фоновой музыки/звуков в оригинальном видео.
    # Используется UVR (Ultimate Vocal Remover) модель.
    audio_separator_url: str = "http://localhost:8310"
    use_voice_separation: bool = True  # Включить/выключить разделение (нужно для PyAnnote + SeedVC)

    # --- Seed-VC V1 (голосовой конвертер, Voice Converter, 44kHz) ---
    # Seed-VC V1 — zero-shot voice conversion с F0 conditioning.
    # Берёт синтезированный TTS голос и накладывает тембр оригинального диктора.
    # Работает на 44kHz (высокое качество), модель 200M параметров.
    seedvc_url: str = "http://localhost:8700"
    seedvc_max_parallel: int = 3  # Макс параллельных SeedVC конвертаций (защита GPU от OOM)

    # --- PyAnnote Speaker Diarization (диаризация спикеров) ---
    # PyAnnote определяет "кто говорит когда" в видео с несколькими людьми.
    # Без этого все спикеры будут озвучены одним голосом.
    # Нужен для multi-speaker (многоспикерного) дубляжа.
    pyannote_url: str = "http://localhost:8500"

    # --- FFmpeg ---
    # Путь к исполняемому файлу FFmpeg (нужен для работы с аудио/видео)
    ffmpeg_path: str = "ffmpeg"

    # --- Настройки загрузки конфигурации ---
    # Pydantic-settings будет читать переменные из файла .env
    model_config = {
        "env_file": ".env",              # Имя файла с переменными окружения
        "env_file_encoding": "utf-8",    # Кодировка файла
        "extra": "ignore",              # Игнорировать лишние переменные из .env
    }

    @property
    def whisper_url(self) -> str:
        """Полный URL для эндпоинта транскрибации Whisper."""
        return f"{self.whisper_base_url}{self.whisper_endpoint}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Синглтон (создаётся один раз и кэшируется).
    Используется вместо прямого создания Settings(), чтобы не читать .env каждый раз.
    Вызов: settings = get_settings()
    """
    return Settings()
