# Dublaj — Автоматический дубляж видео с AI

Система автоматического дубляжа видео с использованием AI: распознавание речи, перевод и синтез речи с клонированием голоса. Поддерживает **многоспикерный дубляж** — каждый спикер говорит переводом СВОИМ голосом.

## Возможности

- **Multi-Speaker Voice Cloning** — каждый спикер в видео говорит своим голосом на новом языке
- **Speaker Diarization** — автоматическое определение кто говорит когда (PyAnnote)
- **Voice Conversion** — наложение тембра оригинального диктора на синтезированную речь (Seed-VC V1)
- **600+ языков** — поддержка большинства языков мира через OmniVoice TTS
- **Audio Separation** — извлечение чистого голоса из видео с музыкой
- **4 провайдера перевода** — Gemini, MiniMax, llama.cpp (локально), MiMo
- Полностью локально — данные не уходят в облако (кроме облачных переводчиков)

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────────┐
│                     Dublaj App (порт 8000)                        │
│                       FastAPI + Python                             │
└────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────┘
     │          │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼          ▼
┌─────────┐┌────────┐┌──────────┐┌────────────┐┌────────┐┌─────────┐
│ Whisper  ││Gemini/ ││OmniVoice ││Windowed    ││PyAnnote││Seed-VC  │
│порт 8100 ││MiniMax ││порт 8200 ││RoFormer    ││порт8500││порт8700 │
│ (Docker) ││  API   ││ (Docker) ││порт 8310   ││(Docker)││ (Docker)│
└─────────┘└────────┘└──────────┘└────────────┘└────────┘└─────────┘
```

### Порты сервисов

| Сервис | Порт | Назначение |
|--------|------|------------|
| **Dublaj App** | 8000 | Основное приложение (FastAPI) |
| **Whisper** | 8100 | Распознавание речи (транскрибация) |
| **OmniVoice** | 8200 | Синтез речи (TTS + Voice Cloning) |
| **Windowed RoFormer** | 8310 | Разделение голоса и музыки (новая быстрая модель) |
| **PyAnnote** | 8500 | Диаризация спикеров (кто когда говорит) |
| **Gemma E4B** | 8600 | Локальный перевод (llama.cpp, опционально) |
| **Seed-VC V1** | 8700 | Конвертация голоса (F0 Base, 200M, 44kHz) |

---

## Требования

- **ОС:** Ubuntu / Debian / WSL2 (Windows)
- **GPU:** NVIDIA с поддержкой CUDA (8+ GB VRAM)
- **Docker:** Docker + Docker Compose + NVIDIA Container Toolkit
- **Python:** 3.10+
- **FFmpeg:** системная установка (`sudo apt install ffmpeg`)

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Izahat/dublaj.git
cd dublaj
```

### 2. Установить системные зависимости

```bash
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version
```

### 3. Установить Python-зависимости

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
source venv/bin/activate.fish   # Fish shell

pip install -r requirements.txt
```

### 4. Настроить `.env`

```bash
cp .env.example .env
```

Отредактируйте `.env` — минимальная конфигурация:

```env
# --- General ---
APP_HOST=0.0.0.0
APP_PORT=8000
MAX_CONCURRENT_JOBS=2

# --- Whisper (Docker, порт 8100) ---
WHISPER_BASE_URL=http://localhost:8100
WHISPER_ENDPOINT=/v1/audio/transcriptions

# --- Translation (выберите провайдер) ---
TRANSLATION_PROVIDER=gemini
GEMINI_API_KEY=ВАШ_GEMINI_API_KEY
TRANSLATION_MODEL=gemini-3.6-flash

# --- OmniVoice TTS (Docker, порт 8200) ---
OMNIVOICE_URL=http://localhost:8200
OMNIVOICE_VOICE=english_male

# --- Audio Separator (windowed-roformer, Docker, порт 8310) ---
AUDIO_SEPARATOR_URL=http://localhost:8310
USE_VOICE_SEPARATION=true

# --- Seed-VC V1 (Docker, порт 8700) ---
SEEDVC_URL=http://localhost:8700

# --- PyAnnote (Docker, порт 8500) ---
PYANNOTE_URL=http://localhost:8500

# --- FFmpeg ---
FFMPEG_PATH=ffmpeg

# --- HuggingFace (для скачивания моделей) ---
HF_TOKEN=ВАШ_HF_TOKEN
```

> **Gemini API Key** — получить бесплатно: [Google AI Studio](https://aistudio.google.com/apikey)

---

### HuggingFace Token (для PyAnnote)

PyAnnote использует модели с HuggingFace, которые требуют принятия лицензии и токена.

**Шаг 1: Создать токен**

1. Зайди на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Нажми **"New token"**
3. Выбери тип **"Fine-grained"**
4. Отметь галочки:
   - ✅ **Read contents of your repos**
   - ✅ **Read contents of public gated repos you can access**
5. Скопируй токен в `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

**Шаг 2: Принять лицензию моделей**

Нужно принять лицензию на страницах моделей (один раз):

1. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) → нажми **"Accept"**
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) → нажми **"Accept"**

**Без принятия лицензии** контейнер `model-downloader` упадёт с ошибкой `403 Forbidden`.

**Шаг 3: Перезапустить контейнер**

```bash
cd docker3/pyannote && docker compose -f docker-compose.pyannote.yml up -d --build
```

---

## Запуск Docker-сервисов

Все модели скачиваются автоматически в Docker volumes при первом запуске. Повторные запуски — мгновенные.

### Whisper (порт 8100) — распознавание речи

```bash
cd ~/dublaj/docker3/whisper
docker compose -f docker-compose.whisper.yml up -d
```

### Whisper Turbo (порт 8100) — faster-whisper-large-v3-turbo

Turbo использует модель `deepdml/faster-whisper-large-v3-turbo-ct2` и
`compute_type=float16`. Downloader скачивает модель в отдельный Docker volume,
после чего сервер загружает ее локально и предоставляет тот же endpoint
`/v1/audio/transcriptions`, поэтому код Dublaj менять не нужно.

Остановите старый Whisper, затем запустите Turbo:

```bash
docker compose -f docker3/whisper/docker-compose.whisper.yml down
cd ~/dublaj/docker3/whisper-turbo
docker compose -f docker-compose.whisper-turbo.yml up -d --build
```

Проверка:

```bash
curl http://localhost:8100/health
```

В ответе должно быть `model=deepdml/faster-whisper-large-v3-turbo-ct2` и
`compute_type=float16`.

### OmniVoice (порт 8200) — синтез речи

```bash
cd ~/dublaj/docker3/omnivoice
docker compose -f docker-compose.omnivoice.yml up -d --build
```

### Windowed RoFormer (порт 8310) — разделение голоса

**Модель:** smulelabs/windowed-roformer (`mbr-win10-sink8.ckpt`) — Mel-Band RoFormer
с оконным sink-вниманием: 92% качества оригинала при 44.5× меньше вычислений
(значительно быстрее и легче по VRAM, чем прежний big_beta4).

Код модели клонируется с GitHub при сборке, модель скачивается в volume один раз.

```bash
cd ~/dublaj/docker3/windowed-roformer
docker compose -f docker-compose.windowed-roformer.yml up -d --build
```

### PyAnnote (порт 8500) — диаризация спикеров

**⚠️ Требуется HuggingFace токен!** Подробности ниже в разделе "HuggingFace Token".

```bash
cd ~/dublaj/docker3/pyannote
docker compose -f docker-compose.pyannote.yml up -d --build
```

### Seed-VC V1 (порт 8700) — конвертация голоса

```bash
cd ~/dublaj/docker3/seedvc
docker compose -f docker-compose.seedvc.yml up -d --build
```

### Gemma E4B (порт 8600) — локальный перевод (опционально)

```bash
cd ~/dublaj/docker3/gemma_e4b
docker compose -f docker-compose.yml up -d --build
```

### Проверить что всё запущено

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Проверить здоровье сервисов

```bash
curl http://localhost:8100/health   # Whisper
curl http://localhost:8200/health   # OmniVoice
curl http://localhost:8310/health   # Windowed RoFormer
curl http://localhost:8500/health   # PyAnnote
curl http://localhost:8700/health   # Seed-VC V1
```

### Логи каждого сервиса

Если контейнер не стартовал или упал — смотри логи. Для каждой модели своя команда:

```bash
cd ~/dublaj/docker3/whisper
docker compose -f docker-compose.whisper.yml logs -f whisper

cd ~/dublaj/docker3/omnivoice
docker compose -f docker-compose.omnivoice.yml logs -f omnivoice

cd ~/dublaj/docker3/windowed-roformer
docker compose -f docker-compose.windowed-roformer.yml logs -f

cd ~/dublaj/docker3/pyannote
docker compose -f docker-compose.pyannote.yml logs -f pyannote-diarization

cd ~/dublaj/docker3/seedvc
docker compose -f docker-compose.seedvc.yml logs -f server
```

---

## Запуск приложения

### Вариант 1: напрямую (Python)

```bash
source venv/bin/activate
python3 main.py
```

### Вариант 2: в Docker (порт 8000)

```bash
docker compose -f docker3/docker-compose.dublaj.yml up -d
```

Контейнер `dublaj-api` сам ставит зависимости из `requirements.txt` и запускает `main.py`.
Сервисы (Whisper и т.д.) должны быть уже запущены — приложение ходит к ним через хост.

Открыть в браузере: **http://localhost:8000/docs** (Swagger UI)

---

## Как работает пайплайн (Multi-Speaker)

```
ВИДЕО с несколькими людьми
    │
    ▼
 FFmpeg ──► original_audio.wav + mute_video.mp4
    │
    ▼
 Windowed RoFormer ──► clean_vocals.wav + instrumental.wav
    │
    ▼
 PyAnnote ──► [{start, end, speaker}, ...]  (кто когда говорит)
    │
    ▼
 Для КАЖДОГО спикера: извлечь 30-сек sample из clean_vocals
    │
    ▼
 Whisper ──► транскрипция с таймкодами
    │
    ▼
 Переводчик (Gemini/MiniMax/llama.cpp/MiMo) ──► переведённые сегменты
    │
    ▼
 Привязать сегменты к спикерам по таймкодам
    │
    ▼
 OmniVoice TTS ──► синтез речи для каждого сегмента
    │
    ▼
 Seed-VC V1 ──► наложение голоса КАЖДОГО спикера (ref из clean_vocals)
    │
    ▼
 Сборка таймлайна + микширование с instrumental
    │
    ▼
 Merge в mute_video ──► ФИНАЛЬНОЕ ВИДЕО
    ✅ Каждый спикер говорит СВОИМ голосом
    ✅ Фоновая музыка сохранена
```

---

## API Endpoints

### Общие

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/docs` | GET | Swagger документация |
| `/api/v1/dubbing/health` | GET | Health check |

### Асинхронный Job API (рекомендуется для интеграций)

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/jobs` | POST | Принять видео → сразу `202` + `job_id`, обработка в фоне |
| `/jobs/{job_id}/status` | GET | Статус: `queued` / `processing` / `completed` / `failed` + этап |
| `/jobs/{job_id}/result` | GET | Скачать готовое видео (стриминг FileResponse) |

### Синхронные эндпоинты (блокирующие, для ручного теста)

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/dubbing/intonation/seedvc-v2/speaker-diarization-v2` | POST | JSON результат (ждёт 3–15+ мин!) |
| `/api/v1/dubbing/intonation/seedvc-v2/speaker-diarization-v2/download` | POST | Скачать видео (ждёт 3–15+ мин!) |

**Pipeline:** Видео → Windowed RoFormer → PyAnnote → Whisper → Translate → OmniVoice → Seed-VC V1 → Merge

**Требуемые Docker сервисы:**
- Whisper (8100)
- OmniVoice (8200)
- Windowed RoFormer (8310)
- PyAnnote (8500)
- Seed-VC V1 (8700)

---

## Асинхронный Job API — как работает

Синхронные эндпоинты держат HTTP-соединение открытым на всё время обработки (3–15+ минут).
Job API решает это: запрос принимается мгновенно, обработка идёт в фоне, результат забирается позже.

```
Клиент                                Dublaj App
  │                                        │
  ├── POST /jobs (video + target_language) ─►
  │                                        ├── сохраняет файл в tmp/{job_id}/
  │                                        ├── ставит задачу в очередь (BackgroundTasks)
  ◄── 202 Accepted {job_id} ───────────────┤   (лишние задачи ждут, статус queued)
  │                                        │
  ├── GET /jobs/{id}/status ──────────────►│  ┌─────────────────────┐
  ◄── {status: processing, stage: diarizing}│  │ Фоновая обработка:  │
  │        (поллить раз в 5–10 сек)        │  │ extract → separate  │
  ├── GET /jobs/{id}/status ──────────────►│  │ → diarize → whisper │
  ◄── {status: completed} ─────────────────│  │ → translate → TTS   │
  │                                        │  │ → Seed-VC → merge   │
  ├── GET /jobs/{id}/result ──────────────►│  └─────────────────────┘
  ◄── ██████ видеопоток (FileResponse) ────┘
```

### 1. Отправить задачу

```bash
curl -X POST "http://localhost:8000/jobs" \
  -F "video=@video.mp4" \
  -F "target_language=ru"
```

Ответ **202 Accepted** (мгновенно):
```json
{
  "job_id": "016546ac73b844a3b19a0e9286528067",
  "status": "queued",
  "status_url": "/jobs/016546ac73b844a3b19a0e9286528067/status",
  "result_url": "/jobs/016546ac73b844a3b19a0e9286528067/result"
}
```

### 2. Следить за статусом

```bash
curl "http://localhost:8000/jobs/016546ac73b844a3b19a0e9286528067/status"
```

Ответ во время обработки:
```json
{
  "job_id": "016546ac73b844a3b19a0e9286528067",
  "status": "processing",
  "stage": "diarizing",
  "target_language": "ru",
  "source_filename": "video.mp4",
  "created_at": "2026-08-18T12:34:56+00:00",
  "num_speakers": 2,
  "source_language": "en",
  "error": null
}
```

**Статусы (`status`):**

| Значение | Значение |
|----------|----------|
| `queued` | В очереди (ждёт свободный слот или сервисы) |
| `processing` | Обрабатывается (см. `stage`) |
| `completed` | Готово — можно качать `/result` |
| `failed` | Ошибка — текст в поле `error` |

**Этапы (`stage`) для прогресс-бара:**

| Этап | Что происходит |
|------|----------------|
| `extracting_audio` | FFmpeg извлекает аудио + mute-видео |
| `separating_vocals` | Windowed RoFormer отделяет голос от музыки |
| `diarizing` | PyAnnote определяет спикеров |
| `transcribing` | Whisper распознаёт речь |
| `translating` | Перевод сегментов (Gemini/MiniMax/…) |
| `synthesizing` | OmniVoice TTS + Seed-VC наложение голоса |
| `merging` | FFmpeg собирает финальное видео |
| `completed` | Готово |

Ответ при ошибке (сервис упал, GPU занят и т.д.):
```json
{
  "status": "failed",
  "stage": "failed",
  "error": "ConnectError: PyAnnote service not responding on :8500"
}
```

### 3. Скачать результат

```bash
curl -o dubbed.mp4 "http://localhost:8000/jobs/016546ac73b844a3b19a0e9286528067/result"
```

- Стриминг через `FileResponse` — не грузит файл в память
- Пока задача не завершена → `409 Conflict` с текущим статусом
- Если задача упала → `409` с текстом ошибки
- Если job_id неизвестен → `404`

### Настройки очереди

| Переменная `.env` | По умолчанию | Описание |
|-------------------|--------------|----------|
| `MAX_CONCURRENT_JOBS` | `2` | Сколько задач обрабатывается одновременно; остальные ждут в `queued` |

> ⚠️ Хранилище задач — **в памяти процесса**. Рестарт приложения = список активных задач
> пропадает (готовые файлы в `output/` остаются на диске). Для одного инстанса этого достаточно.

### Синхронные примеры (для ручного теста)

```bash
# Multi-Speaker дубляж (JSON ответ)
curl -X POST "http://localhost:8000/api/v1/dubbing/intonation/seedvc-v2/speaker-diarization-v2" \
  -F "video=@video.mp4" \
  -F "target_language=ru"

# Multi-Speaker дубляж (скачать видео)
curl -X POST "http://localhost:8000/api/v1/dubbing/intonation/seedvc-v2/speaker-diarization-v2/download" \
  -F "video=@video.mp4" \
  -F "target_language=az" \
  --output dubbed_az.mp4
```

### Ответ синхронного API

```json
{
  "job_id": "016546ac73b844a3b19a0e9286528067",
  "status": "completed",
  "mode": "seedvc_v2_speaker_diarization_v2",
  "num_speakers": 2,
  "source_language": "en",
  "target_language": "ru",
  "output_video": "/path/to/output/dubbed_016546ac73b844a3b19a0e9286528067.mp4"
}
```

---

## Провайдеры перевода

| Провайдер | Модель | Тип | Настройка в `.env` |
|-----------|--------|-----|-------------------|
| **Gemini** | gemini-3.6-flash | Облачный (Google) | `TRANSLATION_PROVIDER=gemini` |
| **MiniMax** | MiniMax-M3 | Облачный | `TRANSLATION_PROVIDER=minimax` |
| **llama.cpp** | gemma-e4b | Локальный (Docker) | `TRANSLATION_PROVIDER=llamacpp` |
| **MiMo** | mimo-v2.5-pro | Облачный (Xiaomi) | `TRANSLATION_PROVIDER=mimo` |

---

## Поддерживаемые языки (ISO 639-1)

| Код | Язык | Код | Язык | Код | Язык |
|-----|------|-----|------|-----|------|
| `az` | Азербайджанский | `tr` | Турецкий | `ru` | Русский |
| `en` | Английский | `es` | Испанский | `fr` | Французский |
| `de` | Немецкий | `it` | Итальянский | `pt` | Португальский |
| `zh` | Китайский | `ja` | Японский | `ko` | Корейский |
| `ar` | Арабский | `hi` | Хинди | `uk` | Украинский |

Полный список 600+ языков: `GET http://localhost:8200/languages`

---

## Управление Docker

```bash
# Остановить конкретный сервис
docker compose -f docker3/whisper/docker-compose.whisper.yml down

# Логи
docker logs -f dublaj-whisper
docker logs -f dublaj-omnivoice
docker logs -f dublaj-windowed-roformer

# Удалить volumes (модели скачаются заново)
docker volume rm whisper-model-data omnivoice-model-data
```

---

## Структура проекта

```
dublaj/
├── main.py                                          # Точка входа
├── config/settings.py                               # Конфигурация (.env)
├── providers/service_factory.py                     # Dependency Injection
├── app/
│   ├── domain/
│   │   ├── entities.py                              # Сущности (DubbingJob, Segments)
│   │   └── interfaces.py                            # Абстрактные интерфейсы
│   └── services/
│       ├── dubbing_orchestrator.py                  # Диспетчер пайплайнов
│       └── orchestrators/
│           ├── output_saver.py                      # Сохранение артефактов
│           └── pipelines/
│               ├── _base.py                         # Базовый пайплайн
│               └── seedvc_v2_speaker_diarization_v2_pipeline.py  # Основной пайплайн
├── infrastructure/
│   ├── api/dubbing_router.py                        # Синхронные эндпоинты
│   ├── api/jobs_router.py                           # Асинхронный Job API (/jobs)
│   ├── whisper_transcriber.py                       # Клиент Whisper
│   ├── llm_translator.py                            # Клиент Gemini
│   ├── llamacpp_translator.py                       # Клиент llama.cpp
│   ├── minimax_translator.py                        # Клиент MiniMax
│   ├── mimo_translator.py                           # Клиент MiMo
│   ├── omnivoice_tts_synthesizer.py                 # Клиент OmniVoice
│   ├── audio_separator_client.py                    # Клиент Windowed RoFormer (разделение)
│   ├── pyannote_diarization_client.py               # Клиент PyAnnote
│   ├── seedvc_voice_converter.py                    # Клиент Seed-VC V1
│   ├── ffmpeg_audio_mixer.py                        # FFmpeg обёртка
│   ├── ffmpeg_audio_assembler.py                    # Сборка таймлайна
│   └── service_manager.py                           # Проверка готовности сервисов
├── docker3/
│   ├── docker-compose.dublaj.yml                    # Само приложение (8000)
│   ├── whisper/docker-compose.whisper.yml           # Whisper (8100)
│   ├── whisper-turbo/                                # Whisper Turbo (8100)
│   │   ├── docker-compose.whisper-turbo.yml         # Compose + downloader
│   │   ├── Dockerfile                               # CUDA + faster-whisper
│   │   ├── download_model.py                        # Скачивание модели
│   │   └── server_whisper_turbo.py                  # OpenAI-compatible API
│   ├── omnivoice/docker-compose.omnivoice.yml       # OmniVoice (8200)
│   ├── windowed-roformer/docker-compose.windowed-roformer.yml  # Разделение (8310)
│   ├── pyannote/docker-compose.pyannote.yml         # PyAnnote (8500)
│   ├── gemma_e4b/docker-compose.yml                 # Gemma E4B (8600, опционально)
│   └── seedvc/docker-compose.seedvc.yml             # Seed-VC V1 (8700)
├── .env                                             # Конфигурация (НЕ коммитить!)
├── .env.example                                     # Шаблон конфигурации
└── requirements.txt                                 # Python зависимости
```

---

## Troubleshooting

### GPU не работает

```bash
docker exec -it dublaj-whisper nvidia-smi

# Установить NVIDIA Container Toolkit:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Модель не скачивается

```bash
docker logs dublaj-whisper-model-downloader
docker logs dublaj-omnivoice-model-downloader
```

### Docker network устарел

```bash
docker network prune -f && docker container prune -f
```

### PyAnnote: 403 Forbidden / model-downloader падает

```bash
# 1. Проверить что токен в .env:
grep HF_TOKEN .env

# 2. Проверить логи:
docker logs dublaj-pyannote-model-downloader

# 3. Если 403 — нужно принять лицензию на HuggingFace:
#    - https://huggingface.co/pyannote/speaker-diarization-community-1 → Accept
#    - https://huggingface.co/pyannote/segmentation-3.0 → Accept

# 4. Перезапустить:
cd docker3/pyannote && docker compose -f docker-compose.pyannote.yml up -d --build
```

---

## Лицензия

Private project.
