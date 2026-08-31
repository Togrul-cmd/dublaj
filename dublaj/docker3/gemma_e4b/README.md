# Gemma 4 E4B Q4 AWQ — Docker Setup

## Запуск

```bash
cd /mnt/c/Users/User/Desktop/dublaj

# Запуск vLLM с Gemma 4 E4B
docker compose -f docker3/gemma_e4b/docker-compose.yml up -d

# Проверить статус (подождать ~5 мин для загрузки модели)
docker logs -f dublaj-gemma-e4b

# Проверить health
curl http://localhost:8600/health
```

## Использование

### OpenAI-compatible API:

```bash
curl http://localhost:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-e4b",
    "messages": [
      {"role": "system", "content": "You are a translator."},
      {"role": "user", "content": "Translate to Russian: Hello world"}
    ],
    "temperature": 0.3
  }'
```

### Интеграция с dublaj:

В `.env` файле:
```env
TRANSLATION_PROVIDER=gemini
GEMINI_API_KEY=not-needed
GEMINI_BASE_URL=http://localhost:8600/v1
TRANSLATION_MODEL=gemma-e4b
```

## Параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `--model` | `google/gemma-4-e4b-it` | Модель |
| `--quantization` | `awq` | Q4 квантование |
| `--max-model-len` | `4096` | Макс длина контекста |
| `--gpu-memory-utilization` | `0.95` | 95% VRAM |
| `--port` | `8600` | Порт сервера |

## VRAM

| Модель | VRAM |
|--------|------|
| Gemma 4 E4B Q4 AWQ | ~4 GB |
| Gemma 4 E12B Q4 AWQ | ~7 GB |
| Gemma 4 E27B Q4 AWQ | ~14 GB |
