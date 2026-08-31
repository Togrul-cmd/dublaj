"""
Gemma 4 E4B Translation Server — llama.cpp OpenAI-compatible API wrapper.

Использует llama.cpp сервер для перевода.
Совместим с infrastructure/llm_translator.py через OpenAI API.

Запуск (автоматически через docker-compose):
  docker compose -f docker3/gemma_e4b/docker-compose.yml up -d

API endpoint:
  POST http://localhost:8600/v1/chat/completions

Использование в dublaj:
  В .env файле:
    TRANSLATION_PROVIDER=gemini
    GEMINI_API_KEY=not-needed
    GEMINI_BASE_URL=http://localhost:8600/v1
    TRANSLATION_MODEL=gemma-e4b
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemma 4 E4B Translation Server")

LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://localhost:8600")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-e4b")
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() == "true"

SYSTEM_PROMPT_THINKING = """You are a professional translator specializing in video dubbing.
Think step by step to find the best translation. In your thinking, analyze each phrase.
In your final answer, return ONLY the translated text, nothing else.

Rules:
- Preserve the original meaning, tone, and emotion.
- Make the translation sound natural for spoken voice-over.
- Keep the sentence structure suitable for lip-sync (similar length).
- CRITICAL: Write ALL numbers as WORDS, never as digits.
  Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two',
  '3 января' → 'third of January', '100' → 'one hundred'.
- CRITICAL: Write ALL single letters as WORDS, never as letters.
  Examples: 'A' → 'ay', 'B' → 'bee', 'C' → 'see'."""

SYSTEM_PROMPT_NO_THINKING = """You are a professional translator specializing in video dubbing.
Return ONLY the translated text, nothing else. No explanations, no quotes, no additional text.

Rules:
- Preserve the original meaning, tone, and emotion.
- Make the translation sound natural for spoken voice-over.
- Keep the sentence structure suitable for lip-sync (similar length).
- CRITICAL: Write ALL numbers as WORDS, never as digits.
  Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two',
  '3 января' → 'third of January', '100' → 'one hundred'.
- CRITICAL: Write ALL single letters as WORDS, never as letters.
  Examples: 'A' → 'ay', 'B' → 'bee', 'C' → 'see'."""

SYSTEM_PROMPT = SYSTEM_PROMPT_THINKING if THINKING_ENABLED else SYSTEM_PROMPT_NO_THINKING


class TranslationRequest(BaseModel):
    text: str
    source_language: str
    target_language: str


class BatchTranslationRequest(BaseModel):
    segments: list[dict]  # [{"index": 0, "text": "...", "start": 0.0, "end": 5.0}, ...]
    source_language: str
    target_language: str


@app.get("/health")
async def health():
    """Health check."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LLAMA_CPP_URL}/health")
            return {"status": "ok", "model": MODEL_NAME, "llama_cpp": resp.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/translate")
async def translate(req: TranslationRequest):
    """Translate single text."""
    prompt = f"Translate from {req.source_language} to {req.target_language}: {req.text}"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{LLAMA_CPP_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 8000,
                },
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"llama.cpp error: {response.text}")

            result = response.json()
            message = result["choices"][0]["message"]
            # Всегда берём content (не reasoning_content)
            translation = message.get("content", "").strip()

            return {
                "original_text": req.text,
                "translated_text": translation,
                "source_language": req.source_language,
                "target_language": req.target_language,
                "model": MODEL_NAME,
                "thinking_enabled": THINKING_ENABLED,
            }

    except Exception as e:
        logger.error("Translation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/batch")
async def translate_batch(req: BatchTranslationRequest):
    """Translate multiple segments in batch."""
    segments_text = "\n".join(
        f"[{s['index']}] {s['text']}" for s in req.segments
    )

    prompt = f"""Translate each numbered segment from {req.source_language} to {req.target_language}.
Return ONLY the translations in the same numbered format.

Segments:
{segments_text}"""

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{LLAMA_CPP_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 8000,
                },
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"llama.cpp error: {response.text}")

            result = response.json()
            message = result["choices"][0]["message"]
            # Всегда берём content (не reasoning_content)
            translation_text = message.get("content", "").strip()

            # Parse numbered translations
            translated_segments = []
            for line in translation_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                for prefix in ["[", ""]:
                    if line.startswith(prefix):
                        try:
                            if prefix == "[":
                                idx_str = line.split("]")[0][1:]
                                translation = line.split("]")[1].strip()
                            else:
                                idx_str = line.split(".")[0]
                                translation = ".".join(line.split(".")[1:]).strip()
                            idx = int(idx_str)
                            translated_segments.append({
                                "index": idx,
                                "translated_text": translation,
                            })
                        except (ValueError, IndexError):
                            continue

            return {
                "segments": translated_segments,
                "source_language": req.source_language,
                "target_language": req.target_language,
                "model": MODEL_NAME,
            }

    except Exception as e:
        logger.error("Batch translation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8601"))
    uvicorn.run(app, host="0.0.0.0", port=port)
