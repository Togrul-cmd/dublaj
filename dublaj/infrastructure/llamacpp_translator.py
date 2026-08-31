"""
llama.cpp Translator — translates text using local llama.cpp server.

Implements ITranslator. Uses llama.cpp OpenAI-compatible API for translation.
Works with any GGUF model (Gemma 4 E4B, Llama 3, Mistral, etc.)

API docs: https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md
"""

import asyncio
import json
import logging
import re
from typing import Optional

import httpx

from app.domain.entities import TranslationResult, TranslatedSegment, TranscriptionSegment
from app.domain.interfaces import ITranslator

logger = logging.getLogger(__name__)

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


class LlamaCppTranslator(ITranslator):
    """
    Translates text via local llama.cpp server (OpenAI-compatible API).

    Requires llama.cpp running locally (default: http://localhost:8600).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8600",
        model_name: str = "gemma-e4b",
        thinking_enabled: bool = False,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._thinking_enabled = thinking_enabled
        self._timeout = timeout
        self._system_prompt = SYSTEM_PROMPT_THINKING if thinking_enabled else SYSTEM_PROMPT_NO_THINKING

    def _build_messages(self, text: str, source_language: str, target_language: str) -> list[dict]:
        """Build translation messages for llama.cpp."""
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Translate from {source_language} to {target_language}: {text}"},
        ]

    async def _call_llamacpp(self, messages: list[dict], max_tokens: int = 8000) -> str:
        """Call llama.cpp /v1/chat/completions endpoint and return the response text."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"llama.cpp error: {response.status_code} - {response.text}"
                )

            data = response.json()
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if content:
                return content

            # Reasoning is for internal analysis; only the final content is translatable.
            if message.get("reasoning_content"):
                logger.warning(
                    "llama.cpp returned reasoning_content but no final content"
                )
            raise RuntimeError("llama.cpp returned an empty final content")

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate full text in one call."""
        messages = self._build_messages(text, source_language, target_language)

        logger.info(
            "Translating %d chars: %s → %s via llama.cpp (%s)",
            len(text), source_language, target_language, self._model_name,
        )

        translated_text = await self._call_llamacpp(messages)

        width = 60
        logger.info("╔%s╗", "═" * width)
        logger.info("║  llama.cpp ПЕРЕВОД: %s → %s  (%s)", source_language, target_language, self._model_name)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  ОРИГИНАЛ:")
        for line in text.splitlines():
            for chunk in [line[i:i+width-4] for i in range(0, len(line), width-4)]:
                logger.info("║  %s", chunk)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  ПЕРЕВОД:")
        for line in translated_text.splitlines():
            for chunk in [line[i:i+width-4] for i in range(0, len(line), width-4)]:
                logger.info("║  %s", chunk)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  Символов: %d → %d", len(text), len(translated_text))
        logger.info("╚%s╝", "═" * width)

        return TranslationResult(
            source_language=source_language,
            target_language=target_language,
            original_text=text,
            translated_text=translated_text,
        )

    async def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        source_language: str,
        target_language: str,
    ) -> list[TranslatedSegment]:
        """
        Translate ALL segments in ONE request — same principle as Gemini:
        JSON array in → JSON array out, same order, same count.
        """
        logger.info(
            "Translating %d segments: %s → %s via llama.cpp (%s, single JSON batch)",
            len(segments), source_language, target_language, self._model_name,
        )

        texts = [seg.text for seg in segments]
        texts_json = json.dumps(texts, ensure_ascii=False)

        prompt = (
            f"Translate each string from {source_language} to {target_language}.\n"
            f"Keep each translation short and natural for voice-over.\n"
            f"IMPORTANT: Keep all numbers as digits (e.g. '12' not 'twelve', '2026' not 'two thousand twenty-six').\n"
            f"Return ONLY a JSON array of translations. Same order, same count.\n"
            f"No explanations, no extra text.\n\n"
            f"Input:\n{texts_json}"
        )

        messages = [
            # JSON batch responses must not spend the output on chain-of-thought.
            {"role": "system", "content": SYSTEM_PROMPT_NO_THINKING},
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = await self._call_llamacpp(messages)
            translations = self._parse_json_response(response_text, len(segments))
        except Exception as exc:
            logger.warning(
                "llama.cpp batch translation failed: %s — falling back to original texts",
                exc,
            )
            translations = list(texts)

        results = [
            TranslatedSegment(
                index=i,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text=translations[i],
            )
            for i, seg in enumerate(segments)
        ]

        for seg in results:
            logger.info(
                "  [%.1f-%.1fs] %s → %s",
                seg.start, seg.end,
                seg.original_text[:40],
                seg.translated_text[:40],
            )

        return results

    @staticmethod
    def _parse_json_response(text: str, expected: int) -> list[str]:
        """Parse JSON array of translations from LLM response (same as Gemini)."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) >= expected:
                    translations = [str(item).strip() for item in parsed[:expected]]
                    if all(translations):
                        return translations
            except json.JSONDecodeError:
                pass

        lines = [l.strip().lstrip('0123456789.)->- ') for l in text.split("\n") if l.strip()]
        if len(lines) < expected:
            raise ValueError(
                f"Expected {expected} translations, got {len(lines)}"
            )
        return lines[:expected]

    async def health_check(self) -> bool:
        """Check if llama.cpp server is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
