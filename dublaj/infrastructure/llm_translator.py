"""
LLM Translator — translates text using a configurable AI model (Google Gemini).

Implements ITranslator. The provider and model are fully configurable via Settings.
Can be swapped for OpenAI, Anthropic, or any other provider by creating a new
ITranslator implementation — zero changes to business logic.
"""

import asyncio
import json
import logging
import re

from google import genai

from app.domain.entities import TranscriptionSegment, TranslationResult, TranslatedSegment
from app.domain.interfaces import ITranslator

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = (
    "You are a professional translator specializing in video dubbing. "
    "NEVER use digits (0-9) in translations. ALL numbers MUST be written as WORDS. "
    "ALL single letters MUST be written as WORDS. "
    "Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two', "
    "'3 января' → 'third of January', '100' → 'one hundred', '50%' → 'fifty percent'. "
    "Return ONLY the translation. No explanations, no quotes, no extra text."
)


class LLMTranslator(ITranslator):
    """
    Translates text via Google Gemini (or compatible) LLM.

    The translation prompt is designed to preserve meaning, tone, and naturalness
    for dubbing/voice-over use cases.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        self._model_name = model_name
        self._client = genai.Client(api_key=api_key)

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text using the configured LLM."""

        logger.info(
            "Translating %d chars: %s → %s via %s",
            len(text),
            source_language,
            target_language,
            self._model_name,
        )

        prompt = self._build_prompt(text, source_language, target_language)

        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_INSTRUCTION},
        )

        translated_text = response.text.strip()

        width = 60
        logger.info("╔%s╗", "═" * width)
        logger.info("║  ПЕРЕВОД: %s → %s  (%s)", source_language, target_language, self._model_name)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  ОРИГИНАЛ:")
        for line in text.splitlines():
            for chunk in _chunks(line, width - 4):
                logger.info("║  %s", chunk)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  ПЕРЕВОД:")
        for line in translated_text.splitlines():
            for chunk in _chunks(line, width - 4):
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

    @staticmethod
    def _build_prompt(text: str, source_language: str, target_language: str) -> str:
        """Build a translation prompt optimized for dubbing/voice-over."""
        return (
            f"Translate from {source_language} to {target_language}.\n"
            f"Return ONLY the translation. No explanations.\n"
            f"Write ALL numbers as words. Write ALL letters as words.\n\n"
            f"Text:\n{text}"
        )

    async def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        source_language: str,
        target_language: str,
    ) -> list[TranslatedSegment]:
        """Translate all segments using JSON format for reliable parsing."""
        logger.info(
            "Translating %d segments: %s → %s via %s (batched)",
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

        # Ретрай на временные ошибки (503 перегрузка, 429, сеть) с бэк-оффом
        import asyncio
        response = None
        last_error = None
        for attempt in range(5):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                )
                break
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                retryable = ("503" in msg or "429" in msg or "UNAVAILABLE" in msg
                             or "RESOURCE_EXHAUSTED" in msg or "timeout" in msg.lower())
                if not retryable or attempt == 4:
                    raise
                wait = 2 ** attempt + 2
                logger.warning(
                    "Gemini attempt %d failed (%s) — retry in %ds",
                    attempt + 1, msg[:80], wait,
                )
                await asyncio.sleep(wait)

        if response is None:
            raise RuntimeError(f"Gemini translation failed after 5 attempts: {last_error}")

        translations = self._parse_json_response(response.text, len(segments))

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
        """Parse JSON array of translations from LLM response."""
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) >= expected:
                    return [str(item).strip() for item in parsed[:expected]]
            except json.JSONDecodeError:
                pass

        lines = [l.strip().lstrip('0123456789.)->- ') for l in text.split("\n") if l.strip()]
        while len(lines) < expected:
            lines.append("")
        return lines[:expected]


def _chunks(text: str, max_len: int) -> list[str]:
    """Split text into lines that fit within max_len characters."""
    if not text:
        return [""]
    parts = []
    while len(text) > max_len:
        cut = text[:max_len].rfind(" ")
        if cut == -1:
            cut = max_len
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts
