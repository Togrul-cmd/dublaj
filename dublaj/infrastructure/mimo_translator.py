"""
MiMo Translator — translates text using Xiaomi MiMo API.

Implements ITranslator. Uses OpenAI-compatible API for translation.
Model: mimo-v2.5-pro
"""

import asyncio
import logging
from typing import Optional

from openai import OpenAI

from app.domain.entities import TranslationResult, TranslatedSegment, TranscriptionSegment
from app.domain.interfaces import ITranslator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a professional translator specializing in video dubbing. "
    "NEVER use digits (0-9) in translations. ALL numbers MUST be written as WORDS. "
    "ALL single letters MUST be written as WORDS. "
    "Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two', "
    "'3 января' → 'third of January', '100' → 'one hundred', '50%' → 'fifty percent'. "
    "Return ONLY the translation. No explanations, no quotes, no extra text."
)


class MiMoTranslator(ITranslator):
    """
    Translates text via Xiaomi MiMo API (OpenAI-compatible).

    Requires API key from token-plan-sgp.xiaomimimo.com.
    """

    def __init__(
        self,
        api_key: str = "REMOVED",
        model_name: str = "mimo-v2.5-pro",
        base_url: str = "https://token-plan-sgp.xiaomimimo.com/v1",
        timeout: float = 120.0,
    ) -> None:
        self._model_name = model_name
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._timeout = timeout

    def _build_messages(self, text: str, source_language: str, target_language: str) -> list[dict]:
        """Build translation messages."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate from {source_language} to {target_language}:\n{text}"},
        ]

    def _call_mimo(self, messages: list[dict], max_tokens: int = 8000) -> str:
        """Call MiMo API and return the response text."""
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0.3,
            top_p=0.95,
            stream=False,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate full text in one call."""
        messages = self._build_messages(text, source_language, target_language)

        logger.info(
            "Translating %d chars: %s → %s via MiMo (%s)",
            len(text), source_language, target_language, self._model_name,
        )

        translated_text = await asyncio.to_thread(self._call_mimo, messages)

        width = 60
        logger.info("╔%s╗", "═" * width)
        logger.info("║  MiMo ПЕРЕВОД: %s → %s  (%s)", source_language, target_language, self._model_name)
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

    async def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        source_language: str,
        target_language: str,
        batch_size: int = 30,
        inter_batch_delay: float = 1.0,
    ) -> list[TranslatedSegment]:
        """Translate segments in batches."""
        results: list[TranslatedSegment] = []

        # Build batch prompt
        segments_text = "\n".join(
            f"[{i}] [{seg.start:.1f}-{seg.end:.1f}s] {seg.text}"
            for i, seg in enumerate(segments)
        )

        prompt = (
            f"Translate each numbered segment from {source_language} to {target_language}.\n"
            f"Return ONLY the translations in the same numbered format.\n"
            f"Write ALL numbers as words. Write ALL letters as words.\n\n"
            f"Segments:\n{segments_text}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "Translating %d segments: %s → %s via MiMo (batched)",
            len(segments), source_language, target_language,
        )

        translated_text = await asyncio.to_thread(self._call_mimo, messages)

        # Parse numbered translations
        for i, seg in enumerate(segments):
            # Find translation for this segment
            translated = seg.text  # fallback
            for line in translated_text.split("\n"):
                line = line.strip()
                if line.startswith(f"[{i}]"):
                    # Extract translation after [i]
                    parts = line.split("]", 1)
                    if len(parts) > 1:
                        translated = parts[1].strip()
                        # Remove timestamp prefix if present
                        if translated.startswith("[") and "]" in translated:
                            translated = translated.split("]", 1)[1].strip()
                    break

            results.append(
                TranslatedSegment(
                    index=i,
                    start=seg.start,
                    end=seg.end,
                    original_text=seg.text,
                    translated_text=translated,
                )
            )

            logger.info(
                "  [%d] [%.1f-%.1fs] %s → %s",
                i, seg.start, seg.end,
                seg.text[:50] + "..." if len(seg.text) > 50 else seg.text,
                translated[:50] + "..." if len(translated) > 50 else translated,
            )

        logger.info("  MiMo: completed %d segments", len(results))
        return results

    async def health_check(self) -> bool:
        """Check if MiMo API is accessible."""
        try:
            result = await asyncio.to_thread(
                self._call_mimo,
                [{"role": "user", "content": "Say OK"}],
                max_tokens=10,
            )
            return len(result) > 0
        except Exception as e:
            logger.warning("MiMo health check failed: %s", e)
            return False


def _chunks(s: str, n: int):
    """Split string into chunks of n characters."""
    for start in range(0, len(s), n):
        yield s[start:start + n]
