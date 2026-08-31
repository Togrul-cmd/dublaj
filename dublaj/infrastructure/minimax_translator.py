"""
MiniMax Translator — translates text using MiniMax API via OpenAI SDK.

Implements ITranslator. Uses MiniMax-M2.7-highspeed model for translation.
API docs: https://api.minimaxi.chat/v1
"""

import asyncio
import json
import logging
import re

from openai import OpenAI

from app.domain.entities import TranscriptionSegment, TranslationResult, TranslatedSegment
from app.domain.interfaces import ITranslator

logger = logging.getLogger(__name__)


class MiniMaxTranslator(ITranslator):
    """Translates text via MiniMax API using OpenAI-compatible SDK."""

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        import httpx
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=3,
        )
        self._model_name = model_name

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        prompt = self._build_prompt(text, source_language, target_language)

        logger.info(
            "Translating %d chars: %s → %s via %s (MiniMax)",
            len(text), source_language, target_language, self._model_name,
        )

        translated_text = await asyncio.to_thread(self._call_sync, prompt)

        width = 60
        logger.info("╔%s╗", "═" * width)
        logger.info("║  MiniMax ПЕРЕВОД: %s → %s  (%s)", source_language, target_language, self._model_name)
        logger.info("╠%s╣", "═" * width)
        logger.info("║  Символов: %d → %d", len(text), len(translated_text))
        logger.info("╠%s╣", "═" * width)
        # Show preview of translated text
        preview = translated_text[:500] if len(translated_text) > 500 else translated_text
        for line in preview.split("\n"):
            logger.info("║  %s", line[:75])
        if len(translated_text) > 500:
            logger.info("║  ... (%d chars total)", len(translated_text))
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
        batch_size: int = 30,  # 30 instead of 50 to avoid MiniMax output truncation
        inter_batch_delay: float = 10.0,
    ) -> list[TranslatedSegment]:
        """
        Translate all segments in batches for better rate-limit handling.

        Each batch is sent as a single API request with a JSON array of texts.
        Default batch size is 50 segments — small enough to avoid MiniMax
        rate limits, large enough to minimize total request count.

        Progress is logged for every batch:
          - batch index / total
          - segment indices in the batch
          - characters count
          - duration of the API call
        """
        total_segments = len(segments)
        total_batches = (total_segments + batch_size - 1) // batch_size

        logger.info("═" * 60)
        logger.info(
            "BATCHED TRANSLATION: %d segments → %d batches of %d",
            total_segments, total_batches, batch_size,
        )
        logger.info(
            "Model: %s | Direction: %s → %s",
            self._model_name, source_language, target_language,
        )
        logger.info("═" * 60)

        all_translations: list[str] = []

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_segments)
            batch = segments[start:end]

            batch_logger = logger.getChild(f"batch{batch_idx + 1}")
            batch_logger.info("─" * 50)
            batch_logger.info(
                "BATCH %d/%d — segments[%d:%d] (%d segments)",
                batch_idx + 1, total_batches, start, end, len(batch),
            )

            texts = [seg.text for seg in batch]
            texts_json = json.dumps(texts, ensure_ascii=False)
            total_chars = sum(len(t) for t in texts)

            batch_logger.info(
                "  chars: %d | model: %s",
                total_chars, self._model_name,
            )

            prompt = self._build_batch_prompt(batch, source_language, target_language)

            import time
            t0 = time.monotonic()

            # Retry loop: keep halving until ALL segments are translated
            # Track how many of the ORIGINAL batch were successfully translated
            original_batch = batch
            original_start = start
            translated_count = 0

            while translated_count < len(original_batch):
                remaining = original_batch[translated_count:]
                batch_logger.info(
                    "  → sending %d segments (offset %d)...",
                    len(remaining), translated_count,
                )

                try:
                    prompt = self._build_batch_prompt(remaining, source_language, target_language)
                    response_text = await asyncio.to_thread(self._call_sync, prompt)
                    elapsed = time.monotonic() - t0
                    batch_logger.info("  ✓ API responded in %.1fs", elapsed)
                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    batch_logger.error("  ✗ API failed: %s", exc)
                    raise

                n_expected = len(remaining)
                translations = self._parse_json_response(response_text, n_expected)
                n_got = len(translations)

                if n_got < n_expected:
                    if n_got == 0:
                        # Total failure — retry single segment
                        batch_logger.warning(
                            "  ⚠ Got 0 translations, retrying with 1 segment...",
                        )
                        translations = self._parse_json_response(
                            response_text, 1,
                        )
                        n_got = max(1, n_got)
                    else:
                        batch_logger.warning(
                            "  ⚠ Partial: got %d/%d, will retry remaining %d",
                            n_got, n_expected, n_expected - n_got,
                        )

                all_translations.extend(translations)
                translated_count += n_got

                if n_got == 0:
                    all_translations.append("")
                    translated_count += 1

                # Log what we got
                for i, (seg, translated) in enumerate(zip(remaining, translations)):
                    global_idx = original_start + translated_count - n_got + i
                    batch_logger.info(
                        "  [%d] [%.1f-%.1fs] %s → %s",
                        global_idx, seg.start, seg.end,
                        seg.text[:40].replace("\n", " "),
                        translated[:40].replace("\n", " ") if translated else "<<<EMPTY>>>",
                    )

            # Delay between batches to avoid MiniMax API rate limiting
            if batch_idx < total_batches - 1:
                batch_logger.info(
                    "  ⏱ sleeping %.1fs before next batch...",
                    inter_batch_delay,
                )
                await asyncio.sleep(inter_batch_delay)

        logger.info("═" * 60)
        logger.info(
            "BATCHED TRANSLATION DONE: %d segments in %d batches",
            total_segments, total_batches,
        )
        logger.info("═" * 60)

        return [
            TranslatedSegment(
                index=i,
                start=seg.start,
                end=seg.end,
                original_text=seg.text,
                translated_text=all_translations[i],
            )
            for i, seg in enumerate(segments)
        ]

    def _call_sync(self, prompt: str, max_tokens: int = 4096, _retry: int = 0) -> str:
        """Synchronous MiniMax call via OpenAI SDK with retry on empty response."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content:
            stripped = content.strip()
            stripped = self._strip_thinking(stripped)

            # Если после удаления thinking остался пустой ответ — retry
            if not stripped and _retry < 3:
                logger.warning(
                    "MiniMax: ответ пустой (вся ответка = thinking), retry %d/3",
                    _retry + 1,
                )
                return self._call_sync(prompt, max_tokens, _retry=_retry + 1)

            return stripped

        # Fallback: reasoning_details если content=None
        reasoning = getattr(response.choices[0].message, "reasoning_details", None)
        if reasoning:
            reason_text = reasoning[0].get("text", "").strip()
            if reason_text:
                return reason_text

        # Retry если content=None
        if _retry < 3:
            logger.warning(
                "MiniMax: content=None, retry %d/3", _retry + 1,
            )
            return self._call_sync(prompt, max_tokens, _retry=_retry + 1)

        return ""

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove thinking/reasoning blocks from MiniMax response.

        MiniMax M2.7 wraps thinking in various formats:
          - 🙄 ... 🗿
          - <think ... </think >
          - \"Thinking...\" followed by reasoning then actual answer
        """
        import re

        original_len = len(text)

        # Pattern 1: 🙄 ... 🗿 (emoji-delimited thinking block)
        text = re.sub(r'🙄.*?🗿', '', text, flags=re.DOTALL)

        # Pattern 2: <think ... </think > (XML-style thinking tags)
        text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.DOTALL)

        # Pattern 3: "Thinking..." prefix line followed by reasoning
        # Remove everything from "Thinking" up to the last empty line before content
        text = re.sub(
            r'Thinking[.\s]*\n.*?(?=\n\n|\n[A-ZА-ЯƏÜÖĞŞÇİ])',
            '', text, flags=re.DOTALL,
        )

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        removed = original_len - len(text)
        if removed > 100:
            logger.info(
                "MiniMax: stripped %d chars of thinking (raw=%d → clean=%d)",
                removed, original_len, len(text),
            )

        return text

    @staticmethod
    def _build_prompt(text: str, source_language: str, target_language: str) -> str:
        return (
            f"You are a professional translator specializing in video dubbing.\n"
            f"Translate the following text from {source_language} to {target_language}.\n\n"
            f"Rules:\n"
            f"- Preserve the original meaning, tone, and emotion.\n"
            f"- Make the translation sound natural for spoken voice-over.\n"
            f"- Keep the sentence structure suitable for lip-sync (similar length).\n"
            f"- Do NOT add any explanations, notes, or commentary.\n"
            f"- Return ONLY the translated text.\n"
            f"- CRITICAL: Write ALL numbers as WORDS, never as digits.\n"
            f"  Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two',\n"
            f"  '3 января' → 'third of January', '100' → 'one hundred'.\n"
            f"  Dates, years, quantities, percentages — ALL numbers spelled out.\n\n"
            f"Text to translate:\n{text}"
        )

    def _build_batch_prompt(
        self,
        batch,
        source_language: str,
        target_language: str,
    ) -> str:
        """Build prompt for batch translation."""
        import json as _json

        texts = [seg.text for seg in batch]
        texts_json = _json.dumps(texts, ensure_ascii=False)
        return (
            f"Translate each string from {source_language} to {target_language}.\n"
            f"Keep each translation short and natural for voice-over.\n"
            f"Return ONLY a JSON array of translations. Same order, same count.\n"
            f"No explanations, no extra text.\n\n"
            f"CRITICAL RULE: Write ALL numbers as WORDS, never as digits.\n"
            f"Examples: '20' → 'twenty', '2022' → 'two thousand twenty-two',\n"
            f"'3 января' → 'third of January', '100' → 'one hundred'.\n"
            f"Dates, years, quantities, percentages — ALL numbers spelled out.\n\n"
            f"Input:\n{texts_json}"
        )

    @staticmethod
    def _parse_json_response(text: str, expected: int) -> list[str]:
        """Parse JSON array of translations from LLM response."""
        # Try to extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and len(parsed) >= expected:
                    return [str(item).strip() for item in parsed[:expected]]
            except json.JSONDecodeError:
                pass

        # Fallback: each non-empty line is a translation
        lines = [l.strip().lstrip('0123456789.)->- ') for l in text.split("\n") if l.strip()]
        while len(lines) < expected:
            lines.append("")
        return lines[:expected]
