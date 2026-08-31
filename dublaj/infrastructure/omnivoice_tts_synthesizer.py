"""
OmniVoice TTS Synthesizer — использует HTTP API сервер OmniVoice.

Только режим Voice Design — описание голоса через текстовые атрибуты.
Референсное аудио не требуется.
Поддерживает 600+ языков.
GitHub: https://github.com/k2-fsa/OmniVoice
"""

import asyncio
import logging
from pathlib import Path

import httpx

from app.domain.interfaces import ITTSSynthesizer

logger = logging.getLogger(__name__)


class OmniVoiceTTSSynthesizer(ITTSSynthesizer):
    """
    Синтезирует речь через HTTP API OmniVoice.

    Режим Voice Design:
    - Референсное аудио не требуется
    - Описание голоса через текстовые атрибуты (пол, возраст, тон, стиль, акцент)
    - Поддерживает 600+ языков

    Атрибуты голоса:
    - gender: "male" или "female"
    - age: "child", "teenager", "young adult", "middle-aged", "elderly"
    - pitch: "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"
    - style: "whisper"
    - accent: "american accent", "british accent", "australian accent" и т.д.

    Примеры:
    - "male" → простой мужской голос
    - "female, young adult, high pitch" → молодая женщина с высоким голосом
    - "male, elderly, low pitch, whisper" → шепчущий старик
    - "female, british accent" → британский женский голос
    """

    # Соответствие кода языка → инструкция голоса
    VOICE_INSTRUCTS = {
        # Русский
        "ru": "male",
        "ru_female": "female",

        # Английский
        "en": "male",
        "en_female": "female",

        # Турецкий
        "tr": "male",
        "tr_female": "female",

        # Азербайджанский
        "az": "male",
        "az_female": "female",

        # Украинский
        "uk": "male",
        "uk_female": "female",

        # Испанский
        "es": "male",
        "es_female": "female",

        # Французский
        "fr": "male",
        "fr_female": "female",

        # Немецкий
        "de": "male",
        "de_female": "female",

        # Итальянский
        "it": "male",
        "it_female": "female",

        # Португальский
        "pt": "male",
        "pt_female": "female",

        # Китайский
        "zh": "male",
        "zh_female": "female",

        # Японский
        "ja": "male",
        "ja_female": "female",

        # Корейский
        "ko": "male",
        "ko_female": "female",

        # Арабский
        "ar": "male",
        "ar_female": "female",

        # Хинди
        "hi": "male",
        "hi_female": "female",
    }

    def __init__(
        self,
        omnivoice_url: str = "http://localhost:8200",
        default_voice: str = "female",
    ):
        self._omnivoice_url = omnivoice_url.rstrip("/")
        self._default_voice = default_voice

    def _get_voice_instruct(self, language: str) -> str:
        """Возвращает инструкцию голоса для кода языка."""
        # Прямое совпадение
        if language in self.VOICE_INSTRUCTS:
            return self.VOICE_INSTRUCTS[language]

        # Пробуем с суффиксом _female
        female_key = f"{language}_female"
        if female_key in self.VOICE_INSTRUCTS:
            return self.VOICE_INSTRUCTS[female_key]

        # Возврат к значению по умолчанию
        logger.warning(f"Unknown language '{language}', using default voice")
        return self._default_voice

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
        Синтезирует речь в режиме Voice Design.

        Примечание: reference_audio_path и reference_text игнорируются в режиме Voice Design.
        Голос создаётся по текстовому описанию (пол, возраст, тон и т.д.).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Получаем инструкцию голоса для языка
        voice_instruct = self._get_voice_instruct(target_language)

        logger.info(
            "Synthesizing %d chars with OmniVoice Voice Design (language: %s, voice: %s, duration: %s)",
            len(text),
            target_language,
            voice_instruct,
            f"{duration}s" if duration else "auto",
        )

        payload = {
            "text": text,
            "voice": voice_instruct,
            "language": target_language,
        }
        if duration:
            payload["duration"] = duration

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._omnivoice_url}/synthesize",
                json=payload,
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"OmniVoice synthesize error: {response.status_code} - {response.text}"
                )
            
            output_path.write_bytes(response.content)

        logger.info("OmniVoice synthesis complete: %s", output_path)
        return output_path

    # Максимум текстов в одном запросе к OmniVoice. Больше — режем на куски:
    # гигантский batch (80+) часто падает "zero-size array" и грузит GPU
    MAX_BATCH_CHUNK = 20

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
        Пакетный синтез — с автоматической нарезкой на куски по MAX_BATCH_CHUNK.

        Если текстов больше 20 — шлём несколько последовательных запросов
        по 20 (меньше нагрузка на модель, мусор в одном тексте валит только
        свой кусок). Результат склеивается в исходном порядке.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(texts) <= self.MAX_BATCH_CHUNK:
            return await self._synthesize_batch_once(
                texts, durations, output_dir, target_language,
            )

        # Режем на куски по 20
        all_paths: list[Path] = []
        total_chunks = (len(texts) + self.MAX_BATCH_CHUNK - 1) // self.MAX_BATCH_CHUNK
        logger.info(
            "Batch of %d texts → %d chunks of ≤%d (sequential)",
            len(texts), total_chunks, self.MAX_BATCH_CHUNK,
        )
        for ci in range(0, len(texts), self.MAX_BATCH_CHUNK):
            chunk_texts = texts[ci:ci + self.MAX_BATCH_CHUNK]
            chunk_durations = durations[ci:ci + self.MAX_BATCH_CHUNK]
            chunk_dir = output_dir / f"chunk_{ci // self.MAX_BATCH_CHUNK:03d}"
            logger.info(
                "Batch chunk %d/%d: texts [%d..%d)",
                ci // self.MAX_BATCH_CHUNK + 1, total_chunks,
                ci, min(ci + self.MAX_BATCH_CHUNK, len(texts)),
            )
            paths = await self._synthesize_batch_once(
                chunk_texts, chunk_durations, chunk_dir, target_language,
            )
            all_paths.extend(paths)

        logger.info("Batch synthesis complete: %d audios in %d chunks",
                    len(all_paths), total_chunks)
        return all_paths

    async def _synthesize_batch_once(
        self,
        texts: list[str],
        durations: list[float],
        output_dir: Path,
        target_language: str = None,
    ) -> list[Path]:
        """Один запрос /synthesize_batch (не больше MAX_BATCH_CHUNK текстов)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        voice_instruct = self._get_voice_instruct(target_language)

        logger.info(
            "Batch default synthesis: %d texts, durations=%s, lang=%s, voice=%s",
            len(texts), durations, target_language or "auto", voice_instruct,
        )

        payload = {
            "texts": texts,
            "durations": durations,
            "voice": voice_instruct,
        }
        if target_language:
            payload["language"] = target_language

        async with httpx.AsyncClient(timeout=1800.0) as client:
            # Повтор до 3 раз — модель иногда падает на определённых текстах
            last_error = None
            for attempt in range(3):
                response = await client.post(
                    f"{self._omnivoice_url}/synthesize_batch",
                    json=payload,
                )
                if response.status_code == 200:
                    break
                last_error = f"OmniVoice batch default error: {response.status_code} - {response.text[:300]}"
                logger.warning(
                    "Batch default attempt %d/%d failed: %s",
                    attempt + 1, 3, last_error,
                )
                await asyncio.sleep(1.0)
            else:
                raise RuntimeError(last_error)

            import base64
            result = response.json()
            audios_b64 = result["audios_base64"]

            paths = []
            for i, audio_b64 in enumerate(audios_b64):
                audio_bytes = base64.b64decode(audio_b64)
                path = output_dir / f"seg_{i:04d}.wav"
                path.write_bytes(audio_bytes)
                paths.append(path)

        logger.info("Batch default synthesis complete: %d audios", len(paths))
        return paths

    async def synthesize_with_instruct(
        self,
        text: str,
        instruct: str,
        output_path: Path,
        language: str = None,
    ) -> Path:
        """
        Продвинутый синтез с пользовательской инструкцией голоса.

        Args:
            text: Текст для синтеза
            instruct: Описание голоса (например, "female, young adult, high pitch, british accent")
            output_path: Куда сохранить аудио
            language: Опциональный код языка

        Примеры:
            - instruct="male" → мужской голос
            - instruct="female, young adult" → молодая женщина
            - instruct="male, elderly, low pitch, whisper" → шепчущий старик
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Synthesizing %d chars with custom instruct: %s",
            len(text),
            instruct,
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._omnivoice_url}/synthesize",
                json={
                    "text": text,
                    "voice": instruct,
                    "language": language,
                },
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"OmniVoice synthesize error: {response.status_code} - {response.text}"
                )
            
            output_path.write_bytes(response.content)

        logger.info("OmniVoice synthesis complete: %s", output_path)
        return output_path

    async def health_check(self) -> bool:
        """Проверяет, работает ли сервер OmniVoice."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._omnivoice_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"OmniVoice health check failed: {e}")
            return False

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
        Синтезирует речь в режиме Voice Cloning.

        Сгенерированный голос будет соответствовать говорящему из референсного аудио.
        Передача target_language помогает модели произнести правильно,
        не перенося акцент из референсного аудио.

        Args:
            text: Текст для синтеза (переведённый текст — что генерировать)
            ref_audio_path: Путь к референсному аудио для клонирования голоса (чистый вокал)
            ref_text: Транскрипция из Whisper (что человек говорил в оригинале)
            output_path: Куда сохранить синтезированное аудио
            target_language: Код целевого языка (например "ru") для вывода без акцента
            duration: Фиксированная длительность вывода в секундах (модель подстраивает скорость)

        Returns:
            Путь к сгенерированному аудиофайлу
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Voice cloning: %d chars | ref_audio: %s | ref_text: %d chars | lang: %s | duration: %s",
            len(text),
            ref_audio_path.name,
            len(ref_text),
            target_language or "auto",
            f"{duration}s" if duration else "auto",
        )

        # Читаем референсное аудио и кодируем в base64
        import base64
        ref_audio_bytes = ref_audio_path.read_bytes()
        ref_audio_base64 = base64.b64encode(ref_audio_bytes).decode('utf-8')

        payload = {
            "text": text,
            "ref_text": ref_text,
            "ref_audio_base64": ref_audio_base64,
        }
        if target_language:
            payload["language"] = target_language
        if duration:
            payload["duration"] = duration

        async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 минут для длинных текстов
            response = await client.post(
                f"{self._omnivoice_url}/synthesize_clone_json",
                json=payload,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OmniVoice voice cloning error: {response.status_code} - {response.text}"
                )

            output_path.write_bytes(response.content)

        logger.info("Voice cloning complete: %s", output_path)
        return output_path

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
        Пакетное клонирование голоса — несколько текстов с индивидуальной длительностью в ОДНОМ запросе.

        OmniVoice генерирует все аудио параллельно, каждое со своей длительностью.
        Возвращает список путей к сгенерированным аудиофайлам.

        Args:
            texts: Список текстов для синтеза.
            durations: Длительность для каждого текста (секунды). Должна совпадать с длиной texts.
            ref_audio_path: Референсное аудио для клонирования голоса.
            ref_text: Транскрипция референсного аудио.
            output_dir: Куда сохранить сгенерированные аудиофайлы.
            target_language: Код целевого языка.

        Returns:
            Список путей к сгенерированным аудиофайлам.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Batch voice cloning: %d texts, durations=%s, lang=%s",
            len(texts), durations, target_language or "auto",
        )

        import base64
        ref_audio_bytes = ref_audio_path.read_bytes()
        ref_audio_base64 = base64.b64encode(ref_audio_bytes).decode('utf-8')

        payload = {
            "texts": texts,
            "durations": durations,
            "ref_audio_base64": ref_audio_base64,
            "ref_text": ref_text,
        }
        if target_language:
            payload["language"] = target_language

        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.post(
                f"{self._omnivoice_url}/synthesize_clone_batch",
                json=payload,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OmniVoice batch error: {response.status_code} - {response.text}"
                )

            result = response.json()
            audios_b64 = result["audios_base64"]

        # Декодируем и сохраняем каждое аудио
        paths: list[Path] = []
        for i, audio_b64 in enumerate(audios_b64):
            audio_path = output_dir / f"seg_{i:04d}.wav"
            audio_path.write_bytes(base64.b64decode(audio_b64))
            paths.append(audio_path)

        logger.info(
            "Batch voice cloning complete: %d audios saved to %s",
            len(paths), output_dir.name,
        )
        return paths
