"""
Windowed RoFormer HTTP Server — vocal separation (Mel-Band RoFormer + WSA).

Модель: smulelabs/windowed-roformer (mbr-win10-sink8.ckpt)
  - Тот же Mel-Band RoFormer, но с оконным sink-вниманием (WSA):
    92% качества оригинала при 44.5x меньше вычислений.
  - num_stems=1 → на выходе только vocals.
    Instrumental считается вычитанием: instrumental = mix - vocals.

API совместим с dublaj-audio-separator (контракт /separate_json),
поэтому клиент infrastructure/audio_separator_client.py работает без правок —
достаточно переключить AUDIO_SEPARATOR_URL на порт 8310.

Порт: 8310
"""

import base64
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Windowed RoFormer Separator (WSA)")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/mbr-win10-sink8.ckpt")
SAMPLE_RATE = 44100
CHUNK_SECONDS = 8      # размер чанка инференса (как в main.py репозитория)
OVERLAP_SECONDS = 1.0  # нахлёст между соседними чанками для кроссфейда на стыках
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = None


def load_model():
    """Загрузка mbr-win10-sink8 (архитектура по умолчанию = config.yaml репо)."""
    global MODEL
    if MODEL is not None:
        return MODEL

    from model import MelBandRoformerWSA

    logger.info("Loading Windowed RoFormer (WSA): %s ...", MODEL_PATH)
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model not found: {MODEL_PATH}. Запустите model-downloader."
        )

    net = MelBandRoformerWSA()  # dim=384, depth=6, bands=60, win=10, sinks=8
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt, strict=True)
    MODEL = net.to(DEVICE).eval()
    logger.info("Model loaded on %s", DEVICE)
    return MODEL


def load_audio_stereo(file_path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Аудио → тензор [2, samples], 44100 Hz, стерео."""
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def _trapezoid_window(chunk_samples: int, overlap_samples: int) -> torch.Tensor:
    """
    Окно = 1.0 в середине чанка, с линейным нарастанием 0→1 на первых
    overlap_samples отсчётах и спадом 1→0 на последних overlap_samples.
    У соседних чанков (шаг = chunk_samples - overlap_samples) сумма окон
    в зоне нахлёста равна ровно 1.0 → линейный кроссфейд вместо жёсткого
    разреза, который давал щелчки/разрывы каждые CHUNK_SECONDS.
    """
    window = torch.ones(chunk_samples)
    if overlap_samples > 0:
        ramp = torch.linspace(0.0, 1.0, overlap_samples)
        window[:overlap_samples] = ramp
        window[-overlap_samples:] = ramp.flip(0)
    return window


@torch.inference_mode()
def demix_vocals(model, mix: torch.Tensor) -> torch.Tensor:
    """
    Разделение: перекрывающиеся чанки по CHUNK_SECONDS с шагом
    (CHUNK_SECONDS - OVERLAP_SECONDS), батчами, fp16-autocast на GPU.
    Результат собирается overlap-add с трапецеидальным окном — это убирает
    слышимые щелчки на границах чанков, которые давал прежний код (чанки
    без нахлёста, просто склеенные встык).
    Возвращает vocals-тензор [2, samples] (исходная длина).
    """
    mix = torch.tensor(mix, dtype=torch.float32)
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    overlap_samples = min(int(OVERLAP_SECONDS * SAMPLE_RATE), chunk_samples // 2)
    step_samples = chunk_samples - overlap_samples
    audio_samples = mix.shape[1]

    if audio_samples <= chunk_samples:
        full_samples = chunk_samples
    else:
        n_steps = int(np.ceil((audio_samples - chunk_samples) / step_samples))
        full_samples = chunk_samples + n_steps * step_samples
    if full_samples > audio_samples:
        pad = full_samples - audio_samples
        mix = torch.nn.functional.pad(mix, (0, pad), mode="constant", value=0)

    chunks = mix.unfold(dimension=1, size=chunk_samples, step=step_samples)
    chunks = chunks.permute(1, 0, 2)  # [num_chunks, channels, chunk_samples]
    clips_num = chunks.shape[0]

    outputs = []
    pointer = 0
    use_amp = DEVICE == "cuda"
    while pointer < clips_num:
        batch = chunks[pointer:pointer + BATCH_SIZE].to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
        outputs.append(out.float().cpu())
        pointer += BATCH_SIZE

    outputs = torch.cat(outputs, dim=0)  # [num_chunks, channels, chunk_samples]
    out_channels = outputs.shape[1]

    base_window = _trapezoid_window(chunk_samples, overlap_samples)
    total_len = (clips_num - 1) * step_samples + chunk_samples
    result = torch.zeros(out_channels, total_len)
    weight = torch.zeros(total_len)

    for i in range(clips_num):
        window = base_window.clone()
        # Первый чанк: нечего кроссфейдить слева → не спадаем в начале.
        if i == 0 and overlap_samples > 0:
            window[:overlap_samples] = 1.0
        # Последний чанк: нечего кроссфейдить справа → не спадаем в конце.
        if i == clips_num - 1 and overlap_samples > 0:
            window[-overlap_samples:] = 1.0

        start = i * step_samples
        result[:, start:start + chunk_samples] += outputs[i] * window
        weight[start:start + chunk_samples] += window

    vocals = result / weight.clamp_min(1e-8)
    return vocals[:, :audio_samples]


def tensor_to_wav_bytes(audio: torch.Tensor) -> bytes:
    """Тензор [2, samples] → WAV bytes."""
    import io
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.cpu(), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


class SeparateJsonRequest(BaseModel):
    """Тот же контракт, что у dublaj-audio-separator /separate_json."""
    audio_base64: str
    filename: str = "audio.wav"


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "windowed-roformer",
        "model": "mbr-win10-sink8",
        "device": DEVICE,
    }


@app.post("/separate_json")
async def separate_json(request: SeparateJsonRequest):
    """
    Разделение голоса и музыки (JSON API, контракт audio-separator).

    Вход:  {audio_base64, filename}
    Выход: {status, vocals_base64, instrumental_base64, filename}
    """
    model = load_model()

    audio_bytes = base64.b64decode(request.audio_base64)
    suffix = Path(request.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        input_path = tmp.name

    logger.info("Separating (JSON): %s (%d bytes)", request.filename, len(audio_bytes))

    try:
        mix = load_audio_stereo(input_path)
        vocals = demix_vocals(model, mix)

        # num_stems=1: инструментал = оригинал минус вокал
        instrumental = (mix - vocals).clamp(-1.0, 1.0)

        logger.info(
            "Separation complete: %.1fs audio → vocals + instrumental",
            mix.shape[1] / SAMPLE_RATE,
        )

        return {
            "status": "ok",
            "vocals_base64": base64.b64encode(tensor_to_wav_bytes(vocals)).decode(),
            "instrumental_base64": base64.b64encode(tensor_to_wav_bytes(instrumental)).decode(),
            "filename": f"vocals_{request.filename}",
        }
    except Exception as exc:
        logger.exception("Separation error")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8310)
