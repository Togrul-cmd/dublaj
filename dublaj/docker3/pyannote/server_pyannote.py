"""
PyAnnote Speaker Diarization HTTP Server.

Использует pyannote/speaker-diarization-community-1 модель.
Модель требует HuggingFace токен для доступа.

Запуск:
    HF_TOKEN=your_token python server_pyannote.py

или через docker compose.
"""

import os
import tempfile
import logging
from pathlib import Path


from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PyAnnote Diarization Server")

HF_TOKEN = os.getenv("HF_TOKEN", "")
PIPELINE = None


def load_pipeline():
    """Load pyannote diarization pipeline."""
    global PIPELINE

    if PIPELINE is None:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is required")

        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote/speaker-diarization-community-1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN
        )

        # Send to GPU if available
        if torch.cuda.is_available():
            try:
                logger.info("Using GPU for diarization")
                pipeline.to(torch.device("cuda"))
            except RuntimeError as e:
                logger.warning("GPU not usable: %s. Using CPU.", e)
        else:
            logger.info("Using CPU for diarization")

        PIPELINE = pipeline
        logger.info("Pipeline loaded successfully")

    return PIPELINE


@app.on_event("startup")
async def startup():
    try:
        load_pipeline()
    except Exception as e:
        logger.error("Failed to load pipeline: %s", e)
        raise


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "pyannote-diarization",
        "model": "pyannote/speaker-diarization-community-1",
        "gpu_available": import_torch_and_check(),
    }


def import_torch_and_check():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """
    Perform speaker diarization on audio file.

    Returns speaker segments with timestamps.
    """
    pipeline = load_pipeline()

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    logger.info("Diarizing: %s", file.filename)

    try:
        # Run diarization — load audio with torchaudio first
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(input_path)

        # Resample to 16kHz if needed (pyannote expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Mix to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Run diarization with waveform dict
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Convert to list of segments using serialize() method
        serialized = output.serialize()
        segments = serialized.get("diarization", [])

        # Get speaker count
        speakers = set(seg["speaker"] for seg in segments)

        logger.info("Diarization complete: %d speaker(s), %d segments",
                    len(speakers), len(segments))

        return {
            "status": "ok",
            "num_speakers": len(speakers),
            "segments": segments,
        }

    except Exception as e:
        logger.error("Diarization error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@app.post("/diarize/batch")
async def diarize_batch(files: list[UploadFile] = File(...)):
    """Diarize multiple audio files."""
    results = []

    for file in files:
        try:
            result = await diarize_audio(file)
            results.append({"filename": file.filename, "result": result})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8500"))
    uvicorn.run(app, host="0.0.0.0", port=port)