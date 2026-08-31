"""
Audio Separator HTTP Server — voice/music separation using audio-separator (UVR models).

Extracts clean vocals from mixed audio.
Used as a pre-processing step before voice cloning.
"""

import logging
import os
import tempfile
import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Separator Server (UVR Models)")

MODEL_NAME = os.getenv("MODEL_NAME", "melband_roformer_big_beta4.ckpt")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEPARATOR = None


def get_separator():
    global SEPARATOR
    if SEPARATOR is None:
        logger.info("Initializing audio-separator with model: %s", MODEL_NAME)
        from audio_separator.separator import Separator
        SEPARATOR = Separator(
            output_dir=str(OUTPUT_DIR),
            model_file_dir="/models",
        )
        # Load model - model_name is positional argument
        SEPARATOR.load_model(MODEL_NAME)
        logger.info("Model loaded: %s", MODEL_NAME)
    return SEPARATOR


@app.on_event("startup")
async def startup():
    get_separator()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "audio-separator",
        "model": MODEL_NAME,
    }


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Separate vocals from background music/noise.

    Accepts a WAV/MP3/any audio file, returns the isolated vocal track as WAV.
    """
    separator = get_separator()

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    logger.info("Separating: %s (%d bytes)", file.filename, os.path.getsize(input_path))

    try:
        output_files = separator.separate(input_path)
        logger.info("Separator output files: %s", output_files)
        # audio-separator returns list of output file paths
        vocals_path = output_files[0] if output_files else None

        if vocals_path and os.path.exists(vocals_path):
            logger.info("Separation complete: %s", vocals_path)
            return FileResponse(
                path=vocals_path,
                media_type="audio/wav",
                filename=f"vocals_{file.filename}",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Separation produced no output files",
            )
    except Exception as e:
        logger.error("Separation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


class SeparateJsonRequest(BaseModel):
    """Request for voice separation with base64 audio."""
    audio_base64: str
    filename: str = "audio.wav"


@app.post("/separate_json")
async def separate_audio_json(request: SeparateJsonRequest):
    """
    Separate vocals from background music/noise (JSON API).

    Accepts base64-encoded audio, returns base64-encoded vocals.
    Used by Dublaj pipeline for voice cloning.
    """
    separator = get_separator()

    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio_base64)

    suffix = Path(request.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        input_path = tmp.name

    logger.info("Separating (JSON): %s (%d bytes)", request.filename, len(audio_bytes))

    try:
        output_files = separator.separate(input_path)

        # UVR outputs two stems: first = Vocals, second = Instrumental/No Vocals
        logger.info("Separator output files: %s", output_files)

        vocals_path = None
        instrumental_path = None

        # Try filename matching first
        for f in output_files:
            fname = f.lower() if f else ""
            if "vocals" in fname and "no_" not in fname and "other" not in fname:
                vocals_path = f
            elif "no_vocals" in fname or "instrumental" in fname or "other" in fname:
                instrumental_path = f

        # Fallback: first file = instrumental/other, second = vocals
        if not vocals_path and len(output_files) >= 1:
            # audio-separator typically outputs: [other, vocals]
            if len(output_files) >= 2:
                instrumental_path = output_files[0]
                vocals_path = output_files[1]
            else:
                vocals_path = output_files[0]

        # Resolve paths: separator may return relative filenames in OUTPUT_DIR
        if vocals_path and not os.path.isabs(vocals_path):
            vocals_path = str(OUTPUT_DIR / vocals_path)
        if instrumental_path and not os.path.isabs(instrumental_path):
            instrumental_path = str(OUTPUT_DIR / instrumental_path)

        logger.info("Resolved paths: vocals=%s, instrumental=%s", vocals_path, instrumental_path)

        if vocals_path and os.path.exists(vocals_path):
            logger.info("Separation complete: vocals=%s, instrumental=%s",
                        vocals_path, instrumental_path)

            vocals_bytes = Path(vocals_path).read_bytes()
            vocals_base64 = base64.b64encode(vocals_bytes).decode('utf-8')

            result = {
                "status": "ok",
                "vocals_base64": vocals_base64,
                "filename": f"vocals_{request.filename}",
            }

            # Also return instrumental if available
            if instrumental_path and os.path.exists(instrumental_path):
                instrumental_bytes = Path(instrumental_path).read_bytes()
                result["instrumental_base64"] = base64.b64encode(instrumental_bytes).decode('utf-8')

            return result
        else:
            raise HTTPException(
                status_code=500,
                detail="Separation produced no output files",
            )
    except Exception as e:
        logger.error("Separation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
