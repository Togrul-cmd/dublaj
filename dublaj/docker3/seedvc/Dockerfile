# ============================================
# Dockerfile — Seed-VC Voice Conversion Server (F0 Base Model)
# ============================================
# Модель: seed-uvit-whisper-base (200M) — F0 conditioning
# Source: https://github.com/Plachtaa/seed-vc
#
# Сборка:
#   docker build -f docker3/seedvc/Dockerfile -t dublaj-seedvc .
# ============================================

FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем PyTorch с CUDA 12.8 (cu128 — поддержка Blackwell: RTX PRO 6000 / RTX 5090, sm_120)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Устанавливаем остальные зависимости для Seed-VC
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    huggingface-hub>=0.28.1 \
    accelerate \
    scipy==1.13.1 \
    librosa==0.10.2 \
    munch==4.0.0 \
    einops==0.8.0 \
    descript-audio-codec==1.0.0 \
    gradio==5.23.0 \
    pydub==0.25.1 \
    resemblyzer \
    jiwer==3.0.3 \
    soundfile==0.12.1 \
    sounddevice==0.5.0 \
    modelscope==1.18.1 \
    funasr==1.1.5 \
    numpy==1.26.4 \
    hydra-core==1.3.2 \
    pyyaml \
    python-dotenv \
    fastapi \
    uvicorn \
    python-multipart \
    onnxruntime

# Клонируем Seed-VC репозиторий (ВСЕ зависимости включены в этот репо)
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git .

# Копируем свой FastAPI сервер
COPY docker3/seedvc/server_seedvc.py /workspace/server_seedvc.py

EXPOSE 8000

WORKDIR /workspace

CMD ["python", "server_seedvc.py"]