# ============================================
# Dockerfile — Windowed RoFormer Separator (WSA)
# ============================================
# Модель: smulelabs/windowed-roformer (mbr-win10-sink8.ckpt)
#   Mel-Band RoFormer + Windowed Sink Attention:
#   92% SDR оригинала при 44.5x меньше FLOPs.
# Код модели клонируется с GitHub при сборке (автономный сервис).
#
# Сборка:
#   docker build -f docker3/windowed-roformer/Dockerfile -t dublaj-windowed-roformer .
# ============================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/mbr-win10-sink8.ckpt

# Системные зависимости
# gcc/g++/make: flex_attention компилирует ядра через Triton (torch.compile)
# — без C-компилятора инференс падает с "Failed to find C compiler"
# git: клонирование кода модели при сборке
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libsndfile1 \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.8 + CUDA 12.8 (cu128: поддержка Blackwell sm_120 и Ada sm_89;
# flex_attention из репозитория требует torch>=2.5, репо пинит 2.8).
# --retries/--timeout: колёса ~3 GB, при обрыве сети pip сам повторяет попытки
RUN pip install --no-cache-dir --retries 10 --timeout 300 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0

# Python-зависимости модели (из requirements репозитория) + API
RUN pip install --no-cache-dir --retries 10 --timeout 300 \
    numpy \
    librosa \
    einops \
    rotary_embedding_torch==0.3.5 \
    soundfile \
    fastapi \
    uvicorn \
    python-multipart

WORKDIR /app

# Код модели клонируется с GitHub при сборке — сервис автономен,
# локальная папка windowed-roformer/ в корне проекта НЕ нужна
RUN git clone --depth 1 https://github.com/smulelabs/windowed-roformer.git /tmp/wr && \
    mv /tmp/wr/model.py /tmp/wr/modules.py /tmp/wr/flex_attention_utils.py /tmp/wr/config.yaml /app/ && \
    rm -rf /tmp/wr

# Наш FastAPI сервер (контракт совместим с dublaj-audio-separator)
COPY docker3/windowed-roformer/server_windowed_roformer.py /app/server.py

EXPOSE 8310

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -f http://localhost:8310/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8310"]
