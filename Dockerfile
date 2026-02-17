# =============================================================================
# FFmpeg Microservice - Simplified Docker Build
# WhisperX + PyAnnote Speaker Diarization (Standard FFmpeg from apt)
# =============================================================================

FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies + standard FFmpeg + build tools for PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    libsndfile1 \
    ca-certificates \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY README.md .

# Install pip and upgrade
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support
ARG USE_CUDA_PYTORCH=true
RUN if [ "$USE_CUDA_PYTORCH" = "true" ]; then \
    pip install --no-cache-dir \
        torch==2.1.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121; \
    else \
    pip install --no-cache-dir \
        torch==2.1.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install the application (includes whisperx)
COPY . .
RUN pip install --no-cache-dir . && \
    rm -rf /root/.cache/pip

# Pre-download WhisperX model (optional - can also download at runtime)
# Valid: tiny, base, small, medium, large-v1, large-v2, large-v3, distil-large-v2
ARG WHISPER_MODEL=large-v3
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download WhisperX model if HF_TOKEN provided
RUN if [ -n "$HF_TOKEN" ]; then \
    python -c "import whisperx; \
import os; \
from huggingface_hub import login; \
token = os.environ.get('HF_TOKEN'); \
login(token=token); \
whisperx.load_model('${WHISPER_MODEL}', device='cpu', compute_type='int8')"; \
    else echo "No HF_TOKEN provided, WhisperX model will be downloaded at runtime"; \
    fi

# Download PyAnnote models
RUN if [ -n "$HF_TOKEN" ]; then \
    python -c "import os; \
from huggingface_hub import login; \
from pyannote.audio import Pipeline; \
token = os.environ.get('HF_TOKEN'); \
login(token=token); \
Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"; \
    else echo "No HF_TOKEN provided, skipping PyAnnote model download"; \
    fi

# Clear HF token from environment for security
ENV HF_TOKEN=

# Environment variables
ENV WHISPER_MODEL=${WHISPER_MODEL}
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "--workers", "2", "--threads", "1", "--bind", "0.0.0.0:8000", "--timeout", "1800", "app.server_app:create_app()"]
