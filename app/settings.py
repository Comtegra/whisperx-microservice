import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings configuration from environment variables"""

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/"

    # Authentication settings
    APP_TOKEN: str = os.getenv("APP_TOKEN", "")
    ENABLE_AUTH: bool = True

    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    GPU_DEVICE: int = int(os.getenv("GPU_DEVICE", "0"))

    # WhisperX settings
    # Valid models: tiny, base, small, medium, large-v1, large-v2, large-v3, distil-large-v2
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "auto")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    WHISPER_BATCH_SIZE: int = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
    WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    WHISPER_TEMPERATURE: float = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
    WHISPER_INITIAL_PROMPT: str = os.getenv(
        "WHISPER_INITIAL_PROMPT",
        ""
    )
    WHISPER_ENABLE_ALIGNMENT: bool = os.getenv("WHISPER_ENABLE_ALIGNMENT", "true").lower() == "true"

    # PyAnnote settings (for diarization)
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    PYANNOTE_MODEL: str = os.getenv(
        "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
    )

    # Processing settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "2048"))  # 2GB default
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp")


settings = Settings()
