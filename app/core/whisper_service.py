import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import IO, List, Optional

import torch
import whisperx
from loguru import logger

from app.settings import settings


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.0
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    segments: List[TranscriptionSegment]
    duration_seconds: float
    language: str


class WhisperService:
    """
    Transcription service using WhisperX.
    Uses faster-whisper backend with VAD for optimal transcription quality.

    Features:
    - initial_prompt support for domain-specific vocabulary guidance
    - temperature=0.0 for deterministic output (reduces hallucinations)
    - beam_size control for accuracy vs speed tradeoff
    - Built-in Silero VAD for hallucination prevention
    - Optional forced alignment for accurate word timestamps
    """

    _model = None
    _model_loaded = False
    _align_model = None
    _align_metadata = None
    _align_language = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "auto",
        batch_size: Optional[int] = None,
        beam_size: Optional[int] = None,
        temperature: Optional[float] = None,
        initial_prompt: Optional[str] = None,
        enable_alignment: Optional[bool] = None,
    ):
        self.model_name = model_name or settings.WHISPER_MODEL
        self.device = device or ("cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        # Use settings.WHISPER_LANGUAGE as default, None for auto-detection
        if language == "auto":
            self.language = settings.WHISPER_LANGUAGE if settings.WHISPER_LANGUAGE != "auto" else None
        else:
            self.language = language
        self.batch_size = batch_size or settings.WHISPER_BATCH_SIZE
        self.beam_size = beam_size or settings.WHISPER_BEAM_SIZE
        self.temperature = temperature if temperature is not None else settings.WHISPER_TEMPERATURE
        self.initial_prompt = initial_prompt or settings.WHISPER_INITIAL_PROMPT
        self.enable_alignment = enable_alignment if enable_alignment is not None else settings.WHISPER_ENABLE_ALIGNMENT

        # Adjust compute_type for CPU
        if self.device == "cpu" and self.compute_type == "float16":
            self.compute_type = "int8"

        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """Load WhisperX model (singleton pattern for efficiency)."""
        if cls._model_loaded:
            return

        try:
            device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
            compute_type = settings.WHISPER_COMPUTE_TYPE

            # Adjust compute_type for CPU
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"

            logger.info(f"Loading WhisperX model: {settings.WHISPER_MODEL} on {device} ({compute_type})")

            # WhisperX passes initial_prompt via asr_options during model load
            asr_options = {}
            if settings.WHISPER_INITIAL_PROMPT:
                asr_options["initial_prompt"] = settings.WHISPER_INITIAL_PROMPT

            cls._model = whisperx.load_model(
                settings.WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
                asr_options=asr_options if asr_options else None,
            )

            cls._model_loaded = True
            logger.info("WhisperX model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise RuntimeError(f"Failed to load WhisperX model: {e}")

    @classmethod
    def _load_align_model(cls, language_code: str, device: str) -> None:
        """Load alignment model for a specific language (lazy loading)."""
        if cls._align_model is not None and cls._align_language == language_code:
            return

        try:
            logger.info(f"Loading alignment model for language: {language_code}")
            cls._align_model, cls._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=device,
            )
            cls._align_language = language_code
            logger.info(f"Alignment model loaded for {language_code}")
        except Exception as e:
            logger.warning(f"Failed to load alignment model for {language_code}: {e}")
            cls._align_model = None
            cls._align_metadata = None

    def _extract_audio_to_wav(self, input_path: str, output_path: str) -> None:
        """Extract and preprocess audio to WAV format for WhisperX."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-af", "highpass=f=80,lowpass=f=8000,aresample=16000",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract audio: {result.stderr}")

    def _get_audio_duration(self, input_path: str) -> float:
        """Get audio duration using ffprobe."""

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            input_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
        return 0.0

    def _rebuild_segment_timestamps_from_words(
        self,
        words: List[WordTimestamp],
        original_start: float,
        original_end: float,
        text: str,
    ) -> tuple[float, float]:
        """
        Rebuild accurate segment timestamps from word-level timestamps.

        Word timestamps from forced alignment are much more accurate than
        segment timestamps. This method uses the first and last word
        boundaries as the true segment boundaries.

        Args:
            words: List of word timestamps from forced alignment
            original_start: Original segment start (fallback)
            original_end: Original segment end (fallback)
            text: Segment text (for duration estimation if no words)

        Returns:
            Tuple of (accurate_start, accurate_end)
        """
        # Filter words with valid timestamps
        valid_words = [
            w for w in words
            if w.start >= 0 and w.end > w.start
        ]

        if valid_words:
            # Use word boundaries as segment boundaries
            segment_start = valid_words[0].start
            segment_end = valid_words[-1].end

            # Sanity check: ensure reasonable bounds
            if segment_end > segment_start:
                return segment_start, segment_end

        # Fallback: use original timestamps if valid
        if original_end > original_start:
            return original_start, original_end

        # Last resort: estimate from text length
        # ~0.3 seconds per word is a reasonable estimate for speech
        word_count = len(text.split()) if text else 1
        estimated_duration = max(0.5, word_count * 0.3)

        logger.warning(
            f"No valid timestamps for segment '{text[:30]}...', "
            f"estimating {estimated_duration:.1f}s duration"
        )

        return original_start, original_start + estimated_duration

    def _convert_whisperx_result(
        self,
        result: dict,
        duration: float,
        detected_language: str,
    ) -> TranscriptionResult:
        """Convert WhisperX output to TranscriptionResult format."""
        segments: List[TranscriptionSegment] = []
        words_used_for_timing = 0
        segments_without_words = 0

        for seg in result.get("segments", []):
            words: List[WordTimestamp] = []

            # Extract word-level timestamps if available
            for word_data in seg.get("words", []):
                word_text = word_data.get("word", "").strip()
                if word_text:
                    # Get word timestamps, handling missing values
                    word_start = word_data.get("start")
                    word_end = word_data.get("end")

                    # Only add words with valid timestamps
                    if word_start is not None and word_end is not None:
                        words.append(WordTimestamp(
                            word=word_text,
                            start=float(word_start),
                            end=float(word_end),
                            confidence=float(word_data.get("score", 0.0)),
                        ))

            # Calculate segment confidence from word scores
            segment_confidence = 0.0
            if words:
                segment_confidence = sum(w.confidence for w in words) / len(words)
                words_used_for_timing += 1
            elif "avg_logprob" in seg:
                # Convert log probability to confidence (rough approximation)
                segment_confidence = min(1.0, max(0.0, math.exp(seg["avg_logprob"])))
                segments_without_words += 1

            text = seg.get("text", "").strip()
            original_start = float(seg.get("start", 0))
            original_end = float(seg.get("end", 0))

            # Rebuild accurate timestamps from words
            accurate_start, accurate_end = self._rebuild_segment_timestamps_from_words(
                words, original_start, original_end, text
            )

            segment = TranscriptionSegment(
                start=accurate_start,
                end=accurate_end,
                text=text,
                confidence=round(segment_confidence, 3),
                words=words if words else [],
            )

            if segment.text:
                segments.append(segment)

        # Log alignment coverage
        total_segments = words_used_for_timing + segments_without_words
        if total_segments > 0:
            coverage = words_used_for_timing / total_segments * 100
            logger.info(
                f"Word alignment coverage: {coverage:.1f}% "
                f"({words_used_for_timing}/{total_segments} segments)"
            )

        full_text = " ".join(seg.text for seg in segments)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration_seconds=duration,
            language=detected_language or "unknown",
        )

    def transcribe(
        self,
        input_file: IO[bytes],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file using WhisperX.

        Args:
            input_file: File-like object containing audio/video data
            language: Override language setting (optional)

        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        transcribe_language = language if language and language != "auto" else self.language

        with tempfile.NamedTemporaryFile(
            suffix=".input", dir=settings.TEMP_DIR, delete=False
        ) as tmp_input:
            input_path = tmp_input.name
            input_file.seek(0)
            tmp_input.write(input_file.read())

        wav_path = input_path + ".wav"

        try:
            logger.info(f"Starting WhisperX transcription (language={transcribe_language or 'auto'})")

            duration = self._get_audio_duration(input_path)
            logger.info(f"Audio duration: {duration:.1f}s")

            self._extract_audio_to_wav(input_path, wav_path)

            audio = whisperx.load_audio(wav_path)

            transcribe_params = {
                "batch_size": self.batch_size,
            }

            if transcribe_language:
                transcribe_params["language"] = transcribe_language


            start_time = time.time()

            result = self._model.transcribe(
                audio,
                **transcribe_params,
            )

            detected_language = result.get("language") or transcribe_language or "unknown"
            logger.info(f"Detected language: {detected_language}")

            if self.enable_alignment and detected_language:
                try:
                    self._load_align_model(detected_language, self.device)
                    if self._align_model is not None:
                        logger.info("Running forced alignment for word timestamps...")
                        result = whisperx.align(
                            result["segments"],
                            self._align_model,
                            self._align_metadata,
                            audio,
                            self.device,
                            return_char_alignments=False,
                        )
                except Exception as e:
                    logger.warning(f"Alignment failed, using original timestamps: {e}")

            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.1f}s")

            transcription_result = self._convert_whisperx_result(
                result, duration, detected_language
            )

            logger.info(
                f"Result: {len(transcription_result.segments)} segments, "
                f"{len(transcription_result.text)} chars"
            )

            return transcription_result

        finally:
            for path in [input_path, wav_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {path}: {e}")

    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio/video file from path.

        Args:
            file_path: Path to audio/video file
            language: Override language setting (optional)

        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        transcribe_language = language if language and language != "auto" else self.language

        wav_path = file_path + ".whisperx.wav"

        try:
            logger.info(f"Starting WhisperX transcription of {file_path}")

            duration = self._get_audio_duration(file_path)
            logger.info(f"Audio duration: {duration:.1f}s")

            self._extract_audio_to_wav(file_path, wav_path)

            audio = whisperx.load_audio(wav_path)

            transcribe_params = {
                "batch_size": self.batch_size,
            }

            if transcribe_language:
                transcribe_params["language"] = transcribe_language


            start_time = time.time()

            result = self._model.transcribe(
                audio,
                **transcribe_params,
            )

            detected_language = result.get("language") or transcribe_language or "unknown"

            if self.enable_alignment and detected_language:
                try:
                    self._load_align_model(detected_language, self.device)
                    if self._align_model is not None:
                        result = whisperx.align(
                            result["segments"],
                            self._align_model,
                            self._align_metadata,
                            audio,
                            self.device,
                            return_char_alignments=False,
                        )
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")

            elapsed = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed:.1f}s")

            return self._convert_whisperx_result(result, duration, detected_language)

        finally:
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup WAV file: {e}")


def check_whisper_available() -> bool:
    """Check if WhisperX model is loaded."""
    return WhisperService._model_loaded


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def preload_whisper_model() -> bool:
    """Preload WhisperX model at startup."""
    try:
        WhisperService._ensure_model_loaded()
        return True
    except Exception as e:
        logger.error(f"Failed to preload WhisperX model: {e}")
        return False
