import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import IO, Dict, List, Optional

import torch
from loguru import logger
from pyannote.audio import Pipeline

from app.settings import settings


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


@dataclass
class SpeakerStats:
    label: str
    total_speaking_time: float
    percentage: float


@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    speakers: Dict[str, SpeakerStats]
    total_duration: float
    total_speech: float
    total_silence: float
    overlap_duration: float
    speaker_turns: int


class DiarizationService:
    """
    Speaker diarization service using PyAnnote 3.1.
    Identifies who speaks when in an audio file.
    """

    _pipeline: Optional[Pipeline] = None
    _pipeline_loaded: bool = False

    def __init__(
        self,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.model_name = model_name or settings.PYANNOTE_MODEL
        self.hf_token = hf_token or settings.HF_TOKEN
        self.use_gpu = use_gpu if use_gpu is not None else settings.USE_GPU
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        self._ensure_pipeline_loaded()

    @classmethod
    def _ensure_pipeline_loaded(cls) -> None:
        """Load PyAnnote pipeline (singleton pattern for efficiency)."""
        if cls._pipeline_loaded:
            return

        try:
            logger.info(f"Loading PyAnnote pipeline: {settings.PYANNOTE_MODEL}")

            cls._pipeline = Pipeline.from_pretrained(
                settings.PYANNOTE_MODEL,
                use_auth_token=settings.HF_TOKEN,
            )

            # Move to GPU if available
            if settings.USE_GPU and torch.cuda.is_available():
                device = torch.device(f"cuda:{settings.GPU_DEVICE}")
                cls._pipeline = cls._pipeline.to(device)
                logger.info(f"PyAnnote pipeline loaded on GPU: {device}")
            else:
                logger.info("PyAnnote pipeline loaded on CPU")

            cls._pipeline_loaded = True

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise RuntimeError(
                f"Failed to load diarization model. "
                f"Ensure HF_TOKEN is set and model is accessible. Error: {e}"
            )

    def _extract_audio_to_wav(self, input_path: str, output_path: str) -> None:
        """Extract audio to WAV format (16kHz mono) for PyAnnote."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract audio: {result.stderr}")

    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
        return 0.0

    def _calculate_overlap(self, segments: List[SpeakerSegment]) -> float:
        """Calculate total overlap duration between speakers."""
        if len(segments) < 2:
            return 0.0

        overlap_duration = 0.0
        sorted_segments = sorted(segments, key=lambda s: s.start)

        for i, seg1 in enumerate(sorted_segments):
            for seg2 in sorted_segments[i + 1:]:
                if seg2.start >= seg1.end:
                    break
                if seg1.speaker != seg2.speaker:
                    overlap_start = max(seg1.start, seg2.start)
                    overlap_end = min(seg1.end, seg2.end)
                    if overlap_end > overlap_start:
                        overlap_duration += overlap_end - overlap_start

        return overlap_duration

    def _count_speaker_turns(self, segments: List[SpeakerSegment]) -> int:
        """Count number of speaker turns (changes)."""
        if len(segments) < 2:
            return len(segments)

        turns = 1
        sorted_segments = sorted(segments, key=lambda s: s.start)
        prev_speaker = sorted_segments[0].speaker

        for seg in sorted_segments[1:]:
            if seg.speaker != prev_speaker:
                turns += 1
                prev_speaker = seg.speaker

        return turns

    def diarize(self, input_file: IO[bytes]) -> DiarizationResult:
        """
        Perform speaker diarization on audio/video file.

        Args:
            input_file: File-like object containing audio/video data

        Returns:
            DiarizationResult with speaker segments and statistics
        """
        # Create temp files
        with tempfile.NamedTemporaryFile(
            suffix=".input", dir=settings.TEMP_DIR, delete=False
        ) as tmp_input:
            input_path = tmp_input.name
            input_file.seek(0)
            tmp_input.write(input_file.read())

        wav_path = input_path + ".wav"

        try:
            logger.info("Extracting audio for diarization...")
            self._extract_audio_to_wav(input_path, wav_path)

            # Get total duration
            total_duration = self._get_audio_duration(wav_path)
            logger.info(f"Audio duration: {total_duration:.1f}s")

            # Run diarization
            logger.info("Running speaker diarization...")

            diarization_params = {}
            if self.num_speakers:
                diarization_params["num_speakers"] = self.num_speakers
            if self.min_speakers:
                diarization_params["min_speakers"] = self.min_speakers
            if self.max_speakers:
                diarization_params["max_speakers"] = self.max_speakers

            diarization = self._pipeline(wav_path, **diarization_params)

            # Convert to segments
            segments: List[SpeakerSegment] = []
            speaker_times: Dict[str, float] = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                )
                segments.append(segment)

                # Accumulate speaker time
                duration = turn.end - turn.start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

            # Calculate statistics
            total_speech = sum(speaker_times.values())
            total_silence = max(0, total_duration - total_speech)
            overlap_duration = self._calculate_overlap(segments)
            speaker_turns = self._count_speaker_turns(segments)

            # Build speaker stats
            speakers: Dict[str, SpeakerStats] = {}
            for speaker, time in speaker_times.items():
                percentage = (time / total_speech * 100) if total_speech > 0 else 0
                speakers[speaker] = SpeakerStats(
                    label=speaker,
                    total_speaking_time=round(time, 2),
                    percentage=round(percentage, 1),
                )

            logger.info(
                f"Diarization complete: {len(speakers)} speakers, "
                f"{len(segments)} segments, {speaker_turns} turns"
            )

            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                total_duration=round(total_duration, 2),
                total_speech=round(total_speech, 2),
                total_silence=round(total_silence, 2),
                overlap_duration=round(overlap_duration, 2),
                speaker_turns=speaker_turns,
            )

        finally:
            # Cleanup temp files
            for path in [input_path, wav_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {path}: {e}")

    def diarize_file(self, file_path: str) -> DiarizationResult:
        """
        Perform speaker diarization on audio/video file from path.

        Args:
            file_path: Path to audio/video file

        Returns:
            DiarizationResult with speaker segments and statistics
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        wav_path = file_path + ".diarize.wav"

        try:
            logger.info(f"Diarizing file: {file_path}")
            self._extract_audio_to_wav(file_path, wav_path)

            total_duration = self._get_audio_duration(wav_path)
            logger.info(f"Audio duration: {total_duration:.1f}s")

            diarization_params = {}
            if self.num_speakers:
                diarization_params["num_speakers"] = self.num_speakers
            if self.min_speakers:
                diarization_params["min_speakers"] = self.min_speakers
            if self.max_speakers:
                diarization_params["max_speakers"] = self.max_speakers

            diarization = self._pipeline(wav_path, **diarization_params)

            segments: List[SpeakerSegment] = []
            speaker_times: Dict[str, float] = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                )
                segments.append(segment)
                duration = turn.end - turn.start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

            total_speech = sum(speaker_times.values())
            total_silence = max(0, total_duration - total_speech)
            overlap_duration = self._calculate_overlap(segments)
            speaker_turns = self._count_speaker_turns(segments)

            speakers: Dict[str, SpeakerStats] = {}
            for speaker, time in speaker_times.items():
                percentage = (time / total_speech * 100) if total_speech > 0 else 0
                speakers[speaker] = SpeakerStats(
                    label=speaker,
                    total_speaking_time=round(time, 2),
                    percentage=round(percentage, 1),
                )

            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                total_duration=round(total_duration, 2),
                total_speech=round(total_speech, 2),
                total_silence=round(total_silence, 2),
                overlap_duration=round(overlap_duration, 2),
                speaker_turns=speaker_turns,
            )

        finally:
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup WAV file: {e}")


def check_pyannote_available() -> bool:
    """Check if PyAnnote is available and model is loaded."""
    try:
        return DiarizationService._pipeline_loaded
    except Exception:
        return False


def preload_pyannote_model() -> bool:
    """Preload PyAnnote model at startup."""
    try:
        DiarizationService._ensure_pipeline_loaded()
        return True
    except Exception as e:
        logger.error(f"Failed to preload PyAnnote model: {e}")
        return False
