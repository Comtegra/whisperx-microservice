from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================


class ExtractAudioModel(BaseModel):
    output_format: Literal["mp3"] = Field(
        "mp3", description="Output audio format. Only 'mp3' is supported."
    )


class TranscribeRequestModel(BaseModel):
    language: str = Field(
        "auto",
        description="Language code for transcription (e.g., 'en', 'pl', 'de', 'auto'). "
        "Use 'auto' for automatic detection.",
    )


class TranscribeSegmentsRequestModel(BaseModel):
    language: str = Field(
        "auto",
        description="Language code for transcription (e.g., 'en', 'pl', 'de', 'auto'). "
        "Use 'auto' for automatic detection.",
    )
    granularity: Literal["segment", "word"] = Field(
        "segment",
        description="Timestamp granularity: 'segment' for phrase-level, 'word' for word-level.",
    )


class DiarizeRequestModel(BaseModel):
    language: str = Field(
        "auto",
        description="Language code for transcription (e.g., 'en', 'pl', 'de', 'auto'). "
        "Use 'auto' for automatic detection.",
    )
    diarization_mode: Literal["pyannote", "channel"] = Field(
        "pyannote",
        description="Diarization method: 'pyannote' for AI-based speaker detection, "
        "'channel' for stereo channel separation (left=SPEAKER_00, right=SPEAKER_01).",
    )
    granularity: Literal["segment", "word"] = Field(
        "segment",
        description="Timestamp granularity: 'segment' for phrase-level, 'word' for word-level.",
    )
    num_speakers: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Expected number of speakers (optional, auto-detect if omitted). "
        "Only used with diarization_mode='pyannote'.",
    )
    min_speakers: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Minimum number of speakers (optional). "
        "Only used with diarization_mode='pyannote'.",
    )
    max_speakers: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum number of speakers (optional). "
        "Only used with diarization_mode='pyannote'.",
    )


# =============================================================================
# Response Models
# =============================================================================


class TranscribeResponse(BaseModel):
    success: bool
    text: str
    duration_seconds: float
    language: str


class WordTimestamp(BaseModel):
    word: str = Field(..., description="The word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")


class TranscriptionSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    words: Optional[List[WordTimestamp]] = Field(
        None, description="Word-level timestamps (only when granularity='word')"
    )


class TranscribeSegmentsResponse(BaseModel):
    success: bool
    text: str
    segments: List[TranscriptionSegment]
    duration_seconds: float
    language: str


class SpeakerInfo(BaseModel):
    label: str = Field(..., description="Speaker label (e.g., 'SPEAKER_00')")
    total_speaking_time: float = Field(..., description="Total speaking time in seconds")
    percentage: float = Field(..., description="Percentage of total speaking time")


class DiarizedSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    speaker: str = Field(..., description="Speaker label")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    words: Optional[List[WordTimestamp]] = Field(
        None, description="Word-level timestamps (only when granularity='word')"
    )


class CallMetrics(BaseModel):
    total_duration: float = Field(..., description="Total audio duration in seconds")
    total_speech: float = Field(..., description="Total speech duration in seconds")
    total_silence: float = Field(..., description="Total silence duration in seconds")
    silence_percentage: float = Field(..., description="Silence as percentage of total")
    overlap_duration: float = Field(..., description="Total overlap duration in seconds")
    overlap_percentage: float = Field(..., description="Overlap as percentage of total")
    speaker_turns: int = Field(..., description="Number of speaker turns")


class DiarizeResponse(BaseModel):
    success: bool
    text: str
    speakers: Dict[str, SpeakerInfo]
    segments: List[DiarizedSegment]
    metrics: CallMetrics
    duration_seconds: float
    language: str
