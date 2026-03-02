import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import IO, Dict, List, Optional, Tuple

from loguru import logger

from app.core.diarization_service import (
    DiarizationResult,
    DiarizationService,
    SpeakerSegment,
    SpeakerStats,
)
from app.core.whisper_service import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperService,
    WordTimestamp,
)
from app.settings import settings


@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 0.0
    words: List[WordTimestamp] = field(default_factory=list)


@dataclass
class CallMetrics:
    total_duration: float
    total_speech: float
    total_silence: float
    silence_percentage: float
    overlap_duration: float
    overlap_percentage: float
    speaker_turns: int


@dataclass
class DiarizedTranscription:
    text: str
    speakers: Dict[str, SpeakerStats]
    segments: List[DiarizedSegment]
    metrics: CallMetrics
    duration_seconds: float
    language: str


class AlignmentService:
    """
    Service for aligning transcription with speaker diarization.
    Combines whisper transcription timestamps with pyannote speaker segments.
    """

    def __init__(
        self,
        language: str = "auto",
        diarization_mode: str = "pyannote",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.language = language
        self.diarization_mode = diarization_mode
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def _find_speaker_for_segment(
        self,
        transcription_segment: TranscriptionSegment,
        diarization_segments: List[SpeakerSegment],
    ) -> str:
        """
        Find the most likely speaker for a transcription segment.
        Uses overlap duration to determine the speaker.
        """
        seg_start = transcription_segment.start
        seg_end = transcription_segment.end
        seg_mid = (seg_start + seg_end) / 2

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for diar_seg in diarization_segments:
            # Calculate overlap
            overlap_start = max(seg_start, diar_seg.start)
            overlap_end = min(seg_end, diar_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg.speaker

        # If no overlap found, find closest segment by midpoint
        if best_speaker == "UNKNOWN" and diarization_segments:
            closest_dist = float("inf")
            for diar_seg in diarization_segments:
                diar_mid = (diar_seg.start + diar_seg.end) / 2
                dist = abs(seg_mid - diar_mid)
                if dist < closest_dist:
                    closest_dist = dist
                    best_speaker = diar_seg.speaker

        return best_speaker

    def _validate_final_output(
        self,
        segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:
        """
        Final validation pass to ensure all segments are valid before output.

        This is the last line of defense - catches any edge cases that
        slipped through earlier processing.

        Validates:
        - All segments have end > start
        - Segments are sorted by start time
        - No negative timestamps
        - All segments have non-empty text
        """
        valid_segments: List[DiarizedSegment] = []
        issues_found = 0

        for seg in segments:
            # Skip segments with invalid timestamps
            if seg.end <= seg.start:
                logger.warning(
                    f"Final validation: Removing invalid segment "
                    f"(end <= start): {seg.start}-{seg.end} '{seg.text[:30]}...'"
                )
                issues_found += 1
                continue

            # Skip segments with negative timestamps
            if seg.start < 0 or seg.end < 0:
                logger.warning(
                    f"Final validation: Removing segment with negative timestamp: "
                    f"{seg.start}-{seg.end}"
                )
                issues_found += 1
                continue

            # Skip empty text
            if not seg.text or not seg.text.strip():
                issues_found += 1
                continue

            valid_segments.append(seg)

        # Ensure sorted by start time
        valid_segments.sort(key=lambda s: (s.start, s.end))

        if issues_found > 0:
            logger.info(f"Final validation: Fixed {issues_found} invalid segments")

        return valid_segments

    def _sanitize_segments(
        self,
        segments: List[DiarizedSegment],
    ) -> List[DiarizedSegment]:
        """
        Sanitize segments by fixing timestamp issues from WhisperX.

        Fixes:
        - Invalid timestamps where end <= start
        - Very short segments that are likely noise
        - Removes empty text segments
        - Removes duplicate/near-duplicate segments
        """
        sanitized: List[DiarizedSegment] = []
        seen_segments: set = set()  # Track (start, speaker, text_hash) to detect duplicates
        min_duration = 0.1  # Minimum 100ms segment

        for seg in segments:
            # Skip empty text segments
            if not seg.text or not seg.text.strip():
                continue

            # Skip segments with just punctuation or single characters
            text_clean = seg.text.strip()
            if len(text_clean) <= 1 and not text_clean.isalnum():
                continue

            start = seg.start
            end = seg.end

            # Fix invalid timestamps (end <= start)
            if end <= start:
                # Use a minimum duration based on text length
                # Rough estimate: ~0.3 seconds per word
                word_count = len(text_clean.split())
                estimated_duration = max(min_duration, word_count * 0.3)
                end = start + estimated_duration
                logger.debug(
                    f"Fixed invalid timestamp: {seg.start}->{seg.end} to {start}->{end} "
                    f"for text: '{text_clean[:30]}...'"
                )

            # Skip very short segments
            if end - start < min_duration:
                continue

            # Detect duplicates: same speaker, similar start time (within 0.5s), same text
            start_bucket = round(start * 2) / 2  # Round to nearest 0.5s
            segment_key = (start_bucket, seg.speaker, text_clean[:50])

            if segment_key in seen_segments:
                logger.debug(f"Skipping duplicate segment: '{text_clean[:30]}...' at {start}")
                continue
            seen_segments.add(segment_key)

            sanitized.append(DiarizedSegment(
                start=round(start, 2),
                end=round(end, 2),
                speaker=seg.speaker,
                text=text_clean,
                confidence=seg.confidence,
                words=seg.words,
            ))

        return sanitized

    def _merge_adjacent_segments(
        self,
        segments: List[DiarizedSegment],
        max_gap: float = 0.5,
    ) -> List[DiarizedSegment]:
        """
        Merge adjacent segments from the same speaker.
        Improves readability by combining short segments.
        """
        if not segments:
            return []

        merged: List[DiarizedSegment] = []
        current = segments[0]

        for next_seg in segments[1:]:
            # Check if should merge: same speaker and small gap
            gap = next_seg.start - current.end
            if next_seg.speaker == current.speaker and gap <= max_gap:
                # Merge segments - combine words and average confidence
                combined_words = current.words + next_seg.words
                combined_confidence = (
                    (current.confidence + next_seg.confidence) / 2
                    if current.confidence and next_seg.confidence
                    else current.confidence or next_seg.confidence
                )
                current = DiarizedSegment(
                    start=current.start,
                    end=next_seg.end,
                    speaker=current.speaker,
                    text=f"{current.text} {next_seg.text}".strip(),
                    confidence=round(combined_confidence, 3),
                    words=combined_words,
                )
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged

    def align(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
    ) -> DiarizedTranscription:
        """
        Align transcription segments with diarization speaker segments.

        Args:
            transcription: Result from WhisperService
            diarization: Result from DiarizationService

        Returns:
            DiarizedTranscription with combined speaker and text information
        """
        logger.info("Aligning transcription with diarization...")

        # Align each transcription segment to a speaker
        diarized_segments: List[DiarizedSegment] = []

        for trans_seg in transcription.segments:
            speaker = self._find_speaker_for_segment(
                trans_seg, diarization.segments
            )
            diarized_segments.append(
                DiarizedSegment(
                    start=round(trans_seg.start, 2),
                    end=round(trans_seg.end, 2),
                    speaker=speaker,
                    text=trans_seg.text,
                    confidence=trans_seg.confidence,
                    words=trans_seg.words,
                )
            )

        # Sanitize segments (fix invalid timestamps, remove noise)
        diarized_segments = self._sanitize_segments(diarized_segments)

        # Merge adjacent segments from same speaker
        merged_segments = self._merge_adjacent_segments(diarized_segments)

        # Final validation - ensure all segments are valid before output
        merged_segments = self._validate_final_output(merged_segments)

        # Calculate metrics
        silence_percentage = (
            (diarization.total_silence / diarization.total_duration * 100)
            if diarization.total_duration > 0
            else 0
        )
        overlap_percentage = (
            (diarization.overlap_duration / diarization.total_duration * 100)
            if diarization.total_duration > 0
            else 0
        )

        metrics = CallMetrics(
            total_duration=diarization.total_duration,
            total_speech=diarization.total_speech,
            total_silence=diarization.total_silence,
            silence_percentage=round(silence_percentage, 1),
            overlap_duration=diarization.overlap_duration,
            overlap_percentage=round(overlap_percentage, 1),
            speaker_turns=diarization.speaker_turns,
        )

        logger.info(
            f"Alignment complete: {len(merged_segments)} segments, "
            f"{len(diarization.speakers)} speakers"
        )

        return DiarizedTranscription(
            text=transcription.text,
            speakers=diarization.speakers,
            segments=merged_segments,
            metrics=metrics,
            duration_seconds=diarization.total_duration,
            language=transcription.language,
        )

    def _extract_stereo_channels(
        self, input_path: str
    ) -> Tuple[str, str]:
        """
        Extract left and right channels from stereo audio file.

        Uses minimal audio processing to preserve signal quality for
        accurate forced alignment. Only resamples to 16kHz (required by Whisper).

        Args:
            input_path: Path to stereo audio file

        Returns:
            Tuple of (left_channel_path, right_channel_path)
        """
        left_path = input_path.replace(".input", "_left.wav")
        right_path = input_path.replace(".input", "_right.wav")

        # Extract left channel (channel 0)
        # Minimal processing: just channel extraction and resampling
        # Avoid aggressive filtering that can hurt alignment accuracy
        left_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "pan=mono|c0=c0,aresample=16000",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            left_path
        ]

        # Extract right channel (channel 1)
        right_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "pan=mono|c0=c1,aresample=16000",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            right_path
        ]

        logger.info("Extracting stereo channels for transcription...")

        # Run both extractions
        for cmd, channel in [(left_cmd, "left"), (right_cmd, "right")]:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to extract {channel} channel: {result.stderr}"
                )

        return left_path, right_path

    def _process_channel_diarization(
        self, input_path: str
    ) -> DiarizedTranscription:
        """
        Process audio using channel-based diarization.
        Left channel = SPEAKER_00, Right channel = SPEAKER_01.

        Args:
            input_path: Path to stereo audio file

        Returns:
            DiarizedTranscription with channel-based speaker assignment
        """
        left_path = None
        right_path = None

        try:
            # Extract stereo channels
            left_path, right_path = self._extract_stereo_channels(input_path)

            # Initialize whisper service with forced alignment ALWAYS enabled
            # for channel diarization - word-level timestamps are critical
            # for accurate segment boundaries
            whisper_service = WhisperService(
                language=self.language,
                enable_alignment=True,  # Force alignment on for accurate timestamps
            )

            # Transcribe channels sequentially to avoid thread-safety issues
            # with shared WhisperX/CTranslate2 model (not safe for concurrent access)
            logger.info("Transcribing left channel...")
            try:
                left_result = whisper_service.transcribe_file(left_path)
                logger.info("Left channel transcription completed")
            except Exception as e:
                logger.error(f"left channel transcription failed: {e}")
                raise RuntimeError(f"left channel transcription failed: {e}")

            logger.info("Transcribing right channel...")
            try:
                right_result = whisper_service.transcribe_file(right_path)
                logger.info("Right channel transcription completed")
            except Exception as e:
                logger.error(f"right channel transcription failed: {e}")
                raise RuntimeError(f"right channel transcription failed: {e}")

            # Combine segments with speaker labels
            all_segments: List[DiarizedSegment] = []

            for seg in left_result.segments:
                all_segments.append(
                    DiarizedSegment(
                        start=round(seg.start, 2),
                        end=round(seg.end, 2),
                        speaker="SPEAKER_00",
                        text=seg.text,
                        confidence=seg.confidence,
                        words=seg.words,
                    )
                )

            for seg in right_result.segments:
                all_segments.append(
                    DiarizedSegment(
                        start=round(seg.start, 2),
                        end=round(seg.end, 2),
                        speaker="SPEAKER_01",
                        text=seg.text,
                        confidence=seg.confidence,
                        words=seg.words,
                    )
                )

            # Sanitize segments (fix invalid timestamps, remove noise)
            all_segments = self._sanitize_segments(all_segments)

            # Sort by start time, then by end time for overlapping segments
            all_segments.sort(key=lambda s: (s.start, s.end))

            # Final validation - ensure all segments are valid before output
            all_segments = self._validate_final_output(all_segments)

            # Calculate speaker stats
            speaker_00_duration = sum(
                seg.end - seg.start for seg in all_segments
                if seg.speaker == "SPEAKER_00"
            )
            speaker_01_duration = sum(
                seg.end - seg.start for seg in all_segments
                if seg.speaker == "SPEAKER_01"
            )
            total_speech = speaker_00_duration + speaker_01_duration

            # Get audio duration
            duration = max(left_result.duration_seconds, right_result.duration_seconds)

            speakers = {
                "SPEAKER_00": SpeakerStats(
                    label="SPEAKER_00",
                    total_speaking_time=round(speaker_00_duration, 2),
                    percentage=round(
                        speaker_00_duration / total_speech * 100, 1
                    ) if total_speech > 0 else 0,
                ),
                "SPEAKER_01": SpeakerStats(
                    label="SPEAKER_01",
                    total_speaking_time=round(speaker_01_duration, 2),
                    percentage=round(
                        speaker_01_duration / total_speech * 100, 1
                    ) if total_speech > 0 else 0,
                ),
            }

            # Calculate metrics
            total_silence = max(0, duration - total_speech)
            silence_percentage = (
                total_silence / duration * 100 if duration > 0 else 0
            )

            # Count speaker turns
            speaker_turns = 1
            for i in range(1, len(all_segments)):
                if all_segments[i].speaker != all_segments[i - 1].speaker:
                    speaker_turns += 1

            metrics = CallMetrics(
                total_duration=round(duration, 2),
                total_speech=round(total_speech, 2),
                total_silence=round(total_silence, 2),
                silence_percentage=round(silence_percentage, 1),
                overlap_duration=0.0,  # Channel separation means no overlap
                overlap_percentage=0.0,
                speaker_turns=speaker_turns,
            )

            # Combine text
            combined_text = " ".join(
                seg.text for seg in all_segments if seg.text
            )

            # Determine output language
            output_language = left_result.language or right_result.language or "unknown"

            logger.info(
                f"Channel diarization complete: {len(all_segments)} segments, "
                f"2 speakers"
            )

            return DiarizedTranscription(
                text=combined_text,
                speakers=speakers,
                segments=all_segments,
                metrics=metrics,
                duration_seconds=duration,
                language=output_language,
            )

        finally:
            # Cleanup temp files
            for path in [left_path, right_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file {path}: {e}")

    def process(self, input_file: IO[bytes]) -> DiarizedTranscription:
        """
        Full diarization pipeline: transcribe + diarize + align.
        Supports both PyAnnote-based and channel-based diarization.

        Args:
            input_file: File-like object containing audio/video data

        Returns:
            DiarizedTranscription with full analysis
        """
        # Save input to temp file (needed for parallel processing)
        with tempfile.NamedTemporaryFile(
            suffix=".input", dir=settings.TEMP_DIR, delete=False
        ) as tmp_input:
            input_path = tmp_input.name
            input_file.seek(0)
            tmp_input.write(input_file.read())

        try:
            # Use channel-based diarization if requested
            if self.diarization_mode == "channel":
                logger.info("Using channel-based diarization (stereo separation)")
                return self._process_channel_diarization(input_path)

            # Default: PyAnnote-based diarization
            logger.info("Starting parallel transcription and diarization (PyAnnote)...")

            # Initialize services
            whisper_service = WhisperService(
                language=self.language,
            )
            diarization_service = DiarizationService(
                num_speakers=self.num_speakers,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

            # Run in parallel
            transcription_result: Optional[TranscriptionResult] = None
            diarization_result: Optional[DiarizationResult] = None

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        whisper_service.transcribe_file, input_path
                    ): "transcription",
                    executor.submit(
                        diarization_service.diarize_file, input_path
                    ): "diarization",
                }

                for future in as_completed(futures):
                    task_name = futures[future]
                    try:
                        result = future.result()
                        if task_name == "transcription":
                            transcription_result = result
                            logger.info("Transcription completed")
                        else:
                            diarization_result = result
                            logger.info("Diarization completed")
                    except Exception as e:
                        logger.error(f"{task_name} failed: {e}")
                        raise RuntimeError(f"{task_name} failed: {e}")

            if not transcription_result or not diarization_result:
                raise RuntimeError("Failed to complete transcription or diarization")

            # Align results
            aligned = self.align(transcription_result, diarization_result)

            return aligned

        finally:
            # Cleanup temp file
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

    def process_file(self, file_path: str) -> DiarizedTranscription:
        """
        Full diarization pipeline from file path.
        Supports both PyAnnote-based and channel-based diarization.

        Args:
            file_path: Path to audio/video file

        Returns:
            DiarizedTranscription with full analysis
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing file for diarized transcription: {file_path}")

        # Use channel-based diarization if requested
        if self.diarization_mode == "channel":
            logger.info("Using channel-based diarization (stereo separation)")
            return self._process_channel_diarization(file_path)

        # Default: PyAnnote-based diarization
        whisper_service = WhisperService(
            language=self.language,
        )
        diarization_service = DiarizationService(
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        # Run in parallel
        transcription_result: Optional[TranscriptionResult] = None
        diarization_result: Optional[DiarizationResult] = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    whisper_service.transcribe_file, file_path
                ): "transcription",
                executor.submit(
                    diarization_service.diarize_file, file_path
                ): "diarization",
            }

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    if task_name == "transcription":
                        transcription_result = result
                    else:
                        diarization_result = result
                except Exception as e:
                    logger.error(f"{task_name} failed: {e}")
                    raise RuntimeError(f"{task_name} failed: {e}")

        if not transcription_result or not diarization_result:
            raise RuntimeError("Failed to complete transcription or diarization")

        aligned = self.align(transcription_result, diarization_result)

        return aligned
