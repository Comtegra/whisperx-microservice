from flask import Blueprint, Response, jsonify, request, stream_with_context
from loguru import logger
from pydantic import ValidationError

from app.api.models import (
    DiarizeRequestModel,
    DiarizeResponse,
    ExtractAudioModel,
    TranscribeRequestModel,
    TranscribeResponse,
    TranscribeSegmentsRequestModel,
    TranscribeSegmentsResponse,
)
from app.core.alignment_service import AlignmentService
from app.core.av_service import AudioExtractor
from app.core.diarization_service import check_pyannote_available
from app.core.whisper_service import (
    WhisperService,
    check_gpu_available,
    check_whisper_available,
)
from app.settings import settings

api_blueprint = Blueprint("api", __name__)


# =============================================================================
# Health Check
# =============================================================================


@api_blueprint.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check with GPU and model status."""
    try:
        gpu_available = check_gpu_available()
        whisper_available = check_whisper_available()
        pyannote_available = check_pyannote_available()

        return jsonify({
            "success": True,
            "message": "Service is healthy",
            "gpu_available": gpu_available,
            "whisper_model_loaded": whisper_available,
            "pyannote_model_loaded": pyannote_available,
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Health check failed: {e}",
        }), 500


# =============================================================================
# Audio Extraction (existing functionality)
# =============================================================================


@api_blueprint.route("/extract", methods=["POST"])
def extract_audio():
    """Extract audio from video file and return as MP3 stream."""
    logger.info("Received request for audio extraction.")

    if "file" not in request.files:
        logger.warning("No file provided in the request.")
        return jsonify({"success": False, "message": "No file provided"}), 400

    video_file = request.files["file"]
    if video_file.filename == "":
        logger.warning("Empty filename provided.")
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Check file size
    video_file.seek(0, 2)  # Seek to end
    file_size = video_file.tell()
    video_file.seek(0)  # Reset to beginning

    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes")
        return jsonify({
            "success": False,
            "message": f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
        }), 400

    logger.info(f"Processing file for extraction: {video_file.filename}")

    try:
        params_json = request.form.get("params", '{"output_format": "mp3"}')
        extraction_params = ExtractAudioModel.model_validate_json(params_json)
        logger.info(f"Extraction parameters parsed: {extraction_params.model_dump()}")
    except ValidationError as e:
        return jsonify(success=False, errors=e.errors()), 400

    try:
        logger.info("Initializing audio extraction.")
        extractor = AudioExtractor(
            input_file_obj=video_file, output_format=extraction_params.output_format
        )
        output_filename = (
            f"{video_file.filename.rsplit('.', 1)[0]}.{extraction_params.output_format}"
        )

        logger.info(f"Streaming extracted audio as {output_filename}")

        def generate():
            try:
                for chunk in extractor.stream_mp3_bytes():
                    yield chunk
                logger.success("Finished streaming audio chunks.")
            except Exception as stream_err:
                logger.error(f"Error during audio stream generation: {stream_err}")

        return Response(
            stream_with_context(generate()),
            mimetype=f"audio/{extraction_params.output_format}",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"},
        )

    except ValueError as ve:
        logger.warning(f"Audio extraction error: {str(ve)}")
        return jsonify({
            "success": False,
            "message": f"Extraction Error: {str(ve)}",
        }), 400
    except Exception as e:
        logger.exception(f"Unexpected error during audio extraction: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Error: Internal server error",
        }), 500


# =============================================================================
# Transcription - Plain Text
# =============================================================================


@api_blueprint.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe audio/video file and return plain text.
    Uses WhisperX with Silero VAD.
    """
    logger.info("Received request for transcription (plain text).")

    if "file" not in request.files:
        logger.warning("No file provided in the request.")
        return jsonify({"success": False, "message": "No file provided"}), 400

    video_file = request.files["file"]
    if video_file.filename == "":
        logger.warning("Empty filename provided.")
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Check file size
    video_file.seek(0, 2)
    file_size = video_file.tell()
    video_file.seek(0)

    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes")
        return jsonify({
            "success": False,
            "message": f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
        }), 400

    # Parse request parameters
    try:
        params_json = request.form.get("params", "{}")
        params = TranscribeRequestModel.model_validate_json(params_json)
    except ValidationError as e:
        logger.error(f"Invalid transcription parameters: {e}")
        return jsonify({"success": False, "errors": e.errors()}), 400

    logger.info(f"Processing file for transcription: {video_file.filename}")

    try:
        whisper_service = WhisperService(
            language=params.language,
        )
        result = whisper_service.transcribe(video_file)

        logger.success(f"Transcription completed: {len(result.text)} characters")

        return jsonify(TranscribeResponse(
            success=True,
            text=result.text,
            duration_seconds=result.duration_seconds,
            language=result.language,
        ).model_dump())

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return jsonify({
            "success": False,
            "message": f"Model configuration error: {e}",
        }), 500
    except RuntimeError as e:
        logger.error(f"Transcription runtime error: {e}")
        return jsonify({
            "success": False,
            "message": f"Transcription failed: {e}",
        }), 500
    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        return jsonify({
            "success": False,
            "message": "Error: Internal server error",
        }), 500


# =============================================================================
# Transcription - With Segments
# =============================================================================


@api_blueprint.route("/transcribe/segments", methods=["POST"])
def transcribe_segments():
    """
    Transcribe audio/video file and return text with timestamps.
    Uses WhisperX with Silero VAD and forced alignment.

    Supports granularity parameter:
    - "segment" (default): Returns phrase-level timestamps with confidence
    - "word": Returns word-level timestamps with confidence for each word
    """
    logger.info("Received request for transcription (with segments).")

    if "file" not in request.files:
        logger.warning("No file provided in the request.")
        return jsonify({"success": False, "message": "No file provided"}), 400

    video_file = request.files["file"]
    if video_file.filename == "":
        logger.warning("Empty filename provided.")
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Check file size
    video_file.seek(0, 2)
    file_size = video_file.tell()
    video_file.seek(0)

    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes")
        return jsonify({
            "success": False,
            "message": f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
        }), 400

    # Parse request parameters
    try:
        params_json = request.form.get("params", "{}")
        params = TranscribeSegmentsRequestModel.model_validate_json(params_json)
    except ValidationError as e:
        logger.error(f"Invalid transcription parameters: {e}")
        return jsonify({"success": False, "errors": e.errors()}), 400

    logger.info(
        f"Processing file for transcription: {video_file.filename} "
        f"(granularity={params.granularity})"
    )

    try:
        whisper_service = WhisperService(
            language=params.language,
        )
        result = whisper_service.transcribe(video_file)

        logger.success(
            f"Transcription completed: {len(result.segments)} segments, "
            f"{len(result.text)} characters"
        )

        # Convert internal segments to response format based on granularity
        include_words = params.granularity == "word"

        segments = []
        for seg in result.segments:
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            if include_words and seg.words:
                segment_data["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.confidence,
                    }
                    for w in seg.words
                ]
            segments.append(segment_data)

        return jsonify(TranscribeSegmentsResponse(
            success=True,
            text=result.text,
            segments=segments,
            duration_seconds=result.duration_seconds,
            language=result.language,
        ).model_dump())

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return jsonify({
            "success": False,
            "message": f"Model configuration error: {e}",
        }), 500
    except RuntimeError as e:
        logger.error(f"Transcription runtime error: {e}")
        return jsonify({
            "success": False,
            "message": f"Transcription failed: {e}",
        }), 500
    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        return jsonify({
            "success": False,
            "message": "Error: Internal server error",
        }), 500


# =============================================================================
# Diarization - Full Speaker Analysis
# =============================================================================


@api_blueprint.route("/diarize", methods=["POST"])
def diarize():
    """
    Transcribe audio/video with speaker diarization.
    Returns text segments labeled by speaker with call metrics.

    Supports granularity parameter:
    - "segment" (default): Returns phrase-level timestamps with confidence
    - "word": Returns word-level timestamps with confidence for each word
    """
    logger.info("Received request for diarization.")

    if "file" not in request.files:
        logger.warning("No file provided in the request.")
        return jsonify({"success": False, "message": "No file provided"}), 400

    video_file = request.files["file"]
    if video_file.filename == "":
        logger.warning("Empty filename provided.")
        return jsonify({"success": False, "message": "Empty filename"}), 400

    # Check file size
    video_file.seek(0, 2)
    file_size = video_file.tell()
    video_file.seek(0)

    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        logger.warning(f"File too large: {file_size} bytes")
        return jsonify({
            "success": False,
            "message": f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB",
        }), 400

    # Parse request parameters
    try:
        params_json = request.form.get("params", "{}")
        params = DiarizeRequestModel.model_validate_json(params_json)
    except ValidationError as e:
        logger.error(f"Invalid diarization parameters: {e}")
        return jsonify({"success": False, "errors": e.errors()}), 400

    logger.info(
        f"Processing file for diarization: {video_file.filename} "
        f"(granularity={params.granularity})"
    )

    try:
        alignment_service = AlignmentService(
            language=params.language,
            diarization_mode=params.diarization_mode,
            num_speakers=params.num_speakers,
            min_speakers=params.min_speakers,
            max_speakers=params.max_speakers,
        )

        result = alignment_service.process(video_file)

        logger.success(
            f"Diarization completed: {len(result.speakers)} speakers, "
            f"{len(result.segments)} segments"
        )

        # Build response
        speakers_dict = {
            speaker_id: {
                "label": stats.label,
                "total_speaking_time": stats.total_speaking_time,
                "percentage": stats.percentage,
            }
            for speaker_id, stats in result.speakers.items()
        }

        # Convert segments based on granularity
        include_words = params.granularity == "word"

        segments_list = []
        for seg in result.segments:
            segment_data = {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            if include_words and seg.words:
                segment_data["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.confidence,
                    }
                    for w in seg.words
                ]
            segments_list.append(segment_data)

        metrics_dict = {
            "total_duration": result.metrics.total_duration,
            "total_speech": result.metrics.total_speech,
            "total_silence": result.metrics.total_silence,
            "silence_percentage": result.metrics.silence_percentage,
            "overlap_duration": result.metrics.overlap_duration,
            "overlap_percentage": result.metrics.overlap_percentage,
            "speaker_turns": result.metrics.speaker_turns,
        }

        return jsonify(DiarizeResponse(
            success=True,
            text=result.text,
            speakers=speakers_dict,
            segments=segments_list,
            metrics=metrics_dict,
            duration_seconds=result.duration_seconds,
            language=result.language,
        ).model_dump())

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return jsonify({
            "success": False,
            "message": f"Model configuration error: {e}",
        }), 500
    except RuntimeError as e:
        logger.error(f"Diarization runtime error: {e}")
        return jsonify({
            "success": False,
            "message": f"Diarization failed: {e}",
        }), 500
    except Exception as e:
        logger.exception(f"Unexpected error during diarization: {e}")
        return jsonify({
            "success": False,
            "message": "Error: Internal server error",
        }), 500
