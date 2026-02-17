# FFmpeg Microservice v3.0

<p align="center">
  <img src="WhisperX-mikroserwis.png" alt="FFmpeg Microservice — Speak. Transcribe. Connect." width="100%">
</p>

<p align="center">
  Made with ❤️ by <strong>Comtegra S.A.</strong>
</p>

---

A high-performance microservice for audio/video transcription and speaker diarization, built with WhisperX and PyAnnote for speaker identification.

## Features

- **Audio Extraction** (`/extract`) - Extract audio from video files as MP3
- **Transcription** (`/transcribe`) - Plain text transcription using WhisperX
- **Timestamped Transcription** (`/transcribe/segments`) - Transcription with precise timestamps
- **Speaker Diarization** (`/diarize`) - Full speaker identification with call metrics

## Key Technologies

- **WhisperX** with faster-whisper backend for fast, accurate transcription
- **Whisper large-v3** model (configurable)
- **Forced alignment** for accurate word-level timestamps
- **Built-in Silero VAD** for voice activity detection (prevents hallucinations)
- **PyAnnote 3.1** for state-of-the-art speaker diarization
- **CUDA GPU acceleration**

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA 12.2+ (tested on A5000)
- HuggingFace account with accepted PyAnnote model license

## Quick Start

### 1. Get HuggingFace Token

1. Create account at https://huggingface.co
2. Accept the PyAnnote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate access token at https://huggingface.co/settings/tokens

### 2. Build Docker Image

```bash
docker build \
  --build-arg HF_TOKEN=your_huggingface_token \
  -t ffmpeg-microservice:3.0 .
```

Note: First build downloads WhisperX and PyAnnote models (~5GB).

### 3. Run Container

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e APP_TOKEN=your_api_token \
  -e HF_TOKEN=your_huggingface_token \
  ffmpeg-microservice:3.0
```

## API Endpoints

### Health Check

```
GET /health
```

Returns service status including GPU and model availability.

```json
{
  "success": true,
  "message": "Service is healthy",
  "gpu_available": true,
  "whisper_model_loaded": true,
  "pyannote_model_loaded": true
}
```

### Extract Audio

```
POST /extract
```

Extract audio from video file as MP3 stream.

```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer your-token" \
  -F "file=@video.mp4" \
  --output audio.mp3
```

### Transcribe (Plain Text)

```
POST /transcribe
```

Transcribe audio/video and return plain text.

**Request:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your-token" \
  -F "file=@recording.mp4" \
  -F 'params={"language":"auto"}'
```

**Response:**
```json
{
  "success": true,
  "text": "Dzień dobry, w czym mogę pomóc?...",
  "duration_seconds": 342.5,
  "language": "pl"
}
```

**Transcription Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| language | string | Language code (e.g., "en", "pl", "de") or "auto" for detection | "auto" |

### Transcribe with Timestamps

```
POST /transcribe/segments
```

Transcribe with precise timestamp segments for UI sync.

**Request:**
```bash
curl -X POST http://localhost:8000/transcribe/segments \
  -H "Authorization: Bearer your-token" \
  -F "file=@recording.mp4" \
  -F 'params={"language":"auto", "granularity":"word"}'
```

**Response:**
```json
{
  "success": true,
  "text": "Full transcription text...",
  "segments": [
    {
      "start": 0.0,
      "end": 2.34,
      "text": "Dzień dobry, w czym mogę pomóc?",
      "confidence": 0.95,
      "words": [
        {"word": "Dzień", "start": 0.0, "end": 0.3, "confidence": 0.98},
        {"word": "dobry", "start": 0.32, "end": 0.6, "confidence": 0.97}
      ]
    }
  ],
  "duration_seconds": 342.5,
  "language": "pl"
}
```

**Segments Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| language | string | Language code (e.g., "en", "pl", "de") or "auto" for detection | "auto" |
| granularity | string | "segment" for phrase-level, "word" for word-level timestamps | "segment" |

### Diarization (Speaker Identification)

```
POST /diarize
```

Full transcription with speaker identification and call metrics.

**Request:**
```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer your-token" \
  -F "file=@call_recording.mp4" \
  -F 'params={"language":"auto","num_speakers":2}'
```

**Response:**
```json
{
  "success": true,
  "text": "Full transcription text...",
  "speakers": {
    "SPEAKER_00": {
      "label": "SPEAKER_00",
      "total_speaking_time": 180.5,
      "percentage": 52.7
    },
    "SPEAKER_01": {
      "label": "SPEAKER_01",
      "total_speaking_time": 145.2,
      "percentage": 42.4
    }
  },
  "segments": [
    {
      "start": 0.0,
      "end": 2.34,
      "speaker": "SPEAKER_00",
      "text": "Dzień dobry, w czym mogę pomóc?"
    },
    {
      "start": 2.89,
      "end": 5.12,
      "speaker": "SPEAKER_01",
      "text": "Chciałbym złożyć reklamację."
    }
  ],
  "metrics": {
    "total_duration": 342.5,
    "total_speech": 325.7,
    "total_silence": 16.8,
    "silence_percentage": 4.9,
    "overlap_duration": 3.2,
    "overlap_percentage": 0.9,
    "speaker_turns": 47
  },
  "duration_seconds": 342.5,
  "language": "pl"
}
```

**Diarization Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| language | string | Language code (e.g., "en", "pl", "de") or "auto" for detection | "auto" |
| diarization_mode | string | "pyannote" for AI-based speaker detection, "channel" for stereo channel separation | "pyannote" |
| granularity | string | "segment" for phrase-level, "word" for word-level timestamps | "segment" |
| num_speakers | int | Expected number of speakers (only with diarization_mode="pyannote") | auto-detect |
| min_speakers | int | Minimum speakers (only with diarization_mode="pyannote") | none |
| max_speakers | int | Maximum speakers (only with diarization_mode="pyannote") | none |

**Diarization Modes:**
- **pyannote** (default): Uses PyAnnote 3.1 AI model to detect and separate speakers. Works with any audio format. Best for conversations where speakers may overlap or switch frequently.
- **channel**: Uses stereo channel separation where left channel = SPEAKER_00 and right channel = SPEAKER_01. Ideal for recordings where each speaker is on a separate audio channel (e.g., call center recordings with agent/customer on separate channels).

## Configuration

### Environment Variables

#### Authentication & Server

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_TOKEN` | API authentication token (required when auth enabled) | - |
| `ENABLE_AUTH` | Enable bearer token auth | true |
| `HF_TOKEN` | HuggingFace token (for PyAnnote and model downloads) | required |
| `HOST` | Server bind address | "0.0.0.0" |
| `PORT` | Server port | 8000 |
| `API_PREFIX` | API route prefix | "/" |

#### GPU Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_GPU` | Enable GPU acceleration | true |
| `GPU_DEVICE` | CUDA device index | 0 |

#### WhisperX Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_MODEL` | Model name (tiny, base, small, medium, large-v1, large-v2, large-v3, distil-large-v2) | "large-v3" |
| `WHISPER_LANGUAGE` | Default transcription language | "auto" |
| `WHISPER_COMPUTE_TYPE` | Compute type (float16, int8) | "float16" |
| `WHISPER_BATCH_SIZE` | Batch size for transcription | 16 |
| `WHISPER_BEAM_SIZE` | Beam size for decoding | 5 |
| `WHISPER_TEMPERATURE` | Sampling temperature (0.0 = deterministic) | 0.0 |
| `WHISPER_INITIAL_PROMPT` | Initial prompt for vocabulary/context guidance | "" |
| `WHISPER_ENABLE_ALIGNMENT` | Enable forced alignment for word timestamps | true |

#### PyAnnote Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PYANNOTE_MODEL` | PyAnnote model name | "pyannote/speaker-diarization-3.1" |

#### Processing Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_FILE_SIZE_MB` | Maximum upload file size in MB | 2048 |
| `TEMP_DIR` | Temporary file directory | "/tmp" |

## Performance

### Benchmarks (A5000 GPU)

| File Duration | Transcription | Diarization | Full Pipeline |
|---------------|---------------|-------------|---------------|
| 30 seconds | ~3s | ~2s | ~4s |
| 5 minutes | ~15s | ~10s | ~20s |
| 30 minutes | ~90s | ~60s | ~120s |
| 2 hours | ~6min | ~4min | ~8min |

## License

GPL-3.0 - see [LICENSE](LICENSE) file

---

## About Comtegra S.A.

Comtegra is an IT systems integrator based in Poland, specializing in various aspects of information technology, including data storage and management, information security, and network construction. Founded in 1999, Comtegra has established itself as a significant player in the Polish IT market, providing services such as backup solutions, cybersecurity, and virtualization technologies. The company emphasizes the integration of artificial intelligence within business operations to enhance data management and decision-making processes.
