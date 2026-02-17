# FFmpeg Microservice User Guide

This guide explains how to use the FFmpeg Microservice for audio/video transcription and speaker diarization.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Authentication](#authentication)
5. [API Endpoints](#api-endpoints)
   - [Health Check](#health-check)
   - [Extract Audio](#extract-audio)
   - [Transcribe (Plain Text)](#transcribe-plain-text)
   - [Transcribe with Timestamps](#transcribe-with-timestamps)
   - [Speaker Diarization](#speaker-diarization)
6. [Common Use Cases](#common-use-cases)
7. [Language Support](#language-support)
8. [Error Handling](#error-handling)
9. [Performance Tips](#performance-tips)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The FFmpeg Microservice provides AI-powered audio and video processing capabilities:

| Feature | Description |
|---------|-------------|
| **Audio Extraction** | Extract audio from video files as MP3 |
| **Transcription** | Convert speech to text using WhisperX |
| **Timestamps** | Get word-level or segment-level timing |
| **Speaker Diarization** | Identify who said what and when |
| **Call Metrics** | Speaking time, silence, overlap statistics |

The service uses:
- **WhisperX** with faster-whisper backend for fast, accurate transcription
- **PyAnnote 3.1** for state-of-the-art speaker identification
- **NVIDIA GPU acceleration** for high performance

---

## Prerequisites

Before using the service, ensure you have:

1. **Service URL** - The endpoint where the microservice is running (e.g., `http://localhost:8000`)
2. **API Token** - Your authentication token (provided by the administrator)
3. **Audio/Video Files** - Supported formats include MP4, MP3, WAV, M4A, FLAC, OGG, WebM, AVI, MKV, MOV

---

## Getting Started

### Quick Test

Verify the service is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "success": true,
  "message": "Service is healthy",
  "gpu_available": true,
  "whisper_model_loaded": true,
  "pyannote_model_loaded": true
}
```

### Your First Transcription

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@recording.mp4"
```

---

## Authentication

All endpoints except `/health` require Bearer token authentication.

### HTTP Header Format

```
Authorization: Bearer YOUR_API_TOKEN
```

### Examples

**cURL:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer my-secret-token" \
  -F "file=@audio.mp3"
```

**Python (requests):**
```python
import requests

headers = {"Authorization": "Bearer my-secret-token"}
files = {"file": open("audio.mp3", "rb")}

response = requests.post(
    "http://localhost:8000/transcribe",
    headers=headers,
    files=files
)
print(response.json())
```

**JavaScript (fetch):**
```javascript
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer my-secret-token'
  },
  body: formData
});

const result = await response.json();
```

### Authentication Errors

| Status Code | Meaning |
|-------------|---------|
| 401 | Missing or invalid token |

---

## API Endpoints

### Health Check

Check if the service is running and models are loaded.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "success": true,
  "message": "Service is healthy",
  "gpu_available": true,
  "whisper_model_loaded": true,
  "pyannote_model_loaded": true
}
```

---

### Extract Audio

Extract audio from video files and download as MP3.

**Endpoint:** `POST /extract`

**Authentication:** Required

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | Video or audio file to extract from |

**Example:**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@video.mp4" \
  --output extracted_audio.mp3
```

**Response:** Binary MP3 audio stream

---

### Transcribe (Plain Text)

Get a plain text transcription of audio/video content.

**Endpoint:** `POST /transcribe`

**Authentication:** Required

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | file | Yes | - | Audio or video file |
| params | JSON | No | - | Optional parameters (see below) |

**Optional params JSON:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| language | string | "auto" | Language code (e.g., "en", "pl", "de") or "auto" for detection |

**Example - Auto-detect language:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@recording.mp4"
```

**Example - Specify language:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@recording.mp4" \
  -F 'params={"language":"en"}'
```

**Response:**
```json
{
  "success": true,
  "text": "Hello, how can I help you today? I would like to place an order...",
  "duration_seconds": 342.5,
  "language": "en"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether the operation succeeded |
| text | string | Full transcription text |
| duration_seconds | number | Audio duration in seconds |
| language | string | Detected or specified language code |

---

### Transcribe with Timestamps

Get transcription with precise timestamps for each segment or word.

**Endpoint:** `POST /transcribe/segments`

**Authentication:** Required

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | file | Yes | - | Audio or video file |
| params | JSON | No | - | Optional parameters (see below) |

**Optional params JSON:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| language | string | "auto" | Language code or "auto" |
| granularity | string | "segment" | "segment" for phrases, "word" for individual words |

**Example - Segment-level timestamps:**
```bash
curl -X POST http://localhost:8000/transcribe/segments \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@recording.mp4" \
  -F 'params={"language":"auto","granularity":"segment"}'
```

**Example - Word-level timestamps:**
```bash
curl -X POST http://localhost:8000/transcribe/segments \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@recording.mp4" \
  -F 'params={"granularity":"word"}'
```

**Response (segment granularity):**
```json
{
  "success": true,
  "text": "Hello, how can I help you today?",
  "segments": [
    {
      "start": 0.0,
      "end": 2.34,
      "text": "Hello, how can I help you today?",
      "confidence": 0.95
    },
    {
      "start": 3.12,
      "end": 5.67,
      "text": "I would like to place an order.",
      "confidence": 0.92
    }
  ],
  "duration_seconds": 342.5,
  "language": "en"
}
```

**Response (word granularity):**
```json
{
  "success": true,
  "text": "Hello, how can I help you today?",
  "segments": [
    {
      "start": 0.0,
      "end": 2.34,
      "text": "Hello, how can I help you today?",
      "confidence": 0.95,
      "words": [
        {"word": "Hello,", "start": 0.0, "end": 0.45, "confidence": 0.98},
        {"word": "how", "start": 0.52, "end": 0.71, "confidence": 0.97},
        {"word": "can", "start": 0.75, "end": 0.92, "confidence": 0.96},
        {"word": "I", "start": 0.95, "end": 1.05, "confidence": 0.99},
        {"word": "help", "start": 1.10, "end": 1.35, "confidence": 0.95},
        {"word": "you", "start": 1.40, "end": 1.58, "confidence": 0.97},
        {"word": "today?", "start": 1.65, "end": 2.34, "confidence": 0.94}
      ]
    }
  ],
  "duration_seconds": 342.5,
  "language": "en"
}
```

**Use Cases:**

| Granularity | Best For |
|-------------|----------|
| segment | Subtitles, meeting notes, general transcription |
| word | Karaoke-style highlighting, precise audio editing, accessibility |

---

### Speaker Diarization

Get transcription with speaker identification and call analytics.

**Endpoint:** `POST /diarize`

**Authentication:** Required

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| file | file | Yes | - | Audio or video file |
| params | JSON | No | - | Optional parameters (see below) |

**Optional params JSON:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| language | string | "auto" | Language code or "auto" |
| diarization_mode | string | "pyannote" | "pyannote" (AI-based) or "channel" (stereo separation) |
| granularity | string | "segment" | "segment" or "word" |
| num_speakers | int | auto | Exact number of speakers (if known) |
| min_speakers | int | none | Minimum expected speakers |
| max_speakers | int | none | Maximum expected speakers |

**Example - Basic diarization:**
```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@call_recording.mp4"
```

**Example - Two-speaker call with word timestamps:**
```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@call_recording.mp4" \
  -F 'params={"language":"en","num_speakers":2,"granularity":"word"}'
```

**Example - Stereo channel separation:**
```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@stereo_call.wav" \
  -F 'params={"diarization_mode":"channel"}'
```

**Response:**
```json
{
  "success": true,
  "text": "Hello, how can I help you today? I would like to place an order...",
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
      "text": "Hello, how can I help you today?"
    },
    {
      "start": 2.89,
      "end": 5.12,
      "speaker": "SPEAKER_01",
      "text": "I would like to place an order."
    },
    {
      "start": 5.45,
      "end": 8.90,
      "speaker": "SPEAKER_00",
      "text": "Sure, I can help you with that. What would you like to order?"
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
  "language": "en"
}
```

**Response Fields:**

| Field | Description |
|-------|-------------|
| speakers | Dictionary of speaker statistics |
| segments | List of transcribed segments with speaker labels |
| metrics | Call analytics (duration, silence, overlap, turns) |

**Metrics Explained:**

| Metric | Description |
|--------|-------------|
| total_duration | Total audio length in seconds |
| total_speech | Time with detected speech |
| total_silence | Time with no speech |
| silence_percentage | Silence as percentage of total duration |
| overlap_duration | Time when multiple speakers talk simultaneously |
| overlap_percentage | Overlap as percentage of total duration |
| speaker_turns | Number of times the speaker changed |

**Diarization Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| pyannote | AI-based speaker detection | General recordings, meetings, interviews |
| channel | Stereo channel separation (left=SPEAKER_00, right=SPEAKER_01) | Call center recordings with separate channels |

---

## Common Use Cases

### 1. Call Center Quality Assurance

Analyze customer service calls with speaker identification:

```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@customer_call.mp3" \
  -F 'params={"num_speakers":2,"granularity":"word"}'
```

**What you get:**
- Full transcription with timestamps
- Agent vs customer identification
- Speaking time ratios
- Silence and overlap metrics
- Turn count for conversation flow analysis

### 2. Meeting Transcription

Transcribe a team meeting with multiple speakers:

```bash
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@team_meeting.mp4" \
  -F 'params={"min_speakers":3,"max_speakers":8}'
```

### 3. Subtitle Generation

Generate subtitles with segment-level timestamps:

```bash
curl -X POST http://localhost:8000/transcribe/segments \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@video.mp4" \
  -F 'params={"granularity":"segment"}'
```

The response segments can be converted to SRT format:

```python
def to_srt(segments):
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        srt_lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(srt_lines)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

### 4. Podcast Processing

Extract and transcribe audio from video podcasts:

```bash
# Step 1: Extract audio
curl -X POST http://localhost:8000/extract \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@podcast.mp4" \
  --output podcast.mp3

# Step 2: Transcribe with speaker identification
curl -X POST http://localhost:8000/diarize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@podcast.mp4" \
  -F 'params={"min_speakers":2,"max_speakers":4}'
```

### 5. Voice Memos / Quick Transcription

Simple text extraction from audio:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@voice_memo.m4a"
```

---

## Language Support

The service supports 99+ languages through WhisperX. Common language codes:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | German | de |
| Polish | pl | French | fr |
| Spanish | es | Italian | it |
| Portuguese | pt | Dutch | nl |
| Russian | ru | Chinese | zh |
| Japanese | ja | Korean | ko |
| Arabic | ar | Hindi | hi |

### Auto-Detection

By default, the service automatically detects the language:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@recording.mp4"
  # language will be auto-detected
```

### Explicit Language

For better accuracy, specify the language if known:

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@german_audio.mp3" \
  -F 'params={"language":"de"}'
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

| HTTP Status | Error | Solution |
|-------------|-------|----------|
| 400 | No file provided | Include a file in your request |
| 400 | Invalid file format | Use a supported audio/video format |
| 400 | File too large | File exceeds maximum size (default: 2GB) |
| 401 | Unauthorized | Check your API token |
| 413 | Payload too large | Reduce file size or contact admin |
| 500 | Processing error | Check server logs, try again |
| 503 | Service unavailable | Models still loading, wait and retry |

### Retry Strategy

For production use, implement exponential backoff:

```python
import time
import requests

def transcribe_with_retry(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/transcribe",
                headers={"Authorization": "Bearer YOUR_TOKEN"},
                files={"file": open(file_path, "rb")},
                timeout=600  # 10 minute timeout for long files
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Service unavailable, wait and retry
                time.sleep(2 ** attempt)
                continue
            else:
                response.raise_for_status()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise Exception("Max retries exceeded")
```

---

## Performance Tips

### 1. Choose the Right Endpoint

| Need | Endpoint | Speed |
|------|----------|-------|
| Just text | `/transcribe` | Fastest |
| Text + timing | `/transcribe/segments` | Fast |
| Speaker ID | `/diarize` | Slower (worth it for multi-speaker) |

### 2. Optimize Audio Before Sending

- **Compress large files** - MP3 at 128kbps is sufficient for speech
- **Trim silence** - Remove long silences at start/end
- **Use mono audio** - Stereo is converted to mono anyway (unless using channel mode)

### 3. Use Appropriate Granularity

| Granularity | Processing Time | Use When |
|-------------|-----------------|----------|
| segment | Faster | Subtitles, general transcription |
| word | Slower | Precise timing needed, karaoke-style |

### 4. Specify Language When Known

Auto-detection adds a small overhead. If you know the language:

```bash
-F 'params={"language":"en"}'
```

### 5. Set Speaker Hints

If you know the number of speakers, tell the service:

```bash
-F 'params={"num_speakers":2}'
# or
-F 'params={"min_speakers":2,"max_speakers":4}'
```

### Expected Processing Times (A5000 GPU)

| Audio Duration | Transcribe | Diarize |
|----------------|------------|---------|
| 30 seconds | ~3s | ~4s |
| 5 minutes | ~15s | ~20s |
| 30 minutes | ~90s | ~120s |
| 2 hours | ~6min | ~8min |

---

## Troubleshooting

### "GPU not available"

The health endpoint shows `"gpu_available": false`:

1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check Docker has GPU access: `docker run --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`
3. Restart the container with `--gpus all`

### Transcription is inaccurate

1. **Check audio quality** - Clear speech produces better results
2. **Specify language** - Don't rely on auto-detection for known languages
3. **Check for background noise** - The service filters noise, but excessive noise affects accuracy

### Speakers not identified correctly

1. **Use `num_speakers`** - If you know the exact count
2. **Try channel mode** - For stereo recordings with separate channels
3. **Check audio levels** - Speakers should have similar volume levels

### Request timeout

For long files (>30 minutes):

1. Increase client timeout to 30+ minutes
2. Consider splitting large files
3. Check server health endpoint first

### "Model not loaded"

The health endpoint shows models not loaded:

1. Wait 60-120 seconds after container start
2. Check container logs: `docker logs <container_id>`
3. Verify HF_TOKEN is set correctly

---

## Python Client Example

Complete example for a production Python client:

```python
import requests
from pathlib import Path
from typing import Optional, Dict, Any


class TranscriptionClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def transcribe(
        self,
        file_path: str,
        language: str = "auto"
    ) -> Dict[str, Any]:
        """Get plain text transcription."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"params": f'{{"language":"{language}"}}'}
            response = requests.post(
                f"{self.base_url}/transcribe",
                headers=self.headers,
                files=files,
                data=data,
                timeout=600
            )
        response.raise_for_status()
        return response.json()

    def transcribe_segments(
        self,
        file_path: str,
        language: str = "auto",
        granularity: str = "segment"
    ) -> Dict[str, Any]:
        """Get transcription with timestamps."""
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"params": f'{{"language":"{language}","granularity":"{granularity}"}}'}
            response = requests.post(
                f"{self.base_url}/transcribe/segments",
                headers=self.headers,
                files=files,
                data=data,
                timeout=600
            )
        response.raise_for_status()
        return response.json()

    def diarize(
        self,
        file_path: str,
        language: str = "auto",
        diarization_mode: str = "pyannote",
        granularity: str = "segment",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get transcription with speaker identification."""
        params = {
            "language": language,
            "diarization_mode": diarization_mode,
            "granularity": granularity
        }
        if num_speakers:
            params["num_speakers"] = num_speakers
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers

        import json
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"params": json.dumps(params)}
            response = requests.post(
                f"{self.base_url}/diarize",
                headers=self.headers,
                files=files,
                data=data,
                timeout=1800  # 30 min for long recordings
            )
        response.raise_for_status()
        return response.json()


# Usage
if __name__ == "__main__":
    client = TranscriptionClient(
        base_url="http://localhost:8000",
        api_token="your-api-token"
    )

    # Check health
    health = client.health_check()
    print(f"Service healthy: {health['success']}")

    # Transcribe a file
    result = client.transcribe("recording.mp4", language="en")
    print(f"Transcription: {result['text'][:200]}...")

    # Diarize a call
    result = client.diarize(
        "call.mp3",
        num_speakers=2,
        granularity="word"
    )
    for speaker, stats in result['speakers'].items():
        print(f"{speaker}: {stats['percentage']:.1f}% speaking time")
```

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review server logs for detailed error messages
3. Contact your system administrator

---

*FFmpeg Microservice v3.0 - Powered by WhisperX and PyAnnote*
