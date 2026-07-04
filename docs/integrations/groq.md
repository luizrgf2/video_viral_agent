# Integration: Groq (Audio Transcription)

**Source:** `src/nodes/transcribe_audio.py` → `transcribe_with_groq`

## When Active

Only when `AUDIO_TRANSCRIPTION_MODE=groq`. Otherwise the local `faster-whisper` path is used.

## Configuration

```env
GROQ_API_KEY=<required in groq mode>
GROQ_WHISPER_MODEL=whisper-large-v3-turbo   # or whisper-large-v3
```

## API Call

```python
client = Groq(api_key=...)
client.audio.transcriptions.create(
    file=<optimized_audio_file>,
    model=model,
    response_format="verbose_json",
    timestamp_granularities=["segment"],
    language="pt",
    temperature=0.0,
)
```

- **Language is hard-coded to Portuguese.** For other languages, edit the `language` argument or remove it for auto-detect.
- `verbose_json` + `segment` granularity is required — the pipeline depends on segment-level timestamps.

## Size Limit

Groq enforces a **25 MB** upload cap. The pipeline mitigates this by pre-compressing audio to 16 kHz / mono / 32 kbps MP3 (see `transcribe-audio.md`). If the optimized file still exceeds 25 MB, a warning is logged but the upload is attempted anyway and may fail.

## Response Shape

```python
{
  "transcription": "...full text...",
  "transcriptionSegments": [{"start": float, "end": float, "text": str}, ...],
  "mode": "groq",
  "model": "whisper-large-v3-turbo",
  "duration": float,
}
```

If the API returns no segments (unexpected), a fallback single-segment spanning the full duration is synthesized.
