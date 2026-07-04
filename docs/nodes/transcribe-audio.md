# Node: `transcribe_audio`

**Source:** `src/nodes/transcribe_audio.py`
**Entry point** of the workflow.

## Purpose

Extract audio from the source video, optimize it for speech recognition, and produce a full transcription with timestamped segments. Segments are consumed by every downstream node (moment identification, context refinement, subtitle overlay).

## Mode Selection

Controlled by env var `AUDIO_TRANSCRIPTION_MODE`:

| Value | Engine | When to use |
|-------|--------|-------------|
| `local` (default) | `faster-whisper` + CTranslate2 | Free, offline, large files OK |
| `groq` | Groq API (`whisper-large-v3-turbo`) | Fastest, most accurate, 25 MB upload cap |

## Audio Optimization (both modes)

`extract_and_compress_audio()` runs before any transcription:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Format | MP3 (`libmp3lame`) | Efficient compression |
| Sample rate | 16 kHz | Whisper's native rate |
| Channels | Mono | Speech needs no stereo |
| Bitrate | 32 kbps | Sufficient for intelligibility |

This typically shrinks a 17 MB video's audio to ~1.8 MB (90%+ reduction), which is critical for staying under Groq's 25 MB limit.

## Local Mode (`transcribe_with_faster_whisper`)

```python
model_size  = "tiny"     # options: tiny, base, small, medium, large
device      = "cpu"      # options: cpu, cuda
compute_type = "int8"    # options: int8, float16, float32
language     = "pt"      # hardcoded Portuguese
vad_filter   = True      # Voice Activity Detection
word_timestamps = True
```

Returns: `{transcription, transcriptionSegments, mode, language, language_probability, duration}`

## Groq Mode (`transcribe_with_groq`)

- Requires `GROQ_API_KEY`.
- Model from `GROQ_WHISPER_MODEL` (default `whisper-large-v3-turbo`).
- `response_format="verbose_json"`, `timestamp_granularities=["segment"]`.
- `temperature=0.0` for deterministic output.
- Language hardcoded to Portuguese.
- Warns (does not fail) if optimized audio still exceeds 25 MB.

## Segment Shape

Each segment is a plain dict:

```python
{"start": 12.34, "end": 14.56, "text": "...spoken text..."}
```

Times are floats in seconds relative to the original video. This shape is the contract every downstream node relies on.

## Cleanup

A `finally` block always deletes the temporary compressed audio file via `cleanup_temp_audio()`.

## Failure Modes

- Video file missing → returns `{error, status: FAILED}`.
- No audio track → raises `ValueError`, caught and propagated as `FAILED`.
- Missing `GROQ_API_KEY` in groq mode → raises `ValueError`.
