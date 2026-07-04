# Environment Configuration

**File:** `.env` (copy `.env.example`)

## Required Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | ✅ always | LLM/VLM access via OpenRouter. **Import fails without it.** |

## Conditional Variables

| Variable | Required when | Default | Purpose |
|----------|---------------|---------|---------|
| `GROQ_API_KEY` | `AUDIO_TRANSCRIPTION_MODE=groq` | — | Groq API auth |
| `GROQ_WHISPER_MODEL` | groq mode | `whisper-large-v3-turbo` | Whisper model name |

## Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUDIO_TRANSCRIPTION_MODE` | `local` | `local` = faster-whisper, `groq` = Groq API |
| `VLM_MODEL_NAME` | `anthropic/claude-3.5-sonnet` | Vision model (only used by disabled `analyze_video` node) |
| `LLM_MODEL_NAME` | `openai/gpt-4o` | Text model for moment ID + context refinement |
| `SITE_URL` | `http://localhost:8000` | Sent to OpenRouter as `HTTP-Referer` |
| `APP_NAME` | `Video Viral Agent` | Sent to OpenRouter as `X-Title` |
| `LOG_LEVEL` | `INFO` | Python logging level (not yet wired into `logging.basicConfig` in `app.py`) |

## System Dependencies (not in `.env`)

- **FFmpeg + FFprobe** on `$PATH` — required for waveform analysis and subtitle burning.
- **Python 3.12+**
- For local mode: CPU with AVX2 (faster-whisper CTranslate2 `int8` benefits from it).

## Model Swap Cheatsheet

OpenRouter exposes many providers — change one env var:

| Provider | Example model string |
|----------|----------------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-3.5-sonnet` |
| Google | `google/gemini-2.0-flash-lite-001` |
| Meta | `meta-llama/llama-3.1-70b-instruct` |

For Whisper local mode, edit `model_size` in `transcribe_audio.py` (`tiny` / `base` / `small` / `medium` / `large`).
