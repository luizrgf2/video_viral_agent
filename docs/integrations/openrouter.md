# Integration: OpenRouter (LLM / VLM)

**Source:** `src/agents.py`

## What It Provides

Two `ChatOpenAI` instances configured against OpenRouter's OpenAI-compatible endpoint, used by `identify_moments` and `refine_clip_context` (and the disabled `analyze_video` node):

| Instance | Model env var | Default | Used by |
|----------|---------------|---------|---------|
| `vlmModel` | `VLM_MODEL_NAME` | `anthropic/claude-3.5-sonnet` | `analyze_video` (disabled) |
| `llmModel` | `LLM_MODEL_NAME` | `openai/gpt-4o` | `identify_moments`, `refine_clip_context` |

Both run at `temperature=0`.

## Configuration

```env
OPENROUTER_API_KEY=<required>     # raises ValueError at import time if missing
VLM_MODEL_NAME=anthropic/claude-3.5-sonnet
LLM_MODEL_NAME=openai/gpt-4o
SITE_URL=http://localhost:8000
APP_NAME=Video Viral Agent
```

The base URL is hard-coded: `https://openrouter.ai/api/v1`.

## Headers

OpenRouter asks clients to identify themselves:

```python
{
  "HTTP-Referer": SITE_URL,
  "X-Title": APP_NAME,
}
```

These are attached as `default_headers` on both models.

## System Prompts

`src/agents.py` defines three system prompt constants:

| Constant | Purpose | Currently used? |
|----------|---------|-----------------|
| `TRANSCRIBE_SYSTEM_PROMPT` | Guidance for transcription-style tasks | ❌ (transcription uses Whisper/Groq directly) |
| `VIDEO_ANALYSIS_SYSTEM_PROMPT` | VLM frame-by-frame description format | ✅ by `analyze_video` (disabled node) |
| `MOMENTS_IDENTIFICATION_SYSTEM_PROMPT` | Viral-scoring rubric | ❌ (`identify_moments` builds its own inline prompt instead) |

> ⚠️ The prompts are defined but `identify_moments` does **not** import `MOMENTS_IDENTIFICATION_SYSTEM_PROMPT` — it uses an inline system message. Editing the constant has no effect on current behavior.

## Failure Modes

- Missing `OPENROUTER_API_KEY` → `ValueError` at **import time** (kills the process before any request).
- Model not available / rate limited → LangChain raises inside `ainvoke`; caught by the node, surfaced as `status: FAILED`.
