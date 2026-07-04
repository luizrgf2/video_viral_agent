# AGENTS.md — Video Viral Agent

> **Read this first.** This file is the operational contract for any AI agent working in this repository.
> Do not skim. Do not assume. Follow the rules below exactly.

---

## 0. Hard Rules (non-negotiable)

1. **The docs are on-demand.** Do NOT proactively read anything under `docs/`. The only time you open a doc is when the user's request matches one of the **explicit triggers** listed in §3. If no trigger fires, you work from the source code, not from the docs.
2. **Do not edit the docs unless explicitly asked.** Docs describe the system as it currently exists. If you change code, you do not touch docs unless the user asks.
3. **Python 3.12+ only.** Do not introduce syntax or libraries that break 3.12 compatibility.
4. **MoviePy 2.0 API.** Use `subclipped()`, not `subclip()`. Use `write_videofile(logger=None)`. Never import `moviepy.editor`.
5. **FFmpeg is the only video tool** outside MoviePy. Do not add ImageMagick, Pillow text rendering, or any other subtitle path. Subtitles go through `drawtext`.
6. **State is sacred.** Every node returns a partial dict merged into `VideoAnalysisState` (`src/state.py`). Never mutate global state. Never add a field without updating the Pydantic model.
7. **Never commit secrets.** `.env` is gitignored. API keys live there.
8. **Language hard-coded to Portuguese** in transcription nodes (`language="pt"`). Do not "fix" this unless the user asks for multi-language support.
9. **No new dependencies** without explicit user approval. Check `pyproject.toml` first.
10. **Do not write comments** unless the user asks. Match existing code style.

---

## 1. Project Summary

A LangGraph multi-agent pipeline that ingests a long video plus user criteria, transcribes the audio, uses an LLM to identify viral-worthy moments, refines cut boundaries to avoid mid-word truncation, renders clips, and burns in synchronized subtitles. Exposed via a Flask web UI and a Python API.

Stack: Python 3.12, LangGraph, LangChain, faster-whisper / Groq, OpenRouter (LLM), MoviePy 2.0, FFmpeg, librosa, Flask.

---

## 2. Repository Map (code, not docs)

```
src/
├── state.py              # VideoAnalysisState — the shared Pydantic state
├── agents.py             # LLM/VLM client config + system prompts
├── workflow.py           # LangGraph graph definition + run_workflow()
├── main.py
├── nodes/
│   ├── transcribe_audio.py     # Stage 1: audio → transcription + segments
│   ├── identify_moments.py     # Stage 2: LLM picks candidate clips
│   ├── refine_clip_context.py  # Stage 3: LLM expands clip boundaries
│   ├── edit_video.py           # Stage 4: waveform-aware cut via MoviePy
│   ├── add_subtitles.py        # Stage 5: FFmpeg drawtext subtitle burn
│   └── analyze_video.py        # DISABLED — VLM frame analysis (not wired)
└── utils/
    └── waveform_analyzer.py    # librosa RMS pause detection
app.py                   # Flask web server
templates/index.html     # Web UI
tests/, src/__tests__/   # pytest
pyproject.toml           # deps + pytest config
.env / .env.example      # env config
```

---

## 3. Documentation Index & Triggers

**Only read a doc when the user's request matches its trigger keywords.** When you do read one, read the whole file before acting.

### Architecture

| Trigger keywords in the user's request | Doc | What it covers |
|---|---|---|
| "pipeline", "how it works", "architecture overview", "stages", "graph flow", "overview" | [`docs/architecture/overview.md`](docs/architecture/overview.md) | End-to-end pipeline diagram, node responsibilities, failure model, concurrency. |
| "state", "VideoAnalysisState", "ClipInfo", "AnalysisStatus", "fields", "validators" | [`docs/architecture/state.md`](docs/architecture/state.md) | Every state field, the `ClipInfo` sub-model, validators, status lifecycle, known quirks. |
| "workflow", "run_workflow", "create_workflow", "how to run", "execution", "edges" | [`docs/architecture/workflow.md`](docs/architecture/workflow.md) | Graph construction, async execution, web-layer threading, no-checkpoint caveat. |

### Nodes (one doc per stage)

| Trigger keywords | Doc | What it covers |
|---|---|---|
| "transcribe", "whisper", "audio extraction", "compression", "faster-whisper" | [`docs/nodes/transcribe-audio.md`](docs/nodes/transcribe-audio.md) | Mode selection, audio optimization params, local vs Groq internals, segment shape. |
| "identify moments", "LLM matching", "criteria", "viral clips", "structured output" | [`docs/nodes/identify-moments.md`](docs/nodes/identify-moments.md) | Prompt construction, 30-second minimum rule, structured-output parsing, helpers. |
| "refine", "context refinement", "expand clip", "surrounding segments", "avoid cutting" | [`docs/nodes/refine-clip-context.md`](docs/nodes/refine-clip-context.md) | Before/after segment gathering, LLM decision JSON, graceful degradation on parse error. |
| "edit video", "cut", "moviepy", "subclipped", "render clip", "output_clips" | [`docs/nodes/edit-video.md`](docs/nodes/edit-video.md) | Waveform-aware cut, MoviePy 2.0 API notes, filename pattern, per-clip error handling. |
| "subtitles", "captions", "drawtext", "burn subtitles", "legendas" | [`docs/nodes/add-subtitles.md`](docs/nodes/add-subtitles.md) | Per-clip algorithm, drawtext filter spec, text wrapping/escaping, FFmpeg command, fallback behavior. |
| "analyze video", "VLM", "frame analysis", "video description" | [`docs/nodes/analyze-video.md`](docs/nodes/analyze-video.md) | Why this node is disabled and how to re-enable it. |

### Integrations

| Trigger keywords | Doc | What it covers |
|---|---|---|
| "openrouter", "LLM model", "VLM model", "agents.py", "system prompt", "temperature" | [`docs/integrations/openrouter.md`](docs/integrations/openrouter.md) | `vlmModel`/`llmModel` config, env vars, headers, unused-prompt gotcha, import-time failure. |
| "groq", "groq api", "whisper api", "cloud transcription", "25MB" | [`docs/integrations/groq.md`](docs/integrations/groq.md) | API call shape, hard-coded Portuguese, 25 MB limit mitigation, response shape. |
| "ffmpeg", "waveform", "librosa", "rms", "pause detection", "natural cut", "ffprobe" | [`docs/integrations/ffmpeg-waveform.md`](docs/integrations/ffmpeg-waveform.md) | Two FFmpeg roles, analyzer params, safety guarantees, subtitle engine binary checks. |

### Web

| Trigger keywords | Doc | What it covers |
|---|---|---|
| "flask", "web", "upload", "endpoint", "api", "status", "session", "routes" | [`docs/web/flask-api.md`](docs/web/flask-api.md) | Endpoints, file limits, extension gotcha, thread-per-request model, polling limits. |

### Setup

| Trigger keywords | Doc | What it covers |
|---|---|---|
| "env", "environment", "config", "variables", "keys", "model name", "AUDIO_TRANSCRIPTION_MODE" | [`docs/setup/environment.md`](docs/setup/environment.md) | Every env var, required vs optional, model swap cheatsheet, system deps. |
| "install", "setup", "run", "how to start", "start server", "tests" | [`docs/setup/install-run.md`](docs/setup/install-run.md) | Prerequisites, install commands, run commands, output locations, test invocation. |

---

## 4. Known Gotchas (always keep in mind)

- **`analyze_video` is dead code.** It exists but is not in the graph. Do not assume frame analysis runs.
- **`identify_moments` ignores `MOMENTS_IDENTIFICATION_SYSTEM_PROMPT`.** It uses an inline prompt. Editing the constant does nothing.
- **`edit_video` sets `COMPLETED` before subtitles.** `add_subtitles` then also sets `COMPLETED`. The `ADDING_SUBTITLES` status enum value is never assigned.
- **`outputClips` is overwritten** by `add_subtitles` to point at the subtitled versions. The original unsubtitted paths are lost unless you copy them first.
- **Upload accepts mov/avi/mkv but state rejects non-MP4.** Web uploads of non-MP4 files will silently fail at workflow start.
- **No retry, no checkpointing, no progress reporting.** A crash loses the run. The web UI cannot tell error from in-progress.
- **Language is hard-coded to Portuguese** in both transcription paths.

---

## 5. Verification Before Claiming Done

Before telling the user a change is complete:

1. Run `uv run pytest` (or `pytest` if not using uv). Tests must pass.
2. If you touched `app.py` or `templates/`, start the Flask server and confirm it boots: `uv run python app.py`.
3. If you touched a node, confirm imports are clean: `python -c "from src.workflow import create_workflow"`.
4. Never claim success without running a verification command.

---

## 6. Commits

- Do **not** commit unless the user explicitly asks.
- When asked, follow conventional-commits style, keep the subject ≤ 50 chars, and never include secrets or `.env`.
