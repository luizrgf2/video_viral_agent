# Setup & Run

## Prerequisites

- Python **3.12+**
- **FFmpeg** (includes `ffprobe`) installed and on `$PATH`
- An **OpenRouter API key** (required — the app refuses to import without it)
- Optional: **Groq API key** if using cloud transcription

## Install

```bash
# Using uv (recommended)
pip install uv
uv sync

# Or plain pip
pip install -e .
```

## Configure

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY at minimum
```

See `setup/environment.md` for the full variable reference.

## Run the Web App

```bash
uv run python app.py
# → http://localhost:5000
```

## Run Programmatically

```python
import asyncio
from src.workflow import run_workflow
from src.state import VideoAnalysisState, AnalysisStatus

state = VideoAnalysisState(
    videoPath="/abs/path/video.mp4",
    analysis=["Extract moments about X", "Funny moments"],
    status=AnalysisStatus.PENDING,
)
result = asyncio.run(run_workflow(state))
print(result["outputClips"])
```

## Output Locations

| Path | Contents |
|------|----------|
| `uploads/` | Raw uploaded source videos |
| `output_clips/` | Unsubtitled clips (per-session subfolders when via web) |
| `output_clips_subtitled/` | Final clips with burned-in subtitles |

## Run Tests

```bash
uv run pytest
# or: pip install -e ".[dev]" && pytest
```

Test paths are configured in `pyproject.toml`: `testpaths = ["tests", "src/__tests__"]`.
