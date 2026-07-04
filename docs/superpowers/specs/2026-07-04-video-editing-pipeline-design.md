# Video Editing Pipeline — Design Spec

**Date:** 2026-07-04
**Status:** Approved (pending implementation)
**Scope:** Add a second bounded context for AI-driven video editing alongside the existing clip-extraction pipeline.

---

## 1. Problem Statement

The existing pipeline (`src/nodes/`, `src/workflow.py`) extracts N short viral clips from a long video. The user wants a **separate, parallel capability** that produces **one edited video** from a long source, by removing irrelevant parts according to a free-form natural-language prompt.

The two capabilities must be architecturally independent: different routes, different scaling characteristics, different change drivers. They share underlying infrastructure (Whisper, OpenRouter, FFmpeg, librosa) but nothing else.

### Non-goals (MVP)

- No real-time/streaming editing.
- No UI controls other than a single text prompt.
- No multi-language support (Portuguese hard-coded, same as today).
- No persistent job storage (in-memory, same model as current Flask app).
- No transition effects, music, or B-roll generation.
- No user accounts / auth.

---

## 2. Architecture

### 2.1 DDD-aligned folder structure

Full reorganization in one cycle. The existing pipeline moves out of `src/` root into `src/clips/`; the new pipeline lives in `src/videos/`; cross-cutting infrastructure moves to `src/shared/`.

```
src/
├── shared/                              # SHARED KERNEL
│   ├── __init__.py
│   ├── llm/
│   │   └── agents.py                    # vlmModel, llmModel (moved from src/agents.py)
│   ├── transcription/
│   │   ├── __init__.py
│   │   └── transcriber.py               # extract_and_compress_audio, faster_whisper, groq
│   │                                    #   (extracted from src/nodes/transcribe_audio.py)
│   ├── audio/
│   │   ├── __init__.py
│   │   └── waveform.py                  # analyze_video_for_natural_cuts et al.
│   │                                    #   (moved from src/utils/waveform_analyzer.py)
│   └── video/
│       ├── __init__.py
│       └── ffmpeg.py                    # NEW: ffprobe helpers, frame extraction, concat
│
├── clips/                               # BOUNDED CONTEXT 1: Clip Extraction (existing)
│   ├── __init__.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── state.py                     # ClipExtractionState (moved from src/state.py)
│   ├── application/
│   │   ├── __init__.py
│   │   └── workflow.py                  # create_clip_workflow, run_clip_workflow
│   │                                    #   (moved from src/workflow.py)
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── transcribe.py                # wrapper: calls shared.transcription
│   │   ├── identify_moments.py
│   │   ├── refine_clip_context.py
│   │   ├── extract_clips.py             # renamed from edit_video.py
│   │   └── burn_subtitles.py            # renamed from add_subtitles.py
│   └── api/
│       ├── __init__.py
│       └── routes.py                    # /clips/* endpoints
│
└── videos/                              # BOUNDED CONTEXT 2: Video Editing (NEW)
    ├── __init__.py
    ├── domain/
    │   ├── __init__.py
    │   └── state.py                     # VideoEditState, EditPlan, EditSegment
    ├── application/
    │   ├── __init__.py
    │   └── workflow.py                  # create_edit_workflow, run_edit_workflow
    ├── nodes/
    │   ├── __init__.py
    │   ├── classify_intent.py           # router LLM → EditPlan
    │   ├── transcribe.py                # wrapper: calls shared.transcription
    │   ├── sample_frames.py             # NEW: FFmpeg frame sampling, conditional
    │   ├── plan_edits.py                # LLM produces keep/remove decisions
    │   ├── assemble_video.py            # cut + concat into 1 output
    │   └── burn_subtitles.py            # drawtext with time-remapping
    └── api/
        ├── __init__.py
        └── routes.py                    # /videos/* endpoints
```

### 2.2 Backward compatibility

The Python API surface for clip extraction is **renamed but preserved**:

| Old | New |
|-----|-----|
| `from src.workflow import run_workflow` | `from src.clips.application.workflow import run_clip_workflow` |
| `from src.state import VideoAnalysisState` | `from src.clips.domain.state import ClipExtractionState` |

A thin shim at `src/workflow.py` and `src/state.py` may re-export under the old names for one release cycle, to avoid breaking external callers. Decision: **provide the shim** (low cost, avoids surprising breaks).

Flask routes for clips keep the existing paths (`/upload`, `/status/<id>`, `/clips/<id>`, `/video/<id>/<f>`) for backward compatibility. New `/clips/*` aliases are added and the old ones are kept as aliases.

---

## 3. New Pipeline: `videos` Context

### 3.1 Pipeline flow

```
                    ┌─────────────────────┐
                    │  classify_intent    │  ← LLM reads prompt → EditPlan
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     transcribe      │  ← shared.transcription
                    └──────────┬──────────┘
                               │
                ┌──────────────▼──────────────┐
                │  editPlan.needsVision?      │
                └──┬──────────────────────┬───┘
                   │ yes                  │ no
         ┌─────────▼─────────┐            │
         │   sample_frames   │            │
         │  (FFmpeg + VLM)   │            │
         └─────────┬─────────┘            │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼──────────┐
                   │     plan_edits      │  ← LLM → List[EditSegment]
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   assemble_video    │  ← waveform-adjust cuts, FFmpeg concat
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │   burn_subtitles    │  ← drawtext, time-remapped
                   └──────────┬──────────┘
                              │
                              END
```

LangGraph conditional edge after `transcribe`: route to `sample_frames` if `editPlan.needsVision` is `True`, otherwise jump to `plan_edits`.

### 3.2 State model

```python
# src/videos/domain/state.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from enum import Enum


class EditStatus(str, Enum):
    PENDING = "pending"
    CLASSIFYING = "classifying"
    TRANSCRIBING = "transcribing"
    SAMPLING_FRAMES = "sampling_frames"
    PLANNING_EDITS = "planning_edits"
    ASSEMBLING = "assembling"
    SUBTITLING = "subtitling"
    COMPLETED = "completed"
    FAILED = "failed"


class EditPlan(BaseModel):
    needsVision: bool = Field(..., description="Whether visual frame analysis is required")
    mode: Literal["direto_ao_ponto", "emotion_peaks", "filler_removal", "custom"]
    editInstructions: str = Field(..., description="Refined directive for the analysis node")
    reasoning: str = Field(..., description="Why the router chose this plan")


class FrameDescription(BaseModel):
    timestamp: float
    description: str


class EditSegment(BaseModel):
    start: float
    end: float
    action: Literal["keep", "remove"]
    reason: str


class TimeRange(BaseModel):
    """Mapping between an original-video time range and the assembled-video time range."""
    original_start: float
    original_end: float
    final_start: float
    final_end: float


class VideoEditState(BaseModel):
    videoPath: str = Field(..., min_length=1)
    userPrompt: str = Field(..., min_length=1)

    editPlan: Optional[EditPlan] = None
    transcription: Optional[str] = None
    transcriptionSegments: Optional[List[dict]] = None
    frameDescriptions: Optional[List[FrameDescription]] = None

    editSegments: Optional[List[EditSegment]] = None
    timeMap: Optional[List[TimeRange]] = None  # ordered list, used to remap timestamps

    outputVideo: Optional[str] = None
    error: Optional[str] = None
    status: EditStatus = EditStatus.PENDING

    @field_validator("videoPath")
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v.endswith(".mp4"):
            raise ValueError("Video file must be in MP4 format")
        return v
```

### 3.3 Nodes

#### 3.3.1 `classify_intent`

Single LLM call. Inputs: `state.userPrompt`. Output: `EditPlan`.

System prompt:
> You are an editing-intent classifier. Given the user's natural-language request for editing a video, output a JSON object describing how to process it. Detect whether the request references visual content (objects on screen, faces, gestures, scene changes, on-screen text) — if so, set `needsVision: true`. If it references only speech, topics, or content, set `needsVision: false`. Map the request to one of the preset modes (`direto_ao_ponto`, `emotion_peaks`, `filler_removal`) when it clearly matches, otherwise use `custom`. Always produce a refined `editInstructions` field suitable for downstream analysis. Return ONLY JSON.

Structured output via `llmModel.with_structured_output(EditPlan)`.

Sets `status = TRANSCRIBING`.

#### 3.3.2 `transcribe`

Wrapper around `shared.transcription`. Same mode selection (local / Groq) via `AUDIO_TRANSCRIPTION_MODE` env var. Same Portuguese hard-coding.

Sets `status = SAMPLING_FRAMES` if `editPlan.needsVision`, else `PLANNING_EDITS`.

#### 3.3.3 `sample_frames` (conditional)

Two phases:

**Phase A — Frame extraction (FFmpeg):**

Frame budget: **150 frames total**. FPS calculated dynamically:

```python
from moviepy import VideoFileClip

def compute_fps(video_path: Path, budget: int = 150) -> float:
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    clip.close()
    return budget / duration

fps = compute_fps(video_path)  # e.g., 10-min video → 150/600 = 0.25 fps
```

FFmpeg command:
```bash
ffmpeg -i <input> -vf fps=<fps> frame_%05d.jpg
```

Frames saved to a temporary directory.

**Phase B — VLM description (batched):**

Frames batched in groups of **20** (so ~8 batches for 150 frames). Each batch is sent to `vlmModel` (default `gemini-2.0-flash-lite`) with:

- The 20 frames as image content parts.
- The timestamp range covered.
- The user's `editInstructions` for context.

VLM returns a description per timestamp. Results merged into `state.frameDescriptions`.

**Cleanup:** all temp JPGs deleted in a `finally` block.

Sets `status = PLANNING_EDITS`.

#### 3.3.4 `plan_edits`

LLM receives:
- `transcriptionSegments` (timestamped text)
- `frameDescriptions` (if present)
- `editPlan.editInstructions`
- `editPlan.mode`

Returns `List[EditSegment]` covering the full video timeline without gaps or overlaps. Consecutive segments of the same action may be merged; the union of all `start`/`end` ranges must equal `[0, video_duration]`.

Structured output via `llmModel.with_structured_output(EditSegmentsOutput)` where:

```python
class EditSegmentsOutput(BaseModel):
    segments: List[EditSegment]
```

Sets `status = ASSEMBLING`.

#### 3.3.5 `assemble_video`

1. **Boundary refinement:** for every cut boundary (start and end of each `keep` and `remove` segment), use `shared.audio.waveform.analyze_video_for_natural_cuts` to snap to the nearest natural pause. This prevents cuts mid-word regardless of which side of the boundary the segment is on.
2. **Filter** to segments where `action == "keep"`.
3. **Cut each keep segment** with FFmpeg re-encode (necessary for arbitrary cut points):
   ```bash
   ffmpeg -ss <start> -to <end> -i <input> -c:v libx264 -preset fast -crf 23 -c:a aac part_N.mp4
   ```
4. **Concat** via FFmpeg concat demuxer (no re-encode at this step):
   ```bash
   ffmpeg -f concat -safe 0 -i list.txt -c copy <output>
   ```
5. **Build `timeMap`:** iterate keep segments in order, accumulating their durations. For each keep segment, append a `TimeRange(original_start=seg.start, original_end=seg.end, final_start=accumulator, final_end=accumulator + seg_duration)`. The result is an ordered list that downstream subtitle remapping can binary-search to translate any original timestamp into its final-video position.

Output: `state.outputVideo` (path to the assembled MP4).

Sets `status = SUBTITLING`.

#### 3.3.6 `burn_subtitles`

Same FFmpeg `drawtext` approach as the clips pipeline (`shared.video.ffmpeg` helpers), but with **time remapping**:

1. Filter `transcriptionSegments` to those that fall within `keep` ranges.
2. For each segment, binary-search `timeMap` to find the matching `TimeRange`, then translate `original_start`/`original_end` into `final_start`/`final_end`.
3. Generate `drawtext` filters with remapped `enable='between(t, final_start, final_end)'`.
4. Run FFmpeg once over the assembled video.

Output: final subtitled MP4 (overwrites `state.outputVideo`).

Sets `status = COMPLETED`.

### 3.4 Workflow definition

```python
# src/videos/application/workflow.py

from langgraph.graph import StateGraph, END
from src.videos.domain.state import VideoEditState
from src.videos.nodes import (
    classify_intent_node,
    transcribe_node,
    sample_frames_node,
    plan_edits_node,
    assemble_video_node,
    burn_subtitles_node,
)


def route_after_transcribe(state: VideoEditState) -> str:
    if state.editPlan and state.editPlan.needsVision:
        return "sample_frames"
    return "plan_edits"


def create_edit_workflow():
    workflow = StateGraph(VideoEditState)

    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("sample_frames", sample_frames_node)
    workflow.add_node("plan_edits", plan_edits_node)
    workflow.add_node("assemble_video", assemble_video_node)
    workflow.add_node("burn_subtitles", burn_subtitles_node)

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "transcribe")
    workflow.add_conditional_edges(
        "transcribe",
        route_after_transcribe,
        {"sample_frames": "sample_frames", "plan_edits": "plan_edits"},
    )
    workflow.add_edge("sample_frames", "plan_edits")
    workflow.add_edge("plan_edits", "assemble_video")
    workflow.add_edge("assemble_video", "burn_subtitles")
    workflow.add_edge("burn_subtitles", END)

    return workflow.compile()


async def run_edit_workflow(initial_state: VideoEditState) -> VideoEditState:
    app = create_edit_workflow()
    result = await app.ainvoke(initial_state)
    return result
```

---

## 4. Flask API

New blueprint `videos_bp` registered on `app.py`. Existing `clips_bp` (or current routes) preserved.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/videos/upload` | Upload video + prompt; starts workflow; returns `session_id` |
| `GET` | `/videos/status/<session_id>` | Polling: `processing` / `completed` / `failed` + current status enum |
| `GET` | `/videos/result/<session_id>` | JSON metadata: output filename, size, duration |
| `GET` | `/videos/video/<session_id>/<filename>` | Serves the final MP4 |

### Request format (`/videos/upload`)

`multipart/form-data`:
- `video` — MP4 file (up to 500 MB, same limit as clips).
- `prompt` — string, the natural-language edit instruction.

### Threading model

Same as current clips endpoint: one thread per request, private asyncio event loop. No shared state across requests. Status inferred from filesystem + a per-session JSON sidecar file written at completion (so failures can be surfaced, unlike the current clips endpoint).

### Error surfacing

Unlike the current clips endpoint (which cannot distinguish "in progress" from "failed"), the videos endpoint writes `<session_id>.json` next to the output folder with the final `status` and `error` fields. `/videos/status/<id>` reads this file and returns `failed` with the error message when applicable.

---

## 5. Shared Kernel extraction

Code currently in `src/agents.py`, `src/nodes/transcribe_audio.py`, and `src/utils/waveform_analyzer.py` is refactored into `src/shared/` as follows.

### `src/shared/llm/agents.py`

Move from `src/agents.py` verbatim. Update imports in `src/clips/` and `src/videos/` nodes to `from src.shared.llm.agents import llmModel, vlmModel`.

### `src/shared/transcription/transcriber.py`

Extract from `src/nodes/transcribe_audio.py`:
- `extract_and_compress_audio`
- `optimize_audio_for_transcription`
- `cleanup_temp_audio`
- `transcribe_with_faster_whisper`
- `transcribe_with_groq`
- `get_transcription_mode`

Public API:
```python
async def transcribe(video_path: Path) -> dict:
    """Returns {transcription, transcriptionSegments, mode, ...}."""
```

### `src/shared/audio/waveform.py`

Move from `src/utils/waveform_analyzer.py` verbatim (all functions). Update importers.

### `src/shared/video/ffmpeg.py` (NEW)

Cross-cutting FFmpeg helpers used by both contexts:
- `check_ffmpeg() -> bool`
- `probe_dimensions(path: Path) -> tuple[int, int]`
- `extract_frames(path: Path, fps: float, output_dir: Path) -> List[Path]` (new)
- `escape_text_for_ffmpeg(text: str) -> str` (extracted from `add_subtitles.py`)
- `wrap_text_for_subtitle(text: str, max_chars: int) -> str` (extracted)
- `create_drawtext_filter(...)` (extracted, parameterized)

---

## 6. Failure Model

Each node catches its own exceptions and returns `{"error": "<msg>", "status": FAILED}`. The Flask endpoint reads the per-session JSON sidecar to surface failures to the client.

| Node | Failure mode | Fallback |
|------|--------------|----------|
| `classify_intent` | LLM error | `FAILED`, no video produced |
| `transcribe` | Whisper/Groq error | `FAILED` |
| `sample_frames` | FFmpeg extraction fails | Skip vision, proceed with text-only (log warning) |
| `sample_frames` | VLM call fails | Skip vision, proceed with text-only (log warning) |
| `plan_edits` | LLM error | `FAILED` |
| `assemble_video` | FFmpeg error | `FAILED` |
| `burn_subtitles` | FFmpeg error | Return assembled video without subtitles (log warning) |

Key principle: **vision failures degrade gracefully** (text-only path) but transcription/planning failures are fatal.

---

## 7. Testing Strategy

Tests live alongside each context:
- `src/clips/__tests__/` — existing tests, updated imports only.
- `src/videos/__tests__/` — new tests.

### Priority tests for `videos`

1. **`classify_intent` unit tests** — given prompt strings, assert `EditPlan` fields (vision detection, mode mapping).
2. **`sample_frames` math test** — given video durations, assert computed FPS and frame count.
3. **`assemble_video` time-map test** — given a list of `EditSegment`, assert `timeMap` is correct.
4. **`burn_subtitles` remapping test** — given a `timeMap` and segments, assert `drawtext` filters use remapped times.
5. **End-to-end smoke test** — small video fixture, prompt "direto ao ponto", assert 1 MP4 output exists.

LLM-dependent nodes are tested with mocked LLM responses (same pattern as existing `src/__tests__/`).

### Verification before claiming done

Same rules as `AGENTS.md` §5:
1. `uv run pytest` passes.
2. `uv run python app.py` boots and both `/clips/upload` and `/videos/upload` respond.
3. Import smoke check: `python -c "from src.videos.application.workflow import create_edit_workflow"`.

---

## 8. Cost & Latency Estimates

For a 10-minute video, prompt requires vision:

| Stage | Time | Cost |
|-------|------|------|
| classify_intent | ~2s | ~$0.001 |
| transcribe (local) | ~2-3 min | $0 |
| transcribe (Groq) | ~30-60s | ~$0.01 |
| sample_frames (150 frames, 8 VLM calls) | ~30-60s | ~$0.03-0.05 |
| plan_edits | ~5-10s | ~$0.005 |
| assemble_video | ~30-60s | $0 (local FFmpeg) |
| burn_subtitles | ~10-30s | $0 |
| **Total (local transcription)** | **~4-6 min** | **~$0.05** |
| **Total (Groq transcription)** | **~2-3 min** | **~$0.06** |

For a prompt without vision, drop the `sample_frames` row: ~$0.005 total, 30-60s faster.

---

## 9. Open Questions Deferred Post-MVP

- User-configurable frame budget (currently hard-coded 150).
- Streaming/progress reporting during long runs.
- Resume after crash (requires checkpointing, not in LangGraph today).
- Multi-language support (currently Portuguese-only).
- Caching of transcription across multiple prompts on the same video.
- Vertical (9:16) output for shorts — currently outputs same aspect ratio as source.

---

## 10. Migration Plan (execution order)

1. Create `src/shared/` skeleton; move `agents.py`, extract transcription helpers, move waveform.
2. Create `src/clips/` skeleton; move existing nodes, state, workflow; update all imports.
3. Add backward-compat shims at `src/workflow.py`, `src/state.py`, `src/agents.py`.
4. Update existing tests to new import paths; run `pytest` — must pass before continuing.
5. Create `src/videos/` skeleton; implement `domain/state.py` and stub all nodes.
6. Implement `classify_intent` and `transcribe` nodes; smoke test.
7. Implement `sample_frames` (FFmpeg + VLM batching); test the math.
8. Implement `plan_edits`; test with mocked LLM.
9. Implement `assemble_video` (waveform + concat + timeMap); test the timeMap.
10. Implement `burn_subtitles` with remapping.
11. Add Flask blueprint `videos_bp` to `app.py`.
12. Update `templates/index.html` (or add new template) for the new `/videos/upload` flow.
13. Update `AGENTS.md` and `docs/` to reflect new structure.
14. Full `pytest` run + manual smoke test of both pipelines.
