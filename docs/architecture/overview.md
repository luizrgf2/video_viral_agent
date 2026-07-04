# Architecture Overview

The **Video Viral Agent** is a multi-agent pipeline that extracts viral-worthy moments from long videos. It is orchestrated by **LangGraph** as a strict linear state machine, with each node performing one discrete stage of the processing.

## Pipeline Stages

```
transcribe_audio  →  identify_moments  →  refine_clip_context  →  edit_video  →  add_subtitles  →  END
```

The pipeline is defined in `src/workflow.py` via `StateGraph(VideoAnalysisState)`. There are no conditional edges or branching — execution is sequential. Every node receives the shared `VideoAnalysisState` and returns a partial dict that is merged back into state by LangGraph.

## Node Responsibilities

| # | Node | Source File | Role |
|---|------|-------------|------|
| 1 | `transcribe_audio` | `src/nodes/transcribe_audio.py` | Extract + compress audio, transcribe (local Whisper or Groq API), produce segments with timestamps |
| 2 | `identify_moments` | `src/nodes/identify_moments.py` | LLM analyzes timestamped transcription against user criteria; emits candidate clips |
| 3 | `refine_clip_context` | `src/nodes/refine_clip_context.py` | LLM decides which surrounding segments to add so clips don't cut mid-sentence |
| 4 | `edit_video` | `src/nodes/edit_video.py` | Waveform analysis finds natural pauses; MoviePy cuts clips at those boundaries |
| 5 | `add_subtitles` | `src/nodes/add_subtitles.py` | FFmpeg `drawtext` overlays synchronized subtitles from transcription segments |

## Runtime Topology

- **Entry point:** `transcribe_audio`
- **Exit:** `END` (LangGraph terminal)
- **State container:** `VideoAnalysisState` (Pydantic model, see `state.md`)
- **Invocation:** `run_workflow()` in `src/workflow.py` compiles and runs the graph asynchronously (`await app.ainvoke(initial_state)`)

## Failure Model

Every node catches its own exceptions and returns:
```python
{"error": "<message>", "status": AnalysisStatus.FAILED}
```
There is **no centralized error recovery**. A `FAILED` status propagates but does not short-circuit downstream nodes by itself — each node must guard against missing inputs (e.g., `if not state.transcription`).

## Concurrency

The workflow is fully async at the LangGraph level, but the web layer (`app.py`) runs each request's workflow inside a **dedicated thread** with its own asyncio event loop. There is no cross-request state sharing.
