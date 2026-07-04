# Workflow & Execution

**Source:** `src/workflow.py`

## Graph Construction

```python
def create_workflow():
    workflow = StateGraph(VideoAnalysisState)
    workflow.add_node("transcribe_audio", transcribe_audio_node)
    workflow.add_node("identify_moments", identify_moments_node)
    workflow.add_node("refine_clip_context", refine_clip_context_node)
    workflow.add_node("edit_video", edit_video_node)
    workflow.add_node("add_subtitles", add_subtitles_node)

    workflow.set_entry_point("transcribe_audio")
    # linear edges → END
    return workflow.compile()
```

The compiled graph is **stateless** — it is recreated on every `run_workflow()` call.

## Running

```python
async def run_workflow(initial_state: VideoAnalysisState) -> VideoAnalysisState:
    app = create_workflow()
    result = await app.ainvoke(initial_state)
    return result
```

### From Python

```python
import asyncio
from src.workflow import run_workflow
from src.state import VideoAnalysisState, AnalysisStatus

state = VideoAnalysisState(
    videoPath="path/to/video.mp4",
    analysis=["Extract funny moments", "Quotes about X"],
    status=AnalysisStatus.PENDING,
)
result = asyncio.run(run_workflow(state))
```

### From the Web Layer

`app.py` wraps `run_workflow` inside a thread with a private event loop:

```python
def process_video_async():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_workflow(initial_state))
    loop.close()
```

This is necessary because Flask is synchronous and LangGraph runs async.

## Important Behaviors

- **No checkpointing.** If the process dies mid-run, all progress is lost.
- **No retry.** Node failures return `FAILED` status; the graph continues to the next node (which should no-op on missing inputs).
- **Single video, single criteria list per run.** The graph is not designed for batch processing.
- The `analyze_video` node (VLM frame analysis) exists in `src/nodes/analyze_video.py` but is **not wired** into the main workflow.
