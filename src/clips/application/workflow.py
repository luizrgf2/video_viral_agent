from langgraph.graph import StateGraph, END
from src.clips.domain.state import ClipExtractionState
from src.clips.nodes.transcribe import transcribe_audio_node
from src.clips.nodes.identify_moments import identify_moments_node
from src.clips.nodes.refine_clip_context import refine_clip_context_node
from src.clips.nodes.extract_clips import edit_video_node
from src.clips.nodes.burn_subtitles import add_subtitles_node
import logging

logger = logging.getLogger(__name__)


def create_workflow():
    workflow = StateGraph(ClipExtractionState)

    workflow.add_node("transcribe_audio", transcribe_audio_node)
    workflow.add_node("identify_moments", identify_moments_node)
    workflow.add_node("refine_clip_context", refine_clip_context_node)
    workflow.add_node("edit_video", edit_video_node)
    workflow.add_node("add_subtitles", add_subtitles_node)

    workflow.set_entry_point("transcribe_audio")

    workflow.add_edge("transcribe_audio", "identify_moments")
    workflow.add_edge("identify_moments", "refine_clip_context")
    workflow.add_edge("refine_clip_context", "edit_video")
    workflow.add_edge("edit_video", "add_subtitles")
    workflow.add_edge("add_subtitles", END)

    return workflow.compile()


# Backward-compat alias
create_clip_workflow = create_workflow


async def run_workflow(initial_state: ClipExtractionState) -> ClipExtractionState:
    logger.info("Starting clip workflow execution", extra={
        "videoPath": initial_state.videoPath,
        "analysis_count": len(initial_state.analysis)
    })

    app = create_workflow()

    result = await app.ainvoke(initial_state)

    logger.info("Clip workflow execution completed", extra={
        "status": result["status"],
        "clip_count": len(result["clips"]) if result["clips"] else 0
    })

    return result


# Backward-compat alias
run_clip_workflow = run_workflow
