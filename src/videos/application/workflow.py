"""LangGraph workflow definition for the videos editing context."""

from langgraph.graph import StateGraph, END
from src.videos.domain.state import VideoEditState, EditStatus
from src.videos.nodes.classify_intent import classify_intent_node
from src.videos.nodes.transcribe import transcribe_node
from src.videos.nodes.sample_frames import sample_frames_node
from src.videos.nodes.plan_edits import plan_edits_node
from src.videos.nodes.assemble_video import assemble_video_node
from src.videos.nodes.burn_subtitles import burn_subtitles_node
import logging

logger = logging.getLogger(__name__)


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


async def run_edit_workflow(initial_state: VideoEditState) -> dict:
    logger.info("Starting video edit workflow", extra={
        "videoPath": initial_state.videoPath,
        "prompt": initial_state.userPrompt,
    })

    app = create_edit_workflow()
    result = await app.ainvoke(initial_state)

    logger.info("Video edit workflow completed", extra={
        "status": result.get("status"),
        "has_output": bool(result.get("outputVideo")),
    })

    return result
