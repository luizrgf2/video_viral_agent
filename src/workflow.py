from langgraph.graph import StateGraph, END
from src.state import VideoAnalysisState
from src.nodes import analyze_video_node, identify_moments_node
import logging

logger = logging.getLogger(__name__)


def create_workflow():
    workflow = StateGraph(VideoAnalysisState)

    workflow.add_node("analyze_video", analyze_video_node)
    workflow.add_node("identify_moments", identify_moments_node)

    workflow.set_entry_point("analyze_video")

    workflow.add_edge("analyze_video", "identify_moments")
    workflow.add_edge("identify_moments", END)

    return workflow.compile()


async def run_workflow(initial_state: VideoAnalysisState) -> VideoAnalysisState:
    logger.info("Starting workflow execution", extra={
        "videoPath": initial_state.videoPath,
        "analysis_count": len(initial_state.analysis)
    })

    app = create_workflow()

    result = await app.ainvoke(initial_state)

    logger.info("Workflow execution completed", extra={
        "status": result["status"],
        "clip_count": len(result["clips"]) if result["clips"] else 0
    })

    return result
