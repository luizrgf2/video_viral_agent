"""Transcription node for the videos context. Wraps shared.transcription."""

import logging
from pathlib import Path
from src.videos.domain.state import VideoEditState, EditStatus
from src.shared.transcription.transcriber import transcribe

logger = logging.getLogger(__name__)

NODE_ID = "transcribe"


async def transcribe_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Starting transcription", extra={"videoPath": state.videoPath})

    try:
        result = await transcribe(Path(state.videoPath))

        next_status = (
            EditStatus.SAMPLING_FRAMES
            if state.editPlan and state.editPlan.needsVision
            else EditStatus.PLANNING_EDITS
        )

        return {
            "transcription": result["transcription"],
            "transcriptionSegments": result["transcriptionSegments"],
            "status": next_status,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Transcription failed", extra={"error": error_message})
        return {
            "error": f"Transcription failed: {error_message}",
            "status": EditStatus.FAILED,
        }
