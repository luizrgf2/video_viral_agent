"""Transcription node for the clips context. Thin wrapper around shared.transcription."""

import logging
from pathlib import Path
from src.clips.domain.state import ClipExtractionState, AnalysisStatus
from src.shared.transcription.transcriber import transcribe

logger = logging.getLogger(__name__)

NODE_ID = "transcribe_audio"


async def transcribe_audio_node(state: ClipExtractionState) -> dict:
    """LangGraph node: transcribes the video and updates state."""
    try:
        result = await transcribe(Path(state.videoPath))
        return {**result, "status": AnalysisStatus.IDENTIFYING_MOMENTS}
    except Exception as e:
        logger.error(f"[{NODE_ID}] Transcription failed", extra={"error": str(e)})
        return {
            "error": f"Transcription failed: {e}",
            "status": AnalysisStatus.FAILED,
        }
