"""Backward-compat shim. Real code lives in src.clips.domain.state."""

from src.clips.domain.state import (
    AnalysisStatus,
    ClipInfo,
    ClipExtractionState,
)

# Deprecated alias — use ClipExtractionState
VideoAnalysisState = ClipExtractionState

__all__ = ["AnalysisStatus", "ClipInfo", "ClipExtractionState", "VideoAnalysisState"]
