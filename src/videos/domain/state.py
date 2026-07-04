"""State model for the videos editing context."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
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
    timeMap: Optional[List[TimeRange]] = None

    outputVideo: Optional[str] = None
    error: Optional[str] = None
    status: EditStatus = EditStatus.PENDING

    @field_validator("videoPath")
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v.endswith(".mp4"):
            raise ValueError("Video file must be in MP4 format")
        return v
