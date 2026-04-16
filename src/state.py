from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    ANALYZING_VIDEO = "analyzing_video"
    IDENTIFYING_MOMENTS = "identifying_moments"
    EDITING_VIDEO = "editing_video"
    COMPLETED = "completed"
    FAILED = "failed"


class ClipInfo(BaseModel):
    startTime: str = Field(..., description="Start time in MM:SS or HH:MM:SS format")
    endTime: str = Field(..., description="End time in MM:SS or HH:MM:SS format")
    reason: str = Field(..., description="Why this moment is viral/important")
    matchedCriterion: str = Field(..., description="Which analysis criterion this matches")


class VideoAnalysisState(BaseModel):
    videoPath: str = Field(..., min_length=1, description="Path to MP4 video file")
    analysis: List[str] = Field(..., min_length=1, description="Analysis criteria")
    videoDescription: Optional[str] = Field(None, description="VLM-generated video description with timestamps")
    clips: Optional[List[ClipInfo]] = Field(None, description="Identified viral moments")
    outputClips: Optional[List[str]] = Field(None, description="Paths to generated clip files")
    error: Optional[str] = Field(None, description="Error message if failed")
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, description="Current workflow status")

    @field_validator('videoPath')
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v.endswith('.mp4'):
            raise ValueError('Video file must be in MP4 format')
        return v

    @field_validator('analysis')
    @classmethod
    def validate_analysis(cls, v: List[str]) -> List[str]:
        if len(v) > 10:
            raise ValueError('Too many analysis criteria (max 10)')
        if any(not criterion.strip() for criterion in v):
            raise ValueError('Analysis criteria cannot be empty')
        return v
