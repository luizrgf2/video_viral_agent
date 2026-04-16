import json
import logging
from src.state import VideoAnalysisState, ClipInfo, AnalysisStatus
from src.agents import llmModel, MOMENTS_IDENTIFICATION_SYSTEM_PROMPT
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NODE_ID = "identify_moments"


class ClipOutput(BaseModel):
    clips: list[ClipInfo] = Field(..., description="List of identified viral moments")


async def identify_moments_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting moment identification")

    try:
        if not state.videoDescription:
            logger.error(f"[{NODE_ID}] No video description available")
            return {
                "error": "Video description is required for moment identification",
                "status": AnalysisStatus.FAILED
            }

        criteria_text = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(state.analysis)])

        prompt = f"""Identify the most viral and impactful moments from this video analysis.

# Analysis Criteria
{criteria_text}

# Video Description with Timestamps
{state.videoDescription}

# Instructions
Based on the video description and analysis criteria, identify the most viral/impactful moments that should be extracted as clips.

IMPORTANT: Focus on NATURAL CONTENT BOUNDARIES, not arbitrary time limits. A moment should be:
- As long as needed to be complete and coherent (could be 15 seconds or 3 minutes)
- Long enough to provide context and deliver the value/punchline
- Short enough to remain engaging (avoid dragging or unnecessary footage)

For each moment, provide:
1. Start time (MM:SS or HH:MM:SS format) - when the moment naturally begins
2. End time (MM:SS or HH:MM:SS format) - when the moment naturally concludes
3. Reason: Why this moment is viral/important
4. Matched criterion: Which analysis criterion this moment matches

Focus on moments that:
- Have high emotional impact
- Contain valuable insights or memorable quotes
- Would perform well on social media (engaging, shareable)
- Match the analysis criteria
- Feel complete and satisfying (not cut off mid-thought)

Return ONLY the clips in JSON format. If no viral moments are found, return an empty array."""

        messages = [
            {"role": "system", "content": MOMENTS_IDENTIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"[{NODE_ID}] Calling LLM", extra={"model": llmModel.model_name})

        structured_llm = llmModel.with_structured_output(ClipOutput)
        response = await structured_llm.ainvoke(messages)

        clips = response.clips

        logger.info(f"[{NODE_ID}] Moments identified", extra={"clip_count": len(clips)})

        return {
            "clips": clips,
            "status": AnalysisStatus.EDITING_VIDEO
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Moment identification failed", extra={"error": error_message})
        return {
            "error": f"Moment identification failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
