"""Plan edits node: LLM decides which segments to keep or remove."""

import logging
from typing import List
from pydantic import BaseModel, Field

from src.videos.domain.state import VideoEditState, EditSegment, EditStatus
from src.shared.llm.agents import llmModel

logger = logging.getLogger(__name__)

NODE_ID = "plan_edits"


class EditSegmentsOutput(BaseModel):
    segments: List[EditSegment] = Field(..., description="Ordered list of keep/remove decisions covering the full video timeline")


SYSTEM_PROMPT = """You are a video editor planning tool. You receive a timestamped transcription (and optionally visual descriptions) of a video, plus an editing directive from the user. Your job is to partition the ENTIRE video timeline into a sequence of non-overlapping segments, each marked as "keep" or "remove", that collectively cover [0, video_duration] with no gaps.

Rules:
- Every second of the video must belong to exactly one segment.
- Adjacent segments with the same action should be merged into a single segment.
- Mark as "remove" any content that does not match the user directive (tangents, off-topic, rambling, dead air, fillers per the directive).
- Mark as "keep" everything that should survive in the final edited video.
- The first segment must start at 0.0 and the last segment must end at the video duration.
- Provide a concise "reason" for each segment.

Return ONLY valid JSON with a "segments" array."""


def build_prompt(state: VideoEditState) -> str:
    parts = [f"# User editing directive\n{state.editPlan.editInstructions}\n"]

    if state.editPlan.mode != "custom":
        parts.append(f"\n# Detected preset mode\n{state.editPlan.mode}\n")

    if state.transcriptionSegments:
        parts.append("\n# Timestamped transcription\n")
        for seg in state.transcriptionSegments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            parts.append(f"[{start:.2f}-{end:.2f}] {text}")
    elif state.transcription:
        parts.append(f"\n# Transcription (no timestamps)\n{state.transcription}")

    if state.frameDescriptions:
        parts.append("\n# Visual frame descriptions\n")
        for fd in state.frameDescriptions:
            parts.append(f"[{fd.timestamp:.2f}] {fd.description}")

    parts.append(
        "\n# Instructions\n"
        "Partition the full timeline into ordered keep/remove segments. "
        "Return ONLY JSON: {\"segments\": [{\"start\": float, \"end\": float, "
        "\"action\": \"keep\"|\"remove\", \"reason\": str}, ...]}"
    )
    return "\n".join(parts)


async def plan_edits_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Planning edits", extra={"videoPath": state.videoPath})

    try:
        if not state.transcription and not state.transcriptionSegments:
            return {
                "error": "Cannot plan edits without transcription",
                "status": EditStatus.FAILED,
            }

        prompt = build_prompt(state)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        structured_llm = llmModel.with_structured_output(EditSegmentsOutput)
        output = await structured_llm.ainvoke(messages)

        segments = output.segments

        if not segments:
            logger.warning(f"[{NODE_ID}] No segments returned, defaulting to keep-all")
            return {
                "editSegments": [],
                "status": EditStatus.ASSEMBLING,
            }

        logger.info(f"[{NODE_ID}] Plan produced", extra={
            "segment_count": len(segments),
            "keep_count": sum(1 for s in segments if s.action == "keep"),
            "remove_count": sum(1 for s in segments if s.action == "remove"),
        })

        return {
            "editSegments": segments,
            "status": EditStatus.ASSEMBLING,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Plan failed", extra={"error": error_message})
        return {
            "error": f"Edit planning failed: {error_message}",
            "status": EditStatus.FAILED,
        }
