import logging
from src.state import VideoAnalysisState, ClipInfo, AnalysisStatus
from src.agents import llmModel
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NODE_ID = "refine_clip_context"


class ContextDecision(BaseModel):
    segments_to_add_before: list[int] = Field(..., description="List of segment indices to add before (0-19, empty if none)")
    segments_to_add_after: list[int] = Field(..., description="List of segment indices to add after (0-19, empty if none)")
    reason: str = Field(..., description="Explanation of the decision")


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = timestamp.split(":")

    if len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def find_surrounding_segments(
    clip_start_seconds: float,
    clip_end_seconds: float,
    transcription_segments: list,
    segments_before: int = 20,
    segments_after: int = 20
) -> dict:
    """
    Find segments before and after a clip.

    Returns:
        dict: {
            "before": [{"index": 0, "start": 0.0, "end": 5.0, "text": "..."}],
            "after": [{"index": 5, "start": 25.0, "end": 30.0, "text": "..."}]
        }
    """
    clip_segments = []
    for i, seg in enumerate(transcription_segments):
        if seg["start"] >= clip_start_seconds and seg["end"] <= clip_end_seconds:
            clip_segments.append(i)
        elif seg["start"] < clip_end_seconds and seg["end"] > clip_start_seconds:
            clip_segments.append(i)

    if not clip_segments:
        return {"before": [], "after": []}

    first_clip_idx = min(clip_segments)
    last_clip_idx = max(clip_segments)

    # Find segments before
    before_segments = []
    for i in range(first_clip_idx - 1, -1, -1):
        if len(before_segments) >= segments_before:
            break
        before_segments.insert(0, {
            "index": i,
            "start": transcription_segments[i]["start"],
            "end": transcription_segments[i]["end"],
            "text": transcription_segments[i]["text"]
        })

    # Find segments after
    after_segments = []
    for i in range(last_clip_idx + 1, len(transcription_segments)):
        if len(after_segments) >= segments_after:
            break
        after_segments.append({
            "index": i,
            "start": transcription_segments[i]["start"],
            "end": transcription_segments[i]["end"],
            "text": transcription_segments[i]["text"]
        })

    return {"before": before_segments, "after": after_segments}


async def refine_clip_context_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting clip context refinement")

    try:
        if not state.clips:
            logger.error(f"[{NODE_ID}] No clips to refine")
            return {
                "error": "No clips to refine",
                "status": AnalysisStatus.FAILED
            }

        if not state.transcriptionSegments:
            logger.warning(f"[{NODE_ID}] No transcription segments available, skipping refinement")
            return {
                "clips": state.clips,
                "status": AnalysisStatus.EDITING_VIDEO
            }

        # Get analysis criteria
        criteria_text = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(state.analysis)])

        refined_clips = []

        for i, clip in enumerate(state.clips):
            logger.info(f"[{NODE_ID}] Refining clip {i+1}/{len(state.clips)}: {clip.startTime}-{clip.endTime}")

        refined_clips = []

        for i, clip in enumerate(state.clips):
            logger.info(f"[{NODE_ID}] Refining clip {i+1}/{len(state.clips)}: {clip.startTime}-{clip.endTime}")

            try:
                clip_start_seconds = parse_timestamp_to_seconds(clip.startTime)
                clip_end_seconds = parse_timestamp_to_seconds(clip.endTime)

                # Find surrounding segments
                surrounding = find_surrounding_segments(
                    clip_start_seconds,
                    clip_end_seconds,
                    state.transcriptionSegments,
                    segments_before=20,
                    segments_after=20
                )

                # Build context text for analysis
                context_text = f"""Current clip ({clip.startTime}-{clip.endTime}):
Reason: {clip.reason}
Matched criterion: {clip.matchedCriterion}

"""

                if surrounding["before"]:
                    context_text += "Segments BEFORE (available to add):\n"
                    for idx, seg in enumerate(surrounding["before"]):
                        timestamp = format_timestamp(seg["start"])
                        context_text += f"[{idx}] [{timestamp}] {seg['text']}\n"
                    context_text += "\n"

                if surrounding["after"]:
                    context_text += "Segments AFTER (available to add):\n"
                    for idx, seg in enumerate(surrounding["after"]):
                        timestamp = format_timestamp(seg["start"])
                        context_text += f"[{idx}] [{timestamp}] {seg['text']}\n"
                    context_text += "\n"

                if not surrounding["before"] and not surrounding["after"]:
                    context_text += "No additional context available.\n"

                # Ask LLM which specific segments to add
                prompt = f"""Analyze which surrounding segments should be added to this clip.

                    # User Analysis Criteria (CRITICAL - MUST RESPECT):
                    {criteria_text}

                    # Clip Information
                    {context_text}

                    # CRITICAL INSTRUCTIONS
                    This clip was selected because it matches the user's criteria. You need to decide which SPECIFIC segments to add.

                    For BEFORE segments (numbered 0-{len(surrounding['before'])-1}):
                    - Analyze EACH segment individually
                    - Select ONLY segments that provide necessary context
                    - Start from closest (0) and work outward
                    - You can select NONE, SOME, or ALL - be selective

                    For AFTER segments (numbered 0-{len(surrounding['after'])-1}):
                    - Analyze EACH segment individually
                    - Select ONLY segments that complete the thought
                    - Start from closest (0) and work outward
                    - You can select NONE, SOME, or ALL - be selective

                    CRITICAL: MUST preserve alignment with user criteria. If adding a segment would make the clip NOT match the criteria, DO NOT select it.

                    Respond ONLY with valid JSON like:
                    {{
                        "segments_to_add_before": [0, 1, 3],
                        "segments_to_add_after": [0, 2],
                        "reason": "Segments 0-1 provide context, segment 2 doesn't match criteria..."
                    }}

                    Where:
                    - segments_to_add_before: Array of indices to add (empty array [] if none)
                    - segments_to_add_after: Array of indices to add (empty array [] if none)"""

                messages = [
                    {
                        "role": "system",
                        "content": "You are a video editor deciding whether to add context to clips. Return ONLY valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ]

                response = await llmModel.ainvoke(messages)
                response_text = response.content.strip()

                # Parse JSON response
                import json
                try:
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()

                    decision = json.loads(response_text)
                    segments_to_add_before = decision.get("segments_to_add_before", [])
                    segments_to_add_after = decision.get("segments_to_add_after", [])
                    reason = decision.get("reason", "")

                    logger.info(f"[{NODE_ID}] Clip {i+1} decision: before={segments_to_add_before}, after={segments_to_add_after}")
                    logger.info(f"[{NODE_ID}] Reason: {reason}")

                    # Calculate new timestamps based on selected segments
                    new_start = clip_start_seconds
                    new_end = clip_end_seconds

                    # Add selected BEFORE segments (in reverse order to maintain continuity)
                    if segments_to_add_before and surrounding["before"]:
                        # Find the earliest segment to include
                        earliest_index = max(segments_to_add_before)
                        if earliest_index < len(surrounding["before"]):
                            new_start = surrounding["before"][earliest_index]["start"]

                    # Add selected AFTER segments
                    if segments_to_add_after and surrounding["after"]:
                        # Find the latest segment to include
                        latest_index = max(segments_to_add_after)
                        if latest_index < len(surrounding["after"]):
                            new_end = surrounding["after"][latest_index]["end"]

                    # Create refined clip
                    refined_clip = ClipInfo(
                        startTime=format_timestamp(new_start),
                        endTime=format_timestamp(new_end),
                        reason=clip.reason,
                        matchedCriterion=clip.matchedCriterion
                    )

                    refined_clips.append(refined_clip)

                    if new_start != clip_start_seconds or new_end != clip_end_seconds:
                        logger.info(f"[{NODE_ID}] Refined clip {i+1}: {clip.startTime}-{clip.endTime} → {refined_clip.startTime}-{refined_clip.endTime}")
                    else:
                        logger.info(f"[{NODE_ID}] Clip {i+1} kept as is")

                except json.JSONDecodeError as e:
                    logger.warning(f"[{NODE_ID}] Failed to parse decision for clip {i+1}: {e}")
                    # Keep original clip
                    refined_clips.append(clip)

            except Exception as e:
                logger.error(f"[{NODE_ID}] Error refining clip {i+1}: {e}")
                # Keep original clip
                refined_clips.append(clip)

        logger.info(f"[{NODE_ID}] Refined {len(refined_clips)} clips")

        return {
            "clips": refined_clips,
            "status": AnalysisStatus.EDITING_VIDEO
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Context refinement failed", extra={"error": error_message})
        return {
            "error": f"Context refinement failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
