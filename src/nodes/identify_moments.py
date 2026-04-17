import logging
from src.state import VideoAnalysisState, ClipInfo, AnalysisStatus
from src.agents import llmModel, MOMENTS_IDENTIFICATION_SYSTEM_PROMPT
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NODE_ID = "identify_moments"


class ClipOutput(BaseModel):
    clips: list[ClipInfo] = Field(..., description="List of identified viral moments")


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


def expand_clip_with_context(clip_start: str, clip_end: str, transcription_segments: list) -> tuple[str, str]:
    """
    Expand clip timestamps to include 2 segments before and 1 segment after.

    Args:
        clip_start: Original start timestamp (MM:SS)
        clip_end: Original end timestamp (MM:SS)
        transcription_segments: List of segments with start/end times

    Returns:
        tuple: (expanded_start, expanded_end) timestamps in MM:SS format
    """
    try:
        clip_start_seconds = parse_timestamp_to_seconds(clip_start)
        clip_end_seconds = parse_timestamp_to_seconds(clip_end)

        # Find the segment indices that overlap with this clip
        clip_segment_indices = []
        for i, seg in enumerate(transcription_segments):
            if seg["start"] >= clip_start_seconds and seg["end"] <= clip_end_seconds:
                clip_segment_indices.append(i)
            elif seg["start"] < clip_end_seconds and seg["end"] > clip_start_seconds:
                # Partial overlap
                clip_segment_indices.append(i)

        if not clip_segment_indices:
            # No segments found, return original
            return clip_start, clip_end

        first_clip_index = min(clip_segment_indices)
        last_clip_index = max(clip_segment_indices)

        # Expand: 2 segments before (if available)
        expanded_start_index = max(0, first_clip_index - 2)

        # Expand: 1 segment after (if available)
        expanded_end_index = min(len(transcription_segments) - 1, last_clip_index + 1)

        # Get new timestamps
        expanded_start = transcription_segments[expanded_start_index]["start"]
        expanded_end = transcription_segments[expanded_end_index]["end"]

        return format_timestamp(expanded_start), format_timestamp(expanded_end)

    except Exception as e:
        logger.warning(f"Failed to expand clip {clip_start}-{clip_end}: {e}")
        return clip_start, clip_end


async def identify_moments_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting moment identification")

    try:
        if not state.transcription:
            logger.error(f"[{NODE_ID}] No transcription available")
            return {
                "error": "Transcription is required for moment identification",
                "status": AnalysisStatus.FAILED
            }

        criteria_text = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(state.analysis)])

        # Build timestamped transcription text
        transcription_with_timestamps = ""
        if state.transcriptionSegments:
            for segment in state.transcriptionSegments:
                start_min, start_sec = divmod(int(segment["start"]), 60)
                timestamp = f"{start_min:02d}:{start_sec:02d}"
                transcription_with_timestamps += f"[{timestamp}] {segment['text']}\n"
        else:
            transcription_with_timestamps = state.transcription

        prompt = f"""Identify moments from this video transcription that EXACTLY match the analysis criteria.

            # Analysis Criteria (MUST FOLLOW EXACTLY)
            {criteria_text}

            # Video Transcription with Timestamps
            {transcription_with_timestamps}

            # CRITICAL INSTRUCTIONS
            1. Identify ONLY moments that EXACTLY match the criteria above
            2. If NO moments match the criteria, return an empty array
            3. Do NOT guess or approximate - be precise
            4. Include sufficient context (2-3 segments before and after) for completeness

            For each matching moment, provide:
            1. Start time (MM:SS or HH:MM:SS format)
            2. End time (MM:SS or HH:MM:SS format)
            3. Reason: Why this moment matches the criteria
            4. Matched criterion: Which analysis criterion this moment matches

            Return ONLY the clips in JSON format. If no moments match, return {{"clips": []}}."""

        messages = [
            {"role": "system", "content": "You are a video content analyst. Identify moments that EXACTLY match the user's specified criteria. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"[{NODE_ID}] Calling LLM", extra={"model": llmModel.model_name})

        structured_llm = llmModel.with_structured_output(ClipOutput)
        response = await structured_llm.ainvoke(messages)

        clips = response.clips

        logger.info(f"[{NODE_ID}] Identified {len(clips)} moments", extra={"clip_count": len(clips)})

        # Expand each clip to include context (2 segments before, 1 after)
        if state.transcriptionSegments and clips:
            expanded_clips = []

            for clip in clips:
                # Get original timestamps
                original_start = clip.startTime
                original_end = clip.endTime

                # Expand with context
                expanded_start, expanded_end = expand_clip_with_context(
                    original_start,
                    original_end,
                    state.transcriptionSegments
                )

                # Create new clip with expanded timestamps
                expanded_clip = ClipInfo(
                    startTime=expanded_start,
                    endTime=expanded_end,
                    reason=clip.reason,
                    matchedCriterion=clip.matchedCriterion
                )

                expanded_clips.append(expanded_clip)

                logger.info(f"[{NODE_ID}] Expanded clip: {original_start}-{original_end} → {expanded_start}-{expanded_end}")

            clips = expanded_clips

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
