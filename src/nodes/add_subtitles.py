"""
Subtitle Generation Node

Adds subtitles to video clips using FFmpeg based on transcription segments.
Overlays text synchronized with segment timestamps.
"""

import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple
from src.state import VideoAnalysisState, AnalysisStatus

logger = logging.getLogger(__name__)

NODE_ID = "add_subtitles"


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available for video processing."""
    return shutil.which("ffmpeg") is not None


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


def escape_text_for_ffmpeg(text: str) -> str:
    """
    Escape text for FFmpeg drawtext filter.
    Special characters need to be escaped: ':', '\', '=', "'", '%'
    """
    # Replace backslashes first
    text = text.replace('\\', '\\\\')
    # Replace colons
    text = text.replace(':', '\\:')
    # Replace equals signs
    text = text.replace('=', '\\=')
    # Replace quotes
    text = text.replace("'", "\\'")
    # Replace percent signs
    text = text.replace('%', '\\%')
    return text


def create_ffmpeg_drawtext_filter(
    text: str,
    start_time: float,
    end_time: float,
    video_width: int,
    video_height: int,
    fontsize: int = 24,
    font_color: str = "white",
    background_color: str = "black@0.5",
    position: str = "bottom"
) -> str:
    """
    Create FFmpeg drawtext filter for a single subtitle.

    Args:
        text: Subtitle text
        start_time: Start time in seconds
        end_time: End time in seconds
        video_width: Video width
        video_height: Video height
        fontsize: Font size in pixels (reduced to fit video)
        font_color: Text color
        background_color: Background color with opacity
        position: Position on screen

    Returns:
        FFmpeg filter string
    """
    # Clean up text and limit length
    text = ' '.join(text.strip().split())

    # Truncate very long text to avoid overflow
    max_chars = 100
    if len(text) > max_chars:
        text = text[:max_chars-3] + "..."

    escaped_text = escape_text_for_ffmpeg(text)

    # Calculate position based on video size
    if position == "top":
        y = f"{int(video_height * 0.1)}"
    elif position == "center":
        y = f"(h-text_h)/2"
    else:  # bottom (default)
        y = f"{int(video_height * 0.85)}"

    # Calculate max line width (80% of video width)
    max_line_width = int(video_width * 0.8)

    # Build the drawtext filter with line wrapping
    filter_str = (
        f"drawtext=text='{escaped_text}':"
        f"fontsize={fontsize}:"
        f"fontcolor={font_color}:"
        f"box=1:boxcolor={background_color}:"
        f"boxborderw=3:"
        f"line_spacing=2:"
        f"x=(w-text_w)/2:"
        f"y={y}:"
        f"enable='between(t,{start_time},{end_time})'"
    )

    return filter_str


def add_subtitles_to_clip(
    clip_path: Path,
    transcription_segments: List[dict],
    clip_start: float,
    clip_end: float,
    output_path: Path
) -> bool:
    """
    Add subtitles to a video clip using FFmpeg based on transcription segments.

    Args:
        clip_path: Path to video clip
        transcription_segments: List of segment dicts with 'start', 'end', 'text'
        clip_start: Clip start time in original video
        clip_end: Clip end time in original video
        output_path: Path for output video with subtitles

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"[{NODE_ID}] Adding subtitles to clip with FFmpeg", extra={
            "clip_path": str(clip_path),
            "segments_count": len(transcription_segments)
        })

        # Check if FFmpeg is available
        if not check_ffmpeg():
            logger.error(f"[{NODE_ID}] FFmpeg not available")
            return False

        # Get video dimensions using FFprobe
        cmd_probe = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(clip_path)
        ]

        try:
            result = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
            dimensions = result.stdout.strip().split(',')
            video_width, video_height = int(dimensions[0]), int(dimensions[1])
        except Exception as e:
            logger.warning(f"[{NODE_ID}] Failed to get video dimensions, using defaults", extra={"error": str(e)})
            video_width, video_height = 1920, 1080

        # Filter segments that fall within this clip's time range
        relevant_segments = []
        for segment in transcription_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]

            # Check if segment overlaps with clip time range
            if seg_start >= clip_start and seg_end <= clip_end:
                # Segment is completely inside clip
                relative_start = seg_start - clip_start
                relative_end = seg_end - clip_start
                relevant_segments.append({
                    "text": segment["text"],
                    "start": relative_start,
                    "end": relative_end
                })
            elif seg_start < clip_end and seg_end > clip_start:
                # Segment partially overlaps with clip
                relative_start = max(0, seg_start - clip_start)
                relative_end = min(clip_end - clip_start, seg_end - clip_start)
                relevant_segments.append({
                    "text": segment["text"],
                    "start": relative_start,
                    "end": relative_end
                })

        if not relevant_segments:
            logger.warning(f"[{NODE_ID}] No relevant segments found for clip")
            # Just copy the clip without subtitles
            shutil.copy2(clip_path, output_path)
            return True

        logger.info(f"[{NODE_ID}] Found {len(relevant_segments)} segments for subtitle overlay")

        # Build FFmpeg command with drawtext filters
        cmd = [
            "ffmpeg",
            "-i", str(clip_path),
            "-vf",  # Video filters
        ]

        # Create drawtext filters for each segment
        filter_complex = []
        for i, segment in enumerate(relevant_segments):
            try:
                filter_str = create_ffmpeg_drawtext_filter(
                    text=segment["text"],
                    start_time=segment["start"],
                    end_time=segment["end"],
                    video_width=video_width,
                    video_height=video_height,
                    fontsize=24,  # Reduced from 48 to fit better
                    font_color="white",
                    background_color="black@0.5",
                    position="bottom"
                )
                filter_complex.append(filter_str)
            except Exception as e:
                logger.warning(f"[{NODE_ID}] Failed to create filter for segment {i}", extra={
                    "error": str(e),
                    "text": segment["text"][:50]
                })
                continue

        if not filter_complex:
            logger.warning(f"[{NODE_ID}] No subtitle filters created, copying original video")
            shutil.copy2(clip_path, output_path)
            return True

        # Combine filters with commas
        cmd.append(",".join(filter_complex))

        # Add output options
        cmd.extend([
            "-c:a", "copy",  # Copy audio without re-encoding
            "-c:v", "libx264",  # Re-encode video with subtitles
            "-preset", "fast",  # Use fast preset
            "-crf", "23",  # Good quality
            str(output_path),
            "-y"  # Overwrite output file
        ])

        logger.info(f"[{NODE_ID}] Running FFmpeg with {len(filter_complex)} subtitle filters")

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"[{NODE_ID}] FFmpeg failed", extra={
                "returncode": result.returncode,
                "stderr": result.stderr[-500:]  # Last 500 chars of error
            })
            return False

        logger.info(f"[{NODE_ID}] Successfully added subtitles", extra={
            "output_path": str(output_path),
            "subtitle_count": len(filter_complex)
        })

        return True

    except Exception as e:
        logger.error(f"[{NODE_ID}] Failed to add subtitles to clip", extra={
            "error": str(e),
            "clip_path": str(clip_path)
        })
        # Fallback: copy the clip without subtitles
        try:
            shutil.copy2(clip_path, output_path)
            logger.info(f"[{NODE_ID}] Copied original clip due to subtitle error")
            return True
        except Exception as copy_error:
            logger.error(f"[{NODE_ID}] Failed to copy original clip", extra={"error": str(copy_error)})
            return False


async def add_subtitles_node(state: VideoAnalysisState) -> dict:
    """
    Add subtitles to all generated clips based on transcription segments.

    This node processes each clip created by edit_video_node and overlays
    subtitles synchronized with the transcription segments.
    """
    logger.info(f"[{NODE_ID}] Starting subtitle generation", extra={
        "videoPath": state.videoPath
    })

    try:
        # Check if we have the necessary data
        if not state.outputClips:
            logger.warning(f"[{NODE_ID}] No output clips to add subtitles to")
            return {
                "subtitledClips": [],
                "status": AnalysisStatus.COMPLETED
            }

        if not state.transcriptionSegments:
            logger.warning(f"[{NODE_ID}] No transcription segments available")
            return {
                "subtitledClips": state.outputClips,  # Return original clips
                "status": AnalysisStatus.COMPLETED
            }

        # Get clips information if available
        clips_info = state.clips or []

        # Create output directory for subtitled clips
        video_path = Path(state.videoPath)
        output_dir = video_path.parent / "output_clips_subtitled"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"[{NODE_ID}] Processing {len(state.outputClips)} clips", extra={
            "output_dir": str(output_dir),
            "segments_available": len(state.transcriptionSegments)
        })

        subtitled_clips = []

        # Process each clip
        for i, clip_path in enumerate(state.outputClips):
            try:
                logger.info(f"[{NODE_ID}] Processing clip {i+1}/{len(state.outputClips)}")

                clip_path = Path(clip_path)
                if not clip_path.exists():
                    logger.warning(f"[{NODE_ID}] Clip file not found: {clip_path}")
                    continue

                # Get clip timing if available
                clip_start = 0.0
                clip_end = 0.0

                if i < len(clips_info):
                    try:
                        clip_start = parse_timestamp_to_seconds(clips_info[i].startTime)
                        clip_end = parse_timestamp_to_seconds(clips_info[i].endTime)
                    except Exception as e:
                        logger.warning(f"[{NODE_ID}] Failed to parse clip timing", extra={"error": str(e)})
                        # Try to get duration from video file
                        clip_video = VideoFileClip(str(clip_path))
                        clip_end = clip_video.duration
                        clip_video.close()

                # Generate output filename
                output_filename = f"subtitled_{clip_path.name}"
                output_path = output_dir / output_filename

                # Add subtitles to the clip
                success = add_subtitles_to_clip(
                    clip_path=clip_path,
                    transcription_segments=state.transcriptionSegments,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    output_path=output_path
                )

                if success:
                    subtitled_clips.append(str(output_path))
                    logger.info(f"[{NODE_ID}] Created subtitled clip {i+1}", extra={
                        "output_path": str(output_path)
                    })
                else:
                    logger.warning(f"[{NODE_ID}] Failed to add subtitles to clip {i+1}")
                    # Keep original clip
                    subtitled_clips.append(str(clip_path))

            except Exception as e:
                logger.error(f"[{NODE_ID}] Error processing clip {i+1}", extra={
                    "error": str(e)
                })
                # Keep original clip on error
                subtitled_clips.append(str(clip_path))
                continue

        logger.info(f"[{NODE_ID}] Subtitle generation completed", extra={
            "total_clips": len(subtitled_clips),
            "output_dir": str(output_dir)
        })

        return {
            "subtitledClips": subtitled_clips,
            "outputClips": subtitled_clips,  # Update outputClips to point to subtitled versions
            "status": AnalysisStatus.COMPLETED
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Subtitle generation failed", extra={"error": error_message})
        return {
            "error": f"Subtitle generation failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }