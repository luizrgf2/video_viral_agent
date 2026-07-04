"""Subtitle burning node for the clips context.

Adds subtitles to video clips using FFmpeg drawtext based on transcription segments.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List
from src.clips.domain.state import ClipExtractionState, AnalysisStatus
from src.shared.video.ffmpeg import (
    check_ffmpeg,
    probe_dimensions,
    create_drawtext_filter,
)

logger = logging.getLogger(__name__)

NODE_ID = "add_subtitles"


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


def add_subtitles_to_clip(
    clip_path: Path,
    transcription_segments: List[dict],
    clip_start: float,
    clip_end: float,
    output_path: Path,
) -> bool:
    """Add subtitles to a video clip using FFmpeg drawtext."""
    try:
        logger.info(f"[{NODE_ID}] Adding subtitles to clip", extra={
            "clip_path": str(clip_path),
            "segments_count": len(transcription_segments),
        })

        if not check_ffmpeg():
            logger.error(f"[{NODE_ID}] FFmpeg not available")
            return False

        video_width, video_height = probe_dimensions(clip_path)

        relevant_segments = []
        for segment in transcription_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]

            if seg_start >= clip_start and seg_end <= clip_end:
                relevant_segments.append({
                    "text": segment["text"],
                    "start": seg_start - clip_start,
                    "end": seg_end - clip_start,
                })
            elif seg_start < clip_end and seg_end > clip_start:
                relevant_segments.append({
                    "text": segment["text"],
                    "start": max(0.0, seg_start - clip_start),
                    "end": min(clip_end - clip_start, seg_end - clip_start),
                })

        if not relevant_segments:
            logger.warning(f"[{NODE_ID}] No relevant segments found for clip")
            shutil.copy2(clip_path, output_path)
            return True

        logger.info(f"[{NODE_ID}] Found {len(relevant_segments)} segments for subtitle overlay")

        filter_complex = []
        for segment in relevant_segments:
            try:
                filter_str = create_drawtext_filter(
                    text=segment["text"],
                    start_time=segment["start"],
                    end_time=segment["end"],
                    video_width=video_width,
                    video_height=video_height,
                    fontsize=14,
                    font_color="white",
                    background_color="black@0.7",
                    position="bottom",
                )
                filter_complex.append(filter_str)
            except Exception as e:
                logger.warning(f"[{NODE_ID}] Failed to create filter for segment", extra={
                    "error": str(e), "text": segment["text"][:50],
                })
                continue

        if not filter_complex:
            logger.warning(f"[{NODE_ID}] No subtitle filters created, copying original video")
            shutil.copy2(clip_path, output_path)
            return True

        cmd = [
            "ffmpeg", "-i", str(clip_path),
            "-vf", ",".join(filter_complex),
            "-c:a", "copy",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(output_path), "-y",
        ]

        logger.info(f"[{NODE_ID}] Running FFmpeg with {len(filter_complex)} subtitle filters")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"[{NODE_ID}] FFmpeg failed", extra={
                "returncode": result.returncode,
                "stderr": result.stderr[-500:],
            })
            return False

        logger.info(f"[{NODE_ID}] Successfully added subtitles", extra={
            "output_path": str(output_path),
            "subtitle_count": len(filter_complex),
        })
        return True

    except Exception as e:
        logger.error(f"[{NODE_ID}] Failed to add subtitles to clip", extra={
            "error": str(e), "clip_path": str(clip_path),
        })
        try:
            shutil.copy2(clip_path, output_path)
            logger.info(f"[{NODE_ID}] Copied original clip due to subtitle error")
            return True
        except Exception as copy_error:
            logger.error(f"[{NODE_ID}] Failed to copy original clip", extra={"error": str(copy_error)})
            return False


async def add_subtitles_node(state: ClipExtractionState) -> dict:
    """Add subtitles to all generated clips based on transcription segments."""
    logger.info(f"[{NODE_ID}] Starting subtitle generation", extra={
        "videoPath": state.videoPath,
    })

    try:
        if not state.outputClips:
            logger.warning(f"[{NODE_ID}] No output clips to add subtitles to")
            return {"subtitledClips": [], "status": AnalysisStatus.COMPLETED}

        if not state.transcriptionSegments:
            logger.warning(f"[{NODE_ID}] No transcription segments available")
            return {"subtitledClips": state.outputClips, "status": AnalysisStatus.COMPLETED}

        clips_info = state.clips or []
        video_path = Path(state.videoPath)
        output_dir = video_path.parent / "output_clips_subtitled"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"[{NODE_ID}] Processing {len(state.outputClips)} clips", extra={
            "output_dir": str(output_dir),
            "segments_available": len(state.transcriptionSegments),
        })

        subtitled_clips = []

        for i, clip_path in enumerate(state.outputClips):
            try:
                logger.info(f"[{NODE_ID}] Processing clip {i+1}/{len(state.outputClips)}")

                clip_path = Path(clip_path)
                if not clip_path.exists():
                    logger.warning(f"[{NODE_ID}] Clip file not found: {clip_path}")
                    continue

                clip_start = 0.0
                clip_end = 0.0

                if i < len(clips_info):
                    try:
                        clip_start = parse_timestamp_to_seconds(clips_info[i].startTime)
                        clip_end = parse_timestamp_to_seconds(clips_info[i].endTime)
                    except Exception as e:
                        logger.warning(f"[{NODE_ID}] Failed to parse clip timing", extra={"error": str(e)})
                        from moviepy import VideoFileClip
                        clip_video = VideoFileClip(str(clip_path))
                        clip_end = clip_video.duration
                        clip_video.close()

                output_filename = f"subtitled_{clip_path.name}"
                output_path = output_dir / output_filename

                success = add_subtitles_to_clip(
                    clip_path=clip_path,
                    transcription_segments=state.transcriptionSegments,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    output_path=output_path,
                )

                if success:
                    subtitled_clips.append(str(output_path))
                    logger.info(f"[{NODE_ID}] Created subtitled clip {i+1}", extra={
                        "output_path": str(output_path),
                    })
                else:
                    logger.warning(f"[{NODE_ID}] Failed to add subtitles to clip {i+1}")
                    subtitled_clips.append(str(clip_path))

            except Exception as e:
                logger.error(f"[{NODE_ID}] Error processing clip {i+1}", extra={"error": str(e)})
                subtitled_clips.append(str(clip_path))
                continue

        logger.info(f"[{NODE_ID}] Subtitle generation completed", extra={
            "total_clips": len(subtitled_clips),
            "output_dir": str(output_dir),
        })

        return {
            "subtitledClips": subtitled_clips,
            "outputClips": subtitled_clips,
            "status": AnalysisStatus.COMPLETED,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Subtitle generation failed", extra={"error": error_message})
        return {
            "error": f"Subtitle generation failed: {error_message}",
            "status": AnalysisStatus.FAILED,
        }
