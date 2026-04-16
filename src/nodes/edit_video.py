import logging
import os
from pathlib import Path
from src.state import VideoAnalysisState, AnalysisStatus
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

NODE_ID = "edit_video"


def parse_timestamp(timestamp: str) -> float:
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


async def edit_video_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting video editing", extra={"videoPath": state.videoPath})

    try:
        if not state.clips:
            logger.warning(f"[{NODE_ID}] No clips to process")
            return {
                "outputClips": [],
                "status": AnalysisStatus.COMPLETED
            }

        video_path = Path(state.videoPath)

        if not video_path.exists():
            logger.error(f"[{NODE_ID}] Video file not found", extra={"videoPath": state.videoPath})
            return {
                "error": f"Video file not found: {state.videoPath}",
                "status": AnalysisStatus.FAILED
            }

        # Create output directory
        output_dir = video_path.parent / "output_clips"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"[{NODE_ID}] Processing {len(state.clips)} clips", extra={"output_dir": str(output_dir)})

        output_clips = []

        # Load the video file
        video_clip = VideoFileClip(str(video_path))

        for i, clip_info in enumerate(state.clips, 1):
            try:
                # Parse timestamps
                start_time = parse_timestamp(clip_info.startTime)
                end_time = parse_timestamp(clip_info.endTime)

                # Validate timestamps
                if start_time >= end_time:
                    logger.warning(f"[{NODE_ID}] Invalid timestamp range for clip {i}", extra={
                        "start": clip_info.startTime,
                        "end": clip_info.endTime
                    })
                    continue

                if end_time > video_clip.duration:
                    logger.warning(f"[{NODE_ID}] End time exceeds video duration for clip {i}", extra={
                        "end": clip_info.endTime,
                        "duration": video_clip.duration
                    })
                    end_time = video_clip.duration

                logger.info(f"[{NODE_ID}] Processing clip {i}", extra={
                    "start": start_time,
                    "end": end_time,
                    "reason": clip_info.reason
                })

                # Extract clip using new MoviePy 2.0 API
                subclip = video_clip.subclipped(start_time, end_time)

                # Generate output filename
                output_filename = f"clip_{i:03d}_{start_time:.0f}-{end_time:.0f}.mp4"
                output_path = output_dir / output_filename

                # Write the clip using new MoviePy 2.0 API
                subclip.write_videofile(
                    str(output_path),
                    logger=None  # Suppress moviepy output
                )

                output_clips.append(str(output_path))

                logger.info(f"[{NODE_ID}] Created clip {i}", extra={
                    "output_path": str(output_path),
                    "size_mb": output_path.stat().st_size / (1024 * 1024)
                })

                # Close subclip to free memory
                subclip.close()

            except Exception as e:
                logger.error(f"[{NODE_ID}] Failed to process clip {i}", extra={
                    "error": str(e),
                    "clip": clip_info.model_dump()
                })
                continue

        # Close the main video clip
        video_clip.close()

        logger.info(f"[{NODE_ID}] Video editing completed", extra={
            "total_clips": len(output_clips),
            "output_dir": str(output_dir)
        })

        return {
            "outputClips": output_clips,
            "status": AnalysisStatus.COMPLETED
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Video editing failed", extra={"error": error_message})
        return {
            "error": f"Video editing failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
