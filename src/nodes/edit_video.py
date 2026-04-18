import logging
import os
from pathlib import Path
from src.state import VideoAnalysisState, AnalysisStatus
from moviepy import VideoFileClip
from src.utils.waveform_analyzer import (
    analyze_video_for_natural_cuts,
    parse_timestamp_to_seconds,
    format_timestamp
)

logger = logging.getLogger(__name__)

NODE_ID = "edit_video"


# Use parse_timestamp_to_seconds from waveform_analyzer
parse_timestamp = parse_timestamp_to_seconds


async def edit_video_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting video editing with waveform analysis", extra={"videoPath": state.videoPath})

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

        # Parse all timestamps first
        original_timestamps = []
        for clip_info in state.clips:
            start_time = parse_timestamp(clip_info.startTime)
            end_time = parse_timestamp(clip_info.endTime)
            original_timestamps.append((start_time, end_time))

        # Analyze waveform to find natural pause points
        logger.info(f"[{NODE_ID}] Analyzing audio waveform for natural cuts...")
        adjusted_timestamps = analyze_video_for_natural_cuts(
            video_path,
            original_timestamps,
            silence_threshold=0.02,  # Low amplitude threshold for silence
            min_pause_length=0.15,    # Minimum 150ms pause
            search_window=2.0         # Search within 2 seconds
        )

        output_clips = []

        # Load the video file
        video_clip = VideoFileClip(str(video_path))

        for i, (clip_info, (adjusted_start, adjusted_end)) in enumerate(zip(state.clips, adjusted_timestamps), 1):
            try:
                # Use adjusted timestamps from waveform analysis
                start_time = adjusted_start
                end_time = adjusted_end

                # Validate timestamps
                if start_time >= end_time:
                    logger.warning(f"[{NODE_ID}] Invalid timestamp range for clip {i}", extra={
                        "start": start_time,
                        "end": end_time
                    })
                    continue

                if end_time > video_clip.duration:
                    logger.warning(f"[{NODE_ID}] End time exceeds video duration for clip {i}", extra={
                        "end": end_time,
                        "duration": video_clip.duration
                    })
                    end_time = video_clip.duration

                logger.info(f"[{NODE_ID}] Processing clip {i} with natural cuts", extra={
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

                logger.info(f"[{NODE_ID}] Created clip {i} with natural cuts", extra={
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

        logger.info(f"[{NODE_ID}] Video editing completed with waveform analysis", extra={
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
