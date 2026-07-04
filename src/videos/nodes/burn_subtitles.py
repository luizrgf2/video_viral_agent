"""Burn subtitles node for the videos context: drawtext with time-remapping."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from src.videos.domain.state import VideoEditState, TimeRange, EditStatus
from src.shared.video.ffmpeg import (
    check_ffmpeg,
    probe_dimensions,
    create_drawtext_filter,
)
from src.videos.nodes.assemble_video import remap_timestamp

logger = logging.getLogger(__name__)

NODE_ID = "burn_subtitles"


def filter_relevant_segments(
    transcription_segments: List[dict],
    time_map: List[TimeRange],
) -> List[dict]:
    """Translate transcription segments into final-video coordinates.

    Returns segments with remapped start/end and original text. Segments whose
    center falls inside a kept range are included; others are dropped.
    """
    out: List[dict] = []

    for seg in transcription_segments:
        original_start = float(seg.get("start", 0.0))
        original_end = float(seg.get("end", original_start))

        final_start = remap_timestamp(time_map, original_start)
        final_end = remap_timestamp(time_map, original_end)

        if final_start < 0 or final_end < 0 or final_end <= final_start:
            continue

        out.append({
            "text": seg.get("text", "").strip(),
            "start": final_start,
            "end": final_end,
        })

    return out


async def burn_subtitles_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Burning subtitles", extra={"videoPath": state.videoPath})

    try:
        if not state.outputVideo:
            logger.warning(f"[{NODE_ID}] No output video to subtitle")
            return {"status": EditStatus.COMPLETED}

        if not state.transcriptionSegments or not state.timeMap:
            logger.warning(f"[{NODE_ID}] Missing segments/timeMap, skipping subtitles")
            return {"status": EditStatus.COMPLETED}

        if not check_ffmpeg():
            logger.warning(f"[{NODE_ID}] FFmpeg unavailable, skipping subtitles")
            return {"status": EditStatus.COMPLETED}

        clip_path = Path(state.outputVideo)
        if not clip_path.exists():
            return {
                "error": f"Assembled video not found: {state.outputVideo}",
                "status": EditStatus.FAILED,
            }

        relevant = filter_relevant_segments(state.transcriptionSegments, state.timeMap)

        if not relevant:
            logger.warning(f"[{NODE_ID}] No segments relevant after remapping, skipping subtitles")
            return {"status": EditStatus.COMPLETED}

        logger.info(f"[{NODE_ID}] Remapped segments", extra={"count": len(relevant)})

        video_width, video_height = probe_dimensions(clip_path)

        filter_complex: List[str] = []
        for seg in relevant:
            try:
                f = create_drawtext_filter(
                    text=seg["text"],
                    start_time=seg["start"],
                    end_time=seg["end"],
                    video_width=video_width,
                    video_height=video_height,
                    fontsize=14,
                    font_color="white",
                    background_color="black@0.7",
                    position="bottom",
                )
                filter_complex.append(f)
            except Exception as e:
                logger.warning(f"[{NODE_ID}] Filter creation failed", extra={
                    "error": str(e), "text": seg["text"][:50],
                })
                continue

        if not filter_complex:
            logger.warning(f"[{NODE_ID}] No subtitle filters created, leaving video unsubtitled")
            return {"status": EditStatus.COMPLETED}

        output_path = clip_path.parent / f"subtitled_{clip_path.name}"

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
            logger.error(f"[{NODE_ID}] FFmpeg failed, keeping unsubtitled video", extra={
                "returncode": result.returncode,
                "stderr": result.stderr[-500:],
            })
            return {"status": EditStatus.COMPLETED}

        try:
            clip_path.unlink()
        except Exception:
            pass

        try:
            output_path.rename(clip_path)
        except Exception as e:
            logger.warning(f"[{NODE_ID}] Could not replace output, leaving subtitled at side path", extra={
                "error": str(e),
            })
            state.outputVideo = str(output_path)

        logger.info(f"[{NODE_ID}] Subtitles burned", extra={"output_path": str(clip_path)})
        return {"status": EditStatus.COMPLETED}

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Subtitle burning failed, returning assembled video", extra={
            "error": error_message,
        })
        return {"status": EditStatus.COMPLETED}
