"""Assemble video node: applies waveform-aware cuts and concatenates kept segments."""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

from src.videos.domain.state import VideoEditState, EditSegment, TimeRange, EditStatus
from src.shared.audio.waveform import analyze_video_for_natural_cuts
from src.shared.video.ffmpeg import cut_segment, concat_segments

logger = logging.getLogger(__name__)

NODE_ID = "assemble_video"


def build_time_map(kept_segments: List[Tuple[float, float]]) -> List[TimeRange]:
    """Build a mapping from original-video time ranges to final-video time ranges.

    kept_segments: ordered list of (original_start, original_end) tuples that will appear in output.
    """
    time_map: List[TimeRange] = []
    accumulator = 0.0

    for original_start, original_end in kept_segments:
        duration = original_end - original_start
        if duration <= 0:
            continue
        time_map.append(TimeRange(
            original_start=original_start,
            original_end=original_end,
            final_start=accumulator,
            final_end=accumulator + duration,
        ))
        accumulator += duration

    return time_map


def remap_timestamp(time_map: List[TimeRange], original_time: float) -> float:
    """Translate an original-video timestamp into its position in the assembled video.

    Linearly interpolates within the matching TimeRange. Returns -1.0 if no range matches.
    """
    for tr in time_map:
        if tr.original_start <= original_time <= tr.original_end:
            original_span = tr.original_end - tr.original_start
            if original_span <= 0:
                return tr.final_start
            ratio = (original_time - tr.original_start) / original_span
            return tr.final_start + ratio * (tr.final_end - tr.final_start)
    return -1.0


async def assemble_video_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Starting assembly", extra={"videoPath": state.videoPath})

    try:
        if not state.editSegments:
            logger.warning(f"[{NODE_ID}] No edit segments, skipping assembly")
            return {"status": EditStatus.SUBTITLING}

        video_path = Path(state.videoPath)
        if not video_path.exists():
            return {
                "error": f"Video file not found: {state.videoPath}",
                "status": EditStatus.FAILED,
            }

        kept = [(s.start, s.end) for s in state.editSegments if s.action == "keep"]

        if not kept:
            logger.warning(f"[{NODE_ID}] Nothing to keep, aborting assembly")
            return {
                "error": "No segments marked as keep; nothing to assemble",
                "status": EditStatus.FAILED,
            }

        logger.info(f"[{NODE_ID}] Adjusting boundaries with waveform", extra={
            "kept_count": len(kept),
        })

        try:
            adjusted = analyze_video_for_natural_cuts(
                video_path,
                kept,
                silence_threshold=0.02,
                min_pause_length=0.15,
                search_window=2.0,
            )
        except Exception as e:
            logger.warning(f"[{NODE_ID}] Waveform adjustment failed, using original cuts", extra={
                "error": str(e),
            })
            adjusted = kept

        output_dir = video_path.parent / "output_videos"
        output_dir.mkdir(exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="assemble_"))

        part_paths: List[Path] = []
        try:
            for i, (start, end) in enumerate(adjusted, 1):
                if start >= end:
                    logger.warning(f"[{NODE_ID}] Skipping zero/negative duration segment {i}")
                    continue
                part_path = tmp_dir / f"part_{i:04d}.mp4"
                logger.info(f"[{NODE_ID}] Cutting segment {i}", extra={
                    "start": start, "end": end,
                })
                cut_segment(video_path, start, end, part_path)
                part_paths.append(part_path)

            if not part_paths:
                return {
                    "error": "Assembly produced no segments after cutting",
                    "status": EditStatus.FAILED,
                }

            output_path = output_dir / f"edited_{video_path.stem}.mp4"
            logger.info(f"[{NODE_ID}] Concatenating {len(part_paths)} parts")
            concat_segments(part_paths, output_path)

            time_map = build_time_map([(s, e) for s, e in adjusted if e > s])

            logger.info(f"[{NODE_ID}] Assembly complete", extra={
                "output_path": str(output_path),
                "time_map_entries": len(time_map),
            })

            return {
                "outputVideo": str(output_path),
                "timeMap": time_map,
                "status": EditStatus.SUBTITLING,
            }

        finally:
            import shutil
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning(f"[{NODE_ID}] Failed to clean temp dir", extra={"error": str(e)})

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Assembly failed", extra={"error": error_message})
        return {
            "error": f"Video assembly failed: {error_message}",
            "status": EditStatus.FAILED,
        }
