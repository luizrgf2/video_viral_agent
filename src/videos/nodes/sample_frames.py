"""Frame sampling node: extracts frames with FFmpeg and describes them with a VLM.

Conditional node: only runs when state.editPlan.needsVision is True.
"""

import logging
import tempfile
from pathlib import Path
from typing import List
import base64

from moviepy import VideoFileClip

from src.videos.domain.state import VideoEditState, FrameDescription, EditStatus
from src.shared.llm.agents import vlmModel
from src.shared.video.ffmpeg import extract_frames

logger = logging.getLogger(__name__)

NODE_ID = "sample_frames"

FRAME_BUDGET = 150
BATCH_SIZE = 20


def compute_fps(video_path: Path, budget: int = FRAME_BUDGET) -> float:
    """Compute FPS so total frames ≈ budget regardless of video duration."""
    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    clip.close()
    if duration <= 0:
        raise ValueError(f"Invalid video duration: {duration}")
    return budget / duration


def encode_frame_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_batch(
    frame_paths: List[Path],
    fps: float,
    edit_instructions: str,
) -> List[FrameDescription]:
    """Send a batch of frames to the VLM and return timestamped descriptions."""
    interval = 1.0 / fps
    start_index = 0  # caller offsets before sending; we use frame index in the batch

    content_parts = [{"type": "text", "text": (
        f"User editing instruction: {edit_instructions}\n\n"
        f"Below are {len(frame_paths)} frames sampled from the video at {fps:.3f} fps "
        f"(one frame every {interval:.2f}s). For EACH frame, output a single line in the format:\n"
        f"<index>: <description>\n\n"
        f"where <index> is the frame's position in this batch (0-based) and <description> is a "
        f"short factual description of what's visible (scene, people, objects, on-screen text, gestures). "
        f"Do not comment on audio. Keep each description under 200 characters."
    )}]

    for i, fp in enumerate(frame_paths):
        b64 = encode_frame_b64(fp)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = [
        {"role": "system", "content": "You are a visual scene describer for video editing. Be terse and factual."},
        {"role": "user", "content": content_parts},
    ]

    response = await vlmModel.ainvoke(messages)
    text = response.content if hasattr(response, "content") else str(response)

    descriptions: List[FrameDescription] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if ":" not in line:
            continue
        idx_str, _, desc = line.partition(":")
        idx_str = idx_str.strip()
        desc = desc.strip()
        if not idx_str.isdigit() or not desc:
            continue
        local_idx = int(idx_str)
        if local_idx < 0 or local_idx >= len(frame_paths):
            continue
        timestamp = start_index + local_idx * interval
        descriptions.append(FrameDescription(timestamp=timestamp, description=desc[:300]))

    return descriptions


async def sample_frames_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Starting frame sampling", extra={"videoPath": state.videoPath})

    if not state.editPlan or not state.editPlan.needsVision:
        logger.warning(f"[{NODE_ID}] Invoked without needsVision=True, skipping")
        return {"status": EditStatus.PLANNING_EDITS}

    video_path = Path(state.videoPath)
    if not video_path.exists():
        return {
            "error": f"Video file not found: {state.videoPath}",
            "status": EditStatus.FAILED,
        }

    tmp_dir = None
    try:
        fps = compute_fps(video_path)
        logger.info(f"[{NODE_ID}] Computed FPS", extra={"fps": fps, "budget": FRAME_BUDGET})

        tmp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
        frames = extract_frames(video_path, fps, tmp_dir)

        if not frames:
            logger.warning(f"[{NODE_ID}] No frames extracted, proceeding without vision")
            return {"status": EditStatus.PLANNING_EDITS}

        all_descriptions: List[FrameDescription] = []
        edit_instructions = state.editPlan.editInstructions

        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch = frames[batch_start:batch_start + BATCH_SIZE]
            try:
                batch_descs = await describe_batch(batch, fps, edit_instructions)
                for d in batch_descs:
                    d.timestamp += batch_start * (1.0 / fps)
                all_descriptions.extend(batch_descs)
                logger.info(f"[{NODE_ID}] Described batch", extra={
                    "batch_start": batch_start, "descriptions": len(batch_descs),
                })
            except Exception as e:
                logger.warning(f"[{NODE_ID}] VLM batch failed, skipping", extra={
                    "batch_start": batch_start, "error": str(e),
                })
                continue

        logger.info(f"[{NODE_ID}] Frame sampling done", extra={
            "total_descriptions": len(all_descriptions),
        })

        if not all_descriptions:
            logger.warning(f"[{NODE_ID}] No descriptions produced, proceeding without vision")
            return {"status": EditStatus.PLANNING_EDITS}

        return {
            "frameDescriptions": all_descriptions,
            "status": EditStatus.PLANNING_EDITS,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Frame sampling failed, degrading to text-only", extra={
            "error": error_message,
        })
        return {"status": EditStatus.PLANNING_EDITS}

    finally:
        if tmp_dir and tmp_dir.exists():
            import shutil
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning(f"[{NODE_ID}] Failed to clean temp frames", extra={"error": str(e)})
