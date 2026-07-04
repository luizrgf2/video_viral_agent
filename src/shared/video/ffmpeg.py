"""Cross-cutting FFmpeg helpers shared by clips and videos contexts."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Check if FFmpeg binary is available on PATH."""
    return shutil.which("ffmpeg") is not None


def check_ffprobe() -> bool:
    """Check if FFprobe binary is available on PATH."""
    return shutil.which("ffprobe") is not None


def probe_dimensions(path: Path) -> Tuple[int, int]:
    """Return (width, height) for the first video stream. Falls back to 1920x1080."""
    if not check_ffprobe():
        logger.warning("[ffmpeg] ffprobe not available, using default 1920x1080")
        return 1920, 1080

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
    except Exception as e:
        logger.warning("[ffmpeg] probe failed, using default 1920x1080", extra={"error": str(e)})
        return 1920, 1080


def extract_frames(video_path: Path, fps: float, output_dir: Path,
                   filename_pattern: str = "frame_%05d.jpg") -> List[Path]:
    """Extract frames from video at given fps. Returns ordered list of frame paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / filename_pattern

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-y",
        str(output_pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("[ffmpeg] frame extraction failed", extra={"stderr": result.stderr[-500:]})
        raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr[-300:]}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info("[ffmpeg] extracted frames", extra={"count": len(frames), "fps": fps})
    return frames


def escape_text_for_ffmpeg(text: str) -> str:
    """Escape text for FFmpeg drawtext filter. Preserves \\n line breaks."""
    text = text.replace('\\n', '___LINE_BREAK___')
    text = text.replace('\\', '\\\\')
    text = text.replace(':', '\\:')
    text = text.replace('=', '\\=')
    text = text.replace("'", "\\'")
    text = text.replace('%', '\\%')
    text = text.replace('___LINE_BREAK___', '\\n')
    return text


def wrap_text_for_subtitle(text: str, max_chars_per_line: int = 30) -> str:
    """Wrap text at word boundaries into multiple lines joined by FFmpeg \\n."""
    words = text.split()
    if not words:
        return text

    lines: List[str] = []
    current_line: List[str] = []
    current_length = 0

    for word in words:
        word_length = len(word)

        if word_length > max_chars_per_line:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
                current_length = 0
            for i in range(0, word_length, max_chars_per_line):
                lines.append(word[i:i + max_chars_per_line])
            continue

        if current_line and current_length + word_length + 1 > max_chars_per_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length + (1 if len(current_line) > 1 else 0)

    if current_line:
        lines.append(' '.join(current_line))

    return '\\n'.join(lines)


def create_drawtext_filter(
    text: str,
    start_time: float,
    end_time: float,
    video_width: int,
    video_height: int,
    fontsize: int = 14,
    font_color: str = "white",
    background_color: str = "black@0.7",
    position: str = "bottom",
    max_total_chars: int = 120,
    max_chars_per_line: int = 30,
) -> str:
    """Build a single drawtext filter string for one subtitle segment."""
    text = ' '.join(text.strip().split())

    if len(text) > max_total_chars:
        text = text[:max_total_chars - 3] + "..."

    wrapped = wrap_text_for_subtitle(text, max_chars_per_line=max_chars_per_line)
    escaped = escape_text_for_ffmpeg(wrapped)

    if position == "top":
        y = f"{int(video_height * 0.1)}"
    elif position == "center":
        y = "(h-text_h)/2"
    else:
        y = f"{int(video_height * 0.85)}"

    return (
        f"drawtext=text='{escaped}':"
        f"fontsize={fontsize}:"
        f"fontcolor={font_color}:"
        f"box=1:boxcolor={background_color}:"
        f"boxborderw=2:"
        f"line_spacing=1:"
        f"x=(w-text_w)/2:"
        f"y={y}:"
        f"enable='between(t,{start_time},{end_time})'"
    )


def cut_segment(input_path: Path, start: float, end: float, output_path: Path,
                preset: str = "fast", crf: int = 23) -> None:
    """Cut a segment [start, end] from input and re-encode to output."""
    cmd = [
        "ffmpeg", "-ss", str(start), "-to", str(end),
        "-i", str(input_path),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "aac",
        "-y", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg cut failed: {result.stderr[-300:]}")


def concat_segments(segment_paths: List[Path], output_path: Path) -> None:
    """Concatenate segments using the concat demuxer (no re-encode)."""
    if not segment_paths:
        raise ValueError("concat_segments: empty segment list")

    list_file = output_path.parent / f"{output_path.stem}_concat.txt"
    try:
        with open(list_file, "w") as f:
            for p in segment_paths:
                f.write(f"file '{p.absolute()}'\n")

        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr[-300:]}")
    finally:
        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass
