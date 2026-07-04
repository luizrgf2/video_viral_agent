"""
Waveform Analysis Module for Natural Pause Detection

This module analyzes audio waveforms to find natural pause points
for smooth video cutting, avoiding mid-word cuts.
"""

import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: Path) -> Optional[str]:
    """
    Extract audio from video file to temporary WAV file.

    Args:
        video_path: Path to video file

    Returns:
        Path to temporary audio file, or None if failed
    """
    try:
        from moviepy import VideoFileClip

        logger.info(f"[waveform] Extracting audio from video", extra={"video_path": str(video_path)})

        # Load video and extract audio
        video_clip = VideoFileClip(str(video_path))
        audio_clip = video_clip.audio

        if audio_clip is None:
            logger.error(f"[waveform] No audio track found in video")
            video_clip.close()
            return None

        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # Write audio to temporary file
        audio_clip.write_audiofile(temp_audio_path, logger=None)

        # Close clips
        audio_clip.close()
        video_clip.close()

        logger.info(f"[waveform] Audio extracted successfully", extra={"audio_path": temp_audio_path})

        return temp_audio_path

    except Exception as e:
        logger.error(f"[waveform] Failed to extract audio", extra={"error": str(e)})
        return None


def analyze_waveform_for_pauses(
    audio_path: str,
    video_duration: float,
    silence_threshold: float = 0.02,
    min_pause_length: float = 0.15,
    frame_length: int = 2048,
    hop_length: int = 512
) -> List[Tuple[float, float]]:
    """
    Analyze audio waveform to detect natural pause points.

    Args:
        audio_path: Path to audio file
        video_duration: Total video duration in seconds
        silence_threshold: Amplitude threshold for silence (0.0-1.0)
        min_pause_length: Minimum pause length in seconds to consider
        frame_length: Window size for FFT analysis
        hop_length: Step size for FFT analysis

    Returns:
        List of (start_time, end_time) tuples for detected pauses
    """
    try:
        import librosa
        import soundfile as sf

        logger.info(f"[waveform] Analyzing waveform for pauses", extra={
            "audio_path": audio_path,
            "silence_threshold": silence_threshold,
            "min_pause_length": min_pause_length
        })

        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Compute RMS energy (amplitude envelope)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        # Find silence regions (amplitude below threshold)
        silence_mask = rms < silence_threshold

        # Find continuous silence regions
        pauses = []
        current_pause_start = None

        for i, is_silent in enumerate(silence_mask):
            if is_silent and current_pause_start is None:
                # Start of a pause
                current_pause_start = times[i]
            elif not is_silent and current_pause_start is not None:
                # End of a pause
                pause_end = times[i]
                pause_length = pause_end - current_pause_start

                # Only keep pauses longer than minimum
                if pause_length >= min_pause_length:
                    pauses.append((current_pause_start, pause_end))

                current_pause_start = None

        # Handle case where video ends with silence
        if current_pause_start is not None:
            pause_end = times[-1]
            pause_length = pause_end - current_pause_start
            if pause_length >= min_pause_length:
                pauses.append((current_pause_start, pause_end))

        logger.info(f"[waveform] Found {len(pauses)} natural pauses", extra={
            "pause_count": len(pauses),
            "total_pauses_time": sum(p[1] - p[0] for p in pauses)
        })

        return pauses

    except Exception as e:
        logger.error(f"[waveform] Failed to analyze waveform", extra={"error": str(e)})
        return []


def find_nearest_pause_point(
    target_time: float,
    pauses: List[Tuple[float, float]],
    search_window: float = 2.0,
    prefer_before: bool = True
) -> Optional[float]:
    """
    Find the nearest natural pause point to target time.

    Args:
        target_time: Target timestamp in seconds
        pauses: List of (start, end) pause tuples
        search_window: Maximum time to search from target (seconds)
        prefer_before: Prefer pause before target (for start cuts)

    Returns:
        Best pause point timestamp, or None if not found
    """
    if not pauses:
        return None

    best_pause = None
    min_distance = float('inf')

    for pause_start, pause_end in pauses:
        # Try start of pause (for start cuts)
        candidate_time = pause_start if prefer_before else pause_end

        distance = abs(candidate_time - target_time)

        # Check if within search window
        if distance <= search_window:
            if distance < min_distance:
                min_distance = distance
                best_pause = candidate_time

    return best_pause


def adjust_timestamps_to_natural_pauses(
    start_time: float,
    end_time: float,
    pauses: List[Tuple[float, float]],
    search_window: float = 2.0
) -> Tuple[float, float]:
    """
    Adjust cut timestamps to align with natural pauses.

    Args:
        start_time: Original start time
        end_time: Original end time
        pauses: List of detected pauses
        search_window: Maximum time to search for pause

    Returns:
        Tuple of (adjusted_start, adjusted_end)
    """
    # Find nearest pause for start time (prefer pause before or at start)
    adjusted_start = find_nearest_pause_point(
        start_time, pauses, search_window, prefer_before=True
    )

    # Find nearest pause for end time (prefer pause at or after end)
    adjusted_end = find_nearest_pause_point(
        end_time, pauses, search_window, prefer_before=False
    )

    # Fallback to original times if no pauses found
    if adjusted_start is None:
        logger.debug(f"[waveform] No pause found for start time {start_time:.2f}, using original")
        adjusted_start = start_time
    else:
        logger.info(f"[waveform] Adjusted start: {start_time:.2f} → {adjusted_start:.2f}")

    if adjusted_end is None:
        logger.debug(f"[waveform] No pause found for end time {end_time:.2f}, using original")
        adjusted_end = end_time
    else:
        logger.info(f"[waveform] Adjusted end: {end_time:.2f} → {adjusted_end:.2f}")

    # Ensure adjusted times maintain minimum duration (1 second)
    if adjusted_end - adjusted_start < 1.0:
        logger.warning(f"[waveform] Adjusted duration too short, using original times")
        return start_time, end_time

    # Ensure we don't extend too far (max 3 seconds from original)
    if abs(adjusted_start - start_time) > 3.0:
        adjusted_start = start_time
    if abs(adjusted_end - end_time) > 3.0:
        adjusted_end = end_time

    return adjusted_start, adjusted_end


def analyze_video_for_natural_cuts(
    video_path: Path,
    timestamps: List[Tuple[float, float]],
    silence_threshold: float = 0.02,
    min_pause_length: float = 0.15,
    search_window: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Main function: analyze video and adjust cut points to natural pauses.

    Args:
        video_path: Path to video file
        timestamps: List of (start, end) tuples to adjust
        silence_threshold: Amplitude threshold for silence detection
        min_pause_length: Minimum pause length to consider
        search_window: Maximum time to search for pauses

    Returns:
        List of adjusted (start, end) tuples
    """
    try:
        from moviepy import VideoFileClip

        logger.info(f"[waveform] Starting waveform-based cut analysis", extra={
            "video_path": str(video_path),
            "num_cuts": len(timestamps)
        })

        # Get video duration
        video_clip = VideoFileClip(str(video_path))
        video_duration = video_clip.duration
        video_clip.close()

        # Extract audio
        temp_audio = extract_audio_from_video(video_path)
        if temp_audio is None:
            logger.warning(f"[waveform] Failed to extract audio, using original timestamps")
            return timestamps

        # Analyze waveform for pauses
        pauses = analyze_waveform_for_pauses(
            temp_audio,
            video_duration,
            silence_threshold=silence_threshold,
            min_pause_length=min_pause_length
        )

        # Clean up temporary audio file
        try:
            Path(temp_audio).unlink()
        except:
            pass

        if not pauses:
            logger.warning(f"[waveform] No pauses detected, using original timestamps")
            return timestamps

        # Adjust each timestamp to natural pauses
        adjusted_timestamps = []
        for i, (start, end) in enumerate(timestamps):
            logger.info(f"[waveform] Adjusting cut {i+1}/{len(timestamps)}")
            adjusted_start, adjusted_end = adjust_timestamps_to_natural_pauses(
                start, end, pauses, search_window
            )
            adjusted_timestamps.append((adjusted_start, adjusted_end))

        logger.info(f"[waveform] Successfully adjusted {len(adjusted_timestamps)} cuts")

        return adjusted_timestamps

    except Exception as e:
        logger.error(f"[waveform] Failed to analyze video for natural cuts", extra={"error": str(e)})
        return timestamps


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


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