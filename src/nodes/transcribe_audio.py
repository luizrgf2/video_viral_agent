import logging
import os
import tempfile
from pathlib import Path
from src.state import VideoAnalysisState, AnalysisStatus
from faster_whisper import WhisperModel
from groq import Groq
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

NODE_ID = "transcribe_audio"


def get_transcription_mode() -> str:
    """Get the transcription mode from environment variable."""
    return os.getenv("AUDIO_TRANSCRIPTION_MODE", "local").lower()


async def transcribe_with_faster_whisper(video_path: Path) -> dict:
    """Transcribe audio using faster-whisper (local mode)."""
    logger.info(f"[{NODE_ID}] Using faster-whisper (local mode)", extra={"videoPath": str(video_path)})

    try:
        # Load faster-whisper model with CTranslate2 optimization
        model_size = "tiny"  # Options: tiny, base, small, medium, large
        device = "cpu"  # Options: cpu, cuda
        compute_type = "int8"  # Options: int8, float16, float32

        logger.info(f"[{NODE_ID}] Loading faster-whisper model", extra={
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type
        })

        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        # Transcribe audio with word timestamps
        logger.info(f"[{NODE_ID}] Transcribing audio with faster-whisper...")

        segments_gen, info = model.transcribe(
            str(video_path),
            language="pt",  # Portuguese by default, can be None for auto-detect
            task="transcribe",
            word_timestamps=True,
            vad_filter=True  # Voice Activity Detection for better accuracy
        )

        # Extract transcription with timestamps
        transcription_segments = []
        full_transcription_parts = []

        for segment in segments_gen:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            transcription_segments.append(segment_data)
            full_transcription_parts.append(segment.text.strip())

        # Build full transcription text
        full_transcription = " ".join(full_transcription_parts)

        logger.info(f"[{NODE_ID}] Transcription completed with faster-whisper", extra={
            "duration": info.duration,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": len(transcription_segments),
            "text_length": len(full_transcription)
        })

        return {
            "transcription": full_transcription,
            "transcriptionSegments": transcription_segments,
            "mode": "local",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] faster-whisper transcription failed", extra={"error": error_message})
        raise


async def transcribe_with_groq(video_path: Path) -> dict:
    """Transcribe audio using Groq API (cloud mode)."""
    logger.info(f"[{NODE_ID}] Using Groq API (cloud mode)", extra={"videoPath": str(video_path)})

    try:
        # Get Groq API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        # Get model name from env or use default
        model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

        logger.info(f"[{NODE_ID}] Initializing Groq client", extra={"model": model})

        client = Groq(api_key=api_key)

        # Open the video file
        logger.info(f"[{NODE_ID}] Transcribing audio with Groq API...")

        with open(video_path, "rb") as file:
            # Create transcription with segments and word timestamps
            transcription = client.audio.transcriptions.create(
                file=file,
                model=model,
                response_format="verbose_json",
                timestamp_granularities=["segment"],  # Get segment-level timestamps
                language="pt",  # Portuguese by default
                temperature=0.0  # Deterministic output
            )

        # Extract segments from Groq response
        transcription_segments = []
        full_transcription_parts = []

        if hasattr(transcription, 'segments') and transcription.segments:
            for segment in transcription.segments:
                segment_data = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                }
                transcription_segments.append(segment_data)
                full_transcription_parts.append(segment["text"].strip())
        else:
            # Fallback if no segments (shouldn't happen with verbose_json)
            logger.warning(f"[{NODE_ID}] No segments in Groq response, using full text only")
            full_transcription_parts = [transcription.text]
            transcription_segments = [{
                "start": 0.0,
                "end": transcription.duration or 0.0,
                "text": transcription.text.strip()
            }]

        # Build full transcription text
        full_transcription = " ".join(full_transcription_parts)

        logger.info(f"[{NODE_ID}] Transcription completed with Groq API", extra={
            "model": model,
            "segments": len(transcription_segments),
            "text_length": len(full_transcription)
        })

        return {
            "transcription": full_transcription,
            "transcriptionSegments": transcription_segments,
            "mode": "groq",
            "model": model,
            "duration": transcription.duration if hasattr(transcription, 'duration') else 0
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Groq transcription failed", extra={"error": error_message})
        raise


async def transcribe_audio_node(state: VideoAnalysisState) -> dict:
    """Main transcription node that routes to appropriate service based on configuration."""
    try:
        video_path = Path(state.videoPath)

        if not video_path.exists():
            logger.error(f"[{NODE_ID}] Video file not found", extra={"videoPath": state.videoPath})
            return {
                "error": f"Video file not found: {state.videoPath}",
                "status": AnalysisStatus.FAILED
            }

        file_size = video_path.stat().st_size
        logger.info(f"[{NODE_ID}] Starting audio transcription", extra={
            "videoPath": state.videoPath,
            "size_mb": file_size / (1024 * 1024)
        })

        # Get transcription mode from environment
        mode = get_transcription_mode()
        logger.info(f"[{NODE_ID}] Transcription mode: {mode}", extra={"mode": mode})

        # Route to appropriate transcription service
        if mode == "groq":
            result = await transcribe_with_groq(video_path)
        else:  # Default to local
            result = await transcribe_with_faster_whisper(video_path)

        # Return success with transcription data
        return {
            **result,
            "status": AnalysisStatus.IDENTIFYING_MOMENTS
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Transcription failed", extra={"error": error_message})
        return {
            "error": f"Transcription failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
