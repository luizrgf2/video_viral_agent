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


def extract_and_compress_audio(
    video_path: Path,
    output_format: str = "mp3",
    sample_rate: int = 16000,
    bitrate: str = "32k",
    n_channels: int = 1
) -> tempfile._TemporaryFileWrapper:
    """
    Extract and compress audio from video for optimized transcription.

    Args:
        video_path: Path to video file
        output_format: Output audio format (mp3, wav, etc.)
        sample_rate: Sample rate in Hz (16000 is sufficient for speech)
        bitrate: Audio bitrate (32k is good for speech transcription)
        n_channels: Number of audio channels (1 = mono is sufficient for speech)

    Returns:
        Temporary file object with compressed audio
    """
    try:
        logger.info(f"[{NODE_ID}] Extracting and compressing audio", extra={
            "video_path": str(video_path),
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "channels": n_channels
        })

        # Load video and extract audio
        video_clip = VideoFileClip(str(video_path))
        audio_clip = video_clip.audio

        if audio_clip is None:
            logger.error(f"[{NODE_ID}] No audio track found in video")
            video_clip.close()
            raise ValueError("No audio track found in video")

        # Create temporary file for compressed audio
        temp_audio = tempfile.NamedTemporaryFile(
            suffix=f".{output_format}",
            delete=False
        )
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # Write compressed audio using MoviePy
        # Parameters optimized for speech transcription:
        # - 16kHz sample rate (Whisper uses 16kHz)
        # - Mono audio (speech doesn't need stereo)
        # - 32kbps bitrate (sufficient for speech intelligibility)
        audio_clip.write_audiofile(
            temp_audio_path,
            fps=sample_rate,
            bitrate=bitrate,
            codec="libmp3lame" if output_format == "mp3" else "pcm_s16le",
            logger=None
        )

        # Get file sizes for logging
        original_size = video_path.stat().st_size / (1024 * 1024)
        compressed_size = Path(temp_audio_path).stat().st_size / (1024 * 1024)
        compression_ratio = (original_size / compressed_size) if compressed_size > 0 else 0

        logger.info(f"[{NODE_ID}] Audio compression completed", extra={
            "original_mb": f"{original_size:.2f}",
            "compressed_mb": f"{compressed_size:.2f}",
            "compression_ratio": f"{compression_ratio:.1f}x",
            "reduction_mb": f"{original_size - compressed_size:.2f}"
        })

        # Close clips
        audio_clip.close()
        video_clip.close()

        return temp_audio_path

    except Exception as e:
        logger.error(f"[{NODE_ID}] Failed to extract and compress audio", extra={"error": str(e)})
        raise


def optimize_audio_for_transcription(video_path: Path) -> Path:
    """
    Optimize audio file for transcription by compressing and converting to optimal format.

    This function extracts audio from video and compresses it significantly
    while maintaining quality sufficient for accurate speech transcription.

    Args:
        video_path: Path to video file

    Returns:
        Path to optimized audio file (temporary, should be cleaned up after use)
    """
    return extract_and_compress_audio(
        video_path,
        output_format="mp3",   # MP3 for efficient compression
        sample_rate=16000,     # 16kHz (Whisper uses 16kHz)
        bitrate="32k",         # 32kbps (sufficient for speech)
        n_channels=1           # Mono (speech doesn't need stereo)
    )


def cleanup_temp_audio(audio_path: Path) -> None:
    """Clean up temporary audio file."""
    try:
        if audio_path and audio_path.exists():
            audio_path.unlink()
            logger.debug(f"[{NODE_ID}] Cleaned up temporary audio: {audio_path}")
    except Exception as e:
        logger.warning(f"[{NODE_ID}] Failed to cleanup temp audio: {e}")


async def transcribe_with_faster_whisper(video_path: Path) -> dict:
    """Transcribe audio using faster-whisper (local mode)."""
    logger.info(f"[{NODE_ID}] Using faster-whisper (local mode)", extra={"videoPath": str(video_path)})

    optimized_audio = None
    try:
        # Optimize audio before transcription
        logger.info(f"[{NODE_ID}] Optimizing audio for transcription...")
        optimized_audio = optimize_audio_for_transcription(video_path)

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

        # Transcribe optimized audio with word timestamps
        logger.info(f"[{NODE_ID}] Transcribing audio with faster-whisper...")

        segments_gen, info = model.transcribe(
            str(optimized_audio),
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
    finally:
        # Clean up optimized audio file
        if optimized_audio:
            cleanup_temp_audio(Path(optimized_audio))


async def transcribe_with_groq(video_path: Path) -> dict:
    """Transcribe audio using Groq API (cloud mode)."""
    logger.info(f"[{NODE_ID}] Using Groq API (cloud mode)", extra={"videoPath": str(video_path)})

    optimized_audio = None
    try:
        # Get Groq API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        # Get model name from env or use default
        model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

        # Optimize audio before sending to Groq (important for 25MB limit!)
        logger.info(f"[{NODE_ID}] Optimizing audio for Groq API (25MB limit)...")
        optimized_audio = optimize_audio_for_transcription(video_path)

        # Check optimized file size
        optimized_size = Path(optimized_audio).stat().st_size / (1024 * 1024)
        logger.info(f"[{NODE_ID}] Optimized audio size: {optimized_size:.2f}MB", extra={
            "size_mb": f"{optimized_size:.2f}"
        })

        if optimized_size > 25:
            logger.warning(f"[{NODE_ID}] Optimized audio still exceeds 25MB limit", extra={
                "size_mb": f"{optimized_size:.2f}"
            })

        logger.info(f"[{NODE_ID}] Initializing Groq client", extra={"model": model})

        client = Groq(api_key=api_key)

        # Open the optimized audio file
        logger.info(f"[{NODE_ID}] Transcribing audio with Groq API...")

        with open(optimized_audio, "rb") as file:
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
    finally:
        # Clean up optimized audio file
        if optimized_audio:
            cleanup_temp_audio(Path(optimized_audio))


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
