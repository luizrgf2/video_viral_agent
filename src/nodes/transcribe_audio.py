import logging
from pathlib import Path
from src.state import VideoAnalysisState, AnalysisStatus
import whisper

logger = logging.getLogger(__name__)

NODE_ID = "transcribe_audio"


async def transcribe_audio_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting audio transcription", extra={"videoPath": state.videoPath})

    try:
        video_path = Path(state.videoPath)

        if not video_path.exists():
            logger.error(f"[{NODE_ID}] Video file not found", extra={"videoPath": state.videoPath})
            return {
                "error": f"Video file not found: {state.videoPath}",
                "status": AnalysisStatus.FAILED
            }

        file_size = video_path.stat().st_size
        logger.info(f"[{NODE_ID}] Video file size", extra={"size_mb": file_size / (1024 * 1024)})

        # Load Whisper model
        model_size = "tiny"  # Options: tiny, base, small, medium, large
        logger.info(f"[{NODE_ID}] Loading Whisper model", extra={"model_size": model_size})

        model = whisper.load_model(model_size)

        # Transcribe audio
        logger.info(f"[{NODE_ID}] Transcribing audio...")

        result = model.transcribe(
            str(video_path),
            language="pt",  # Portuguese by default, can be None for auto-detect
            task="transcribe",
            word_timestamps=True,
            verbose=False
        )

        # Extract transcription with timestamps
        transcription_segments = []

        for segment in result["segments"]:
            transcription_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })

        # Build full transcription text
        full_transcription = result["text"]

        logger.info(f"[{NODE_ID}] Transcription completed", extra={
            "duration": result.get("duration", 0),
            "segments": len(transcription_segments),
            "text_length": len(full_transcription)
        })

        return {
            "transcription": full_transcription,
            "transcriptionSegments": transcription_segments,
            "status": AnalysisStatus.IDENTIFYING_MOMENTS
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Transcription failed", extra={"error": error_message})
        return {
            "error": f"Transcription failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
