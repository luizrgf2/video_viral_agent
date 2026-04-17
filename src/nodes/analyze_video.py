import base64
import logging
import os
from pathlib import Path
from src.state import VideoAnalysisState, AnalysisStatus
from src.agents import vlmModel, VIDEO_ANALYSIS_SYSTEM_PROMPT
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

NODE_ID = "analyze_video"


async def analyze_video_node(state: VideoAnalysisState) -> dict:
    logger.info(f"[{NODE_ID}] Starting video analysis", extra={"videoPath": state.videoPath})

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

        if file_size > 25 * 1024 * 1024:
            logger.warning(f"[{NODE_ID}] Video file is large, may exceed API limits", extra={"size_mb": file_size / (1024 * 1024)})

        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')

        criteria_text = "\n".join([f"{i+1}. {criterion}" for i, criterion in enumerate(state.analysis)])

        prompt = f"""Analyze this video and identify the most impactful moments.

        Analysis Criteria:
        {criteria_text}

        Instructions:
        {VIDEO_ANALYSIS_SYSTEM_PROMPT}

        Provide timestamped descriptions highlighting moments with high viral potential."""

        messages = [
            {"role": "system", "content": VIDEO_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    }
                ]
            }
        ]

        logger.info(f"[{NODE_ID}] Calling VLM", extra={"model": vlmModel.model_name})

        response = await vlmModel.ainvoke(messages)

        video_description = response.content

        logger.info(f"[{NODE_ID}] Video analysis completed", extra={
            "description_length": len(video_description)
        })

        return {
            "videoDescription": video_description,
            "status": AnalysisStatus.IDENTIFYING_MOMENTS
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Video analysis failed", extra={"error": error_message})
        return {
            "error": f"Video analysis failed: {error_message}",
            "status": AnalysisStatus.FAILED
        }
