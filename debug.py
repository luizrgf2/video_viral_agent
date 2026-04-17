import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.state import VideoAnalysisState, AnalysisStatus
from src.workflow import run_workflow
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def debug_main():
    video_path = "/home/luiz/Downloads/video_to_test.mp4"
    analysis = ["Capture somente partes onde ele cita desenvolvedores plenos e somente isso, se não citar desenvolvedor pleno eu não quero"]

    load_dotenv()

    initial_state = VideoAnalysisState(
        videoPath=video_path,
        analysis=analysis,
        status=AnalysisStatus.PENDING
    )

    try:
        logger.info("Starting DEBUG workflow execution")
        logger.info(f"Video: {video_path}")
        logger.info(f"Analysis: {analysis}")

        result = await run_workflow(initial_state)

        # Show transcription if available
        if result.get("transcription"):
            print(f"\n📝 Transcription completed:")
            print(f"   {len(result["transcription"])} characters")
            print(f"   {len(result.get("transcriptionSegments", []))} segments")
            print()

        if result["status"] == AnalysisStatus.COMPLETED and result["clips"]:
            print(f"\n✅ Selected {len(result["clips"])} best viral moments:\n")

            for i, clip in enumerate(result["clips"], 1):
                print(f"{i}. {clip.startTime} - {clip.endTime}")
                print(f"   Reason: {clip.reason}")
                print(f"   Matched: {clip.matchedCriterion}")
                print()

            # Show generated clips
            if result.get("outputClips"):
                print(f"\n🎬 Generated {len(result["outputClips"])} video clips:\n")
                for clip_path in result["outputClips"]:
                    print(f"   📹 {clip_path}")
                print()

        elif result["status"] == AnalysisStatus.FAILED:
            print(f"\n❌ Analysis failed: {result["error"]}\n")

        else:
            print("\n⚠️  No viral moments found\n")

        return result

    except Exception as e:
        logger.exception("Unexpected error")
        raise


if __name__ == "__main__":
    asyncio.run(debug_main())
