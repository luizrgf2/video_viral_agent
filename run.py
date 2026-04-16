#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.state import VideoAnalysisState, AnalysisStatus
from src.workflow import run_workflow
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <video_path> <analysis1> [analysis2] ...")
        sys.exit(1)

    video_path = sys.argv[1]
    analysis = sys.argv[2:]

    load_dotenv()

    initial_state = VideoAnalysisState(
        videoPath=video_path,
        analysis=analysis,
        status=AnalysisStatus.PENDING
    )

    try:
        result = await run_workflow(initial_state)

        if result.status == AnalysisStatus.COMPLETED and result.clips:
            print(f"\n✅ Found {len(result.clips)} viral moments:\n")

            for i, clip in enumerate(result.clips, 1):
                print(f"{i}. {clip.startTime} - {clip.endTime}")
                print(f"   Reason: {clip.reason}")
                print(f"   Matched: {clip.matchedCriterion}")
                print()

        elif result.status == AnalysisStatus.FAILED:
            print(f"\n❌ Analysis failed: {result.error}\n")

        else:
            print("\n⚠️  No viral moments found\n")

        sys.exit(0 if result.status == AnalysisStatus.COMPLETED else 1)

    except Exception as e:
        logger.exception("Unexpected error")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
