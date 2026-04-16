import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.state import VideoAnalysisState, AnalysisStatus
from src.workflow import run_workflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level))


def print_results(state: VideoAnalysisState):
    if state.status == AnalysisStatus.COMPLETED and state.clips:
        print(f"\n✅ Found {len(state.clips)} viral moments:\n")

        for i, clip in enumerate(state.clips, 1):
            print(f"{i}. {clip.startTime} - {clip.endTime}")
            print(f"   Reason: {clip.reason}")
            print(f"   Matched: {clip.matchedCriterion}")
            print()

    elif state.status == AnalysisStatus.FAILED:
        print(f"\n❌ Analysis failed: {state.error}\n")

    else:
        print("\n⚠️  No viral moments found\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Identify viral moments in videos using AI"
    )
    parser.add_argument(
        "video_path",
        help="Path to MP4 video file"
    )
    parser.add_argument(
        "analysis",
        nargs="+",
        help="Analysis criteria (e.g., 'identify funny moments' 'identify impactful moments')"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)

    if not video_path.exists():
        print(f"❌ Error: Video file not found: {args.video_path}")
        sys.exit(1)

    if not video_path.suffix == ".mp4":
        print(f"❌ Error: Video file must be in MP4 format")
        sys.exit(1)

    load_dotenv()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable is required")
        sys.exit(1)

    setup_logging()

    initial_state = VideoAnalysisState(
        videoPath=str(video_path),
        analysis=args.analysis,
        status=AnalysisStatus.PENDING
    )

    try:
        result = await run_workflow(initial_state)
        print_results(result)

        if args.output and result.clips:
            output_path = Path(args.output)
            output_path.write_text(
                json.dumps(
                    [clip.model_dump() for clip in result.clips],
                    indent=2
                )
            )
            print(f"💾 Results saved to: {args.output}\n")

        sys.exit(0 if result.status == AnalysisStatus.COMPLETED else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.exception("Unexpected error")
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
