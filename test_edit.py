"""Test script for video editing node."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.state import VideoAnalysisState, ClipInfo, AnalysisStatus

# Import edit_video node directly to avoid importing all nodes
import importlib.util
spec = importlib.util.spec_from_file_location("edit_video", "src/nodes/edit_video.py")
edit_video_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edit_video_module)

edit_video_node = edit_video_module.edit_video_node
parse_timestamp = edit_video_module.parse_timestamp

# Load environment variables (not needed for edit node, but good practice)
load_dotenv()


async def test_timestamp_parsing():
    """Test timestamp parsing."""
    print("Testing timestamp parsing...")

    # Test MM:SS format
    assert parse_timestamp("01:30") == 90.0, "Failed to parse MM:SS"
    assert parse_timestamp("00:45") == 45.0, "Failed to parse MM:SS"

    # Test HH:MM:SS format
    assert parse_timestamp("01:30:45") == 5445.0, "Failed to parse HH:MM:SS"
    assert parse_timestamp("00:02:30") == 150.0, "Failed to parse HH:MM:SS"

    print("✅ Timestamp parsing tests passed!")


async def test_edit_node():
    """Test edit_video_node with sample data."""
    print("\nTesting edit_video_node...")

    # Create test state with sample clips
    test_state = VideoAnalysisState(
        videoPath="/home/luiz/Downloads/videoplayback.1776280508826.publer.com.mp4",
        analysis=["test"],
        clips=[
            ClipInfo(
                startTime="00:00",
                endTime="00:10",
                reason="Test clip 1",
                matchedCriterion="test"
            ),
            ClipInfo(
                startTime="00:30",
                endTime="00:45",
                reason="Test clip 2",
                matchedCriterion="test"
            ),
            ClipInfo(
                startTime="01:00",
                endTime="01:15",
                reason="Test clip 3",
                matchedCriterion="test"
            )
        ],
        status=AnalysisStatus.EDITING_VIDEO
    )

    # Run the edit node
    result = await edit_video_node(test_state)

    # Check results
    if result["status"] == AnalysisStatus.COMPLETED:
        print(f"✅ Edit node completed successfully!")
        print(f"   Generated {len(result['outputClips'])} clips:")

        for clip_path in result["outputClips"]:
            clip_file = Path(clip_path)
            size_mb = clip_file.stat().st_size / (1024 * 1024)
            print(f"   - {clip_path} ({size_mb:.2f} MB)")
    else:
        print(f"❌ Edit node failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(test_timestamp_parsing())
    asyncio.run(test_edit_node())
