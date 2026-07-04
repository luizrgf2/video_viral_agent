"""Tests for the videos editing pipeline (pure logic, no LLM/FFmpeg calls)."""

from src.videos.nodes.assemble_video import build_time_map, remap_timestamp
from src.videos.nodes.sample_frames import FRAME_BUDGET
from src.videos.nodes.burn_subtitles import filter_relevant_segments
from src.videos.domain.state import TimeRange


class TestBuildTimeMap:
    def test_three_segments_accumulate_correctly(self):
        kept = [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]
        tm = build_time_map(kept)

        assert len(tm) == 3
        assert tm[0].original_start == 0.0
        assert tm[0].final_start == 0.0
        assert tm[0].final_end == 10.0

        assert tm[1].original_start == 20.0
        assert tm[1].final_start == 10.0
        assert tm[1].final_end == 20.0

        assert tm[2].original_start == 40.0
        assert tm[2].final_start == 20.0
        assert tm[2].final_end == 30.0

    def test_zero_duration_segments_are_skipped(self):
        kept = [(0.0, 5.0), (5.0, 5.0), (10.0, 15.0)]
        tm = build_time_map(kept)
        assert len(tm) == 2

    def test_empty_input_returns_empty(self):
        assert build_time_map([]) == []


class TestRemapTimestamp:
    def setup_method(self):
        self.tm = build_time_map([(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)])

    def test_at_boundaries(self):
        assert remap_timestamp(self.tm, 0.0) == 0.0
        assert remap_timestamp(self.tm, 10.0) == 10.0
        assert remap_timestamp(self.tm, 20.0) == 10.0
        assert remap_timestamp(self.tm, 50.0) == 30.0

    def test_middle_of_ranges(self):
        assert remap_timestamp(self.tm, 5.0) == 5.0
        assert remap_timestamp(self.tm, 25.0) == 15.0
        assert remap_timestamp(self.tm, 45.0) == 25.0

    def test_in_removed_range_returns_negative(self):
        assert remap_timestamp(self.tm, 15.0) == -1.0
        assert remap_timestamp(self.tm, 35.0) == -1.0
        assert remap_timestamp(self.tm, 60.0) == -1.0

    def test_empty_time_map(self):
        assert remap_timestamp([], 5.0) == -1.0


class TestFilterRelevantSegments:
    def test_keeps_segments_inside_kept_ranges(self):
        time_map = build_time_map([(0.0, 10.0), (20.0, 30.0)])
        segments = [
            {"start": 0.0, "end": 5.0, "text": "first"},
            {"start": 5.0, "end": 8.0, "text": "second"},
            {"start": 22.0, "end": 28.0, "text": "third"},
        ]

        out = filter_relevant_segments(segments, time_map)

        assert len(out) == 3
        assert out[0]["text"] == "first"
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 5.0
        assert out[2]["text"] == "third"
        # [22,28] in original range (20,30) which maps to final (10,20):
        # ratio of 22 = (22-20)/10 = 0.2 → 10 + 0.2*10 = 12.0
        # ratio of 28 = (28-20)/10 = 0.8 → 10 + 0.8*10 = 18.0
        assert out[2]["start"] == 12.0
        assert out[2]["end"] == 18.0

    def test_drops_segments_in_removed_ranges(self):
        time_map = build_time_map([(0.0, 10.0), (20.0, 30.0)])
        segments = [
            {"start": 12.0, "end": 18.0, "text": "removed"},
        ]

        out = filter_relevant_segments(segments, time_map)
        assert len(out) == 0


class TestFrameBudget:
    def test_budget_constant(self):
        assert FRAME_BUDGET == 150
