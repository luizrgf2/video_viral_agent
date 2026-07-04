# Node: `edit_video`

**Source:** `src/nodes/edit_video.py`
**Fourth stage** of the workflow.

## Purpose

Cut the source video into individual clip files, aligning cut boundaries to **natural pauses** detected in the audio waveform so clips never start/end mid-word.

## Pipeline Inside the Node

1. Parse each `ClipInfo.startTime`/`endTime` into seconds.
2. Call `analyze_video_for_natural_cuts()` (see `integrations/ffmpeg-waveform.md`) with:
   - `silence_threshold=0.02`
   - `min_pause_length=0.15` (150 ms)
   - `search_window=2.0` (±2 s)
3. Load the source video once with `VideoFileClip`.
4. For each (clip, adjusted_timestamps) pair:
   - Validate `start < end`.
   - Clamp `end` to `video_clip.duration`.
   - `subclip = video_clip.subclipped(start, end)` (MoviePy 2.0 API).
   - Write to `output_clips/clip_NNN_<start>-<end>.mp4`.
   - Close the subclip to free memory.
5. Close the source video.

## MoviePy 2.0 API Notes

- Uses `subclipped()` (not `subclip()`) — MoviePy 2.x renamed the method.
- Uses `write_videofile()` with `logger=None` to suppress stdout.

## Output Directory

`{video_path.parent}/output_clips/` — created if missing.

Filename pattern: `clip_001_30-45.mp4` (zero-padded index + integer second offsets).

## Output

```python
{"outputClips": ["/path/to/clip_001_...mp4", ...], "status": COMPLETED}
```

> Note: this node sets `COMPLETED` even though subtitles have not yet been added. The `add_subtitles` node runs afterwards and re-emits `COMPLETED`.

## Failure Modes

- No clips → returns empty `outputClips`, `COMPLETED`.
- Missing video → `FAILED`.
- Per-clip errors are logged and skipped; remaining clips still render.
