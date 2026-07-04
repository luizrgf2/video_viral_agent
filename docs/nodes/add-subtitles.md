# Node: `add_subtitles`

**Source:** `src/nodes/add_subtitles.py`
**Final stage** of the workflow.

## Purpose

Burn synchronized subtitles into each clip produced by `edit_video`, using FFmpeg `drawtext` filters. Subtitles come from the original `transcriptionSegments`, re-timed to be relative to each clip.

## Per-Clip Algorithm

For each `outputClips[i]`:

1. Resolve the clip's absolute start/end in the source video from `clips[i]` timestamps. Fallback: probe the file with MoviePy to read `duration`.
2. **Filter** `transcriptionSegments` to those overlapping `[clip_start, clip_end]`.
3. **Translate** each segment's absolute times into clip-relative times:
   - `relative_start = seg_start - clip_start`
   - `relative_end   = seg_end   - clip_start`
4. Probe video dimensions with `ffprobe` (fallback `1920×1080`).
5. Build one `drawtext` filter per segment.
6. Concatenate filters with `,` and run FFmpeg once per clip.

## drawtext Filter

Hardcoded defaults (override in `create_ffmpeg_drawtext_filter`):

| Option | Value |
|--------|-------|
| `fontsize` | **14** |
| `fontcolor` | `white` |
| `boxcolor` | `black@0.7` |
| `boxborderw` | `2` |
| `line_spacing` | `1` |
| position | bottom (`y = height * 0.85`) |
| `enable` | `between(t,start,end)` |

### Text Processing

- `wrap_text_for_subtitle(text, max_chars_per_line=30)` — word-boundary wrapping.
- Total text truncated to **120 chars** with `...` suffix.
- `escape_text_for_ffmpeg()` escapes `: \ = ' %` while preserving literal `\n` line breaks.

## FFmpeg Command

```
ffmpeg -i <clip> -vf <filter1,filter2,...> \
  -c:a copy -c:v libx264 -preset fast -crf 23 <out> -y
```

- Audio is **copied** (no re-encode).
- Video is re-encoded to H.264 (fast preset, CRF 23).

## Output Directory

`{video_path.parent}/output_clips_subtitled/`

Filename: `subtitled_<original_clip_name>`.

On any failure the node **falls back to copying** the unsubtitled clip so the user always gets output. The returned `outputClips` list is **overwritten** to point at the subtitled versions:

```python
{
  "subtitledClips": [...],
  "outputClips": subtitled_clips,   # <-- overwrites prior list
  "status": COMPLETED,
}
```

## Failure Modes

- No output clips → empty list, `COMPLETED`.
- No transcription segments → returns original clips as `subtitledClips`.
- FFmpeg missing → copies clips unsubtitled.
- FFmpeg nonzero exit → copies clips unsubtitled.
