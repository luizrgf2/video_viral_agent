# Integration: FFmpeg + Waveform Analysis

Two distinct uses of FFmpeg in this project:

1. **Audio extraction & waveform analysis** (pre-cut) — `src/utils/waveform_analyzer.py`
2. **Subtitle burning** (post-cut) — `src/nodes/add_subtitles.py`

---

## 1. Waveform Analyzer

**Source:** `src/utils/waveform_analyzer.py`

### `analyze_video_for_natural_cuts(video_path, timestamps, ...)`

Main entry point called by `edit_video_node`. Pipeline:

1. `extract_audio_from_video()` — MoviePy writes a temp WAV.
2. `analyze_waveform_for_pauses()` — `librosa` computes RMS energy per frame; consecutive low-energy frames ≥ `min_pause_length` are treated as pauses.
3. `adjust_timestamps_to_natural_pauses()` — for each cut, find the nearest pause within `search_window` and snap to it.

### Parameters (as called by `edit_video`)

| Param | Value | Meaning |
|-------|-------|---------|
| `silence_threshold` | `0.02` | RMS below this = silence |
| `min_pause_length` | `0.15` s | Min pause to be a cut candidate |
| `search_window` | `2.0` s | Max distance to search from original cut |
| `frame_length` | `2048` | FFT window (default) |
| `hop_length` | `512` | FFT step (default) |

### Safety Guarantees

- Adjusted clip is rejected if duration drops below **1 s** → original times reused.
- Adjusted cut is rejected if it drifts more than **3 s** from original → original reused.
- All errors degrade gracefully to original timestamps.

### Cleanup

Temp WAV is deleted in a `try/except` (failures are swallowed).

---

## 2. FFmpeg Subtitle Engine

**Source:** `src/nodes/add_subtitles.py` — see `nodes/add-subtitles.md` for the full filter spec.

### Binary Check

`check_ffmpeg()` uses `shutil.which("ffmpeg")`. If FFmpeg is missing the node falls back to copying the unsubtitled clip.

### ffprobe

Used to read video dimensions (`width,height`) before building `drawtext` filters. Falls back to `1920×1080`.

### FFmpeg Requirements

- `ffmpeg` and `ffprobe` on `$PATH`.
- `libx264` encoder available.
- No ImageMagick dependency (subtitles use `drawtext`, not MoviePy `TextClip`).
