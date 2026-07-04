# State Model (`VideoAnalysisState`)

**Source:** `src/state.py`

The entire pipeline shares a single Pydantic state object. LangGraph merges each node's return dict back into this model.

## Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `videoPath` | `str` | ✅ | Path to source `.mp4` file |
| `analysis` | `List[str]` | ✅ | User-supplied analysis criteria (1–10 items) |
| `videoDescription` | `Optional[str]` | — | VLM-generated description (unused in current pipeline) |
| `transcription` | `Optional[str]` | — | Full transcription text |
| `transcriptionSegments` | `Optional[List[dict]]` | — | Segment list, each `{start, end, text}` |
| `candidateMoments` | `Optional[List[dict]]` | — | Reserved for deeper analysis (currently unused) |
| `clips` | `Optional[List[ClipInfo]]` | — | Identified viral moments |
| `outputClips` | `Optional[List[str]]` | — | Filesystem paths to rendered clips |
| `subtitledClips` | `Optional[List[str]]` | — | Paths to clips with burned-in subtitles |
| `error` | `Optional[str]` | — | Error message if status is `FAILED` |
| `status` | `AnalysisStatus` | ✅ (default `PENDING`) | Current pipeline status |

## `ClipInfo` Sub-model

```python
class ClipInfo(BaseModel):
    startTime: str          # "MM:SS" or "HH:MM:SS"
    endTime: str            # same format
    reason: str             # why this moment matches
    matchedCriterion: str   # which analysis criterion it matched
```

## Validators

- `videoPath` **must** end with `.mp4` — other formats are rejected at the state boundary.
- `analysis` enforces: non-empty list, max 10 criteria, no empty strings.

> **Note:** The Flask web layer (`app.py`) accepts `mp4|mov|avi|mkv` for upload but converts nothing; downstream nodes assume MP4. Non-MP4 uploads will fail the state validator.

## `AnalysisStatus` Lifecycle

```
PENDING
  → ANALYZING_VIDEO (reserved — analyze_video node not wired into main graph)
  → IDENTIFYING_MOMENTS   (set by transcribe_audio)
  → REFINING_CONTEXT      (set by identify_moments)
  → EDITING_VIDEO         (set by refine_clip_context)
  → COMPLETED             (set by edit_video / add_subtitles)
  → FAILED                (set by any node on exception)
  → ADDING_SUBTITLES      (declared but not actively set in current code)
```

Note: `ADDING_SUBTITLES` is declared in the enum but never assigned — `edit_video` jumps straight to `COMPLETED`, then `add_subtitles` also returns `COMPLETED`.
