# Node: `refine_clip_context`

**Source:** `src/nodes/refine_clip_context.py`
**Third stage** of the workflow.

## Purpose

For each clip identified by `identify_moments`, ask the LLM which neighboring transcription segments should be merged in so the final cut does not truncate a thought mid-sentence.

## Algorithm

For each clip:

1. **Locate** clip's segment indices inside `transcriptionSegments` (full or partial overlap).
2. **Gather context** — up to 20 segments before and 20 segments after (`find_surrounding_segments`).
3. **Ask the LLM** which specific segment **indices** (0-based, relative to the before/after lists) to include.

The LLM prompt enforces:
- Respect the user's criteria — do not add segments that break the match.
- Be selective: none, some, or all.
- Respond **only** with JSON.

4. **Apply the decision:**
   - `segments_to_add_before` → new start = `before[max(indices)]["start"]`
   - `segments_to_add_after` → new end = `after[max(indices)]["end"]`
5. Build a new `ClipInfo` preserving `reason` and `matchedCriterion`.

## Parsing

The response is parsed as raw text (not structured output). The code strips ` ```json ` fences and `json.loads` the body. On `JSONDecodeError`, the original clip is kept unchanged (graceful degradation).

## Output

```python
{"clips": [ClipInfo(...), ...], "status": EDITING_VIDEO}
```

## Failure Modes

- No clips → `FAILED`.
- No transcription segments → skips refinement, passes clips through to `EDITING_VIDEO`.
- Per-clip errors are caught; the original clip is preserved.
