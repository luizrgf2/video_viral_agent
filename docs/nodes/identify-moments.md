# Node: `identify_moments`

**Source:** `src/nodes/identify_moments.py`
**Second stage** of the workflow.

## Purpose

Feed the timestamped transcription + user criteria into an LLM and get back candidate viral clips. This is where semantic matching happens.

## Input Contract

- `state.transcription` — full text (required; failure if missing).
- `state.transcriptionSegments` — preferred; builds a timestamped prompt.
- `state.analysis` — list of criteria strings.

## Prompt Construction

1. Criteria are enumerated:
   ```
   1. <criterion one>
   2. <criterion two>
   ```
2. Transcription is formatted with `[MM:SS]` prefixes per segment.
3. Hard rules baked into the prompt:
   - Match criteria **exactly**.
   - **Group** adjacent moments into coherent clips.
   - **Minimum 30 seconds** per clip — shorter moments are dropped.
   - If nothing qualifies, return `{"clips": []}`.

## LLM Call

```python
structured_llm = llmModel.with_structured_output(ClipOutput)
response = await structured_llm.ainvoke(messages)
```

Output is parsed into `ClipOutput` → `list[ClipInfo]`. Structured output is enforced by Pydantic, not by JSON parsing.

## Output

```python
{"clips": [ClipInfo(...), ...], "status": REFINING_CONTEXT}
```

## Helper Utilities (also exported)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `parse_timestamp_to_seconds` | `(ts: str) -> float` | `MM:SS` / `HH:MM:SS` → seconds |
| `format_timestamp` | `(sec: float) -> str` | seconds → `MM:SS` |
| `detect_natural_boundaries` | `(segments, pause_threshold=2.0) -> list[int]` | Indices where gap ≥ threshold |
| `expand_clip_with_context` | `(start, end, segments) -> (start, end)` | Heuristic context expansion (unused by the node itself, available for manual use) |

## Failure Modes

- Missing transcription → `FAILED`.
- LLM/structured-output error → `FAILED`.
