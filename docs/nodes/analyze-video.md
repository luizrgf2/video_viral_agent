# Node: `analyze_video` (NOT WIRED)

**Source:** `src/nodes/analyze_video.py`

> ⚠️ This node **exists but is not connected** to the main workflow graph in `src/workflow.py`. It is retained for future use when visual (VLM) frame analysis is desired.

## Intended Purpose

Base64-encode the entire video and send it to a vision LLM to produce a timestamped visual+audio description. The description would then feed into moment identification.

## API

- Uses `vlmModel` (default `anthropic/claude-3.5-sonnet` via OpenRouter).
- Sends a `data:video/mp4;base64,...` payload as an `image_url` content part.
- Warns if the file exceeds 25 MB.

## Why It Is Disabled

1. Most clips are long enough that base64 encoding blows past provider context/size limits.
2. The transcription-only path produces better timestamp precision for the current use cases.

To re-enable: add `workflow.add_node("analyze_video", analyze_video_node)` and insert it between `transcribe_audio` and `identify_moments`, then read `state.videoDescription` inside `identify_moments`.
