# Flask Web Application

**Source:** `app.py`
**Templates:** `templates/index.html`

A minimal Flask server that accepts video uploads, kicks off the workflow on a background thread, and exposes polling endpoints for the UI.

## Configuration

| Key | Value |
|-----|-------|
| `MAX_CONTENT_LENGTH` | 500 MB |
| `UPLOAD_FOLDER` | `uploads/` |
| `OUTPUT_FOLDER` | `output_clips/` |
| `ALLOWED_EXTENSIONS` | `mp4`, `mov`, `avi`, `mkv` |

> ⚠️ Although `mov|avi|mkv` are accepted for upload, `VideoAnalysisState` rejects anything not ending in `.mp4`. Non-MP4 uploads will save to disk but fail when the workflow starts.

## Endpoints

### `GET /`
Renders `index.html`.

### `POST /upload`
- **Form fields:** `video` (file), `criteria` (string).
- Saves file as `uploads/<YYYYMMDD_HHMMSS>_<filename>`.
- Starts a `threading.Thread` that creates a private asyncio loop and runs `run_workflow`.
- Moves finished clips into `output_clips/<timestamp>/`.
- **Returns:** `{message, session_id: <timestamp>, filename}`.

### `GET /status/<session_id>`
- Looks for `output_clips/<session_id>/`.
- Returns `completed` with clip list once any `.mp4` exists, else `processing`.

### `GET /clips/<session_id>`
Returns JSON list of clips (filename, URL, size MB) for the session.

### `GET /video/<session_id>/<filename>`
Serves the raw MP4 file.

## Concurrency Model

Each request spawns its own thread + its own event loop. There is no shared in-memory state — status is inferred purely from filesystem presence. This means:

- No progress reporting beyond `processing` / `completed`.
- No way to surface errors to the client (a failed run simply never produces files, so the UI polls forever).
- Restarting the server loses all in-flight sessions.
