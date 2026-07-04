"""Flask blueprint for the videos editing context."""

import asyncio
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint,
    request,
    jsonify,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from src.videos.domain.state import VideoEditState, EditStatus
from src.videos.application.workflow import run_edit_workflow

videos_bp = Blueprint("videos", __name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_videos"
ALLOWED_EXTENSIONS = {"mp4"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024


def _ensure_dirs():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _status_file(session_id: str) -> Path:
    return Path(OUTPUT_FOLDER) / session_id / "status.json"


def _write_status(session_id: str, status: dict) -> None:
    sf = _status_file(session_id)
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(status))


def _read_status(session_id: str) -> dict | None:
    sf = _status_file(session_id)
    if not sf.exists():
        return None
    try:
        return json.loads(sf.read_text())
    except Exception:
        return None


@videos_bp.route("/upload", methods=["POST"])
def upload_video():
    _ensure_dirs()

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only MP4 is allowed"}), 400

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

    session_output_dir = os.path.join(OUTPUT_FOLDER, timestamp)
    os.makedirs(session_output_dir, exist_ok=True)

    _write_status(timestamp, {
        "status": EditStatus.PENDING.value,
        "error": None,
        "output_video": None,
    })

    def process_video_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            initial_state = VideoEditState(
                videoPath=filepath,
                userPrompt=prompt,
                status=EditStatus.PENDING,
            )
            result = loop.run_until_complete(run_edit_workflow(initial_state))

            if result.get("status") == EditStatus.FAILED.value:
                _write_status(timestamp, {
                    "status": "failed",
                    "error": result.get("error", "unknown error"),
                    "output_video": None,
                })
                return

            output_video = result.get("outputVideo")
            if output_video and os.path.exists(output_video):
                target_path = os.path.join(session_output_dir, os.path.basename(output_video))
                try:
                    import shutil
                    shutil.copy(output_video, target_path)
                except Exception:
                    target_path = output_video

                _write_status(timestamp, {
                    "status": "completed",
                    "error": None,
                    "output_video": os.path.basename(target_path),
                })
            else:
                _write_status(timestamp, {
                    "status": "failed",
                    "error": "No output video produced",
                    "output_video": None,
                })

        except Exception as e:
            _write_status(timestamp, {
                "status": "failed",
                "error": str(e),
                "output_video": None,
            })
        finally:
            loop.close()

    thread = threading.Thread(target=process_video_async)
    thread.start()

    return jsonify({
        "message": "Video uploaded successfully! Processing started...",
        "session_id": timestamp,
        "filename": unique_filename,
    })


@videos_bp.route("/status/<session_id>")
def check_status(session_id: str):
    status_data = _read_status(session_id)
    if status_data is None:
        return jsonify({"status": "processing"})

    return jsonify({
        "status": status_data.get("status", "processing"),
        "error": status_data.get("error"),
        "output_video": status_data.get("output_video"),
    })


@videos_bp.route("/result/<session_id>")
def result(session_id: str):
    status_data = _read_status(session_id)
    if status_data is None:
        return jsonify({"error": "Session not found"}), 404

    output_video = status_data.get("output_video")
    if not output_video:
        return jsonify({"error": "No output video available"}), 404

    video_path = os.path.join(OUTPUT_FOLDER, session_id, output_video)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 2)

    return jsonify({
        "session_id": session_id,
        "status": status_data.get("status"),
        "filename": output_video,
        "url": url_for("videos.serve_video", session_id=session_id, filename=output_video),
        "size_mb": size_mb,
    })


@videos_bp.route("/video/<session_id>/<filename>")
def serve_video(session_id: str, filename: str):
    safe_name = secure_filename(filename)
    video_path = os.path.join(OUTPUT_FOLDER, session_id, safe_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    return send_file(video_path, mimetype="video/mp4")
