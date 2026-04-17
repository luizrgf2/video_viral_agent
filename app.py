"""
Flask Application for Video Viral Agent
Provides web interface for video upload and viral moment extraction
"""

import os
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import video processing modules
from src.workflow import run_workflow
from src.state import VideoAnalysisState, AnalysisStatus

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_clips'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing."""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']

        # Check if filename is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only MP4, MOV, AVI, MKV are allowed'}), 400

        # Get analysis criteria
        analysis_criteria = request.form.get('criteria', '')

        if not analysis_criteria:
            return jsonify({'error': 'Analysis criteria is required'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Process video asynchronously
        def process_video_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                initial_state = VideoAnalysisState(
                    videoPath=filepath,
                    analysis=[analysis_criteria],
                    status=AnalysisStatus.PENDING
                )

                result = loop.run_until_complete(run_workflow(initial_state))

                # Move output clips to a session-specific folder
                if result.get("outputClips"):
                    session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], timestamp)
                    os.makedirs(session_output_dir, exist_ok=True)

                    for clip_path in result["outputClips"]:
                        if os.path.exists(clip_path):
                            clip_name = os.path.basename(clip_path)
                            shutil.copy(clip_path, os.path.join(session_output_dir, clip_name))

                    # Clean up original output folder
                    for clip_path in result.get("outputClips", []):
                        if os.path.exists(clip_path):
                            try:
                                os.remove(clip_path)
                            except:
                                pass

            finally:
                loop.close()

        # Start async processing
        import threading
        thread = threading.Thread(target=process_video_async)
        thread.start()

        return jsonify({
            'message': 'Video uploaded successfully! Processing started...',
            'session_id': timestamp,
            'filename': unique_filename
        })

    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500


@app.route('/status/<session_id>')
def check_status(session_id):
    """Check processing status for a session."""
    try:
        session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)

        if os.path.exists(session_output_dir):
            # Check for video files
            video_files = [f for f in os.listdir(session_output_dir) if f.endswith('.mp4')]

            if video_files:
                return jsonify({
                    'status': 'completed',
                    'clips_count': len(video_files),
                    'clips': video_files
                })
            else:
                return jsonify({'status': 'processing'})
        else:
            return jsonify({'status': 'processing'})

    except Exception as e:
        return jsonify({'error': f'Error checking status: {str(e)}'}), 500


@app.route('/video/<session_id>/<filename>')
def serve_video(session_id, filename):
    """Serve processed video files."""
    try:
        video_path = os.path.join(app.config['OUTPUT_FOLDER'], session_id, filename)

        if os.path.exists(video_path):
            return send_file(video_path, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Video not found'}), 404

    except Exception as e:
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500


@app.route('/clips/<session_id>')
def list_clips(session_id):
    """List all clips for a session."""
    try:
        session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)

        if not os.path.exists(session_output_dir):
            return jsonify({'error': 'Session not found'}), 404

        video_files = []
        for filename in sorted(os.listdir(session_output_dir)):
            if filename.endswith('.mp4'):
                filepath = os.path.join(session_output_dir, filename)
                file_size = os.path.getsize(filepath)
                file_size_mb = round(file_size / (1024 * 1024), 2)

                video_files.append({
                    'filename': filename,
                    'url': url_for('serve_video', session_id=session_id, filename=filename),
                    'size_mb': file_size_mb
                })

        return jsonify({
            'session_id': session_id,
            'clips': video_files,
            'total_count': len(video_files)
        })

    except Exception as e:
        return jsonify({'error': f'Error listing clips: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
