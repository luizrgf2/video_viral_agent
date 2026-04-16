#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "❌ Error: .env file not found"
    echo "Copy .env.example to .env and add your OPENROUTER_API_KEY"
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh <video_path> <analysis1> [analysis2] ..."
    exit 1
fi

VIDEO_PATH="$1"
shift
ANALYSIS="$@"

cd "$SCRIPT_DIR"

/home/luiz/.local/bin/uv run python run.py "$VIDEO_PATH" $ANALYSIS
