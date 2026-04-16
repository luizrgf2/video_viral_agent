# Video Viral Agent

Multi-agent system that identifies viral moments in videos using VLM and LLM.

## Features

- 🎬 **Direct Video Analysis**: VLM processes video + audio together (no separate transcription)
- 🤖 **Multi-Agent Architecture**: LangGraph workflow with specialized agents
- 🎯 **Custom Analysis Criteria**: Define what kind of moments to identify
- 📊 **Precise Timestamps**: Get exact cut points for each viral moment
- 🔒 **Type-Safe**: Built with Python 3.12 and Pydantic
- 🌐 **OpenRouter Integration**: Use any model via OpenRouter

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with pip (development)
pip install -e ".[dev]"

# Or with UV (recommended)
pip install uv
uv pip install -e .
```

## Configuration

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
VLM_MODEL_NAME=anthropic/claude-3.5-sonnet
LLM_MODEL_NAME=openai/gpt-4o
SITE_URL=http://localhost:8000
APP_NAME=Video Viral Agent
LOG_LEVEL=INFO
```

## Usage

### Basic Usage

```bash
python src/main.py video.mp4 "identify funny moments" "identify impactful moments"
```

### Save Results to JSON

```bash
python src/main.py video.mp4 "identify funny moments" -o results.json
```

### Multiple Analysis Criteria

```bash
python src/main.py video.mp4 \
  "identify the most impactful moments" \
  "identify funny moments" \
  "find emotional peaks" \
  "locate audience reactions"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐                                        │
│  │  Analyze Video  │ -> Output: Timestamped description     │
│  │      (VLM)       │                                        │
│  └─────────────────┘                                        │
│         │                                                    │
│         v                                                    │
│  ┌─────────────────┐                                        │
│  │ Identify Moments│ -> Output: Clips with timestamps     │
│  │      (LLM)       │                                        │
│  └─────────────────┘                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Supported Models

Via OpenRouter, you can use:
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **OpenAI**: GPT-4o, GPT-4 Turbo
- **Google**: Gemini 2.0 Flash, Gemini Pro
- **Meta**: Llama 3.1 70B
- And many more...

## License

See LICENSE file for details.
