---
title: Video Viral Moments Detector - Python Multi-Agent System
status: draft
created: 2026-04-15
author: spec-writer
tags: [langgraph, langchain, vlm, llm, video-analysis, multi-agent, python, openrouter]
---

# Video Viral Moments Detector - Python Multi-Agent System

## SPEC

### One-line purpose
A multi-agent system that analyzes MP4 videos using VLM and LLM to automatically identify the most viral/impactful moments and generate precise cut points for clip creation.

### Users and use cases
- As a **content creator**, I want to automatically identify the most impactful moments in my videos so that I can create short viral clips for social media.
- As a **video editor**, I want to receive precise timestamps of the best moments so that I can quickly locate and extract clips without watching the entire video.
- As a **marketing professional**, I want to analyze videos based on custom criteria (e.g., "funny moments", "high impact statements") so that I can create targeted content for different audiences.

### Requirements
1. The system must accept MP4 video files as input
2. The system must use a VLM (Vision-Language Model) to analyze video content with audio directly (no separate transcription step)
3. The system must use an LLM to identify viral moments based on custom analysis criteria
4. The system must support custom analysis criteria provided as an array of strings
5. The system must output precise cut points (start and end timestamps) for each identified moment
6. The system must provide a reason/explanation for each identified moment
7. The system must use LangGraph for workflow orchestration
8. The system must use LangChain for agent implementation
9. The system must support configurable VLM and LLM models via environment variables
10. The system must use OpenRouter as the provider for VLM/LLM models
11. The state must be validated using Pydantic schemas
12. [ASSUMPTION: Python 3.12 with UV package manager]

### Edge cases
- **Video without audio**: System should still process using visual analysis only (VLM handles this)
- **Video with poor quality frames**: System should handle and still provide analysis
- **No viral moments found**: System should return empty clips array with appropriate message
- **Invalid video file**: System should validate file exists and is readable
- **Overlapping cut points**: System should merge or prioritize overlapping moments
- **Very short video (< 30 seconds)**: System should handle gracefully and identify if full video is the moment
- **Very long video (> 2 hours)**: System should process efficiently without timeout
- **Empty analysis array**: System should use default analysis criteria or error appropriately
- **OpenRouter API failure**: System should handle API errors gracefully and provide helpful error messages
- **Model not available on OpenRouter**: System should validate model availability before processing

### Acceptance criteria

**Happy Path - Complete Analysis:**
```
Given a valid MP4 video file
And an analysis array with criteria ["identify the most impactful moments", "identify funny moments"]
And valid OpenRouter API credentials
When the system processes the video
Then it should analyze the video directly using VLM (no separate transcription)
And it should generate a description with timestamps
And it should return clips with startTime, endTime, and reason
And each clip should match at least one analysis criterion
```

**Invalid Video File:**
```
Given a non-existent or corrupted MP4 file
When the system attempts to process
Then it should return a clear error message
And the workflow should terminate gracefully
```

**No Viral Moments Found:**
```
Given a video with no identifiable viral moments
When the system completes analysis
Then it should return an empty clips array
And it should provide a reason why no moments were identified
```

**Custom Analysis Criteria:**
```
Given an analysis array with specific criteria
When the system analyzes the video
Then each returned clip should include which criterion it matches
And the reason should explain how it relates to the criterion
```

**Overlapping Cut Points:**
```
Given a video where multiple moments overlap in time
When the system generates cut points
Then it should merge overlapping clips
Or it should prioritize by relevance score
```

**OpenRouter Model Selection:**
```
Given a configured OpenRouter API key
And a specific model name (e.g., "anthropic/claude-3.5-sonnet")
When the system initializes the VLM/LLM
Then it should use OpenRouter as the provider
And it should route requests to the specified model
```

### Non-functional requirements
- **Performance**: Should process a 10-minute video in under 5 minutes
- **Reliability**: Should handle video processing failures gracefully without crashing
- **Configurability**: All models and parameters should be configurable via environment variables
- **Type safety**: All state must be validated with Pydantic schemas
- **Maintainability**: Code should follow the specified folder structure (src/nodes, src/state.py, src/workflow.py)
- **Provider flexibility**: System should support switching between different models via OpenRouter
- **Cost efficiency**: Use appropriate models for each task (VLM for video, faster LLM for moment identification)

---

## PLAN

### Stack and architecture

**[ASSUMPTION: Python 3.12 with UV package manager, following modern Python project structure]**

- **Runtime**: Python 3.12
- **Package Manager**: UV (fast Python package manager)
- **Framework**: LangGraph + LangChain
- **Schema Validation**: Pydantic v2
- **AI Provider**: OpenRouter (unified API for multiple models)
- **AI Models**:
  - VLM: Configurable via `VLM_MODEL_NAME` (e.g., "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp")
  - LLM: Configurable via `LLM_MODEL_NAME` (e.g., "openai/gpt-4o", "meta-llama/llama-3.1-70b")
- **Video Processing**: Direct video upload to VLM (no separate audio extraction)
- **Pattern**: Multi-agent workflow with LangGraph StateGraph

**Architectural Pattern:**
- **State-driven workflow**: LangGraph manages state transitions between nodes
- **Separation of concerns**: Each node has a single responsibility
- **Type-safe state**: Pydantic schemas validate all state transitions
- **Provider abstraction**: LangChain's OpenRouter integration for model flexibility

**Why OpenRouter:**
- Single API key for multiple providers (Anthropic, OpenAI, Google, Meta, etc.)
- Access to latest models without multiple API keys
- Unified pricing and rate limiting
- Easy model switching via environment variables

### Data model changes

**State Schema (with Pydantic):**

```python
# src/state.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    ANALYZING_VIDEO = "analyzing_video"
    IDENTIFYING_MOMENTS = "identifying_moments"
    COMPLETED = "completed"
    FAILED = "failed"

class ClipInfo(BaseModel):
    """Information about a viral moment clip"""
    startTime: str = Field(..., description="Start time in MM:SS or HH:MM:SS format")
    endTime: str = Field(..., description="End time in MM:SS or HH:MM:SS format")
    reason: str = Field(..., description="Why this moment is viral/important")
    matchedCriterion: str = Field(..., description="Which analysis criterion this matches")

class VideoAnalysisState(BaseModel):
    """State for video analysis workflow"""
    # Input
    videoPath: str = Field(..., min_length=1, description="Path to MP4 video file")
    analysis: List[str] = Field(..., min_length=1, description="Analysis criteria")

    # Intermediate state (NO transcription - VLM processes video directly)
    videoDescription: Optional[str] = Field(None, description="VLM-generated video description with timestamps")

    # Output
    clips: Optional[List[ClipInfo]] = Field(None, description="Identified viral moments")

    # Metadata
    error: Optional[str] = Field(None, description="Error message if failed")
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, description="Current workflow status")

    @field_validator('videoPath')
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v.endswith('.mp4'):
            raise ValueError('Video file must be in MP4 format')
        return v

    @field_validator('analysis')
    @classmethod
    def validate_analysis(cls, v: List[str]) -> List[str]:
        if len(v) > 10:
            raise ValueError('Too many analysis criteria (max 10)')
        if any(not criterion.strip() for criterion in v):
            raise ValueError('Analysis criteria cannot be empty')
        return v
```

### API contracts

**Not applicable for this phase** - This is a library/agent system, not a REST API. Future work may add:
- CLI interface
- HTTP API for video submission
- WebSocket for progress updates

### Patterns to follow

**LangGraph Workflow Pattern:**
```python
# src/workflow.py
from langgraph.graph import StateGraph, END
from src.state import VideoAnalysisState
from src.nodes.analyze_video import analyze_video_node
from src.nodes.identify_moments import identify_moments_node

# Define workflow
workflow = StateGraph(VideoAnalysisState)

# Add nodes
workflow.add_node("analyze_video", analyze_video_node)
workflow.add_node("identify_moments", identify_moments_node)

# Define edges
workflow.set_entry_point("analyze_video")
workflow.add_edge("analyze_video", "identify_moments")
workflow.add_edge("identify_moments", END)

# Compile
app = workflow.compile()
```

**Node Pattern:**
```python
# src/nodes/analyze_video.py
from src.state import VideoAnalysisState
from langchain_openai import ChatOpenAI

def analyze_video_node(state: VideoAnalysisState) -> dict:
    """Analyze video using VLM"""
    # Implementation
    return {
        "videoDescription": "...",
        "status": "identifying_moments"
    }
```

**OpenRouter Integration Pattern:**
```python
from langchain_openai import ChatOpenAI

# Configure OpenRouter
openrouter_base_url = "https://openrouter.ai/api/v1"

vlmModel = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",  # via OpenRouter
    api_key=OPENROUTER_API_KEY,
    base_url=openrouter_base_url,
    defaultHeaders={
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME
    }
)
```

**[ASSUMPTION: Using LangChain's OpenAI-compatible client for OpenRouter]**

### Testing strategy

**Unit Tests:**
- Test each node independently with mocked inputs
- Test state schema validation with Pydantic
- Test error handling in each node

**Integration Tests:**
- Test complete workflow with sample videos
- Test state transitions between nodes
- Test with different analysis criteria
- Test OpenRouter API mocking

**E2E Tests:**
- Test with real short video files (< 1 minute)
- Verify output format matches schema
- Test edge cases (no audio, corrupted video, etc.)

**Test Files Structure:**
```
src/nodes/__tests__/
  - test_analyze_video.py
  - test_identify_moments.py
src/__tests__/
  - test_workflow.py
  - test_state.py
tests/
  - test_integration.py
  - fixtures/videos/
    - short-video.mp4
    - funny-moment.mp4
```

### Security considerations
- **API Key Security**: All API keys must be loaded from environment variables, never hardcoded
- **File Path Validation**: Validate video paths to prevent directory traversal attacks
- **Input Sanitization**: Sanitize analysis criteria to prevent prompt injection
- **Rate Limiting**: [ASSUMPTION: OpenRouter handles rate limiting, add retry logic for transient failures]
- **File Size Limits**: Validate video file size before processing (OpenRouter has limits)
- **Cost Management**: Add warnings for expensive models (Claude Opus, GPT-4 Vision)

### Scalability plan
- **Async Processing**: Use Python asyncio for all I/O operations
- **Streaming**: [FUTURE] Stream video processing for large files if OpenRouter supports it
- **Queue System**: [FUTURE] Add job queue (Celery/Redis) for processing multiple videos
- **Caching**: [FUTURE] Cache video descriptions to avoid reprocessing
- **Parallel Processing**: [FUTURE] Process multiple videos in parallel using asyncio

### Observability plan
- **Logging**: Use Python's logging module with structured logging (e.g., structlog)
- **Progress Tracking**: Update status field in state for each stage
- **Error Tracking**: Capture and log errors with full context
- **Metrics**: Track processing time per video, API call counts, token usage
- **[FUTURE]**: Add OpenTelemetry tracing for LangGraph workflows

---

## TASKS

## Task 1: Project Setup with UV

**What to build:** Initialize the Python project structure with UV, install all required dependencies for LangGraph, LangChain, Pydantic, and OpenRouter integration.

**Files likely affected:**
- `pyproject.toml` (UV project configuration)
- `.python-version` (Pin Python 3.12)
- `.env.example` (Environment variables template)
- `src/__init__.py` (Package initialization)
- `src/state.py` (Create initial state schema)

**Patterns to follow:** UV package management, Python 3.12 features, modern Python project structure

**Acceptance criteria:**
1. Initialize project with `uv init --python 3.12`
2. Install dependencies: `langgraph`, `langchain-openai`, `langchain-core`, `pydantic`, `python-dotenv`
3. Create `.env.example` with OPENROUTER_API_KEY, VLM_MODEL_NAME, LLM_MODEL_NAME variables
4. Create `src/state.py` with initial Pydantic VideoAnalysisState model
5. Create base folder structure: `src/nodes/`, `src/__tests__/`
6. Verify Python 3.12 compatibility with `uv run python --version`
7. Create `README.md` with setup instructions using UV

**Dependencies:** none
**Estimated complexity:** LOW

---

## Task 2: Implement State Schema with Pydantic

**What to build:** Complete the state schema in `src/state.py` with all required fields, Pydantic v2 validation, and proper type hints.

**Files likely affected:**
- `src/state.py`

**Patterns to follow:** Pydantic v2 patterns, field validators, proper type hints

**Acceptance criteria:**
1. Define VideoAnalysisState with all required fields (videoPath, analysis, videoDescription, clips, error, status)
2. Define ClipInfo model for individual clips
3. Add AnalysisStatus enum for status tracking
4. Add field validators for videoPath (.mp4 check) and analysis (max 10, non-empty)
5. Export models and types properly
6. Add comprehensive docstrings for all fields
7. Create unit tests in `src/__tests__/test_state.py` for schema validation
8. Test edge cases (invalid paths, empty analysis, too many criteria)

**Dependencies:** Task 1
**Estimated complexity:** LOW

---

## Task 3: Implement Analyze Video Node (VLM)

**What to build:** Create the VLM node that analyzes video content directly (frames + audio) using OpenRouter-compatible models like Claude 3.5 Sonnet or Gemini 2.0 Flash.

**Files likely affected:**
- `src/nodes/analyze_video.py`
- `src/nodes/__tests__/test_analyze_video.py`

**Patterns to follow:** LangGraph node pattern, VLM integration via LangChain, OpenRouter configuration

**Acceptance criteria:**
1. Create `analyze_video_node` function that accepts VideoAnalysisState and returns dict with state updates
2. Configure VLM using ChatOpenAI with OpenRouter base URL and headers
3. Read MP4 video file and prepare for VLM upload (base64 or direct file reference)
4. Construct prompt with VLM including analysis criteria and request for timestamped descriptions
5. Call VLM API (model from VLM_MODEL_NAME env var)
6. Parse VLM response to extract timestamped video description
7. Return state dict with videoDescription populated and status updated to "identifying_moments"
8. Handle VLM API errors gracefully (timeout, rate limit, invalid model)
9. Add unit tests with mocked VLM calls
10. Add logging for progress tracking using Python logging module
11. Validate video file size before uploading (warn if > 25MB for some models)

**Dependencies:** Task 2
**Estimated complexity:** HIGH

---

## Task 4: Implement Identify Moments Node (LLM)

**What to build:** Create the LLM node that analyzes the video description and identifies viral moments based on the analysis criteria using faster/cheaper models via OpenRouter.

**Files likely affected:**
- `src/nodes/identify_moments.py`
- `src/nodes/__tests__/test_identify_moments.py`

**Patterns to follow:** LangGraph node pattern, LLM integration, structured output with Pydantic

**Acceptance criteria:**
1. Create `identify_moments_node` function that accepts VideoAnalysisState and returns dict
2. Configure LLM using ChatOpenAI with OpenRouter (potentially different model than VLM)
3. Construct prompt with LLM including videoDescription and analysis criteria array
4. Configure LLM to return structured output (Pydantic model with clips)
5. Call LLM API (model from LLM_MODEL_NAME env var)
6. Parse LLM response and validate against ClipInfo schema using Pydantic
7. Return state dict with clips populated and status updated to "completed"
8. Handle LLM API errors gracefully
9. Add logic to merge overlapping clips if needed
10. Add unit tests with mocked LLM calls
11. Add logging for progress tracking

**Dependencies:** Task 2
**Estimated complexity:** HIGH

---

## Task 5: Implement LangGraph Workflow

**What to build:** Create the LangGraph workflow that orchestrates both nodes in sequence with proper state management.

**Files likely affected:**
- `src/workflow.py`
- `src/__tests__/test_workflow.py`

**Patterns to follow:** LangGraph StateGraph pattern, workflow definition, state management

**Acceptance criteria:**
1. Import both node functions and VideoAnalysisState model
2. Create StateGraph with VideoAnalysisState as the state schema
3. Add both nodes (analyze_video, identify_moments)
4. Define entry point (analyze_video) and edges between nodes
5. Compile the workflow into a runnable app
6. Export a `run_workflow` function that accepts initial state and returns final state
7. Add integration test in `tests/test_integration.py` that runs complete workflow with mocked nodes
8. Add error handling for workflow failures
9. Add proper type hints for all functions
10. Test state transitions between nodes

**Dependencies:** Task 3, Task 4
**Estimated complexity:** MEDIUM

---

## Task 6: Implement Main Entry Point and CLI

**What to build:** Create the main entry point that allows running the workflow with a video file and analysis criteria via CLI.

**Files likely affected:**
- `src/main.py` (or `__main__.py`)
- `cli.py` (optional, for richer CLI interface)

**Patterns to follow:** CLI argument parsing (argparse or click), environment variable loading

**Acceptance criteria:**
1. Create main function that loads environment variables from .env using python-dotenv
2. Accept video path and analysis criteria as command-line arguments
3. Initialize VideoAnalysisState with videoPath and analysis
4. Invoke workflow via `run_workflow` and wait for completion
5. Display results (clips with timestamps and reasons) in user-friendly format
6. Handle errors and display user-friendly messages
7. Add option to save results to JSON file
8. Test with a sample short video file
9. Add proper signal handling (Ctrl+C gracefully)
10. Create run script using UV (`uv run src/main.py`)

**Dependencies:** Task 5
**Estimated complexity:** MEDIUM

---

## Task 7: Add Error Handling and Validation

**What to build:** Implement comprehensive error handling and validation across all nodes and the workflow.

**Files likely affected:**
- All node files
- `src/workflow.py`
- `src/validation.py` (new)

**Patterns to follow:** Try-except blocks, validation before processing, graceful degradation

**Acceptance criteria:**
1. Validate video file exists and is readable before processing
2. Validate analysis array is not empty
3. Add timeout handling for API calls (default 30s for VLM, 15s for LLM)
4. Implement retry logic with exponential backoff for transient API failures (max 3 retries)
5. Add validation for clip timestamps (startTime < endTime, valid format)
6. Add validation for overlapping clips
7. Ensure all errors populate state.error field
8. Add tests for all error scenarios
9. Add user-friendly error messages with actionable suggestions
10. Validate OpenRouter API key is present before starting workflow
11. Validate model names are supported by OpenRouter

**Dependencies:** Task 6
**Estimated complexity:** MEDIUM

---

## Task 8: Add Logging and Observability

**What to build:** Implement structured logging and progress tracking throughout the workflow using Python logging.

**Files likely affected:**
- All node files
- `src/logging_config.py` (new)

**Patterns to follow:** Structured logging, progress updates, Python logging module

**Acceptance criteria:**
1. Create logging configuration with different log levels (DEBUG, INFO, WARNING, ERROR)
2. Log entry and exit of each node with timing information
3. Log state transitions and key values (videoPath, clip count, etc.)
4. Log API calls with model names, token usage (if available), and timing
5. Add progress indicators (e.g., "Analyzing video...", "Identifying moments...")
6. Log errors with full context (traceback, state info)
7. Add option to enable debug logging via LOG_LEVEL env var
8. Document logging output format in README
9. Use structured logging format (JSON or key-value) for easier parsing
10. Add logging to main entry point for workflow start/end

**Dependencies:** Task 7
**Estimated complexity:** LOW

---

## Task 9: Add Integration Tests

**What to build:** Create comprehensive integration tests that test the complete workflow with real or mocked dependencies.

**Files likely affected:**
- `tests/test_integration.py` (new)
- `tests/fixtures/videos/` (new, with sample videos)

**Patterns to follow:** Integration testing, test fixtures, mocking external APIs

**Acceptance criteria:**
1. Create test fixtures directory with sample short videos (< 30s each)
2. Create integration test that runs complete workflow with mocked OpenRouter APIs
3. Test with different analysis criteria (funny, impactful, emotional)
4. Test error scenarios (invalid video, API failures, timeout)
5. Test state transitions throughout workflow
6. Verify final output matches expected Pydantic schema
7. Add test for overlapping clips scenario
8. Add test for no viral moments found scenario
9. All tests should pass with `uv run pytest`
10. Add test configuration in `pyproject.toml`

**Dependencies:** Task 8
**Estimated complexity:** MEDIUM

---

## Task 10: Documentation and Examples

**What to build:** Create comprehensive documentation for setup, usage, and configuration.

**Files likely affected:**
- `README.md` (update)
- `docs/OPENROUTER_SETUP.md` (new)
- `examples/` (new)

**Patterns to follow:** Clear documentation, working examples, troubleshooting guides

**Acceptance criteria:**
1. Update README.md with Python 3.12 + UV setup instructions
2. Document OpenRouter configuration and model selection
3. Document all environment variables with examples
4. Create OPENROUTER_SETUP.md with detailed OpenRouter configuration
5. Add example usage code snippets
6. Document recommended VLM/LLM model combinations
7. Add troubleshooting section for common issues
8. Document cost considerations for different models
9. Add example analysis criteria for different use cases
10. Create example scripts in `examples/` directory

**Dependencies:** Task 9
**Estimated complexity:** LOW

---

## Review Checkpoint

Before handing these tasks to a coding agent, verify:
1. ✅ All environment variables are documented in `.env.example`
2. ✅ State schema is complete and validated with Pydantic
3. ✅ Each node has a single, clear responsibility
4. ✅ Error handling is comprehensive and user-friendly
5. ✅ Tests cover all major scenarios and edge cases
6. ✅ Logging provides visibility into workflow execution
7. ✅ Code follows the specified folder structure
8. ✅ Python 3.12 compatibility verified
9. ✅ OpenRouter integration is properly configured
10. ✅ No separate transcription step (VLM processes video directly)

---

## Assumptions to Review

1. [ASSUMPTION: Python 3.12 with UV package manager] — Impact: MEDIUM
   - Correct this if: You need to support older Python versions or use pip/poetry instead
   - Alternative: Use Python 3.11+ with pip/poetry/uv

2. [ASSUMPTION: VLM accepts MP4 files directly without audio extraction] — Impact: HIGH
   - Correct this if: Your chosen VLM doesn't support direct video upload
   - Alternative: Extract frames and use audio separately, or use frame-by-frame analysis

3. [ASSUMPTION: Using LangChain's OpenAI-compatible client for OpenRouter] — Impact: LOW
   - Correct this if: OpenRouter requires a different client library
   - Alternative: Use OpenRouter's native Python client if available

4. [ASSUMPTION: OpenRouter handles rate limiting automatically] — Impact: MEDIUM
   - Correct this if: You need custom rate limiting logic
   - Alternative: Implement exponential backoff with retry logic

5. [ASSUMPTION: Single video processing per workflow run] — Impact: LOW
   - Correct this if: You need batch processing from the start
   - Alternative: Add batch input to state and process in loop with asyncio

6. [ASSUMPTION: Output format in MM:SS or HH:MM:SS] — Impact: LOW
   - Correct this if: You prefer seconds (integer) or frame numbers
   - Alternative: Use multiple formats or make it configurable

7. [ASSUMPTION: Overlapping clips should be merged] — Impact: MEDIUM
   - Correct this if: You prefer to keep all clips even if overlapping
   - Alternative: Prioritize by score or keep all overlaps

8. [ASSUMPTION: No separate transcription step - VLM processes video+audio together] — Impact: HIGH
   - Correct this if: Your chosen VLM doesn't support audio analysis
   - Alternative: Add separate audio transcription step before VLM analysis

9. [ASSUMPTION: Using OpenRouter for all model access] — Impact: MEDIUM
   - Correct this if: You need direct API access to specific providers
   - Alternative: Use provider-specific SDKs alongside OpenRouter

10. [ASSUMPTION: Video file size limit of 25MB for OpenRouter] — Impact: MEDIUM
    - Correct this if: You need to process larger videos
    - Alternative: Implement video chunking or compression before upload
