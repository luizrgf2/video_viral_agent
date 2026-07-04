"""Backward-compat shim. Real code lives in src.shared.llm.agents."""

from src.shared.llm.agents import (
    vlmModel,
    llmModel,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    SITE_URL,
    APP_NAME,
    VLM_MODEL_NAME,
    LLM_MODEL_NAME,
    get_openrouter_headers,
    TRANSCRIBE_SYSTEM_PROMPT,
    VIDEO_ANALYSIS_SYSTEM_PROMPT,
    MOMENTS_IDENTIFICATION_SYSTEM_PROMPT,
)

__all__ = [
    "vlmModel",
    "llmModel",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_API_KEY",
    "SITE_URL",
    "APP_NAME",
    "VLM_MODEL_NAME",
    "LLM_MODEL_NAME",
    "get_openrouter_headers",
    "TRANSCRIBE_SYSTEM_PROMPT",
    "VIDEO_ANALYSIS_SYSTEM_PROMPT",
    "MOMENTS_IDENTIFICATION_SYSTEM_PROMPT",
]
