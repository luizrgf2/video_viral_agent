"""Backward-compat shim. Real code lives in src.clips.application.workflow."""

from src.clips.application.workflow import (
    create_workflow,
    create_clip_workflow,
    run_workflow,
    run_clip_workflow,
)

__all__ = [
    "create_workflow",
    "create_clip_workflow",
    "run_workflow",
    "run_clip_workflow",
]
