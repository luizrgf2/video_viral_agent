"""Router node: classifies the user prompt and produces an EditPlan.

Robust against LLM structured-output failures: tries structured output first,
falls back to manual JSON parsing, then to a sane default plan.
"""

import json
import logging
import re
from src.videos.domain.state import VideoEditState, EditPlan, EditStatus
from src.shared.llm.agents import llmModel

logger = logging.getLogger(__name__)

NODE_ID = "classify_intent"

VALID_MODES = {"direto_ao_ponto", "emotion_peaks", "filler_removal", "custom"}

SYSTEM_PROMPT = """You are an editing-intent classifier. Given the user's natural-language request for editing a video, output a JSON object describing how to process it.

Output schema (return ONLY valid JSON, no markdown fences):
{
  "needsVision": <boolean>,
  "mode": <one of "direto_ao_ponto" | "emotion_peaks" | "filler_removal" | "custom">,
  "editInstructions": <string, in Portuguese, prescriptive directive for downstream analysis>,
  "reasoning": <string, brief explanation>
}

Detection rules:
- needsVision = true if the request references visual content (objects on screen, faces, gestures, scene changes, on-screen text, "when X appears", "showing Y").
- needsVision = false if it references only speech, topics, content, or audio cues.

Mode mapping:
- "direto_ao_ponto" — user wants to remove tangents, off-topic, rambling, "direto ao ponto", "cut the fluff".
- "emotion_peaks" — user wants to keep high-energy/emotional moments.
- "filler_removal" — user wants to remove "é", "né", "tipo", long pauses, breathing gaps.
- "custom" — anything else.

Always produce editInstructions in Portuguese. Return ONLY the JSON object, no extra text."""


def _parse_json_lenient(text: str) -> dict | None:
    """Extract a JSON object from a possibly-noisy LLM response."""
    if not text:
        return None

    text = text.strip()

    # Strip markdown fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            text = text.split("```")[1] if "```" in text else text

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _coerce_plan(data: dict, fallback_prompt: str) -> EditPlan:
    """Build an EditPlan from a possibly-partial dict, with safe defaults."""
    raw_mode = data.get("mode", "custom")
    if raw_mode not in VALID_MODES:
        raw_mode = "custom"

    needs_vision = bool(data.get("needsVision", False))
    instructions = data.get("editInstructions") or data.get("edit_instructions") or fallback_prompt
    reasoning = data.get("reasoning") or f"mode={raw_mode}, needsVision={needs_vision}"

    return EditPlan(
        needsVision=needs_vision,
        mode=raw_mode,
        editInstructions=str(instructions).strip(),
        reasoning=str(reasoning).strip(),
    )


def _default_plan(prompt: str) -> EditPlan:
    """Last-resort plan: treat the prompt as custom instructions, no vision."""
    return EditPlan(
        needsVision=False,
        mode="custom",
        editInstructions=prompt,
        reasoning="Fallback plan (LLM classifier failed); using user prompt verbatim.",
    )


async def classify_intent_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Classifying intent", extra={"prompt": state.userPrompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state.userPrompt},
    ]

    # Attempt 1: structured output
    try:
        structured_llm = llmModel.with_structured_output(EditPlan)
        plan = await structured_llm.ainvoke(messages)
        if plan is not None and isinstance(plan, EditPlan):
            logger.info(f"[{NODE_ID}] Classification via structured_output", extra={
                "needsVision": plan.needsVision, "mode": plan.mode,
            })
            return {"editPlan": plan, "status": EditStatus.TRANSCRIBING}
    except Exception as e:
        logger.warning(f"[{NODE_ID}] structured_output failed, trying manual JSON", extra={
            "error": str(e),
        })

    # Attempt 2: raw LLM + manual JSON parse
    try:
        response = await llmModel.ainvoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        data = _parse_json_lenient(text)
        if data:
            plan = _coerce_plan(data, state.userPrompt)
            logger.info(f"[{NODE_ID}] Classification via manual JSON parse", extra={
                "needsVision": plan.needsVision, "mode": plan.mode,
            })
            return {"editPlan": plan, "status": EditStatus.TRANSCRIBING}
        logger.warning(f"[{NODE_ID}] JSON parse returned None", extra={"raw": text[:200]})
    except Exception as e:
        logger.warning(f"[{NODE_ID}] Manual JSON attempt failed", extra={"error": str(e)})

    # Attempt 3: default fallback
    logger.warning(f"[{NODE_ID}] All classification attempts failed, using default plan")
    plan = _default_plan(state.userPrompt)
    return {"editPlan": plan, "status": EditStatus.TRANSCRIBING}
