"""Router node: classifies the user prompt and produces an EditPlan."""

import logging
from src.videos.domain.state import VideoEditState, EditPlan, EditStatus
from src.shared.llm.agents import llmModel

logger = logging.getLogger(__name__)

NODE_ID = "classify_intent"

SYSTEM_PROMPT = """You are an editing-intent classifier. Given the user's natural-language request for editing a video, output a JSON object describing how to process it.

Detection rules:
- Set needsVision to true if the request references visual content (objects on screen, faces, gestures, scene changes, on-screen text, "when X appears", "showing Y", visual descriptions).
- Set needsVision to false if it references only speech, topics, content, or audio cues.

Mode mapping:
- "direto_ao_ponto" when the user wants to remove tangents, off-topic, rambling, filler talk, "vai direto ao ponto", "direto ao ponto", "cut the fluff".
- "emotion_peaks" when the user wants to keep high-energy/emotional moments (laughs, shouting, emphasis, excitement).
- "filler_removal" when the user wants to remove "é", "né", "tipo", long pauses, "hum", breathing gaps.
- "custom" for anything else.

Always produce a refined, prescriptive editInstructions field in Portuguese that downstream analysis can follow directly. Return ONLY valid JSON."""


async def classify_intent_node(state: VideoEditState) -> dict:
    logger.info(f"[{NODE_ID}] Classifying intent", extra={"prompt": state.userPrompt})

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.userPrompt},
        ]

        structured_llm = llmModel.with_structured_output(EditPlan)
        plan = await structured_llm.ainvoke(messages)

        logger.info(f"[{NODE_ID}] Classification done", extra={
            "needsVision": plan.needsVision,
            "mode": plan.mode,
        })

        return {
            "editPlan": plan,
            "status": EditStatus.TRANSCRIBING,
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{NODE_ID}] Classification failed", extra={"error": error_message})
        return {
            "error": f"Intent classification failed: {error_message}",
            "status": EditStatus.FAILED,
        }
