"""Tests for classify_intent resilience: JSON parsing and plan coercion."""

from src.videos.nodes.classify_intent import (
    _parse_json_lenient,
    _coerce_plan,
    _default_plan,
)
from src.videos.domain.state import EditPlan


class TestParseJsonLenient:
    def test_clean_json(self):
        assert _parse_json_lenient('{"a": 1}') == {"a": 1}

    def test_markdown_fences(self):
        result = _parse_json_lenient('```json\n{"needsVision": true}\n```')
        assert result == {"needsVision": True}

    def test_markdown_fences_no_lang(self):
        result = _parse_json_lenient('```\n{"mode": "custom"}\n```')
        assert result == {"mode": "custom"}

    def test_noisy_text_around_json(self):
        result = _parse_json_lenient('Here is the JSON: {"mode": "custom"} done')
        assert result == {"mode": "custom"}

    def test_empty_returns_none(self):
        assert _parse_json_lenient("") is None
        assert _parse_json_lenient(None) is None

    def test_invalid_json_returns_none(self):
        assert _parse_json_lenient("not json at all") is None
        assert _parse_json_lenient("{broken") is None


class TestCoercePlan:
    def test_full_valid_dict(self):
        data = {
            "needsVision": True,
            "mode": "direto_ao_ponto",
            "editInstructions": "corte divagações",
            "reasoning": "user asked",
        }
        plan = _coerce_plan(data, "fallback")
        assert plan.needsVision is True
        assert plan.mode == "direto_ao_ponto"
        assert plan.editInstructions == "corte divagações"
        assert plan.reasoning == "user asked"

    def test_invalid_mode_coerced_to_custom(self):
        plan = _coerce_plan({"mode": "nonexistent"}, "fallback")
        assert plan.mode == "custom"

    def test_snake_case_edit_instructions(self):
        plan = _coerce_plan({"edit_instructions": "corte X"}, "fallback")
        assert plan.editInstructions == "corte X"

    def test_missing_instructions_uses_fallback(self):
        plan = _coerce_plan({"mode": "custom"}, "use this")
        assert plan.editInstructions == "use this"

    def test_missing_needsvision_defaults_false(self):
        plan = _coerce_plan({}, "fallback")
        assert plan.needsVision is False


class TestDefaultPlan:
    def test_default_is_custom_no_vision(self):
        plan = _default_plan("corte as pausas")
        assert plan.mode == "custom"
        assert plan.needsVision is False
        assert plan.editInstructions == "corte as pausas"

    def test_default_returns_edit_plan_instance(self):
        plan = _default_plan("anything")
        assert isinstance(plan, EditPlan)
