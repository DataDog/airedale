# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Tests for airedale.scoring.rubric — RubricEvaluator and helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from ddtrace.llmobs import EvaluatorContext

from airedale.scoring.rubric import (
    RubricEvaluator,
    _extract_answer,
    _extract_criteria,
    _extract_question,
    _normalize_score,
    _parse_judge_json,
)

# ---------------------------------------------------------------------------
# _parse_judge_json
# ---------------------------------------------------------------------------


class TestParseJudgeJson:
    def test_valid_json(self):
        result = _parse_judge_json('{"score": 0.9, "reasoning": "ok"}')
        assert result["score"] == 0.9
        assert result["reasoning"] == "ok"

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"score": 0.5, "reasoning": "partial"} done.'
        result = _parse_judge_json(text)
        assert result["score"] == 0.5

    def test_invalid_json_returns_raw_text(self):
        result = _parse_judge_json("not json at all")
        assert result.get("score") == 0
        assert result.get("reasoning") == "not json at all"

    def test_non_dict_json_returns_empty(self):
        # e.g. a JSON array at top level
        result = _parse_judge_json("[1, 2, 3]")
        assert result == {}

    def test_empty_string_returns_fallback(self):
        result = _parse_judge_json("")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _normalize_score
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    def test_zero_to_one_preserved(self):
        assert _normalize_score(0.75) == 0.75

    def test_above_one_clamped(self):
        assert _normalize_score(1.5) == 1.0

    def test_below_zero_clamped(self):
        assert _normalize_score(-0.5) == 0.0

    def test_exact_zero(self):
        assert _normalize_score(0) == 0.0

    def test_exact_one(self):
        assert _normalize_score(1) == 1.0

    def test_string_number(self):
        assert _normalize_score("0.8") == 0.8

    def test_none_returns_zero(self):
        assert _normalize_score(None) == 0.0

    def test_invalid_string_returns_zero(self):
        assert _normalize_score("bad") == 0.0


# ---------------------------------------------------------------------------
# _extract_criteria / _extract_answer / _extract_question
# ---------------------------------------------------------------------------


class TestExtractHelpers:
    def test_extract_criteria_from_dict(self):
        criteria = _extract_criteria({"criteria": ["c1", "c2"]})
        assert criteria == ["c1", "c2"]

    def test_extract_criteria_from_list(self):
        criteria = _extract_criteria(["c1", "c2"])
        assert criteria == ["c1", "c2"]

    def test_extract_criteria_from_string(self):
        criteria = _extract_criteria("single criterion")
        assert criteria == ["single criterion"]

    def test_extract_criteria_empty_list(self):
        assert _extract_criteria([]) == []

    def test_extract_criteria_none(self):
        assert _extract_criteria(None) == []

    def test_extract_answer_from_dict(self):
        answer = _extract_answer({"answer": "The answer is X"})
        assert answer == "The answer is X"

    def test_extract_answer_from_string(self):
        assert _extract_answer("direct answer") == "direct answer"

    def test_extract_answer_none(self):
        assert _extract_answer(None) == ""

    def test_extract_question_from_dict(self):
        question = _extract_question({"prompt": "What is X?"})
        assert question == "What is X?"

    def test_extract_question_from_string(self):
        assert _extract_question("What is X?") == "What is X?"

    def test_extract_question_none(self):
        assert _extract_question(None) == ""


# ---------------------------------------------------------------------------
# RubricEvaluator.evaluate — end-to-end with fake Anthropic client
# ---------------------------------------------------------------------------


def _make_anthropic_client(responses: list[str]) -> MagicMock:
    """Build a mock Anthropic async client returning canned JSON text per call."""
    mock_client = MagicMock()
    side_effects = [MagicMock(content=[MagicMock(text=resp)]) for resp in responses]
    mock_client.messages.create = AsyncMock(side_effect=side_effects)
    return mock_client


def _make_openai_client(responses: list[str]) -> MagicMock:
    """Build a mock OpenAI async client returning canned JSON text per call."""
    mock_client = MagicMock()
    side_effects = [MagicMock(choices=[MagicMock(message=MagicMock(content=resp))]) for resp in responses]
    mock_client.chat.completions.create = AsyncMock(side_effect=side_effects)
    return mock_client


class TestRubricEvaluatorEndToEnd:
    async def test_all_pass_anthropic(self):
        responses = [
            '{"score": 1.0, "reasoning": "correct"}',
            '{"score": 1.0, "reasoning": "also correct"}',
        ]
        fake_client = _make_anthropic_client(responses)
        evaluator = RubricEvaluator(
            judge_model="anthropic/claude-3-haiku-20240307",
            anthropic_client=fake_client,
        )
        ctx = EvaluatorContext(
            input_data={"prompt": "What is SSI?"},
            output_data={"answer": "SSI is Single Step Instrumentation."},
            expected_output={"criteria": ["Defines SSI correctly", "Mentions instrumentation"]},
        )
        result = await evaluator.evaluate(ctx)
        assert result.value == pytest.approx(1.0)
        assert result.assessment == "pass"
        assert fake_client.messages.create.call_count == 2

    async def test_partial_score_fails(self):
        responses = [
            '{"score": 1.0, "reasoning": "ok"}',
            '{"score": 0.4, "reasoning": "incomplete"}',
        ]
        fake_client = _make_anthropic_client(responses)
        evaluator = RubricEvaluator(
            judge_model="anthropic/claude-3-haiku-20240307",
            anthropic_client=fake_client,
            pass_threshold=0.8,
        )
        ctx = EvaluatorContext(
            input_data={"prompt": "What is X?"},
            output_data={"answer": "X is partial."},
            expected_output={"criteria": ["c1", "c2"]},
        )
        result = await evaluator.evaluate(ctx)
        # mean of 1.0 and 0.4 = 0.7 < 0.8 threshold
        assert result.value == pytest.approx(0.7)
        assert result.assessment == "fail"

    async def test_no_criteria_returns_fail(self):
        fake_client = _make_anthropic_client([])
        evaluator = RubricEvaluator(
            judge_model="anthropic/claude-3-haiku-20240307",
            anthropic_client=fake_client,
        )
        ctx = EvaluatorContext(
            input_data={"prompt": "Q"},
            output_data={"answer": "A"},
            expected_output={"criteria": []},
        )
        result = await evaluator.evaluate(ctx)
        assert result.assessment == "fail"
        assert "No rubric criteria" in result.reasoning

    async def test_evaluate_with_openai_judge(self):
        responses = ['{"score": 0.9, "reasoning": "good"}']
        fake_client = _make_openai_client(responses)
        evaluator = RubricEvaluator(
            judge_model="openai/gpt-4o",
            openai_client=fake_client,
        )
        ctx = EvaluatorContext(
            input_data={"prompt": "What is X?"},
            output_data={"answer": "X is great."},
            expected_output={"criteria": ["Explains X"]},
        )
        result = await evaluator.evaluate(ctx)
        assert result.value == pytest.approx(0.9)
        assert result.assessment == "pass"
        assert fake_client.chat.completions.create.call_count == 1

    async def test_result_metadata_contains_judge_info(self):
        fake_client = _make_anthropic_client(['{"score": 1.0, "reasoning": "ok"}'])
        evaluator = RubricEvaluator(
            judge_model="anthropic/claude-3-haiku-20240307",
            anthropic_client=fake_client,
        )
        ctx = EvaluatorContext(
            input_data="question",
            output_data="answer",
            expected_output={"criteria": ["one criterion"]},
        )
        result = await evaluator.evaluate(ctx)
        assert result.metadata is not None
        assert result.metadata["judge_model"] == "anthropic/claude-3-haiku-20240307"
        assert result.metadata["criteria_count"] == 1

    async def test_evaluator_name_is_rubric_score(self):
        fake_client = _make_anthropic_client([])
        evaluator = RubricEvaluator(
            judge_model="anthropic/claude-3-haiku-20240307",
            anthropic_client=fake_client,
        )
        assert evaluator.name == "rubric_score"
