# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Mocked tests for the opt-in Decisions rubric evaluator."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ddtrace.llmobs import EvaluatorContext

from airedale.config.experiment import ConfigError, ExperimentConfig
from airedale.scoring.decisions import MAX_QUESTIONS, DecisionsRubricEvaluator


def ctx(criteria=("c1", "c2")):
    return EvaluatorContext(
        input_data={"prompt": "q"}, output_data={"answer": "a"}, expected_output={"criteria": list(criteria)}
    )


def client(answers=None, error=None):
    result = SimpleNamespace(http_response=SimpleNamespace(json=lambda: {"answers": answers or []}))
    mock = MagicMock()
    mock.decisions.with_raw_response.create = AsyncMock(side_effect=error) if error else AsyncMock(return_value=result)
    return mock


def answer(name, probability=0.5, kind="predicate"):
    return {"name": name, "probability": probability, "type": kind}


class TestDecisions:
    async def test_boundary_and_concurrent_calls(self):
        fake = client([answer("0"), answer("1", 0.6)])
        evaluator = DecisionsRubricEvaluator(threshold=0.5, openai_client=fake)
        results = await asyncio.gather(evaluator.evaluate(ctx()), evaluator.evaluate(ctx()))
        assert all(result.value == 1 for result in results)
        assert fake.decisions.with_raw_response.create.call_count == 2

    @pytest.mark.parametrize(
        "answers",
        [
            [answer("0")],
            [answer("0"), answer("1"), answer("2")],
            [answer("1"), answer("0")],
            [answer("0"), answer("0")],
            [answer("0", True), answer("1")],
        ],
    )
    async def test_malformed_responses_fail_and_count(self, answers):
        result = await DecisionsRubricEvaluator(openai_client=client(answers)).evaluate(ctx())
        assert result.assessment == "fail"
        assert result.metadata["invalid_answer_count"] > 0

    async def test_refusal_is_counted_and_fails(self):
        result = await DecisionsRubricEvaluator(
            openai_client=client([answer("0", 1), answer("1", kind="refusal")])
        ).evaluate(ctx())
        assert result.assessment == "fail"
        assert result.metadata["refusal_count"] == 1

    async def test_error_is_safe(self):
        with pytest.raises(RuntimeError, match="Decisions provider request failed") as exc:
            await DecisionsRubricEvaluator(openai_client=client(error=RuntimeError("sk-secret-do-not-store"))).evaluate(
                ctx()
            )
        assert "sk-secret" not in str(exc.value)
        assert exc.value.__context__ is None

    async def test_limit_rejected_without_call(self):
        fake = client()
        result = await DecisionsRubricEvaluator(openai_client=fake).evaluate(
            ctx(tuple("x" for _ in range(MAX_QUESTIONS + 1)))
        )
        assert result.assessment == "fail"
        fake.decisions.with_raw_response.create.assert_not_called()


def test_config_threshold_validation():
    base = {"project": "p", "models": ("openai/gpt-4.1",), "scenarios": (MagicMock(),), "tasks": (MagicMock(),)}
    assert ExperimentConfig(**base, judge_type="decisions", decisions_threshold=0.0).decisions_threshold == 0.0
    for value in (True, "x", 1.1):
        with pytest.raises(ConfigError):
            ExperimentConfig(**base, judge_type="decisions", decisions_threshold=value)
    with pytest.raises(ConfigError):
        ExperimentConfig(**base, decisions_threshold=0.5)
