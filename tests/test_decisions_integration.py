# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Offline integration tests for Decisions grading."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx2
import openai
import pytest
from ddtrace.llmobs import EvaluatorContext
from ddtrace.llmobs._experiment import Experiment

from airedale.config.experiment import ConfigError, load_experiment
from airedale.config.gateway import GatewayConfig, ProviderGatewayConfig
from airedale.experiment import _run_cell
from airedale.scoring.decisions import DecisionsRubricEvaluator
from airedale.scoring.rubric import RubricEvaluator
from airedale.types import ModelSpec


def context(n=1):
    return EvaluatorContext(
        input_data={"prompt": "q"},
        output_data={"answer": "a"},
        expected_output={"criteria": [f"c{i}" for i in range(n)]},
    )


def sdk_client(handler):
    return openai.AsyncOpenAI(
        api_key="synthetic-key",
        base_url="https://gateway.example/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


@pytest.mark.parametrize(
    "answers",
    [
        123,
        None,
        [{"name": "0", "type": "predicate", "probability": True}],
        [{"name": "0", "type": "predicate", "probability": "0.9"}],
        [{"name": "wrong", "type": "predicate", "probability": 0.9}],
        [],
        [
            {"name": "0", "type": "predicate", "probability": 0.9},
            {"name": "1", "type": "predicate", "probability": 0.9},
        ],
    ],
)
async def test_real_sdk_rejects_raw_bad_answers(answers):
    fake = sdk_client(lambda request: httpx2.Response(200, json={"answers": answers}))
    try:
        result = await DecisionsRubricEvaluator(openai_client=fake).evaluate(context())
        assert result.assessment == "fail"
        assert result.metadata["invalid_answer_count"] > 0
    finally:
        await fake.close()


async def test_refusal_at_high_mean():
    answers = [{"name": str(i), "type": "predicate", "probability": 1.0} for i in range(4)] + [
        {"name": "4", "type": "refusal"}
    ]
    fake = sdk_client(lambda request: httpx2.Response(200, json={"answers": answers}))
    try:
        result = await DecisionsRubricEvaluator(openai_client=fake).evaluate(context(5))
        assert result.value == 0.8
        assert result.assessment == "fail"
        assert result.metadata["refusal_count"] == 1
    finally:
        await fake.close()


async def test_concurrent_calls_keep_results_separate():
    entered = 0
    all_entered = asyncio.Event()

    async def handler(request):
        nonlocal entered
        entered += 1
        if entered == 2:
            all_entered.set()
        await all_entered.wait()
        input_text = json.loads(request.content)["input"]
        probability = 1 if "first" in input_text else 0
        return httpx2.Response(200, json={"answers": [{"name": "0", "type": "predicate", "probability": probability}]})

    fake = sdk_client(handler)
    evaluator = DecisionsRubricEvaluator(openai_client=fake)
    first = EvaluatorContext(input_data={"prompt": "first"}, output_data={}, expected_output={"criteria": ["c"]})
    second = EvaluatorContext(input_data={"prompt": "second"}, output_data={}, expected_output={"criteria": ["c"]})
    try:
        results = await asyncio.gather(evaluator.evaluate(first), evaluator.evaluate(second))
        assert [result.value for result in results] == [1, 0]
    finally:
        await fake.close()


async def test_201_criteria_never_reaches_sdk():
    fake = sdk_client(lambda request: pytest.fail("Unexpected request"))
    try:
        result = await DecisionsRubricEvaluator(openai_client=fake).evaluate(context(201))
        assert result.metadata["invalid_answer_count"] == 201
        assert result.assessment == "fail"
    finally:
        await fake.close()


async def test_gateway_client_real_sdk(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json={"answers": [{"name": "0", "type": "predicate", "probability": 0.5}]})

    transport = httpx2.MockTransport(handler)
    monkeypatch.setattr(httpx2.AsyncClient, "_transport_for_url", lambda self, url: transport)
    monkeypatch.setenv("TEST_OPENAI_KEY", "synthetic-key")
    gateway = GatewayConfig(
        providers={
            "openai": ProviderGatewayConfig(
                provider="openai",
                base_url="https://gateway.example/v1",
                api_key_env="TEST_OPENAI_KEY",
                headers={"x-test-source": "test"},
            )
        }
    )
    evaluator = DecisionsRubricEvaluator(gateway=gateway)
    try:
        result = await evaluator.evaluate(context())
        assert result.value == 1
        assert str(requests[0].url) == "https://gateway.example/v1/decisions"
        assert requests[0].headers["x-test-source"] == "test"
        assert requests[0].headers["authorization"] == "Bearer synthetic-key"
    finally:
        await evaluator.client.close()


@pytest.mark.parametrize("status", [307, 308])
async def test_gateway_cross_origin_redirect_never_sends_request(monkeypatch, status):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "gateway.example":
            return httpx2.Response(status, headers={"location": "https://api.openai.com/v1/decisions"})
        return httpx2.Response(200, json={"answers": [{"name": "0", "type": "predicate", "probability": 1}]})

    transport = httpx2.MockTransport(handler)
    monkeypatch.setattr(httpx2.AsyncClient, "_transport_for_url", lambda self, url: transport)
    monkeypatch.setenv("TEST_OPENAI_KEY", "synthetic-key")
    gateway = GatewayConfig(
        providers={
            "openai": ProviderGatewayConfig(
                provider="openai",
                base_url="https://gateway.example/v1",
                api_key_env="TEST_OPENAI_KEY",
                headers={"X-Api-Key": "synthetic-header-secret"},
            )
        }
    )
    evaluator = DecisionsRubricEvaluator(gateway=gateway)
    try:
        with pytest.raises(RuntimeError, match="Decisions provider request failed") as error:
            await evaluator.evaluate(context())
        assert error.value.__context__ is None
        assert [request.url.host for request in requests] == ["gateway.example"]
    finally:
        await evaluator.client.close()


async def test_ddtrace_provider_error_is_safe_and_failfast():
    fake = sdk_client(lambda request: httpx2.Response(401, json={"error": {"message": "synthetic-secret"}}))
    evaluator = DecisionsRubricEvaluator(openai_client=fake)
    experiment = Experiment(name="test", project_name="test", evaluators=[evaluator])
    record = {"input_data": {"prompt": "q"}, "expected_output": {"criteria": ["c"]}}
    task = {"idx": 0, "output": {"answer": "a"}}
    try:
        stored = await experiment._evaluate_record(record, task, asyncio.Semaphore(1), raise_errors=False)
        assert experiment._has_errors
        assert "synthetic-secret" not in str(stored)
        assert "Decisions provider request failed" in str(stored)
        with pytest.raises(RuntimeError) as error:
            await experiment._evaluate_record(record, task, asyncio.Semaphore(1), raise_errors=True)
        import traceback

        assert "synthetic-secret" not in "".join(traceback.format_exception(error.value))
    finally:
        await fake.close()


@pytest.mark.parametrize("threshold", ["true", "false", '"0.5"', "1.1", "-0.1"])
def test_config_loader_rejects_invalid_threshold(tmp_path, threshold):
    config_file = tmp_path / "experiment.toml"
    config_file.write_text(
        'project = "p"\nmodels = ["openai/gpt-4.1"]\njudge_type = "decisions"\n'
        f'decisions_threshold = {threshold}\n[scenarios.basic]\n[tasks.one]\nprompt = "q"\ncriteria = ["c"]\n'
    )
    with pytest.raises(ConfigError):
        load_experiment(config_file)


def test_config_loader_rejects_threshold_without_decisions(tmp_path):
    config_file = tmp_path / "experiment.toml"
    config_file.write_text(
        'project = "p"\nmodels = ["openai/gpt-4.1"]\ndecisions_threshold = 0.5\n'
        '[scenarios.basic]\n[tasks.one]\nprompt = "q"\ncriteria = ["c"]\n'
    )
    with pytest.raises(ConfigError):
        load_experiment(config_file)


@pytest.mark.parametrize("extra", ["", 'judge_type = "decisions"\ndecisions_threshold = 0.0\n'])
async def test_config_loader_selection_and_metadata(tmp_path, monkeypatch, extra):
    config_file = tmp_path / "experiment.toml"
    config_file.write_text(
        'project = "p"\nmodels = ["openai/gpt-4.1"]\n'
        + extra
        + '\n[scenarios.basic]\n\n[tasks.one]\nprompt = "q"\ncriteria = ["c"]\n'
    )
    config = load_experiment(config_file)
    experiment = MagicMock()
    experiment.run = AsyncMock(return_value={"rows": []})
    factory = MagicMock(return_value=experiment)
    monkeypatch.setattr("airedale.experiment.LLMObs.async_experiment", factory)
    monkeypatch.setattr("airedale.experiment.LLMObs.flush", lambda: None)
    monkeypatch.setattr("airedale.experiment.dataset_module.dataset_for_task", lambda *args: None)
    progress = MagicMock()
    progress.message = AsyncMock()
    await _run_cell(
        (ModelSpec.parse(config.models[0]), config.scenarios[0], config.tasks[0]),
        config=config,
        gateway=None,
        dataset=None,
        judge_model=config.judge_model,
        runs=1,
        fail_fast=False,
        progress=progress,
        workspace_manager=MagicMock(),
    )
    kwargs = factory.call_args.kwargs
    if extra:
        assert isinstance(kwargs["evaluators"][0], DecisionsRubricEvaluator)
        assert kwargs["evaluators"][0].threshold == 0.0
        assert kwargs["config"]["judge_type"] == "decisions"
        assert kwargs["config"]["judge_model"] == "gpt-6-luna"
        assert kwargs["config"]["decisions_threshold"] == 0.0
    else:
        assert isinstance(kwargs["evaluators"][0], RubricEvaluator)
        assert "judge_type" not in kwargs["config"]
        assert "decisions_threshold" not in kwargs["config"]
        assert kwargs["config"]["judge_model"] == config.judge_model
        assert kwargs["tags"] == {
            "scenario": "basic",
            "prompt_version": "basic",
            "model_name": "openai/gpt-4.1",
            "task_slug": "one",
            "judge_model": config.judge_model,
        }
