# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Task and judge errors reach ddtrace only as safe text (ROMAIN-65, O7).

The tests run the real ``Experiment.run`` of ddtrace (a real retry, with ``max_retries=1``) on a fake LLM Obs service
(``ddtrace_experiment`` in ``conftest.py``). The spans are real ``ddtrace.trace.Span`` objects. A marker is raised from a
variable, not from a literal in the ``raise`` line: ``error.stack`` has source lines.

Each test checks that the retry summary record exists (ddtrace rate-limits its records, the fixture sets DEBUG) before it
checks that a marker is absent.
"""

from __future__ import annotations

import asyncio
import re
import traceback
from typing import Any

import httpx2
import openai
import pytest
from ddtrace.llmobs import EvaluatorContext

from airedale.experiment import ExperimentTaskError
from airedale.scoring.rubric import RubricEvaluator, RubricJudgeError
from airedale.types import HarnessResult, UsageMetrics

TASK_MARKER = "sk-TASK-MARKER"
JUDGE_MARKER = "sk-JUDGE-MARKER"
SAFE_MESSAGE = re.compile(r"^ValueError at \S+:\d+$")


class FailingRunner:
    """A runner that raises ``ValueError(<marker>)`` on each attempt."""

    async def run(self, **kwargs: Any) -> HarnessResult:
        marker = TASK_MARKER
        raise ValueError(marker)


class PassingRunner:
    async def run(self, **kwargs: Any) -> HarnessResult:
        return HarnessResult(answer="the answer", usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2))


def summary_record(caplog) -> str:
    """Return the retry summary of ddtrace. It fails if the record is not there (a marker check would pass for no reason)."""
    [message] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ddtrace.llmobs._experiment" and "Retries (" in record.getMessage()
    ]
    return message


def everything_that_ddtrace_has(service, caplog, summary) -> str:
    """The text of the 6 sinks and of the rest: log records, span tags, annotations, backend calls, and rows."""
    records = "\n".join(f"{r.name}: {r.getMessage()}|{r.exc_text}" for r in caplog.records)
    spans = "\n".join(f"{span.get_tags()!r}" for span in service.spans)
    return "\n".join(
        [records, caplog.text, spans, repr(service.annotations), repr(service.backend_calls), repr(summary.rows)]
    )


async def test_a_task_error_reaches_ddtrace_as_safe_text(ddtrace_experiment, run_cell, caplog):
    ddtrace_experiment.max_retries = 1
    calls: list[str] = []

    def make_runner(cwd):
        calls.append(str(cwd))
        return FailingRunner()

    summary = await run_cell(make_runner)

    assert len(calls) == 2  # A real retry.
    retries = summary_record(caplog)  # The record exists (the next check would pass for no reason without it).
    assert "Retries (1)" in retries
    # Sinks 1 to 5 (the failed-status text goes to ``experiment_update``): no marker.
    assert TASK_MARKER not in everything_that_ddtrace_has(ddtrace_experiment, caplog, summary)
    assert "task row 0: attempt 1/2 failed: ValueError at test_experiment_errors.py:" in retries
    [row] = summary.rows
    assert SAFE_MESSAGE.match(row["error"]["message"])  # Sink 3 (and the error.message tag of sink 2).
    assert row["error"]["type"] == "airedale.experiment.ExperimentTaskError"
    assert "raise ValueError(marker)" in row["error"]["stack"]  # The frames stay (and the source line has no marker).
    [span] = ddtrace_experiment.spans  # One span for the record. The retries are inside it.
    assert SAFE_MESSAGE.match(span.get_tag("error.message"))
    assert "raise ValueError(marker)" in span.get_tag("error.stack")
    assert span.get_tag("error.type") == "airedale.experiment.ExperimentTaskError"
    update_calls = [call for call in ddtrace_experiment.backend_calls if "experiment_update" in str(call)]
    assert any("failed" in str(call) for call in update_calls)


async def test_a_task_retry_still_runs_and_can_pass(ddtrace_experiment, run_cell, caplog):
    ddtrace_experiment.max_retries = 1
    cwds: list[str] = []

    def make_runner(cwd):
        cwds.append(str(cwd))
        return FailingRunner() if len(cwds) == 1 else PassingRunner()

    summary = await run_cell(make_runner)

    assert len(cwds) == 2
    assert len(set(cwds)) == 2  # A fresh workspace for each attempt.
    [row] = summary.rows
    assert not row["error"]["message"]
    assert row["output"]["answer"] == "the answer"
    assert any(annotation.get("tags", {}).get("retries") == "1" for annotation in ddtrace_experiment.annotations)
    retries = summary_record(caplog)
    assert "Retries (1)" in retries
    assert TASK_MARKER not in everything_that_ddtrace_has(ddtrace_experiment, caplog, summary)
    assert "attempt 1/2 failed: ValueError at test_experiment_errors.py:" in retries


async def test_fail_fast_still_stops_with_safe_text(ddtrace_experiment, run_cell, caplog):
    ddtrace_experiment.max_retries = 1

    with pytest.raises(RuntimeError) as raised:
        await run_cell(lambda cwd: FailingRunner(), fail_fast=True)

    assert "Retries (1)" in summary_record(caplog)
    assert TASK_MARKER not in str(raised.value)  # Sink 6: ``cli.py`` prints ``str(error)``.
    assert TASK_MARKER not in "".join(traceback.format_exception(raised.value))
    assert re.match(r"^Error on record 0: ValueError at \S+:\d+\n", str(raised.value))


def judge_client(requests: list[httpx2.Request]) -> openai.AsyncOpenAI:
    """An OpenAI client whose provider answers HTTP 400 with the marker in the body (the SDK does not retry a 400)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(400, json={"error": {"message": f"bad request {JUDGE_MARKER}", "type": "invalid"}})

    return openai.AsyncOpenAI(
        api_key="synthetic-key",
        base_url="https://gateway.example/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


@pytest.mark.parametrize("raise_errors", [False, True], ids=["keep-going", "fail-fast"])
async def test_a_judge_error_reaches_ddtrace_as_safe_text(ddtrace_experiment, run_cell, caplog, raise_errors):
    requests: list[httpx2.Request] = []
    ddtrace_experiment.max_retries = 1
    ddtrace_experiment.evaluators = (
        RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client(requests)),
    )

    summary = None
    raised_text = ""  # (``str(error)`` and the whole traceback with its chain, when ``fail_fast`` raises.)
    if raise_errors:
        with pytest.raises(RuntimeError) as raised:
            await run_cell(lambda cwd: PassingRunner(), fail_fast=True)
        raised_text = str(raised.value) + "\n" + "".join(traceback.format_exception(raised.value))
        assert str(raised.value) == "Evaluator rubric_score failed on row 0"
    else:
        summary = await run_cell(lambda cwd: PassingRunner())

    assert len(requests) == 2  # A real retry of the judge.
    retries = summary_record(caplog)  # The record exists (the next checks would pass for no reason without it).
    assert "Retries (1)" in retries
    texts = [
        caplog.text,
        "\n".join(f"{r.getMessage()}|{r.exc_text}" for r in caplog.records),
        repr(ddtrace_experiment.backend_calls),  # The eval error and the failed-status text (sinks 4 and 5).
    ]
    if summary is not None:
        texts.append(repr(summary.rows))
    else:
        texts.append(raised_text)  # Sink 6 (judge).
    for text in texts:
        assert JUDGE_MARKER not in text
    assert "evaluator 'rubric_score' row 0: attempt 1/2 failed: BadRequestError at " in retries
    if summary is not None:
        [row] = summary.rows
        error = row["evaluations"]["rubric_score"]["error"]
        assert re.match(r"^BadRequestError at \S+:\d+$", error["message"])
        assert error["type"] == "RubricJudgeError"
        assert "_judge_with_openai" in error["stack"]  # The frames stay.
        assert any("rubric_score" in str(call) and "failed" in str(call) for call in ddtrace_experiment.backend_calls)


@pytest.mark.parametrize("guard", ["task", "judge"])
async def test_the_guards_do_not_chain_or_wrap_cancellation(run_cell, ddtrace_experiment, monkeypatch, guard):
    from airedale import experiment as experiment_module

    if guard == "task":
        # Call the ``experiment_task`` that ``_run_cell`` gives to ddtrace, with no ddtrace in between.
        class CancelledRunner:
            async def run(self, **kwargs: Any) -> None:
                raise asyncio.CancelledError

        async def call_task(runner: Any) -> BaseException:
            outcome: list[BaseException] = []

            class DirectExperiment:
                def __init__(self, task: Any) -> None:
                    self.task = task

                async def run(self, jobs: int = 1, raise_errors: bool = False) -> dict[str, Any]:
                    try:
                        await self.task({"prompt": "hi"}, {})
                    except BaseException as error:  # noqa: BLE001 (A cancellation is what the test looks for.)
                        outcome.append(error)
                    return {"rows": []}

            monkeypatch.setattr(
                experiment_module.LLMObs,
                "async_experiment",
                staticmethod(lambda **kwargs: DirectExperiment(kwargs["task"])),
            )
            await run_cell(lambda cwd: runner)
            [error] = outcome
            return error

        error = await call_task(FailingRunner())
        assert type(error) is ExperimentTaskError
        cancel = await call_task(CancelledRunner())
    else:
        evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))
        context = EvaluatorContext(
            input_data={"prompt": "q"}, output_data={"answer": "a"}, expected_output={"criteria": ["c"]}
        )
        with pytest.raises(RubricJudgeError) as raised:
            await evaluator.evaluate(context)
        error = raised.value

        async def cancelled(_context: EvaluatorContext) -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(evaluator, "_evaluate", cancelled)
        with pytest.raises(asyncio.CancelledError) as cancel_raised:
            await evaluator.evaluate(context)
        cancel = cancel_raised.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert type(cancel) is asyncio.CancelledError  # Not wrapped.
