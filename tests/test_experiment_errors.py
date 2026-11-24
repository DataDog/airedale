# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Task and judge errors reach ddtrace only as safe text.

The tests run the real ``Experiment.run`` of ddtrace (a real retry, with ``max_retries=1``) on a fake LLM Obs service
(``ddtrace_experiment`` in ``conftest.py``). The spans are real ``ddtrace.trace.Span`` objects. A marker is raised from a
variable, not from a literal in the ``raise`` line: ``error.stack`` has source lines.

Each test checks that the retry summary record exists (ddtrace rate-limits its records, the fixture sets DEBUG) before it
checks that a marker is absent.
"""

from __future__ import annotations

import asyncio
import logging
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


def _text(error: BaseException) -> str:
    """Everything that a library can copy from ``error``."""
    return "\n".join(
        [str(error), repr(error), repr(getattr(error, "__notes__", None)), "".join(traceback.format_exception(error))]
    )


def summary_record(caplog) -> str:
    """Return the retry summary of ddtrace. It fails if the record is not there (a marker check would pass for no reason)."""
    [message] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ddtrace.llmobs._experiment" and "Retries (" in record.getMessage()
    ]
    return message


def experiment_summary_record(caplog, *, level: int | None = None) -> str:
    """Return the summary record of ddtrace (with or without retries). It fails if the record is not there."""
    [message] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ddtrace.llmobs._experiment"
        and "Experiment '" in record.getMessage()
        and (level is None or record.levelno == level)
    ]
    return message


def everything_that_ddtrace_has(service, caplog, summary) -> str:
    """The text of the 6 sinks and of the rest: log records, span tags, annotations, backend calls, and rows."""
    records = "\n".join(f"{r.name}: {r.getMessage()}|{r.exc_text}" for r in caplog.records)
    spans = "\n".join(f"{span.get_tags()!r}" for span in service.spans)
    return "\n".join(
        [
            records,
            caplog.text,
            spans,
            repr(service.annotations),
            repr(service.backend_calls),
            repr(summary.rows if summary else []),
        ]
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
            """A runner that is cancelled while it handles a raw error (the raw error is the ``__context__``)."""

            def __init__(self) -> None:
                self.raised: asyncio.CancelledError | None = None

            async def run(self, **kwargs: Any) -> None:
                marker = TASK_MARKER
                try:
                    raise ValueError(marker)
                except ValueError:
                    self.raised = asyncio.CancelledError("Cancelled via cancel scope 0x1")
                    self.raised.add_note(marker)
                    raise self.raised  # noqa: B904 (The context is what the test is about.)

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
        runner = CancelledRunner()
        cancel = await call_task(runner)
        assert cancel is runner.raised  # The same object, so asyncio and anyio still see their cancellation.
    else:
        evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))
        context = EvaluatorContext(
            input_data={"prompt": "q"}, output_data={"answer": "a"}, expected_output={"criteria": ["c"]}
        )
        with pytest.raises(RubricJudgeError) as raised:
            await evaluator.evaluate(context)
        error = raised.value

        raised_cancellation: list[asyncio.CancelledError] = []

        async def cancelled(_context: EvaluatorContext) -> None:
            marker = JUDGE_MARKER
            try:
                raise ValueError(marker)
            except ValueError:
                raised_cancellation.append(asyncio.CancelledError("Cancelled via cancel scope 0x1"))
                raised_cancellation[0].add_note(marker)
                raise raised_cancellation[0]  # noqa: B904 (The context is what the test is about.)

        monkeypatch.setattr(evaluator, "_evaluate", cancelled)
        with pytest.raises(asyncio.CancelledError) as cancel_raised:
            await evaluator.evaluate(context)
        cancel = cancel_raised.value
        assert cancel is raised_cancellation[0]
    assert error.__cause__ is None
    assert error.__context__ is None
    assert type(cancel) is asyncio.CancelledError  # Not wrapped.
    assert cancel.args == ("Cancelled via cancel scope 0x1",)  # The message is not changed.
    assert cancel.__cause__ is None
    assert cancel.__context__ is None  # The raw error is cut.
    assert getattr(cancel, "__notes__", None) is None
    assert TASK_MARKER not in "".join(traceback.format_exception(cancel))
    assert JUDGE_MARKER not in "".join(traceback.format_exception(cancel))


# --------------------------------------------------------------------------- #
# A mixed exception group, and a real cancellation
# --------------------------------------------------------------------------- #
def mixed_group(marker: str) -> BaseExceptionGroup:
    """A group of an ordinary error and a cancellation. The marker is in the message, in a leaf and in a note."""
    group = BaseExceptionGroup(marker, [ValueError(marker), asyncio.CancelledError()])
    group.add_note(marker)
    return group


class MixedGroupRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> HarnessResult:
        self.calls += 1
        group = mixed_group(TASK_MARKER)
        raise group


@pytest.mark.parametrize("fail_fast", [False, True], ids=["keep-going", "fail-fast"])
async def test_a_mixed_group_task_error_reaches_ddtrace_as_safe_text(ddtrace_experiment, run_cell, caplog, fail_fast):
    ddtrace_experiment.max_retries = 1
    runner = MixedGroupRunner()
    summary = None
    escaped: BaseExceptionGroup | None = None
    if fail_fast:
        with pytest.raises(BaseExceptionGroup) as raised:
            await run_cell(lambda cwd: runner, fail_fast=True)
        escaped = raised.value
    else:
        summary = await run_cell(lambda cwd: runner)

    assert runner.calls == 1  # No retry: ddtrace retries only an ``Exception``.
    experiment_summary_record(caplog, level=logging.WARNING if fail_fast else logging.INFO)  # The record exists.
    assert TASK_MARKER not in everything_that_ddtrace_has(ddtrace_experiment, caplog, summary)
    [span] = ddtrace_experiment.spans
    assert re.match(r"^CancelledError, ValueError at \S+:\d+ \(2 sub-exceptions\)$", span.get_tag("error.message"))
    assert span.get_tag("error.type") == "builtins.BaseExceptionGroup"
    assert "raise group" in span.get_tag("error.stack")  # The frames stay.
    if escaped is not None:
        assert TASK_MARKER not in _text(escaped)
        assert escaped.subgroup(asyncio.CancelledError) is not None  # The cancellation is still in the group.
        assert escaped.subgroup(ExperimentTaskError) is not None  # The failure too.


@pytest.mark.parametrize("raise_errors", [False, True], ids=["keep-going", "fail-fast"])
async def test_a_mixed_group_judge_error_reaches_ddtrace_as_safe_text(
    ddtrace_experiment, run_cell, caplog, monkeypatch, raise_errors
):
    ddtrace_experiment.max_retries = 1
    evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))
    calls: list[int] = []

    async def mixed(_context: EvaluatorContext) -> None:
        calls.append(1)
        group = mixed_group(JUDGE_MARKER)
        raise group

    monkeypatch.setattr(evaluator, "_evaluate", mixed)
    ddtrace_experiment.evaluators = (evaluator,)
    summary = None
    escaped: BaseExceptionGroup | None = None
    if raise_errors:
        with pytest.raises(BaseExceptionGroup) as raised:
            await run_cell(lambda cwd: PassingRunner(), fail_fast=True)
        escaped = raised.value
    else:
        summary = await run_cell(lambda cwd: PassingRunner())

    assert calls == [1]  # No retry.
    experiment_summary_record(caplog, level=logging.WARNING if raise_errors else logging.INFO)  # The record exists.
    assert JUDGE_MARKER not in everything_that_ddtrace_has(ddtrace_experiment, caplog, summary)
    if escaped is not None:
        assert JUDGE_MARKER not in _text(escaped)
        assert escaped.subgroup(asyncio.CancelledError) is not None
        assert escaped.subgroup(RubricJudgeError) is not None


@pytest.mark.parametrize("where", ["task", "judge"])
async def test_a_cancellation_keeps_propagating_without_its_raw_chain(
    ddtrace_experiment, run_cell, caplog, monkeypatch, where
):
    """A real ``task.cancel()`` during the cleanup of a failed run: the cancellation has the raw error as ``__context__``."""
    ddtrace_experiment.max_retries = 1
    marker = TASK_MARKER if where == "task" else JUDGE_MARKER
    in_cleanup, release = asyncio.Event(), asyncio.Event()

    async def fail_then_wait_in_cleanup() -> None:
        text = marker
        try:
            raise ValueError(text)
        finally:
            in_cleanup.set()
            await release.wait()  # The cancellation arrives here, while ``ValueError`` is being handled.

    class CleanupRunner:
        async def run(self, **kwargs: Any) -> None:
            await fail_then_wait_in_cleanup()

    if where == "judge":
        evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))

        async def failing_evaluate(_context: EvaluatorContext) -> None:
            await fail_then_wait_in_cleanup()

        monkeypatch.setattr(evaluator, "_evaluate", failing_evaluate)
        ddtrace_experiment.evaluators = (evaluator,)
        make_runner: Any = lambda cwd: PassingRunner()  # noqa: E731
    else:
        make_runner = lambda cwd: CleanupRunner()  # noqa: E731

    cell = asyncio.ensure_future(run_cell(make_runner))
    await asyncio.wait_for(in_cleanup.wait(), timeout=30)
    cell.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(cell, timeout=30)

    # ddtrace saw an interruption (the status and the WARNING summary), and the raw error is nowhere.
    experiment_summary_record(caplog, level=logging.WARNING)  # The record exists.
    assert any("interrupted" in str(call) for call in ddtrace_experiment.backend_calls)
    assert marker not in everything_that_ddtrace_has(ddtrace_experiment, caplog, None)
    assert marker not in _text(raised.value)
    if where == "task":
        [span] = ddtrace_experiment.spans
        assert span.get_tag("error.stack")  # The span has the error (the check above is not empty).
        assert span.get_tag("error.type") == "asyncio.exceptions.CancelledError"


@pytest.mark.parametrize("guard", ["task", "judge"])
async def test_a_failing_helper_does_not_attach_the_raw_error(run_cell, ddtrace_experiment, monkeypatch, guard):
    """``safe_error`` runs outside the ``except`` block: if it fails, the new error has no raw ``__context__``."""
    from airedale import experiment as experiment_module
    from airedale.scoring import rubric as rubric_module

    def failing_helper(error: BaseException, wrapper: type[Exception]) -> BaseException:
        raise OSError("the helper failed")

    if guard == "task":
        monkeypatch.setattr(experiment_module, "safe_error", failing_helper)
        outcome: list[BaseException] = []

        class DirectExperiment:
            def __init__(self, task: Any) -> None:
                self.task = task

            async def run(self, jobs: int = 1, raise_errors: bool = False) -> dict[str, Any]:
                try:
                    await self.task({"prompt": "hi"}, {})
                except BaseException as error:  # noqa: BLE001
                    outcome.append(error)
                return {"rows": []}

        monkeypatch.setattr(
            experiment_module.LLMObs,
            "async_experiment",
            staticmethod(lambda **kwargs: DirectExperiment(kwargs["task"])),
        )
        await run_cell(lambda cwd: FailingRunner())
        [error] = outcome
    else:
        monkeypatch.setattr(rubric_module, "safe_error", failing_helper)
        evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))
        context = EvaluatorContext(
            input_data={"prompt": "q"}, output_data={"answer": "a"}, expected_output={"criteria": ["c"]}
        )
        with pytest.raises(OSError) as raised:  # noqa: PT011
            await evaluator.evaluate(context)
        error = raised.value
    assert isinstance(error, OSError)
    assert error.__context__ is None
    assert error.__cause__ is None
    assert TASK_MARKER not in _text(error)
    assert JUDGE_MARKER not in _text(error)


class CustomGroup(BaseExceptionGroup):
    """A legal group that is not an ``Exception``, though all its leaves are (ddtrace does not retry it)."""


@pytest.mark.parametrize("fail_fast", [False, True], ids=["keep-going", "fail-fast"])
@pytest.mark.parametrize("where", ["task", "judge"])
async def test_a_custom_group_of_exceptions_is_still_not_an_exception_and_not_retried(
    ddtrace_experiment, run_cell, caplog, monkeypatch, where, fail_fast
):
    ddtrace_experiment.max_retries = 1
    marker = TASK_MARKER if where == "task" else JUDGE_MARKER
    calls: list[int] = []

    def make_error() -> BaseException:
        calls.append(1)
        group = CustomGroup(marker, [ValueError(marker), KeyError(marker)])
        group.add_note(marker)
        return group

    class CustomGroupRunner:
        async def run(self, **kwargs: Any) -> None:
            error = make_error()
            raise error

    make_runner: Any = lambda cwd: CustomGroupRunner()  # noqa: E731
    if where == "judge":
        evaluator = RubricEvaluator(judge_model="openai/gpt-4.1", openai_client=judge_client([]))

        async def custom_evaluate(_context: EvaluatorContext) -> None:
            error = make_error()
            raise error

        monkeypatch.setattr(evaluator, "_evaluate", custom_evaluate)
        ddtrace_experiment.evaluators = (evaluator,)
        make_runner = lambda cwd: PassingRunner()  # noqa: E731

    summary = None
    escaped: BaseException | None = None
    if fail_fast:
        with pytest.raises(BaseExceptionGroup) as raised:
            await run_cell(make_runner, fail_fast=True)
        escaped = raised.value
    else:
        summary = await run_cell(make_runner)

    assert len(calls) == 1  # No retry, as for the original group.
    experiment_summary_record(caplog, level=logging.WARNING if fail_fast else logging.INFO)  # The record exists.
    assert marker not in everything_that_ddtrace_has(ddtrace_experiment, caplog, summary)
    if escaped is not None:
        assert not isinstance(escaped, Exception)  # The same propagation as the original group.
        assert marker not in _text(escaped)
        assert escaped.subgroup(ExperimentTaskError if where == "task" else RubricJudgeError) is not None  # type: ignore[attr-defined]
    if where == "task":
        [span] = ddtrace_experiment.spans
        assert re.match(r"^KeyError, ValueError at \S+:\d+ \(2 sub-exceptions\)$", span.get_tag("error.message"))
        assert span.get_tag("error.stack")  # The span has the error (the checks above are not empty).
