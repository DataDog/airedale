# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Shared pytest fixtures for airedale unit tests."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def private_report_dir(tmp_path_factory, monkeypatch) -> Path:
    """Send the private reports of failed pi runs to a temporary directory (and give the tests its path)."""
    directory = tmp_path_factory.mktemp("airedale-output")
    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(directory))
    return directory


@pytest.fixture
def clear_credential_cache():
    """Clear the gateway credential cache before and after each test.

    Import is deferred so only gateway tests pay the import cost.
    """
    from airedale.gateway import _credential_cache

    _credential_cache.clear()
    yield
    _credential_cache.clear()


class FakeSpan:
    """A span stand-in: records its kind, name, parent, times, annotations and error."""

    def __init__(self, tracer: FakeSpanLLMObs, kind: str, name: str | None, **kwargs) -> None:
        self.tracer = tracer
        self.kind = kind
        self.name = name
        self.kwargs = kwargs
        self.parent = tracer.active[-1] if tracer.active else None
        self.start_ns = 0
        self.end_ns: int | None = None
        self.error = 0
        self.tags: dict = {}
        self.annotations: dict = {}

    def set_tag(self, key, value) -> None:
        self.tags[key] = value

    def finish(self, finish_time: float | None = None) -> None:
        assert self.tracer.active and self.tracer.active[-1] is self, "spans must finish in order"
        self.tracer.active.pop()
        self.end_ns = round(finish_time * 1e9) if finish_time is not None else self.start_ns


class FakeSpanLLMObs:
    """LLMObs stand-in for manual spans: ``llm`` / ``tool`` / ``task`` + ``annotate``."""

    enabled = True

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.active: list[FakeSpan] = []
        self.annotations: list[dict] = []

    def _start(self, kind: str, name: str | None, **kwargs) -> FakeSpan:
        span = FakeSpan(self, kind, name, **kwargs)
        self.spans.append(span)
        self.active.append(span)
        return span

    def llm(self, *, model_name=None, model_provider=None, name=None):
        return self._start("llm", name, model_name=model_name, model_provider=model_provider)

    def tool(self, *, name=None):
        return self._start("tool", name)

    def task(self, *, name=None):
        return self._start("task", name)

    def annotate(self, span=None, **kwargs) -> None:
        self.annotations.append(kwargs)
        if span is not None:
            span.annotations.update(kwargs)


@pytest.fixture
def fake_span_llmobs() -> FakeSpanLLMObs:
    return FakeSpanLLMObs()


PI_REFERENCE_SCRIPT = Path(__file__).parent / "support" / "pi_reference.mjs"
PI_REFERENCE_FIXTURES = Path(__file__).parent / "fixtures" / "pi_reference"
PI_REFERENCE_VERSIONS = ("1.4.0", "1.5.0", "1.5.1")


def _canonical(request: dict) -> str:
    return json.dumps(request, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _live_pi_packages() -> dict[str, Path] | None:
    """Return the package directory of each installed pi release, or ``None`` without ``node`` or the releases."""
    if os.environ.get("AIREDALE_PI_REFERENCE") == "fixtures" or shutil.which("node") is None:
        return None
    releases = Path(os.environ.get("PI_RELEASES", Path.home() / ".pi" / "agent" / "install" / "releases"))
    packages = {
        version: releases / version / "node_modules" / "@earendil-works" / "pi-coding-agent"
        for version in PI_REFERENCE_VERSIONS
    }
    return packages if all(package.is_dir() for package in packages.values()) else None


@pytest.fixture(scope="session")
def pi_reference():
    """Return ``run(request)``: what pi 1.4.0, 1.5.0 and 1.5.1 answer to ``request``, by version.

    The tests that use it compare airedale with the code of pi. The answers are
    recorded in ``tests/fixtures/pi_reference/<action>.json``, so that the tests
    run everywhere, without ``node``. When ``node`` and the releases are installed
    (``$PI_RELEASES``, default ``~/.pi/agent/install/releases``), each answer is
    also computed live and must be the recorded one (set
    ``AIREDALE_PI_REFERENCE=fixtures`` to skip that check).

    To record the answers of new requests (or after a new pi release), run the
    tests with ``AIREDALE_RECORD_PI_REFERENCE=1``: the answers of the live run are
    written to the fixture files.
    """
    packages = _live_pi_packages()
    record = os.environ.get("AIREDALE_RECORD_PI_REFERENCE") == "1"
    if record and packages is None:
        pytest.fail("AIREDALE_RECORD_PI_REFERENCE=1 needs node and the pi releases 1.4.0, 1.5.0 and 1.5.1")
    loaded: dict[str, dict] = {}

    def fixtures_of(action: str) -> dict:
        if action not in loaded:
            path = PI_REFERENCE_FIXTURES / f"{action}.json"
            loaded[action] = json.loads(path.read_text()) if path.is_file() else {"cases": {}}
        return loaded[action]

    def live(request: dict) -> dict[str, dict]:
        assert packages is not None
        answers = {}
        for version, package in packages.items():
            process = subprocess.run(
                ["node", str(PI_REFERENCE_SCRIPT), str(package)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            answers[version] = json.loads(process.stdout)
        return answers

    def run(request: dict) -> dict[str, dict]:
        key, cases = _canonical(request), fixtures_of(request["action"])["cases"]
        if record:
            cases[key] = live(request)
            return cases[key]
        if key not in cases:
            pytest.fail(
                "no recorded answer of pi for this request; record it with "
                "AIREDALE_RECORD_PI_REFERENCE=1 (needs node and pi 1.4.0, 1.5.0 and 1.5.1): " + key[:200]
            )
        if packages is not None:
            assert live(request) == cases[key], "the recorded answer of pi is out of date; record it again"
        return cases[key]

    yield run
    if record:
        PI_REFERENCE_FIXTURES.mkdir(parents=True, exist_ok=True)
        node_version = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        for action, data in loaded.items():
            document = {
                "_comment": "Answers of pi to the requests of the tests. Record again: see tests/conftest.py (pi_reference).",
                "pi_versions": list(PI_REFERENCE_VERSIONS),
                "node": node_version,
                "cases": dict(sorted(data["cases"].items())),
            }
            (PI_REFERENCE_FIXTURES / f"{action}.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# A real ddtrace experiment on a fake LLM Obs service (no network)
# --------------------------------------------------------------------------- #
class FakeExperimentService:
    """The LLM Obs service that ``ddtrace.llmobs._experiment.Experiment`` needs, with no network.

    It follows the ``_ExperimentService`` contract of ddtrace. The spans are **real** ``ddtrace.trace.Span`` objects, so
    ``set_exc_info`` and ``get_tag`` are the code of ddtrace (that is how ``error.message``, ``error.stack`` and
    ``error.type`` are made). The backend client is a ``MagicMock``: ``repr(service.backend_calls)`` holds everything
    that ddtrace sends to LLM Obs (rows, evaluations, status text).
    """

    enabled = True

    def __init__(self, *, max_retries: int = 1, evaluators: tuple[Any, ...] = ()) -> None:
        from ddtrace.llmobs._experiment import Dataset

        self.max_retries = max_retries  #: Retries of ``Experiment.run`` (the airedale code never sets it).
        self.evaluators = evaluators  #: The evaluators that replace the real judge of the cell.
        self.spans: list[Any] = []
        self.annotations: list[dict[str, Any]] = []
        self._dne_client = MagicMock(name="dne_client")
        self._dne_client.project_create_or_get.return_value = {"_id": "project", "name": "p"}
        self._dne_client.experiment_create.return_value = ("experiment-id", "experiment-name")
        self.dataset = Dataset(
            name="dataset",
            project={"_id": "project", "name": "p"},
            dataset_id="dataset-id",
            records=[
                {
                    "record_id": "record-0",
                    "input_data": {"prompt": "hi"},
                    "expected_output": {"criteria": ["c"]},
                    "metadata": {},
                }
            ],
            description="",
            latest_version=1,
            version=1,
            _dne_client=self._dne_client,
        )

    @property
    def backend_calls(self) -> list[Any]:
        """Every call that ddtrace made to the LLM Obs backend client."""
        return list(self._dne_client.mock_calls)

    def _experiment(self, **kwargs: Any) -> Any:
        from ddtrace.trace import Span

        span = Span(kwargs.get("name") or "experiment")
        self.spans.append(span)
        return span

    def annotate(self, span: Any = None, **kwargs: Any) -> None:
        self.annotations.append(kwargs)

    def export_span(self, span: Any = None) -> None:
        return None

    def flush(self) -> None:
        pass


@pytest.fixture
def ddtrace_experiment(monkeypatch, caplog) -> FakeExperimentService:
    """Make ``_run_cell`` run the real ddtrace ``Experiment.run`` (retries included) on a ``FakeExperimentService``.

    Set ``service.max_retries`` and ``service.evaluators`` before the run. The logger of the experiment is at DEBUG:
    ddtrace rate-limits its log records (one for each message every 60 s, except at DEBUG), so a test must see the
    summary record before it can say that a marker is absent from the records.
    """
    from ddtrace.llmobs._experiment import Experiment

    from airedale import experiment as experiment_module

    service = FakeExperimentService()

    class RetryingExperiment(Experiment):
        async def run(  # type: ignore[override]
            self, jobs: int = 10, raise_errors: bool = False, *args: Any, **kwargs: Any
        ) -> Any:
            return await super().run(
                jobs=jobs, raise_errors=raise_errors, max_retries=service.max_retries, retry_delay=lambda _: 0
            )

    class FakeLLMObs:
        enabled = False

        @staticmethod
        def async_experiment(*, evaluators: Any, **kwargs: Any) -> Experiment:
            return RetryingExperiment(**kwargs, evaluators=service.evaluators, _llmobs_instance=service)

        @staticmethod
        def flush() -> None:
            pass

        @staticmethod
        def annotate(**kwargs: Any) -> None:
            pass

    monkeypatch.setattr(experiment_module, "LLMObs", FakeLLMObs)
    monkeypatch.setattr(experiment_module.dataset_module, "dataset_for_task", lambda dataset, task: service.dataset)
    caplog.set_level(logging.DEBUG, logger="ddtrace.llmobs._experiment")
    return service


@pytest.fixture
def run_cell(ddtrace_experiment, monkeypatch, tmp_path):
    """Return ``run(make_runner, fail_fast=False)``: run one real ``_run_cell`` whose runners come from ``make_runner``.

    ``make_runner(cwd)`` is called once for each attempt (a runner or a transport can be single-use). The cell uses the
    real ``experiment_task``, a real ``WorkspaceManager``, and the real ddtrace ``Experiment.run`` with
    ``ddtrace_experiment.max_retries`` retries.
    """
    from airedale import experiment as experiment_module
    from airedale.config.experiment import ExperimentConfig, ScenarioConfig, TaskConfig
    from airedale.progress import ProgressReporter
    from airedale.types import ModelSpec
    from airedale.workdir import WorkspaceManager

    async def run(make_runner: Any, *, fail_fast: bool = False) -> Any:
        monkeypatch.setattr(
            experiment_module, "create_runner", lambda model, *, scenario, gateway, cwd: make_runner(Path(cwd))
        )
        model = ModelSpec.parse("anthropic/claude-3-haiku-20240307")
        scenario = ScenarioConfig(name="s1")  # (No workdir: an empty temporary directory for each attempt.)
        task = TaskConfig(id="t1", prompt="Q", criteria=("c",))
        config = ExperimentConfig(project="p", models=(model.label,), scenarios=(scenario,), tasks=(task,))
        async with WorkspaceManager(config_dir=tmp_path) as workspace_manager:
            return await experiment_module._run_cell(
                (model, scenario, task),
                config=config,
                gateway=None,
                dataset=None,
                judge_model="openai/gpt-4.1",
                runs=1,
                fail_fast=fail_fast,
                progress=ProgressReporter(enabled=False),
                workspace_manager=workspace_manager,
            )

    return run
