# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.experiment — pure helper functions."""

from __future__ import annotations

import asyncio

import pytest

from airedale.config.experiment import ExperimentConfig, McpServerConfig, ScenarioConfig, TaskConfig
from airedale.experiment import _experiment_name, _managed_server_specs, _normalize_filter
from airedale.types import ModelSpec

# ---------------------------------------------------------------------------
# _normalize_filter
# ---------------------------------------------------------------------------


class TestNormalizeFilter:
    def test_none_returns_empty_set(self):
        assert _normalize_filter(None) == set()

    def test_empty_list_returns_empty_set(self):
        assert _normalize_filter([]) == set()

    def test_single_value(self):
        assert _normalize_filter(["alpha"]) == {"alpha"}

    def test_multiple_values(self):
        assert _normalize_filter(["alpha", "beta"]) == {"alpha", "beta"}

    def test_comma_separated_split(self):
        assert _normalize_filter(["alpha,beta"]) == {"alpha", "beta"}

    def test_comma_and_repeat_deduped(self):
        assert _normalize_filter(["alpha,beta", "alpha"]) == {"alpha", "beta"}

    def test_whitespace_stripped(self):
        assert _normalize_filter([" alpha , beta "]) == {"alpha", "beta"}

    def test_empty_parts_ignored(self):
        assert _normalize_filter(["alpha,,beta"]) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# _experiment_name
# ---------------------------------------------------------------------------


class TestExperimentName:
    def _make(self, scenario_name: str, model_label: str, task_id: str) -> str:
        scenario = ScenarioConfig(name=scenario_name)
        model = ModelSpec.parse(model_label)
        task = TaskConfig(id=task_id, prompt="Q", criteria=("c",))
        return _experiment_name(scenario, model, task)

    def test_basic_format(self):
        name = self._make("fat-mcp", "anthropic/claude-sonnet-4-6", "ssi_overview")
        # slugify turns "/" to "-"
        assert "fat-mcp" in name
        assert "anthropic-claude-sonnet-4-6" in name
        assert "ssi_overview" in name
        assert name.count("|") == 2

    def test_length_at_most_180(self):
        long_scenario = "s" * 90
        long_task = "t" * 90
        name = self._make(long_scenario, "anthropic/claude-3-haiku-20240307", long_task)
        assert len(name) <= 180

    def test_special_chars_slugified(self):
        name = self._make("my scenario!", "openai/gpt-4o", "task/one")
        # Spaces and special chars become dashes; pipes in parts become dashes
        assert "!" not in name
        assert " " not in name


# ---------------------------------------------------------------------------
# _managed_server_specs
# ---------------------------------------------------------------------------


class TestManagedServerSpecs:
    def _make_server(
        self, name: str, *, url: str = "http://localhost/mcp", command: str | None = None
    ) -> McpServerConfig:
        return McpServerConfig(name=name, url=url, command=command)

    def test_empty_scenarios_returns_empty(self):
        assert _managed_server_specs([]) == []

    def test_servers_without_command_excluded(self):
        scenario = ScenarioConfig(
            name="s1",
            mcp_servers=(self._make_server("s1", url="http://localhost:8000/mcp"),),
        )
        specs = _managed_server_specs([scenario])
        assert specs == []

    def test_server_with_command_included(self):
        scenario = ScenarioConfig(
            name="s1",
            mcp_servers=(self._make_server("managed", command="my-server"),),
        )
        specs = _managed_server_specs([scenario])
        assert len(specs) == 1
        assert specs[0].name == "managed"

    def test_deduplication_by_name_and_url(self):
        server = self._make_server("managed", url="http://localhost:8000/mcp", command="my-server")
        s1 = ScenarioConfig(name="s1", mcp_servers=(server,))
        s2 = ScenarioConfig(name="s2", mcp_servers=(server,))
        specs = _managed_server_specs([s1, s2])
        assert len(specs) == 1

    def test_different_servers_not_deduped(self):
        s1_server = self._make_server("s1-srv", url="http://localhost:8001/mcp", command="start-s1")
        s2_server = self._make_server("s2-srv", url="http://localhost:8002/mcp", command="start-s2")
        sc1 = ScenarioConfig(name="sc1", mcp_servers=(s1_server,))
        sc2 = ScenarioConfig(name="sc2", mcp_servers=(s2_server,))
        specs = _managed_server_specs([sc1, sc2])
        assert len(specs) == 2

    def test_mix_of_managed_and_unmanaged(self):
        managed = self._make_server("mgd", command="start")
        unmanaged = self._make_server("nope")
        scenario = ScenarioConfig(name="s", mcp_servers=(managed, unmanaged))
        specs = _managed_server_specs([scenario])
        assert len(specs) == 1
        assert specs[0].name == "mgd"


# ---------------------------------------------------------------------------
# per-run workspaces (_run_cell)
# ---------------------------------------------------------------------------


class TestPerRunWorkspaces:
    def test_fresh_workspace_and_runner_per_run(self, tmp_path, monkeypatch):
        from airedale import experiment as exp_mod
        from airedale.progress import ProgressReporter
        from airedale.types import HarnessResult, UsageMetrics
        from airedale.workdir import WorkspaceManager

        runs = 3

        class FakeExperiment:
            url = "http://example/exp"

            def __init__(self, task, n):
                self._task = task
                self._n = n

            async def run(self, jobs=1, raise_errors=False):
                # LLMObs invokes the task once per (record x run).
                for _ in range(self._n):
                    await self._task({"prompt": "hi"})
                return {"rows": []}

        captured_experiment_kwargs: dict = {}

        class FakeLLMObs:
            enabled = False

            @staticmethod
            def async_experiment(*, task, runs, **kwargs):
                captured_experiment_kwargs.update(kwargs)
                return FakeExperiment(task, runs)

            @staticmethod
            def flush():
                pass

            @staticmethod
            def annotate(**kwargs):
                pass

        monkeypatch.setattr(exp_mod, "LLMObs", FakeLLMObs)
        monkeypatch.setattr(exp_mod.dataset_module, "dataset_for_task", lambda ds, task: None)

        constructed_cwds: list[str] = []

        class FakeRunner:
            def __init__(self, cwd):
                self.cwd = cwd

            async def run(self, **kwargs):
                return HarnessResult(
                    answer="a",
                    usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2),
                    tool_calls=[],
                    harness="s1",
                )

        def fake_create_runner(model, *, scenario, gateway, cwd):
            constructed_cwds.append(cwd)
            return FakeRunner(cwd)

        monkeypatch.setattr(exp_mod, "create_runner", fake_create_runner)

        model = ModelSpec.parse("anthropic/claude-3-haiku-20240307")
        scenario = ScenarioConfig(name="s1")  # workdir=None -> empty temp dirs
        task = TaskConfig(id="t1", prompt="Q", criteria=("c",))
        config = ExperimentConfig(
            project="p", models=("anthropic/claude-3-haiku-20240307",), scenarios=(scenario,), tasks=(task,)
        )

        workspace_entries = 0

        async def run():
            nonlocal workspace_entries
            async with WorkspaceManager(config_dir=tmp_path) as wm:
                original = wm.workspace

                def counting_workspace(workdir):
                    nonlocal workspace_entries
                    workspace_entries += 1
                    return original(workdir)

                wm.workspace = counting_workspace
                await exp_mod._run_cell(
                    (model, scenario, task),
                    config=config,
                    gateway=None,
                    dataset=None,
                    judge_model="anthropic/claude-3-haiku-20240307",
                    runs=runs,
                    fail_fast=False,
                    progress=ProgressReporter(enabled=False),
                    workspace_manager=wm,
                )

        asyncio.run(run())

        # The experiment must carry prompt_version (= scenario name) so LLMObs
        # Experiments can compare scenarios against each other.
        assert captured_experiment_kwargs["tags"]["prompt_version"] == "s1"
        assert captured_experiment_kwargs["config"]["prompt_version"] == "s1"
        # The harness (native / pi) is part of the experiment data.
        assert captured_experiment_kwargs["tags"]["harness"] == "native"
        assert captured_experiment_kwargs["config"]["harness"] == "native"

        assert workspace_entries == runs
        assert len(constructed_cwds) == runs
        # Each run gets a distinct, fresh workspace.
        assert len(set(constructed_cwds)) == runs


# ---------------------------------------------------------------------------
# A pi scenario that cannot run is refused before anything starts
# ---------------------------------------------------------------------------


class TestPiScenarioRefusedBeforeStart:
    def _config(self, scenario: ScenarioConfig) -> ExperimentConfig:
        return ExperimentConfig(
            project="p",
            models=("anthropic/claude-sonnet-4-6",),
            scenarios=(scenario,),
            tasks=(TaskConfig(id="t", prompt="p", criteria=("c",)),),
        )

    def _spy(self, monkeypatch):
        import airedale.experiment as experiment_module

        calls: list[str] = []

        class _Managed:
            def __init__(self, specs):
                calls.append(f"managed_servers({[spec.name for spec in specs]})")

            async def __aenter__(self):
                return None

            async def __aexit__(self, *exc):
                return None

        monkeypatch.setattr(experiment_module, "managed_servers", _Managed)
        monkeypatch.setattr(experiment_module, "enable_llmobs", lambda *a, **k: calls.append("enable_llmobs"))
        monkeypatch.setattr(
            experiment_module.dataset_module, "get_or_create_dataset", lambda *a, **k: calls.append("dataset")
        )
        return calls

    @pytest.mark.parametrize(
        ("servers", "skills", "message"),
        [
            # A name that pi does not accept: the managed server would be started for nothing.
            ((McpServerConfig(name="bad.name", url="http://localhost:8000/mcp", command="serve"),), (), "bad.name"),
            (
                (McpServerConfig(name="ok", url="http://localhost:8000/mcp", command="serve", tool_names=("a*",)),),
                (),
                "ok",
            ),
            (
                (
                    McpServerConfig(name="docs-api", url="http://localhost:8000/mcp", command="serve"),
                    McpServerConfig(name="docs_api", url="http://localhost:8001/mcp"),
                ),
                (),
                "same name for pi",
            ),
            (
                (McpServerConfig(name="ok", url="http://localhost:8000/mcp", command="serve"),),
                ("/does/not/exist",),
                "does not exist",
            ),
        ],
    )
    def test_run_experiments_starts_nothing(self, monkeypatch, servers, skills, message):
        from airedale.experiment import run_experiments

        calls = self._spy(monkeypatch)
        scenario = ScenarioConfig(name="s", harness="pi", mcp_servers=servers, skills=skills)
        with pytest.raises(ValueError, match=message) as error:
            asyncio.run(run_experiments(self._config(scenario), gateway=None, show_progress=False))
        assert "Scenario 's' cannot run with pi" in str(error.value)
        assert calls == []

    def test_possible_tool_name_collision_starts_nothing(self, monkeypatch):
        from airedale.experiment import run_experiments

        calls = self._spy(monkeypatch)
        servers = (
            McpServerConfig(name="a", url="http://localhost:8000/mcp", command="serve", tool_names=("b__c",)),
            McpServerConfig(name="a__b", url="http://localhost:8001/mcp", tool_names=("c",)),
        )
        scenario = ScenarioConfig(name="s", harness="pi", mcp_servers=servers)
        with pytest.raises(ValueError, match="collision"):
            asyncio.run(run_experiments(self._config(scenario), gateway=None, show_progress=False))
        assert calls == []

    def test_other_harness_is_not_checked(self):
        from airedale.harness import check_scenario_for_pi

        servers = (McpServerConfig(name="bad.name", url="http://localhost:8000/mcp", command="serve"),)
        check_scenario_for_pi(ScenarioConfig(name="s", harness="native", mcp_servers=servers))

    def test_a_valid_pi_scenario_passes(self):
        from airedale.harness import check_scenario_for_pi

        servers = (McpServerConfig(name="ok", url="http://localhost:8000/mcp", command="serve"),)
        check_scenario_for_pi(ScenarioConfig(name="s", harness="pi", mcp_servers=servers))
