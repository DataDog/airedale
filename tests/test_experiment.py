"""Tests for dd_ai_devx_evals.experiment — pure helper functions."""

from __future__ import annotations

from dd_ai_devx_evals.config.experiment import McpServerConfig, ScenarioConfig, TaskConfig
from dd_ai_devx_evals.experiment import _experiment_name, _managed_server_specs, _normalize_filter
from dd_ai_devx_evals.types import ModelSpec

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
        self, name: str, *, url: str = "http://localhost/mcp", start_command: str | None = None
    ) -> McpServerConfig:
        return McpServerConfig(name=name, url=url, start_command=start_command)

    def test_empty_scenarios_returns_empty(self):
        assert _managed_server_specs([]) == []

    def test_servers_without_start_command_excluded(self):
        scenario = ScenarioConfig(
            name="s1",
            mcp_servers=(self._make_server("s1", url="http://localhost:8000/mcp"),),
        )
        specs = _managed_server_specs([scenario])
        assert specs == []

    def test_server_with_start_command_included(self):
        scenario = ScenarioConfig(
            name="s1",
            mcp_servers=(self._make_server("managed", start_command="python -m server"),),
        )
        specs = _managed_server_specs([scenario])
        assert len(specs) == 1
        assert specs[0].name == "managed"

    def test_deduplication_by_name_and_url(self):
        server = self._make_server("managed", url="http://localhost:8000/mcp", start_command="python -m server")
        s1 = ScenarioConfig(name="s1", mcp_servers=(server,))
        s2 = ScenarioConfig(name="s2", mcp_servers=(server,))
        specs = _managed_server_specs([s1, s2])
        assert len(specs) == 1

    def test_different_servers_not_deduped(self):
        s1_server = self._make_server("s1-srv", url="http://localhost:8001/mcp", start_command="start-s1")
        s2_server = self._make_server("s2-srv", url="http://localhost:8002/mcp", start_command="start-s2")
        sc1 = ScenarioConfig(name="sc1", mcp_servers=(s1_server,))
        sc2 = ScenarioConfig(name="sc2", mcp_servers=(s2_server,))
        specs = _managed_server_specs([sc1, sc2])
        assert len(specs) == 2

    def test_mix_of_managed_and_unmanaged(self):
        managed = self._make_server("mgd", start_command="start")
        unmanaged = self._make_server("nope")
        scenario = ScenarioConfig(name="s", mcp_servers=(managed, unmanaged))
        specs = _managed_server_specs([scenario])
        assert len(specs) == 1
        assert specs[0].name == "mgd"
