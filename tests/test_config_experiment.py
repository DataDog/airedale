"""Tests for dd_ai_devx_evals.config.experiment — load_experiment parsing and validation."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dd_ai_devx_evals.config import ConfigError
from dd_ai_devx_evals.config.experiment import load_experiment

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "experiment.toml"
    p.write_text(textwrap.dedent(content))
    return p


MINIMAL = """
project = "test-project"
models = ["anthropic/claude-3-haiku-20240307"]

[scenarios.default]
description = "Default scenario"

[[tasks]]
id = "task1"
prompt = "What is X?"
criteria = ["Defines X correctly"]
"""


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------


class TestLoadExperimentValid:
    def test_minimal_loads(self, tmp_path):
        config = load_experiment(write_toml(tmp_path, MINIMAL))
        assert config.project == "test-project"
        assert len(config.models) == 1
        assert len(config.scenarios) == 1
        assert len(config.tasks) == 1

    def test_dataset_name_defaults_to_project(self, tmp_path):
        config = load_experiment(write_toml(tmp_path, MINIMAL))
        assert config.dataset_name == "test-project"

    def test_explicit_dataset_name(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        dataset_name = "my-dataset"

        [scenarios.s1]

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.dataset_name == "my-dataset"

    def test_defaults_applied_to_scenario(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [defaults]
        max_turns = 32
        effort = "low"

        [scenarios.s1]
        description = "scenario"

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        scenario = config.scenarios[0]
        assert scenario.max_turns == 32
        assert scenario.effort == "low"

    def test_scenario_overrides_defaults(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [defaults]
        max_turns = 32
        effort = "low"

        [scenarios.s1]
        max_turns = 64
        effort = "high"

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.scenarios[0].max_turns == 64
        assert config.scenarios[0].effort == "high"

    def test_mcp_http_server_parsed(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [scenarios.s1.mcp_servers.apm]
        url = "http://localhost:8000/mcp"
        headers = { source = "evals" }
        bearer_token_env_var = "APM_TOKEN"
        tool_names = ["search_apm"]
        start_command = "python -m server"

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        server = config.scenarios[0].mcp_servers[0]
        assert server.name == "apm"
        assert server.url == "http://localhost:8000/mcp"
        assert server.headers == {"source": "evals"}
        assert server.bearer_token_env_var == "APM_TOKEN"
        assert server.tool_names == ("search_apm",)
        assert server.start_command == "python -m server"

    def test_mcp_stdio_server_parsed(self, tmp_path):
        toml = """
        project = "p"
        models = ["openai/gpt-4o"]

        [scenarios.local.mcp_servers.tools]
        command = "python"
        args = ["-m", "my_server"]
        env = { FOO = "bar" }
        tool_names = ["do_thing"]

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        server = config.scenarios[0].mcp_servers[0]
        assert server.command == "python"
        assert server.args == ("-m", "my_server")
        assert server.env == {"FOO": "bar"}
        assert server.tool_names == ("do_thing",)

    def test_multiple_models(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307", "openai/gpt-4o"]

        [scenarios.s1]
        description = "s"

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert len(config.models) == 2
        assert "anthropic/claude-3-haiku-20240307" in config.models
        assert "openai/gpt-4o" in config.models

    def test_task_with_context(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [scenarios.s1]

        [[tasks]]
        id = "t1"
        prompt = "What is X?"
        context = "Context goes here"
        criteria = ["c1", "c2"]
        latency_threshold_ms = 5000
        """
        config = load_experiment(write_toml(tmp_path, toml))
        task = config.tasks[0]
        assert task.context == "Context goes here"
        assert task.criteria == ("c1", "c2")
        assert task.latency_threshold_ms == 5000

    def test_scenario_with_skills_and_tools(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [scenarios.s1]
        skills = ["./skills/apm"]
        allowed_builtin_tools = ["Read", "Grep"]

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        scenario = config.scenarios[0]
        assert scenario.skills == ("./skills/apm",)
        assert scenario.allowed_builtin_tools == ("Read", "Grep")


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestLoadExperimentErrors:
    def test_unknown_top_level_key(self, tmp_path):
        # extra_key must appear BEFORE [[tasks]] to land at the top level in TOML
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        extra_key = "boom"

        [scenarios.s1]

        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Unknown top-level keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_project(self, tmp_path):
        toml = """
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="project"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_models(self, tmp_path):
        toml = """
        project = "p"
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="models"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_scenarios(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="scenarios"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_tasks(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        """
        with pytest.raises(ConfigError, match="tasks"):
            load_experiment(write_toml(tmp_path, toml))

    def test_bad_model_string_no_slash(self, tmp_path):
        toml = """
        project = "p"
        models = ["gpt-4"]
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Invalid model"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_provider(self, tmp_path):
        toml = """
        project = "p"
        models = ["gemini/pro"]
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Invalid model"):
            load_experiment(write_toml(tmp_path, toml))

    def test_mcp_server_both_url_and_command(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.mcp_servers.bad]
        url = "http://localhost:8000/mcp"
        command = "python"
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="exactly one of"):
            load_experiment(write_toml(tmp_path, toml))

    def test_mcp_server_neither_url_nor_command(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.mcp_servers.bad]
        tool_names = ["tool"]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="exactly one of"):
            load_experiment(write_toml(tmp_path, toml))

    def test_task_empty_criteria(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = []
        """
        with pytest.raises(ConfigError, match="criterion"):
            load_experiment(write_toml(tmp_path, toml))

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_experiment(tmp_path / "nonexistent.toml")
