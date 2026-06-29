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

[tasks.task1]
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

        [tasks.t1]
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

        [tasks.t1]
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

        [tasks.t1]
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

        [mcp_servers.apm]
        url = "http://localhost:8000/mcp"
        headers = { source = "evals" }
        tool_names = ["search_apm"]

        [scenarios.s1]
        mcp_servers = ["apm"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        server = config.scenarios[0].mcp_servers[0]
        assert server.name == "apm"
        assert server.type == "http"
        assert server.url == "http://localhost:8000/mcp"
        assert server.headers == {"source": "evals"}
        assert server.tool_names == ("search_apm",)
        assert not server.is_managed

    def test_mcp_stdio_server_parsed(self, tmp_path):
        toml = """
        project = "p"
        models = ["openai/gpt-4o"]

        [mcp_servers.tools]
        command = "python"
        args = ["-m", "my_server"]
        env = { FOO = "bar" }
        tool_names = ["do_thing"]

        [scenarios.local]
        mcp_servers = ["tools"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        server = config.scenarios[0].mcp_servers[0]
        assert server.type == "stdio"
        assert server.command == "python"
        assert server.args == ("-m", "my_server")
        assert server.env == {"FOO": "bar"}
        assert server.tool_names == ("do_thing",)
        assert not server.is_managed

    def test_multiple_models(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307", "openai/gpt-4o"]

        [scenarios.s1]
        description = "s"

        [tasks.t1]
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

        [tasks.t1]
        prompt = "What is X?"
        context = "Context goes here"
        criteria = ["c1", "c2"]
        latency_threshold_ms = 5000
        """
        config = load_experiment(write_toml(tmp_path, toml))
        task = config.tasks[0]
        assert task.id == "t1"  # id comes from the [tasks.<id>] table key
        assert task.context == "Context goes here"
        assert task.criteria == ("c1", "c2")
        assert task.latency_threshold_ms == 5000

    def test_scenario_with_skills_and_tools(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [skills]
        apm = "./skills/apm"

        [scenarios.s1]
        skills = ["apm"]
        allowed_builtin_tools = ["Read", "Grep"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        scenario = config.scenarios[0]
        assert scenario.skills == (str((tmp_path / "skills" / "apm").resolve()),)
        assert scenario.allowed_builtin_tools == ("Read", "Grep")

    def test_shared_registry_referenced_by_multiple_scenarios(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [skills]
        apm = "./skills/apm"

        [mcp_servers.apm]
        url = "http://localhost:8000/mcp"
        tool_names = ["search_apm"]

        [scenarios.a]
        mcp_servers = ["apm"]
        skills = ["apm"]

        [scenarios.b]
        mcp_servers = ["apm"]
        skills = ["apm"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        a, b = config.scenarios
        assert a.mcp_servers[0].name == "apm"
        assert b.mcp_servers[0].name == "apm"
        apm_dir = str((tmp_path / "skills" / "apm").resolve())
        assert a.skills == (apm_dir,) == b.skills

    def test_allowed_builtin_tools_unset_is_none(self, tmp_path):
        config = load_experiment(write_toml(tmp_path, MINIMAL))
        # Omitted -> None sentinel meaning "all built-in tools allowed".
        assert config.scenarios[0].allowed_builtin_tools is None

    def test_allowed_builtin_tools_empty_list_is_empty_tuple(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [scenarios.s1]
        allowed_builtin_tools = []

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        # Explicit empty list -> no built-in tools (distinct from unset).
        assert config.scenarios[0].allowed_builtin_tools == ()

    def test_defaults_apply_extended_fields(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [skills]
        apm = "./skills/apm"

        [mcp_servers.apm]
        url = "http://localhost:8000/mcp"

        [defaults]
        system_prompt = "base prompt"
        skills = ["apm"]
        allowed_builtin_tools = ["Read"]
        mcp_servers = ["apm"]

        [scenarios.inherits]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        s = config.scenarios[0]
        assert s.system_prompt == "base prompt"
        assert s.skills == (str((tmp_path / "skills" / "apm").resolve()),)
        assert s.allowed_builtin_tools == ("Read",)
        assert s.mcp_servers[0].name == "apm"

    def test_scenario_replaces_defaults_entirely(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [skills]
        apm = "./skills/apm"
        k8s = "./skills/k8s"

        [defaults]
        skills = ["apm"]

        [scenarios.override]
        skills = ["k8s"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        # Scenario value wins entirely; no union with defaults.
        assert config.scenarios[0].skills == (str((tmp_path / "skills" / "k8s").resolve()),)

    def test_skill_paths_resolved_relative_to_config_dir(self, tmp_path, monkeypatch):
        """A relative [skills] path resolves against the config dir, not the CWD."""
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]

        [skills]
        apm = "./skills/apm"

        [scenarios.s1]
        skills = ["apm"]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config_file = write_toml(tmp_path, toml)
        # Run the loader from an unrelated CWD to prove resolution is config-relative.
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)
        config = load_experiment(config_file)
        assert config.scenarios[0].skills == (str((tmp_path / "skills" / "apm").resolve()),)
        assert config.config_dir == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestLoadExperimentErrors:
    def test_unknown_top_level_key(self, tmp_path):
        # extra_key must appear BEFORE [tasks.<id>] to land at the top level in TOML
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        extra_key = "boom"

        [scenarios.s1]

        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Unknown top-level keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_project(self, tmp_path):
        toml = """
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="project"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_models(self, tmp_path):
        toml = """
        project = "p"
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="models"):
            load_experiment(write_toml(tmp_path, toml))

    def test_missing_scenarios(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [tasks.t1]
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
        [tasks.t1]
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
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Invalid model"):
            load_experiment(write_toml(tmp_path, toml))

    def test_mcp_server_managed_localhost_url_valid(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.managed]
        url = "http://localhost:8000/mcp"
        command = "my-server"
        args = ["--port", "8000"]
        [scenarios.s1]
        mcp_servers = ["managed"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        server = config.scenarios[0].mcp_servers[0]
        assert server.type == "http"
        assert server.is_managed
        assert server.command == "my-server"
        assert server.args == ("--port", "8000")

    def test_mcp_server_managed_non_localhost_url_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.bad]
        url = "http://example.com:8000/mcp"
        command = "my-server"
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="not localhost"):
            load_experiment(write_toml(tmp_path, toml))

    def test_mcp_server_managed_loopback_ipv4_valid(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.managed]
        url = "http://127.0.0.5:8000/mcp"
        command = "my-server"
        [scenarios.s1]
        mcp_servers = ["managed"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.scenarios[0].mcp_servers[0].is_managed

    def test_mcp_server_managed_loopback_ipv6_valid(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.managed]
        url = "http://[::1]:8000/mcp"
        command = "my-server"
        [scenarios.s1]
        mcp_servers = ["managed"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.scenarios[0].mcp_servers[0].is_managed

    def test_mcp_server_neither_url_nor_command(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.bad]
        tool_names = ["tool"]
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="must define 'url'"):
            load_experiment(write_toml(tmp_path, toml))

    def test_mcp_server_unsupported_type_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [mcp_servers.bad]
        type = "sse"
        url = "http://localhost:8000/mcp"
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unsupported type"):
            load_experiment(write_toml(tmp_path, toml))

    def test_task_empty_criteria(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = []
        """
        with pytest.raises(ConfigError, match="criterion"):
            load_experiment(write_toml(tmp_path, toml))

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_experiment(tmp_path / "nonexistent.toml")

    def test_scenario_references_unknown_mcp_server(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        mcp_servers = ["missing"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown MCP server 'missing'"):
            load_experiment(write_toml(tmp_path, toml))

    def test_scenario_references_unknown_skill(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        skills = ["./skills/apm"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown skill"):
            load_experiment(write_toml(tmp_path, toml))

    def test_inline_mcp_server_table_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.mcp_servers.apm]
        url = "http://localhost:8000/mcp"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="not an inline table"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_scenario_key_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        bogus = "x"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_defaults_key_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [defaults]
        bogus = "x"
        [scenarios.s1]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Unknown \\[defaults\\] keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_tasks_array_of_tables_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [[tasks]]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="must be a table keyed by id"):
            load_experiment(write_toml(tmp_path, toml))

    def test_duplicate_task_id_rejected(self, tmp_path):
        # TOML forbids duplicate keys, so a repeated [tasks.<id>] is a parse error.
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [tasks.dup]
        prompt = "a"
        criteria = ["c"]
        [tasks.dup]
        prompt = "b"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_task_key_rejected(self, tmp_path):
        # An in-body `id` is now an unknown key (the id comes from the table key).
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [tasks.t1]
        id = "t1"
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_task_missing_prompt_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1]
        [tasks.t1]
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="missing required 'prompt'"):
            load_experiment(write_toml(tmp_path, toml))
