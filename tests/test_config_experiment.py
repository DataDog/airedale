# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Tests for airedale.config.experiment — load_experiment parsing and validation."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from airedale.config import ConfigError
from airedale.config.experiment import load_experiment

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
# workdir
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    """Initialize a throwaway local git repo (offline; harness mechanism)."""
    import os
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)


class TestWorkdir:
    def test_repo_self_resolves_in_git_repo(self, tmp_path):
        _init_git_repo(tmp_path)
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        repo = "self"
        ref = "main"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        wd = config.scenarios[0].workdir
        assert wd is not None
        assert wd.repo == "self"
        assert wd.ref == "main"
        assert wd.steps == ()

    def test_repo_self_outside_git_repo_errors(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        repo = "self"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        # tmp_path is not a git repo (and pytest tmp dirs are outside this repo).
        with pytest.raises(ConfigError, match="not inside a git repository"):
            load_experiment(write_toml(tmp_path, toml))

    def test_repo_url_kept_verbatim(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        repo = "https://github.com/example/repo.git"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.scenarios[0].workdir.repo == "https://github.com/example/repo.git"

    def test_repo_local_path_resolved_against_config_dir(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        repo = "./fixture-repo"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        assert config.scenarios[0].workdir.repo == str((tmp_path / "fixture-repo").resolve())

    def test_no_repo_starts_empty(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "main.py"
        content = "print('hi')\\n"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        wd = config.scenarios[0].workdir
        assert wd.repo is None
        assert len(wd.steps) == 1

    def test_all_step_types_parsed(self, tmp_path):
        _init_git_repo(tmp_path)
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        repo = "self"
        ref = "v2.3.0"
        [[scenarios.s1.workdir.steps]]
        op = "restore"
        from = "v2.2.0"
        paths = ["src/api/**", "README.md"]
        [[scenarios.s1.workdir.steps]]
        op = "remove"
        paths = ["secrets/**"]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "NOTES.md"
        content = "evaluate"
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "fixtures/input.json"
        source = "./fixtures/input.json"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        from airedale.config.experiment import RemoveStep, RestoreStep, WriteStep

        config = load_experiment(write_toml(tmp_path, toml))
        steps = config.scenarios[0].workdir.steps
        assert isinstance(steps[0], RestoreStep)
        assert steps[0].from_ref == "v2.2.0"
        assert steps[0].paths == ("src/api/**", "README.md")
        assert isinstance(steps[1], RemoveStep)
        assert steps[1].paths == ("secrets/**",)
        assert isinstance(steps[2], WriteStep)
        assert steps[2].content == "evaluate"
        assert steps[2].source_path is None
        assert isinstance(steps[3], WriteStep)
        assert steps[3].content is None
        assert steps[3].source_path == str((tmp_path / "fixtures" / "input.json").resolve())

    def test_workdir_defaultable_and_override_entirely(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [defaults.workdir]
        repo = "https://github.com/example/base.git"
        [scenarios.inherits]
        [scenarios.override.workdir]
        repo = "https://github.com/example/other.git"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        config = load_experiment(write_toml(tmp_path, toml))
        by_name = {s.name: s for s in config.scenarios}
        assert by_name["inherits"].workdir.repo == "https://github.com/example/base.git"
        assert by_name["override"].workdir.repo == "https://github.com/example/other.git"

    def test_restore_without_repo_errors(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "restore"
        from = "main"
        paths = ["x"]
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="no 'repo' to restore from"):
            load_experiment(write_toml(tmp_path, toml))

    def test_write_path_absolute_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "/etc/passwd"
        content = "x"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="absolute"):
            load_experiment(write_toml(tmp_path, toml))

    def test_write_path_dotdot_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "../escape.txt"
        content = "x"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="escape"):
            load_experiment(write_toml(tmp_path, toml))

    def test_write_both_content_and_source_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "f.txt"
        content = "x"
        source = "./f.txt"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="exactly one of 'content' or 'source'"):
            load_experiment(write_toml(tmp_path, toml))

    def test_write_neither_content_nor_source_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "write"
        path = "f.txt"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="exactly one of 'content' or 'source'"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_op_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "frobnicate"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown op"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_step_key_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        [[scenarios.s1.workdir.steps]]
        op = "remove"
        paths = ["x"]
        bogus = 1
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="unknown keys"):
            load_experiment(write_toml(tmp_path, toml))

    def test_unknown_workdir_key_rejected(self, tmp_path):
        toml = """
        project = "p"
        models = ["anthropic/claude-3-haiku-20240307"]
        [scenarios.s1.workdir]
        bogus = "x"
        [tasks.t1]
        prompt = "p"
        criteria = ["c"]
        """
        with pytest.raises(ConfigError, match="workdir has unknown keys"):
            load_experiment(write_toml(tmp_path, toml))


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
