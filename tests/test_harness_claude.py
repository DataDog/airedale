# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the Claude runner's tool allow-list construction.

``_claude_available_tools`` feeds ``ClaudeAgentOptions.allowed_tools`` under
``permission_mode="dontAsk"`` (only pre-approved tools may run), so it encodes
the built-in-tool semantics: omitted -> all built-ins; ``[]`` -> none; an
explicit list -> exactly those.
"""

from __future__ import annotations

import json

from airedale.config.experiment import ScenarioConfig
from airedale.harness import create_runner
from airedale.harness.claude import CLAUDE_BUILTIN_TOOLS, ClaudeRunner
from airedale.mcp import McpServerSpec
from airedale.types import ModelSpec


class TestClaudeAvailableTools:
    def test_unset_allows_all_builtins(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path, allowed_builtin_tools=None)
        tools = runner._claude_available_tools()
        assert set(CLAUDE_BUILTIN_TOOLS) <= set(tools)

    def test_empty_list_allows_no_builtins(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path, allowed_builtin_tools=[])
        assert runner._claude_available_tools() == []

    def test_explicit_list_is_exact(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path, allowed_builtin_tools=["Read", "Grep"])
        assert runner._claude_available_tools() == ["Read", "Grep"]

    def test_mcp_tools_appended_with_allow_list(self, tmp_path):
        spec = McpServerSpec(name="apm", url="http://localhost:8000/mcp", tool_names=("search_apm",))
        runner = ClaudeRunner(cwd=tmp_path, allowed_builtin_tools=["Read"], mcp_servers=[spec])
        tools = runner._claude_available_tools()
        assert tools == ["Read", "mcp__apm__search_apm"]

    def test_mcp_server_without_tool_names_uses_prefix(self, tmp_path):
        spec = McpServerSpec(name="apm", url="http://localhost:8000/mcp")
        runner = ClaudeRunner(cwd=tmp_path, allowed_builtin_tools=[], mcp_servers=[spec])
        assert runner._claude_available_tools() == ["mcp__apm"]


ANTHROPIC_MODEL = ModelSpec.parse("anthropic/claude-3-haiku-20240307")


class TestClaudeBuildOptions:
    def test_setting_sources_always_project(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path)
        options = runner._build_options(model=ANTHROPIC_MODEL, system_prompt="", trace_headers={})
        assert options.setting_sources == ["project"]

    def test_strict_mcp_config_stays_true(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path)
        options = runner._build_options(model=ANTHROPIC_MODEL, system_prompt="", trace_headers={})
        assert options.strict_mcp_config is True

    def test_skills_allow_list_from_staged_dir(self, tmp_path):
        # A scenario skill plus a repo-provided skill both end up in the allow-list.
        source = tmp_path / "sources" / "scenario-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("# scenario\n")
        repo_skill = tmp_path / "cwd" / ".claude" / "skills" / "repo-skill"
        repo_skill.mkdir(parents=True)
        (repo_skill / "SKILL.md").write_text("# repo\n")
        runner = ClaudeRunner(cwd=tmp_path / "cwd", skills=[str(source)])
        options = runner._build_options(model=ANTHROPIC_MODEL, system_prompt="", trace_headers={})
        assert set(options.skills) == {"scenario-skill", "repo-skill"}


class TestCreateRunnerProjectMcpMerge:
    def _scenario(self, **kwargs) -> ScenarioConfig:
        return ScenarioConfig(name="s1", **kwargs)

    def test_discovered_servers_merged(self, tmp_path):
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"repo-http": {"url": "http://localhost:9000/mcp"}}})
        )
        runner = create_runner(ANTHROPIC_MODEL, scenario=self._scenario(), gateway=None, cwd=str(tmp_path))
        names = {s.name for s in runner.mcp_servers}
        assert "repo-http" in names
        # Discovered http server's mcp__ allow-list entry is emitted.
        assert "mcp__repo-http" in runner._claude_available_tools()

    def test_scenario_wins_on_name_collision(self, tmp_path):
        from airedale.config.experiment import McpServerConfig

        (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"apm": {"url": "http://localhost:9999/repo"}}}))
        scenario_server = McpServerConfig(name="apm", url="http://localhost:8000/scenario")
        runner = create_runner(
            ANTHROPIC_MODEL,
            scenario=self._scenario(mcp_servers=(scenario_server,)),
            gateway=None,
            cwd=str(tmp_path),
        )
        apm = [s for s in runner.mcp_servers if s.name == "apm"]
        assert len(apm) == 1
        assert apm[0].url == "http://localhost:8000/scenario"

    def test_discovered_http_gets_trace_headers(self, tmp_path):
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"repo-http": {"url": "http://localhost:9000/mcp"}}})
        )
        runner = create_runner(ANTHROPIC_MODEL, scenario=self._scenario(), gateway=None, cwd=str(tmp_path))
        spec = next(s for s in runner.mcp_servers if s.name == "repo-http")
        config = spec.to_claude_config({"x-dd-trace-id": "123"})
        assert config["headers"]["x-dd-trace-id"] == "123"

    def test_codex_does_not_read_mcp_json(self, tmp_path):
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"repo-http": {"url": "http://localhost:9000/mcp"}}})
        )
        openai_model = ModelSpec.parse("openai/gpt-4o")
        runner = create_runner(openai_model, scenario=self._scenario(), gateway=None, cwd=str(tmp_path))
        assert runner.mcp_servers == []
