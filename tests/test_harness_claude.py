"""Tests for the Claude runner's tool allow-list construction.

``_claude_available_tools`` feeds ``ClaudeAgentOptions.allowed_tools`` under
``permission_mode="dontAsk"`` (only pre-approved tools may run), so it encodes
the built-in-tool semantics: omitted -> all built-ins; ``[]`` -> none; an
explicit list -> exactly those.
"""

from __future__ import annotations

from dd_ai_devx_evals.harness.claude import CLAUDE_BUILTIN_TOOLS, ClaudeRunner
from dd_ai_devx_evals.mcp import McpServerSpec


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
