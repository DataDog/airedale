"""Tests for the Codex runner's environment wiring.

`CodexRunner._codex_env()` **always** isolates `CODEX_HOME` to a fresh per-run
dir so the operator's global `~/.codex/config.toml` (MCP servers etc.) never
leaks into a run. Auth is preserved either via env (gateway token /
`OPENAI_API_KEY`) or by seeding only `auth.json` from the operator's real
`CODEX_HOME` into the isolated dir (so `codex login` keeps working without the
global config).
"""

from __future__ import annotations

from types import SimpleNamespace

from dd_ai_devx_evals.harness.base import AgentThought, AgentToolCall
from dd_ai_devx_evals.harness.codex import (
    CODEX_BUILTIN_TOOLS,
    ISOLATED_CODEX_HOME_DIRNAME,
    CodexRunner,
    _codex_output_segments,
)


def _item(**fields):
    """Build a Codex ThreadItem-like object (attribute access, ``.root`` = self)."""
    ns = SimpleNamespace(**fields)
    ns.root = ns
    return ns


def _operator_home_with_auth(tmp_path, monkeypatch, *, with_auth: bool):
    """Point CODEX_HOME at a throwaway operator home, optionally with auth.json."""
    operator = tmp_path / "operator-codex-home"
    operator.mkdir()
    if with_auth:
        (operator / "auth.json").write_text('{"tokens": "operator"}')
    # Also drop a config.toml the run must NOT pick up.
    (operator / "config.toml").write_text("[mcp_servers.leaky]\nurl = 'http://x'\n")
    monkeypatch.setenv("CODEX_HOME", str(operator))
    return operator


class TestCodexOutputSegments:
    """The extraction must capture built-in tool activity, not just MCP calls.

    Regression: a Codex run that only shelled into a checkout previously reported
    zero tool usage because ``commandExecution`` items were dropped.
    """

    def test_command_execution_becomes_shell_tool_call(self):
        item = _item(
            type="commandExecution",
            id="cmd_1",
            command="ls -la",
            cwd="/work",
            aggregated_output="total 0",
            exit_code=0,
            status=SimpleNamespace(value="completed"),
            duration_ms=12,
            source=SimpleNamespace(value="agent"),
        )
        (call,) = _codex_output_segments([item])
        assert isinstance(call, AgentToolCall)
        assert call.name == "shell"
        assert call.arguments == {"command": "ls -la", "cwd": "/work"}
        assert call.result == "total 0"
        assert call.error is None
        assert call.metadata["exit_code"] == 0

    def test_failed_command_records_error(self):
        item = _item(
            type="commandExecution",
            id="cmd_2",
            command="false",
            cwd="/work",
            aggregated_output="",
            exit_code=1,
            status=SimpleNamespace(value="failed"),
        )
        (call,) = _codex_output_segments([item])
        assert call.error == "failed"

    def test_file_change_becomes_apply_patch_tool_call(self):
        item = _item(
            type="fileChange",
            id="fc_1",
            status=SimpleNamespace(value="completed"),
            # kind is a PatchChangeKind RootModel: .root.type carries the literal.
            changes=[SimpleNamespace(path="a.py", kind=SimpleNamespace(root=SimpleNamespace(type="add")), diff="+x")],
        )
        (call,) = _codex_output_segments([item])
        assert call.name == "apply_patch"
        assert call.arguments == {"changes": [{"path": "a.py", "kind": "add"}]}
        assert call.result[0]["kind"] == "add"
        assert call.metadata["file_count"] == 1

    def test_declined_command_records_error(self):
        item = _item(
            type="commandExecution",
            id="cmd_3",
            command="rm -rf /",
            cwd="/work",
            aggregated_output="",
            exit_code=None,
            status=SimpleNamespace(value="declined"),
        )
        (call,) = _codex_output_segments([item])
        assert call.error == "declined"

    def test_web_search_becomes_tool_call(self):
        item = _item(type="webSearch", id="ws_1", query="what is ssi", action=None)
        (call,) = _codex_output_segments([item])
        assert call.name == "web_search"
        assert call.arguments == {"query": "what is ssi"}

    def test_reasoning_and_plan_become_thoughts_and_preserve_order(self):
        items = [
            _item(type="reasoning", id="r1", summary=["first I look"], content=[]),
            _item(
                type="commandExecution",
                id="c1",
                command="cat README",
                cwd="/w",
                aggregated_output="docs",
                exit_code=0,
                status=SimpleNamespace(value="completed"),
            ),
            _item(type="plan", id="p1", text="1. summarize"),
        ]
        segments = _codex_output_segments(items)
        assert [type(s).__name__ for s in segments] == ["AgentThought", "AgentToolCall", "AgentThought"]
        assert segments[0] == AgentThought(text="first I look", kind="reasoning")
        assert segments[2] == AgentThought(text="1. summarize", kind="plan")

    def test_empty_reasoning_is_dropped(self):
        assert _codex_output_segments([_item(type="reasoning", id="r", summary=[], content=[])]) == []

    def test_unknown_item_types_are_skipped(self):
        items = [
            _item(type="userMessage", id="u"),
            _item(type="agentMessage", id="a", text="the answer"),
            _item(type="contextCompaction", id="cc"),
        ]
        assert _codex_output_segments(items) == []


class TestCodexBuiltinTools:
    def test_all_allowed_expands_to_full_builtin_set(self, tmp_path):
        runner = CodexRunner(cwd=tmp_path)
        assert runner._effective_builtin_tools() == CODEX_BUILTIN_TOOLS

    def test_explicit_allow_list_still_reports_full_set(self, tmp_path):
        # Codex cannot gate built-ins, so the available-tools report is the full
        # set regardless of the scenario's (informational) allow-list.
        runner = CodexRunner(cwd=tmp_path, allowed_builtin_tools=["shell"])
        assert runner._effective_builtin_tools() == CODEX_BUILTIN_TOOLS


class TestCodexEnv:
    def test_always_isolates_codex_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        isolated = cwd / ISOLATED_CODEX_HOME_DIRNAME
        assert env["CODEX_HOME"] == str(isolated)
        assert isolated.is_dir()
        # The isolated home does not contain the operator's global config.
        assert not (isolated / "config.toml").exists()

    def test_codex_login_auth_seeded_without_env_auth(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        seeded = cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json"
        # auth.json is copied in (codex login preserved) but config.toml is not.
        assert seeded.read_text() == '{"tokens": "operator"}'
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "config.toml").exists()
        assert "OPENAI_BASE_URL" not in env

    def test_env_api_key_does_not_seed_auth_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        # With env auth we rely on OPENAI_API_KEY, not the operator's auth.json.
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)

    def test_gateway_sets_base_url_token_and_isolates(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(
            cwd=cwd,
            gateway_base_url="https://gw.example.com/v1",
            gateway_token="bearer-xyz",
        )
        env = runner._codex_env()
        assert env["OPENAI_BASE_URL"] == "https://gw.example.com/v1"
        assert env["OPENAI_API_KEY"] == "bearer-xyz"
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)
        # Gateway token is env auth -> no auth.json seeding.
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()

    def test_missing_operator_auth_is_tolerated(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        # No env auth and no operator auth.json: still isolates, nothing seeded.
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()
