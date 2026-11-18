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

import asyncio
import json
import logging
import os
import stat

import claude_agent_sdk
import pytest
from claude_agent_sdk._internal.transport import Transport

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
        assert isinstance(options.skills, list)
        assert set(options.skills) == {"scenario-skill", "repo-skill"}

    @pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
    def test_supported_effort_passed_through(self, tmp_path, effort):
        runner = ClaudeRunner(cwd=tmp_path, effort=effort)
        options = runner._build_options(model=ANTHROPIC_MODEL, system_prompt="", trace_headers={})
        assert options.effort == effort

    def test_unsupported_effort_rejected(self, tmp_path):
        runner = ClaudeRunner(cwd=tmp_path, effort="minimal")
        with pytest.raises(ValueError, match="unsupported Claude effort 'minimal'"):
            runner._build_options(model=ANTHROPIC_MODEL, system_prompt="", trace_headers={})


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
        assert isinstance(runner, ClaudeRunner)
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
        assert config == {"type": "http", "url": "http://localhost:9000/mcp", "headers": {"x-dd-trace-id": "123"}}

    def test_codex_does_not_read_mcp_json(self, tmp_path):
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"repo-http": {"url": "http://localhost:9000/mcp"}}})
        )
        openai_model = ModelSpec.parse("openai/gpt-4o")
        runner = create_runner(openai_model, scenario=self._scenario(), gateway=None, cwd=str(tmp_path))
        assert runner.mcp_servers == []


# --------------------------------------------------------------------------- #
# A failed run keeps the raw text of the Claude CLI and of the provider out of the logs
# --------------------------------------------------------------------------- #
STDERR_MARKER = "sk-STDERR-MARKER"
RESULT_MARKER = "sk-RESULT-MARKER"
READER_MARKER = "sk-READER-MARKER"
STDOUT_MARKER = "sk-STDOUT-MARKER"
RETRY_MARKER = "sk-RETRY-MARKER"
_END = object()


class _FakeTransport(Transport):
    """A Claude Code CLI that answers ``initialize``, then fails the run. ``options`` is set by the test.

    It writes the CLI stderr (``options.stderr``), emits an error ``ResultMessage`` and then either raises an exception
    in the reader (path 1) or ends the stream (path 2).
    """

    def __init__(self, *, raises: bool) -> None:
        self.options: claude_agent_sdk.ClaudeAgentOptions | None = None
        self._raises = raises
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready = False

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            message = json.loads(line)
            if message.get("type") == "control_request" and message["request"]["subtype"] == "initialize":
                response = {"subtype": "success", "request_id": message["request_id"], "response": {"commands": []}}
                self._queue.put_nowait({"type": "control_response", "response": response})
            elif message.get("type") == "user":
                assert self.options is not None and self.options.stderr is not None
                self.options.stderr(STDERR_MARKER)
                self._queue.put_nowait(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "duration_ms": 1,
                        "duration_api_ms": 1,
                        "is_error": True,
                        "num_turns": 1,
                        "session_id": "s",
                        "errors": [RESULT_MARKER],
                    }
                )
                self._queue.put_nowait(RuntimeError(f"stream broke {READER_MARKER}") if self._raises else _END)

    async def read_messages(self):
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def close(self) -> None:
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        pass


def _claude_runner(tmp_path, *, path: int):
    """``(runner, transport)``: path 1 uses ``query()`` with the transport, path 2 uses ``ClaudeSDKClient``."""
    transport = _FakeTransport(raises=path == 1)
    if path == 1:

        def query_func(*, prompt, options):
            transport.options = options
            return claude_agent_sdk.query(prompt=prompt, options=options, transport=transport)

        return ClaudeRunner(cwd=tmp_path, claude_query_func=query_func), transport

    def client_factory(options):
        transport.options = options
        return claude_agent_sdk.ClaudeSDKClient(options, transport=transport)

    return ClaudeRunner(cwd=tmp_path, claude_client_factory=client_factory), transport


def _failed_run(tmp_path, monkeypatch, *, path: int):
    """Run a Claude run that fails; return the raised error."""
    from airedale.progress import ProgressReporter

    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(tmp_path / "reports"))
    (tmp_path / "cwd").mkdir()
    runner, _transport = _claude_runner(tmp_path / "cwd", path=path)
    reporter = ProgressReporter(enabled=True)  # (Not started: plain mode, one line for each message.)
    with pytest.raises(Exception) as raised:  # noqa: PT011
        asyncio.run(
            asyncio.wait_for(
                runner.run(
                    model=ANTHROPIC_MODEL,
                    system_prompt="S",
                    user_prompt="Q",
                    prompt_version="v",
                    harness="h",
                    progress=reporter.message,
                ),
                timeout=60,
            )
        )
    return raised.value


def _records_text(caplog) -> str:
    return "\n".join(f"{record.name}: {record.getMessage()}|{record.exc_text}" for record in caplog.records)


@pytest.mark.parametrize("path", [1, 2], ids=["exception-in-the-stream", "error-result-message"])
def test_a_failed_claude_run_has_no_raw_text_in_the_logs_and_a_private_report(
    tmp_path, monkeypatch, caplog, capfd, path
):
    caplog.set_level(logging.DEBUG)
    error = _failed_run(tmp_path, monkeypatch, path=path)
    captured = capfd.readouterr()
    # O1 and O2: the log lines of Airedale and of the SDK have no raw text, no exception text and no traceback.
    text = _records_text(caplog)
    for marker in (STDERR_MARKER, RESULT_MARKER, READER_MARKER):
        assert marker not in text, marker
    assert all(record.exc_info is None for record in caplog.records if record.name.startswith("airedale"))
    [failure] = [r for r in caplog.records if r.getMessage().startswith("Claude Agent SDK eval run failed (")]
    assert failure.levelno == logging.ERROR
    reports = list((tmp_path / "reports").glob("claude-run-*.log"))
    if os.name == "nt":  # Windows writes no private report: the line says so, and a fixed warning is logged.
        assert reports == []
        assert "no private file was written" in failure.getMessage()
        assert any(
            r.getMessage() == "Unable to write the private report of the failed Claude run" for r in caplog.records
        )
    else:
        [report] = reports
        assert str(report) in failure.getMessage()
        assert stat.S_IMODE(report.stat().st_mode) == 0o600
        contents = report.read_text()
        assert STDERR_MARKER in contents and RESULT_MARKER in contents  # O2: the private report has them.
    # O5: the raised error is not changed (accepted). O6: the progress and the console output (accepted).
    assert (READER_MARKER if path == 1 else RESULT_MARKER) in str(error)
    assert RESULT_MARKER in captured.out + captured.err
    if path == 1:
        # O3: the SDK log of the error that stops its reader has a fixed text.
        [sdk_record] = [r for r in caplog.records if r.name == "claude_agent_sdk._internal.query" and r.levelno == 40]
        assert sdk_record.getMessage().startswith("Claude Agent SDK ERROR at query:")
        assert sdk_record.getMessage().endswith("(details hidden)")


def test_the_claude_sdk_stdout_debug_log_has_a_fixed_text(caplog):
    """O4: the SDK logs the first 200 characters of a stdout line of the CLI that is not JSON."""
    from claude_agent_sdk._internal.transport import subprocess_cli

    from airedale.safe_log import ensure_mcp_log_guard

    ensure_mcp_log_guard()
    caplog.set_level(logging.DEBUG)
    assert subprocess_cli._parse_stdout_line(f"not json {STDOUT_MARKER}") is None
    [record] = [r for r in caplog.records if r.name == "claude_agent_sdk._internal.transport.subprocess_cli"]
    assert STDOUT_MARKER not in record.getMessage()
    assert record.getMessage().startswith("Claude Agent SDK DEBUG at subprocess_cli:")
    assert record.getMessage().endswith("(details hidden)")


def test_the_ddtrace_retry_summary_is_not_changed(caplog):
    """O7 (accepted): the experiment summary of ddtrace has the text of the retries. Airedale does not change it."""
    from ddtrace.llmobs._experiment import Experiment

    experiment = object.__new__(Experiment)
    experiment.name = "experiment"
    experiment._runs = 1
    experiment._interrupted = False
    experiment._retries = [f"task row 0: attempt 1/2 failed: {RETRY_MARKER}"]
    with caplog.at_level(logging.DEBUG):
        experiment._log_experiment_summary({"summary_evaluations": {}, "rows": [], "runs": []})
    assert any(
        RETRY_MARKER in record.getMessage() and record.name == "ddtrace.llmobs._experiment" for record in caplog.records
    )
