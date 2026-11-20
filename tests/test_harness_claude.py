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
import re
import stat
from types import SimpleNamespace
from typing import Any

import claude_agent_sdk
import pytest
from claude_agent_sdk import SystemMessage, TaskStartedMessage
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk._internal.transport import Transport

from airedale.config.experiment import ScenarioConfig
from airedale.harness import create_runner
from airedale.harness.claude import (
    CLAUDE_BUILTIN_TOOLS,
    ClaudeRunner,
    _claude_progress_message,
    _claude_result_error_summary,
)
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
API_RETRY_MARKER = "sk-API-RETRY-MARKER"
DENIAL_MARKER = "sk-DENIAL-MARKER"
DEFERRED_MARKER = "sk-DEFERRED-MARKER"
PROGRESS_MARKER = "sk-PROGRESS-MARKER"
_END = object()

# The exact progress lines of a failed run (design section 3.4).
LABEL = "anthropic/claude-3-haiku-20240307"
RUN_LINE = "h/anthropic/claude-3-haiku-20240307: claude-agent-sdk run"
RESULT_LINE = (
    "anthropic/claude-3-haiku-20240307: Claude error result - subtype=error_during_execution; turns=1; errors=1"
)
LATE_RESULT_LINE = (
    "anthropic/claude-3-haiku-20240307: Claude error result - subtype=error_during_execution; turns=1; errors=1; "
    "permission_denials=1; deferred_tool_use=yes"
)
RETRY_LINE = "anthropic/claude-3-haiku-20240307: api_retry status=529 error=<other> attempt=1/10 retry_delay_ms=500"


def _result_dict(**fields: Any) -> dict[str, Any]:
    """A raw ``result`` message of the Claude CLI that is an error. It has no ``errors``; add them with ``fields``."""
    message: dict[str, Any] = {
        "type": "result",
        "subtype": "error_during_execution",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": True,
        "num_turns": 1,
        "session_id": "s",
    }
    message.update(fields)
    return message


def _retry_dict(**data: Any) -> dict[str, Any]:
    """A raw ``api_retry`` system message of the Claude CLI. ``data`` replaces or adds fields."""
    message: dict[str, Any] = {
        "type": "system",
        "subtype": "api_retry",
        "attempt": 1,
        "max_retries": 10,
        "retry_delay_ms": 500,
        "error_status": 529,
        "error": "rate_limit",
        "uuid": "u",
        "session_id": "s",
    }
    message.update(data)
    return message


def _late_result_dict() -> dict[str, Any]:
    """The DD-35 error result, plus a permission denial and a deferred tool call that have a marker."""
    return _result_dict(
        errors=[RESULT_MARKER],
        permission_denials=[
            {"tool_name": "Bash", "tool_use_id": "t1", "tool_input": {"command": f"echo {DENIAL_MARKER}"}}
        ],
        deferred_tool_use={"id": "d1", "name": "Bash", "input": {"command": f"echo {DEFERRED_MARKER}"}},
    )


def _assert_absent(texts: list[str], *markers: str) -> None:
    """Check, in the given order, that no text has a marker. The failure message is the marker."""
    for marker in markers:
        assert not [text for text in texts if marker in text], marker


def _private_report_text(tmp_path) -> str | None:
    """The text of the private report of a failed run, or ``None`` on Windows (no report is written there)."""
    if os.name == "nt":
        return None
    [report] = list((tmp_path / "reports").glob("claude-run-*.log"))
    return report.read_text()


class _FakeTransport(Transport):
    """A Claude Code CLI that answers ``initialize``, then fails the run. ``options`` is set by the test.

    It writes the CLI stderr (``options.stderr``), emits ``messages`` (by default one error result with a marker) and
    then either raises an exception in the reader (path 1) or ends the stream (path 2).
    """

    def __init__(self, *, raises: bool, messages: list[dict[str, Any]] | None = None) -> None:
        self.options: claude_agent_sdk.ClaudeAgentOptions | None = None
        self._raises = raises
        self._messages = messages if messages is not None else [_result_dict(errors=[RESULT_MARKER])]
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
                for raw_message in self._messages:
                    self._queue.put_nowait(raw_message)
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


def _claude_runner(tmp_path, *, path: int, messages: list[dict[str, Any]] | None = None):
    """``(runner, transport)``: path 1 uses ``query()`` with the transport, path 2 uses ``ClaudeSDKClient``."""
    transport = _FakeTransport(raises=path == 1, messages=messages)
    if path == 1:

        def query_func(*, prompt, options):
            transport.options = options
            return claude_agent_sdk.query(prompt=prompt, options=options, transport=transport)

        return ClaudeRunner(cwd=tmp_path, claude_query_func=query_func), transport

    def client_factory(options):
        transport.options = options
        return claude_agent_sdk.ClaudeSDKClient(options, transport=transport)

    return ClaudeRunner(cwd=tmp_path, claude_client_factory=client_factory), transport


def _failed_run(tmp_path, monkeypatch, *, path: int, messages: list[dict[str, Any]] | None = None):
    """Run a Claude run that fails; return ``(raised error, every progress line that the runner sent)``."""
    from airedale.progress import ProgressReporter

    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(tmp_path / "reports"))
    (tmp_path / "cwd").mkdir()
    runner, _transport = _claude_runner(tmp_path / "cwd", path=path, messages=messages)
    reporter = ProgressReporter(enabled=True)  # (Not started: plain mode, one line for each message.)
    calls: list[str] = []

    async def progress(label: str) -> None:
        calls.append(label)
        await reporter.message(label)

    with pytest.raises(Exception) as raised:  # noqa: PT011
        asyncio.run(
            asyncio.wait_for(
                runner.run(
                    model=ANTHROPIC_MODEL,
                    system_prompt="S",
                    user_prompt="Q",
                    prompt_version="v",
                    harness="h",
                    progress=progress,
                ),
                timeout=60,
            )
        )
    return raised.value, calls


def _records_text(caplog) -> str:
    return "\n".join(f"{record.name}: {record.getMessage()}|{record.exc_text}" for record in caplog.records)


@pytest.mark.parametrize("path", [1, 2], ids=["exception-in-the-stream", "error-result-message"])
def test_a_failed_claude_run_has_no_raw_text_in_the_logs_and_a_private_report(
    tmp_path, monkeypatch, caplog, capfd, path
):
    caplog.set_level(logging.DEBUG)
    error, calls = _failed_run(tmp_path, monkeypatch, path=path)
    captured = capfd.readouterr()
    # O6: the progress callback and the console output have only the safe text.
    _assert_absent(calls, RESULT_MARKER, STDERR_MARKER, READER_MARKER)
    assert calls == [RUN_LINE, RESULT_LINE]
    _assert_absent([captured.out + captured.err], RESULT_MARKER, STDERR_MARKER, READER_MARKER)
    assert RESULT_LINE in captured.out
    # O1 and O2: the log lines of Airedale and of the SDK have no raw text, no exception text and no traceback.
    text = _records_text(caplog)
    for marker in (STDERR_MARKER, RESULT_MARKER, READER_MARKER):
        assert marker not in text, marker
    assert not any(
        record.exc_info for record in caplog.records if record.name.startswith("airedale")
    )  # (No traceback.)
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
    # O5: the raised error is not changed (accepted for DD-35). O6 is checked above.
    if path == 1:
        assert READER_MARKER in str(error)
    else:
        assert str(error) == (
            "claude-agent-sdk returned an error result for anthropic/claude-3-haiku-20240307: "
            "sk-RESULT-MARKER; subtype=error_during_execution; turns=1"
        )
    if path == 1:
        # O3: the SDK log of the error that stops its reader has a fixed text.
        [sdk_record] = [r for r in caplog.records if r.name == "claude_agent_sdk._internal.query" and r.levelno == 40]
        assert sdk_record.getMessage().startswith("Claude Agent SDK ERROR at query:")
        assert sdk_record.getMessage().endswith("(details hidden)")


@pytest.mark.parametrize("path", [1, 2], ids=["exception-in-the-stream", "error-result-message"])
def test_an_api_retry_has_no_raw_text_in_the_progress_and_keeps_it_in_the_private_report(
    tmp_path, monkeypatch, capfd, path
):
    messages = [_retry_dict(error={"message": API_RETRY_MARKER}), _result_dict(errors=[RESULT_MARKER])]
    _error, calls = _failed_run(tmp_path, monkeypatch, path=path, messages=messages)
    captured = capfd.readouterr()
    _assert_absent(calls, API_RETRY_MARKER, RESULT_MARKER)
    assert calls == [RUN_LINE, RETRY_LINE, RESULT_LINE]
    _assert_absent([captured.out + captured.err], API_RETRY_MARKER, RESULT_MARKER)
    assert RETRY_LINE in captured.out
    report = _private_report_text(tmp_path)
    if report is not None:
        assert API_RETRY_MARKER in report  # The private report keeps the raw text.


@pytest.mark.parametrize("path", [1, 2], ids=["exception-in-the-stream", "error-result-message"])
def test_late_result_fields_have_no_raw_text_in_the_progress(tmp_path, monkeypatch, capfd, path):
    late = _late_result_dict()
    # The markers are after the first 120 characters of the raw text: only the fixed text keeps them out.
    base = f"{LABEL}: Claude error result - " + _claude_result_error_summary(parse_message(late), max_turns=None)
    assert base.index(DENIAL_MARKER) >= 120 and base.index(DEFERRED_MARKER) >= 120
    _error, calls = _failed_run(tmp_path, monkeypatch, path=path, messages=[late])
    captured = capfd.readouterr()
    _assert_absent(calls, DENIAL_MARKER, DEFERRED_MARKER, RESULT_MARKER)
    assert calls == [RUN_LINE, LATE_RESULT_LINE]
    _assert_absent([captured.out + captured.err], DENIAL_MARKER, DEFERRED_MARKER, RESULT_MARKER)
    assert LATE_RESULT_LINE[:120] in captured.out
    report = _private_report_text(tmp_path)
    if report is not None:
        assert DENIAL_MARKER in report and DEFERRED_MARKER in report


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


@pytest.mark.parametrize("path", [1, 2], ids=["exception-in-the-stream", "error-result-message"])
async def test_the_ddtrace_retry_summary_has_no_raw_text(
    tmp_path, monkeypatch, caplog, ddtrace_experiment, run_cell, path
):
    """O7 (ROMAIN-65): a real failed Claude run, retried 1 time by ddtrace, leaves no raw text in the retry summary."""
    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(tmp_path / "reports"))
    ddtrace_experiment.max_retries = 1
    runners = []

    def make_runner(cwd):
        runner, _transport = _claude_runner(
            cwd, path=path
        )  # (A transport is single-use: a new runner for each attempt.)
        runners.append(runner)
        return runner

    summary = await run_cell(make_runner)

    assert len(runners) == 2  # A real retry.
    [retries] = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ddtrace.llmobs._experiment" and "Retries (" in record.getMessage()
    ]
    assert "Retries (1)" in retries
    _assert_absent(
        [_records_text(caplog), caplog.text, repr(summary.rows)], READER_MARKER, RESULT_MARKER, STDERR_MARKER
    )
    [row] = summary.rows
    assert re.match(r"^\w+(, \w+)* at \S+:\d+$", row["error"]["message"])


class _FailsToConnect(_FakeTransport):
    """A transport that cannot connect: no stderr, no result, only an exception with a cause."""

    async def connect(self) -> None:
        try:
            raise OSError(f"cause {CAUSE_MARKER}")
        except OSError as cause:
            raise claude_agent_sdk.CLIConnectionError(f"cannot connect {CONNECT_MARKER}") from cause


CONNECT_MARKER = "sk-CONNECT-MARKER"
CAUSE_MARKER = "sk-CAUSE-MARKER"


def test_a_connection_failure_with_no_stderr_and_no_result_still_has_a_private_report(tmp_path, monkeypatch, caplog):
    """Copilot, finding 3: the report has the error (type, message, traceback, cause), also with no diagnostics."""
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(tmp_path / "reports"))
    (tmp_path / "cwd").mkdir()
    transport = _FailsToConnect(raises=False)

    def query_func(*, prompt, options):
        transport.options = options
        return claude_agent_sdk.query(prompt=prompt, options=options, transport=transport)

    runner = ClaudeRunner(cwd=tmp_path / "cwd", claude_query_func=query_func)
    with pytest.raises(claude_agent_sdk.CLIConnectionError):
        asyncio.run(
            asyncio.wait_for(
                runner.run(model=ANTHROPIC_MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"),
                timeout=60,
            )
        )
    text = _records_text(caplog)
    assert CONNECT_MARKER not in text and CAUSE_MARKER not in text  # The log line has only the type and the place.
    [failure] = [r for r in caplog.records if r.getMessage().startswith("Claude Agent SDK eval run failed (")]
    assert "CLIConnectionError" in failure.getMessage()
    reports = list((tmp_path / "reports").glob("claude-run-*.log"))
    if os.name == "nt":
        assert reports == [] and "no private file was written" in failure.getMessage()
        return
    [report] = reports
    assert str(report) in failure.getMessage()
    contents = report.read_text()
    assert "CLIConnectionError" in contents and f"cannot connect {CONNECT_MARKER}" in contents
    assert "Traceback" in contents and f"cause {CAUSE_MARKER}" in contents  # The traceback and the cause chain.
    assert "no diagnostic from the Claude CLI" in contents


# --------------------------------------------------------------------------- #
# The safe progress text of a failed run (design sections 3.1 to 3.3)
# --------------------------------------------------------------------------- #
# These lists are written again here on purpose: they are an independent check of the lists in ``claude.py``.
RESULT_SUBTYPES = (
    "success",
    "error_during_execution",
    "error_max_turns",
    "error_max_budget_usd",
    "error_max_structured_output_retries",
)
STOP_REASONS = (
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "compaction",
    "refusal",
    "model_context_window_exceeded",
    "tool_deferred",
    "tool_deferred_unavailable",
)
RETRY_ERRORS = (
    "authentication_failed",
    "oauth_org_not_allowed",
    "account_on_hold",
    "verification_required",
    "billing_error",
    "rate_limit",
    "overloaded",
    "invalid_request",
    "model_not_found",
    "server_error",
    "unknown",
    "max_output_tokens",
    "cloud_credential_error",
)
_RESULT_PREFIX = f"{LABEL}: Claude error result - "
_RETRY_PREFIX = f"{LABEL}: api_retry"
OTHER = "<other>"


class _Str(str):
    """A ``str`` subclass: it is not a plain string, so it must not be shown."""


class _Int(int):
    """An ``int`` subclass: it is not a plain integer, so it must not be shown."""


def _result_line(**fields: Any) -> str | None:
    return _claude_progress_message(parse_message(_result_dict(**fields)), ANTHROPIC_MODEL)


def _retry_line(**data: Any) -> str | None:
    return _claude_progress_message(parse_message(_retry_dict(**data)), ANTHROPIC_MODEL)


def _not_a_listed_string(listed: str) -> list[Any]:
    """Values that look like ``listed`` or that are not strings. None of them is a listed string."""
    return [
        PROGRESS_MARKER,
        f"{listed} {PROGRESS_MARKER}",
        f"{listed}\n",
        listed.upper(),
        1,
        True,
        [PROGRESS_MARKER],
        {"message": PROGRESS_MARKER},
        _Str(listed),
    ]


# For each string field: the expected line for a value, and how to make the line.
_STRING_FIELDS = {
    "subtype": (
        RESULT_SUBTYPES,
        lambda value: _result_line(subtype=value),
        lambda shown: f"{_RESULT_PREFIX}subtype={shown}; turns=1",
    ),
    "stop_reason": (
        STOP_REASONS,
        lambda value: _result_line(stop_reason=value),
        lambda shown: f"{_RESULT_PREFIX}subtype=error_during_execution; stop_reason={shown}; turns=1",
    ),
    "error": (
        RETRY_ERRORS,
        lambda value: _retry_line(error=value),
        lambda shown: f"{_RETRY_PREFIX} status=529 error={shown} attempt=1/10 retry_delay_ms=500",
    ),
}


class TestClaudeProgressSafeText:
    """The progress text of an error result and of an ``api_retry`` message shows only safe values."""

    @pytest.mark.parametrize("field", list(_STRING_FIELDS))
    def test_a_listed_string_is_shown(self, field):
        listed, make_line, expected = _STRING_FIELDS[field]
        for value in listed:
            assert make_line(value) == expected(value), value

    @pytest.mark.parametrize("field", list(_STRING_FIELDS))
    def test_another_value_of_a_string_field_is_other(self, field):
        listed, make_line, expected = _STRING_FIELDS[field]
        values = _not_a_listed_string(listed[0])
        if field == "error":
            values += [{"type": "rate_limit"}, {"error": {"message": PROGRESS_MARKER}}]
        for value in values:
            line = make_line(value)
            assert line == expected(OTHER), repr(value)
            assert line is not None and PROGRESS_MARKER not in line

    @pytest.mark.parametrize(
        ("make_line", "expected"),
        [
            (
                lambda value: _result_line(api_error_status=value),
                lambda shown: f"{_RESULT_PREFIX}subtype=error_during_execution; api_error_status={shown}; turns=1",
            ),
            (
                lambda value: _retry_line(error_status=value),
                lambda shown: f"{_RETRY_PREFIX} status={shown} error=rate_limit attempt=1/10 retry_delay_ms=500",
            ),
        ],
        ids=["api_error_status", "error_status"],
    )
    def test_http_status_rule(self, make_line, expected):
        for value in (100, 429, 599):
            assert make_line(value) == expected(value), value
        for value in (99, 600, -1, True, False, 429.0, "429", PROGRESS_MARKER, _Int(429)):
            line = make_line(value)
            assert line == expected(OTHER), repr(value)
            assert line is not None and PROGRESS_MARKER not in line

    @pytest.mark.parametrize(
        ("make_line", "expected"),
        [
            (
                lambda value: _result_line(num_turns=value),
                lambda shown: f"{_RESULT_PREFIX}subtype=error_during_execution; turns={shown}",
            ),
            (
                lambda value: _retry_line(attempt=value),
                lambda shown: f"{_RETRY_PREFIX} status=529 error=rate_limit attempt={shown}/10 retry_delay_ms=500",
            ),
            (
                lambda value: _retry_line(max_retries=value),
                lambda shown: f"{_RETRY_PREFIX} status=529 error=rate_limit attempt=1/{shown} retry_delay_ms=500",
            ),
            (
                lambda value: _retry_line(retry_delay_ms=value),
                lambda shown: f"{_RETRY_PREFIX} status=529 error=rate_limit attempt=1/10 retry_delay_ms={shown}",
            ),
        ],
        ids=["num_turns", "attempt", "max_retries", "retry_delay_ms"],
    )
    def test_number_rule(self, make_line, expected):
        for value in (0, 1, 1_000_000_000):
            assert make_line(value) == expected(value), value
        for value in (-1, 1_000_000_001, True, False, 1.0, "1", PROGRESS_MARKER, _Int(1)):
            line = make_line(value)
            assert line == expected(OTHER), repr(value)
            assert line is not None and PROGRESS_MARKER not in line

    @pytest.mark.parametrize("field", ["errors", "permission_denials"])
    def test_count_rule(self, field):
        def line(value):
            return _result_line(**{field: value})

        for value in (None, [], [""], ["", None], "", 0, {}):
            assert line(value) == f"{_RESULT_PREFIX}subtype=error_during_execution; turns=1", repr(value)
        for value in ([PROGRESS_MARKER], ["", PROGRESS_MARKER], PROGRESS_MARKER, 5, {"k": PROGRESS_MARKER}):
            assert line(value) == f"{_RESULT_PREFIX}subtype=error_during_execution; turns=1; {field}=1", repr(value)
        for value in ([PROGRESS_MARKER, PROGRESS_MARKER], [PROGRESS_MARKER, "", PROGRESS_MARKER]):
            assert line(value) == f"{_RESULT_PREFIX}subtype=error_during_execution; turns=1; {field}=2", repr(value)
        if field == "permission_denials":
            denial = {"tool_name": "Bash", "tool_use_id": "t", "tool_input": {"command": PROGRESS_MARKER}}
            assert line([{}, denial]) == f"{_RESULT_PREFIX}subtype=error_during_execution; turns=1; {field}=1"

    def test_deferred_tool_use_presence(self):
        no_field = f"{_RESULT_PREFIX}subtype=error_during_execution; turns=1"
        assert _result_line() == no_field  # (No key.)
        assert _result_line(deferred_tool_use=None) == no_field
        assert _result_line(deferred_tool_use={}) == no_field
        for deferred_input in ({}, {"command": PROGRESS_MARKER}):
            line = _result_line(deferred_tool_use={"id": "d1", "name": "Bash", "input": deferred_input})
            assert line == f"{no_field}; deferred_tool_use=yes"

    def test_field_presence_and_empty_forms(self):
        assert _result_line(subtype="", num_turns=None, stop_reason=None, api_error_status=None) == (
            f"{LABEL}: Claude error result"
        )
        assert _result_line(subtype="", num_turns=None, stop_reason="", api_error_status="") == (
            f"{LABEL}: Claude error result"
        )
        no_data = SystemMessage(subtype="api_retry", data=None)  # pyright: ignore[reportArgumentType]
        assert _claude_progress_message(no_data, ANTHROPIC_MODEL) == f"{LABEL}: api_retry"
        assert _retry_line(error_status=None) == f"{_RETRY_PREFIX} error=rate_limit attempt=1/10 retry_delay_ms=500"
        assert _retry_line(error=None) == f"{_RETRY_PREFIX} status=529 attempt=1/10 retry_delay_ms=500"
        assert (
            _retry_line(max_retries=None) == f"{_RETRY_PREFIX} status=529 error=rate_limit attempt=1 retry_delay_ms=500"
        )
        assert _retry_line(attempt=None) == f"{_RETRY_PREFIX} status=529 error=rate_limit retry_delay_ms=500"
        assert _retry_line(retry_delay_ms=None) == f"{_RETRY_PREFIX} status=529 error=rate_limit attempt=1/10"
        usage = {"input_tokens": 5, "output_tokens": 2}
        assert _result_line(is_error=False, usage=usage) == f"{LABEL}: final answer ready (7 tokens)"

    def test_dispatch(self):
        # A result is a result, also when its ``subtype`` is ``api_retry`` (the SDK does not check the value).
        assert _result_line(subtype="api_retry") == f"{_RESULT_PREFIX}subtype={OTHER}; turns=1"
        assert _result_line(subtype="api_retry", is_error=False) == f"{LABEL}: final answer ready"
        # Only a ``SystemMessage`` is an ``api_retry`` message.
        nested = TaskStartedMessage(
            subtype="api_retry",
            data={"error": PROGRESS_MARKER},
            task_id="t",
            description="d",
            uuid="u",
            session_id="s",
        )
        assert _claude_progress_message(nested, ANTHROPIC_MODEL) == f"{LABEL}: nested agent started - d"
        other = SimpleNamespace(subtype="api_retry", data={"error": PROGRESS_MARKER})
        assert _claude_progress_message(other, ANTHROPIC_MODEL) is None
