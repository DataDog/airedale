# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Log lines about an exception have no message and no traceback values (data of MCP servers and of pi)."""

from __future__ import annotations

import asyncio
import logging

import pytest

import airedale.mcp as mcp_module
from airedale.harness.pi_spans import PiSpanRecorder
from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog
from airedale.safe_log import exception_summary
from airedale.skills import _pi_settings_skill_paths

SECRET = "sk-ADVERTISED-SECRET-0123456789"


def _raise_in_helper(error):
    raise error


def test_summary_has_the_type_and_the_place_but_no_message():
    with pytest.raises(ValueError) as raised:
        _raise_in_helper(ValueError(f"bad value {SECRET}"))
    summary = exception_summary(raised.value)
    assert summary == "ValueError at test_safe_log.py:" + summary.rsplit(":", 1)[1]
    assert SECRET not in summary


def test_summary_of_a_group_lists_the_types_of_the_leaves():
    group = ExceptionGroup("g", [ValueError(SECRET), ExceptionGroup("h", [KeyError(SECRET), ValueError("x")])])
    assert exception_summary(group) == "KeyError, ValueError"


def test_metadata_probe_does_not_log_the_tool_object_of_a_rejected_list(monkeypatch, caplog):
    """The MCP SDK validates ``tools/list``: its error holds the whole tool object (a tool without ``inputSchema``)."""
    from mcp.types import ListToolsResult
    from pydantic import ValidationError

    secret = "sk-LEAK-0123456789"  # (Pydantic cuts long values in its message.)
    with pytest.raises(ValidationError) as raised:
        ListToolsResult.model_validate({"tools": [{"name": secret}]})
    assert secret in str(raised.value)  # (This is the leak that the log line must not repeat.)

    async def failing_list(server, headers, *, all_pages=False, errlog=None, **_):
        raise raised.value

    monkeypatch.setattr(mcp_module, "_list_mcp_tools", failing_list)
    server = McpServerSpec(name="srv", url="http://localhost/mcp")
    with caplog.at_level(logging.DEBUG):
        catalog = asyncio.run(_mcp_tool_metadata_catalog([server], {}))
    assert catalog == {}
    assert "Unable to fetch MCP tool metadata for server srv (ValidationError" in caplog.text
    assert secret not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_span_recorder_does_not_log_the_message_of_its_error(monkeypatch, caplog):
    recorder = PiSpanRecorder.__new__(PiSpanRecorder)
    recorder._last_event_ns = 0

    def failing(event, now):
        raise ValueError(f"cannot read {SECRET}")

    monkeypatch.setattr(recorder, "_feed", failing, raising=False)
    with caplog.at_level(logging.DEBUG):
        recorder.feed({"type": "x"})
    assert "Ignoring a pi event that the span recorder cannot use (ValueError at test_safe_log.py:" in caplog.text
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_project_files_that_do_not_parse_are_logged_without_exc_info(tmp_path, caplog):
    config = tmp_path / ".mcp.json"
    config.write_text("{not json")
    with caplog.at_level(logging.DEBUG):
        assert mcp_module._parse_mcp_json(config) == []
    assert "Ignoring unreadable project MCP config" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_pi_settings_that_do_not_parse_are_logged_without_exc_info(tmp_path, caplog):
    (tmp_path / ".pi").mkdir()
    (tmp_path / ".pi" / "settings.json").write_text('{"skills": [' + SECRET)
    with caplog.at_level(logging.DEBUG):
        assert _pi_settings_skill_paths(tmp_path) == []
    assert "Unable to parse project pi settings" in caplog.text
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


SERVER_SCRIPT = """
import sys
sys.stderr.write("starting with token sk-SERVER-STDERR-0123456789\\n")
sys.stderr.flush()
sys.exit(3)
"""


def test_stderr_of_a_stdio_server_goes_to_the_notes_when_the_probe_fails_and_never_to_the_terminal(capfd):
    import sys

    server = McpServerSpec(name="noisy", command=sys.executable, args=("-c", SERVER_SCRIPT))
    notes: list[str] = []
    assert asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes)) == {}
    out, err = capfd.readouterr()
    assert "sk-SERVER-STDERR" not in out + err
    assert len(notes) == 1 and "sk-SERVER-STDERR-0123456789" in notes[0] and "'noisy'" in notes[0]


def test_stderr_of_a_stdio_server_is_dropped_without_notes_and_still_not_on_the_terminal(capfd):
    import sys

    server = McpServerSpec(name="noisy", command=sys.executable, args=("-c", SERVER_SCRIPT))
    assert asyncio.run(_mcp_tool_metadata_catalog([server], {})) == {}
    out, err = capfd.readouterr()
    assert "sk-SERVER-STDERR" not in out + err


def test_the_stderr_tail_is_bounded(capfd):
    import sys

    from airedale.mcp import MCP_PROBE_STDERR_TAIL_BYTES

    script = "import sys; sys.stderr.write('x' * 200000 + 'END'); sys.exit(1)"
    server = McpServerSpec(name="loud", command=sys.executable, args=("-c", script))
    notes: list[str] = []
    asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes))
    assert notes and notes[0].endswith("END")
    assert len(notes[0].encode()) < MCP_PROBE_STDERR_TAIL_BYTES + 200
