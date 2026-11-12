# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Log records made inside the MCP SDK carry no data that an MCP server sent.

The tests use a real stdio transport: a small Python MCP server, started by the metadata probe.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import textwrap

import pytest

from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog

MARKER = "sk-SDK-LOG-MARKER-0123456789"

SERVER = textwrap.dedent(
    """
    import json, sys
    secret = sys.argv[1]
    mode = sys.argv[2]

    def send(message):
        sys.stdout.write(json.dumps(message) + chr(10))
        sys.stdout.flush()

    if mode == "invalid-output":
        sys.stdout.write("this is not json " + secret + chr(10))
        sys.stdout.flush()
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "notifications/initialized" and mode == "unsupported-request":
            send({"jsonrpc": "2.0", "id": 99, "method": "unsupported/" + secret, "params": {"token": secret}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "tools": [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}]}})
    """
)


def _probe(mode: str) -> dict:
    server = McpServerSpec(name="fake", command=sys.executable, args=("-c", SERVER, MARKER, mode))
    return asyncio.run(_mcp_tool_metadata_catalog([server], {}))


@pytest.fixture
def everything_logged():
    """A handler on the root logger that takes every record at every level, formatted with the exception text."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG], ids=["warning", "debug"])
@pytest.mark.parametrize("mode", ["invalid-output", "unsupported-request"])
def test_the_sdk_logs_no_data_of_the_server(mode, level, caplog, everything_logged):
    with caplog.at_level(level):
        catalog = _probe(mode)
    # The probe still works (a line that is not JSON, or a request that is not supported, does not stop it).
    assert ("fake", "t") in catalog or mode == "invalid-output"
    texts = [record.getMessage() for record in caplog.records]
    for record in caplog.records:
        texts.append(str(record.exc_info) + str(record.exc_text) + str(record.stack_info) + str(record.args))
    assert MARKER not in "\n".join(texts) + everything_logged.getvalue() + caplog.text
    sdk_records = [r for r in caplog.records if "MCP SDK" in r.getMessage()]
    assert sdk_records, "the SDK made no log record in this case: the test does not cover the guard"
    for record in sdk_records:
        assert record.getMessage().endswith("(details hidden)")
        assert record.args == () and record.exc_info is None and record.stack_info is None


def test_the_level_and_the_logger_of_a_rewritten_record_are_kept(caplog):
    with caplog.at_level(logging.DEBUG):
        _probe("invalid-output")
    [record] = [r for r in caplog.records if r.name == "mcp.client.stdio" and r.levelno == logging.ERROR]
    assert record.getMessage().startswith("MCP SDK ERROR at __init__:")


def test_a_record_from_another_file_is_not_changed(caplog):
    other = logging.getLogger("not.mcp")
    with caplog.at_level(logging.DEBUG):
        try:
            raise ValueError(f"detail {MARKER}")
        except ValueError:
            other.exception("message %s", MARKER)
        logging.getLogger("mcp.something").warning("a record that the SDK did not make: %s", MARKER)
    texts = [record.getMessage() for record in caplog.records]
    assert any(MARKER in text for text in texts) and all("details hidden" not in text for text in texts)
    assert any(record.exc_info is not None for record in caplog.records if record.name == "not.mcp")


def test_the_guard_is_installed_once():
    from airedale.safe_log import install_mcp_log_guard

    factory = logging.getLogRecordFactory()
    install_mcp_log_guard()
    install_mcp_log_guard()
    assert logging.getLogRecordFactory() is factory
