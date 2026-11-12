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
    import json, os, sys, time
    marker = sys.argv[1]
    mode = sys.argv[2]
    sync = sys.argv[3] if len(sys.argv) > 3 else ""

    def send(message):
        sys.stdout.write(json.dumps(message) + chr(10))
        sys.stdout.flush()

    if mode == "invalid-output":
        sys.stdout.write("this is not json " + marker + chr(10))
        sys.stdout.flush()
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "notifications/initialized" and mode == "unsupported-request-paused":
            # Tell the test that the session is open, wait until it has changed the logging, then send the request.
            open(os.path.join(sync, "reached"), "w").close()
            for _ in range(600):
                if os.path.exists(os.path.join(sync, "release")):
                    break
                time.sleep(0.05)
            send({"jsonrpc": "2.0", "id": 99, "method": "unsupported/" + marker, "params": {"value": marker}})
        elif method == "notifications/initialized" and mode == "unsupported-request":
            send({"jsonrpc": "2.0", "id": 99, "method": "unsupported/" + marker, "params": {"value": marker}})
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
        logging.getLogger("airedale.mcp").warning("a record of Airedale, not of the SDK: %s", MARKER)
    texts = [record.getMessage() for record in caplog.records]
    assert any(MARKER in text for text in texts) and all("details hidden" not in text for text in texts)
    assert any(record.exc_info is not None for record in caplog.records if record.name == "not.mcp")


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG], ids=["warning", "debug"])
@pytest.mark.parametrize("copy_attributes", [False, True], ids=["plain-factory", "functools-wraps-factory"])
def test_a_factory_set_from_another_thread_during_a_probe_does_not_remove_the_protection(
    tmp_path, caplog, level, copy_attributes
):
    """The session is open and the server waits; the logging changes; then the server sends the request."""
    import functools
    import threading
    import time

    from airedale.safe_log import ensure_mcp_log_guard

    ensure_mcp_log_guard()
    previous = logging.getLogRecordFactory()

    def app_factory(*args, **kwargs):
        record = logging.LogRecord(*args, **kwargs)  # (it does not call the factory before it)
        record.app_attribute = "from the application"
        return record

    if copy_attributes:
        app_factory = functools.wraps(previous)(app_factory)
    server = McpServerSpec(
        name="fake",
        command=sys.executable,
        args=("-c", SERVER, MARKER, "unsupported-request-paused", str(tmp_path)),
    )
    probe = threading.Thread(target=lambda: asyncio.run(_mcp_tool_metadata_catalog([server], {})))
    try:
        with caplog.at_level(level):
            probe.start()
            deadline = time.monotonic() + 20
            while not (tmp_path / "reached").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (tmp_path / "reached").exists()
            logging.setLogRecordFactory(app_factory)  # From this thread, while the probe runs in another one.
            (tmp_path / "release").touch()
            probe.join(30)
    finally:
        logging.setLogRecordFactory(previous)
    assert not probe.is_alive()
    texts = [record.getMessage() + str(record.exc_info) + str(record.args) for record in caplog.records]
    assert MARKER not in "\n".join(texts) + caplog.text
    sdk = [record for record in caplog.records if "MCP SDK" in record.getMessage()]
    assert sdk, "the SDK made no log record: the test does not cover the guard"
    assert any(getattr(record, "app_attribute", None) for record in sdk)  # The new factory made them.


def test_records_on_the_root_logger_and_on_other_loggers_are_unchanged_with_the_filters(caplog):
    import logging as std_logging

    with caplog.at_level(std_logging.DEBUG):
        std_logging.getLogger().warning("root record of the application: %s", MARKER)
        std_logging.getLogger("not.mcp.at.all").error("record on another logger: %s", MARKER)
        std_logging.getLogger("airedale.mcp").info("record of Airedale: %s", MARKER)
    texts = [record.getMessage() for record in caplog.records]
    assert sum(MARKER in text for text in texts) == 3 and all("details hidden" not in text for text in texts)


def test_a_logger_called_mcp_is_hidden_even_when_the_file_is_unknown(caplog):
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("mcp.client.other").warning("data of a server: %s", MARKER)
    assert MARKER not in caplog.text and "MCP SDK WARNING at test_mcp_log_guard:" in caplog.text


def test_a_guard_is_recognized_by_identity_not_by_an_attribute():
    import functools

    from airedale.safe_log import _guards, ensure_mcp_log_guard

    ensure_mcp_log_guard()
    guard = logging.getLogRecordFactory()
    assert guard in _guards

    def copy_of_the_attributes(*args, **kwargs):
        return logging.LogRecord(*args, **kwargs)

    copy_of_the_attributes = functools.wraps(guard)(copy_of_the_attributes)
    assert getattr(copy_of_the_attributes, "_airedale_mcp_log_guard", False)  # (The mark was copied.)
    logging.setLogRecordFactory(copy_of_the_attributes)
    try:
        ensure_mcp_log_guard()
        wrapped = logging.getLogRecordFactory()
        assert wrapped is not copy_of_the_attributes and wrapped in _guards
        ensure_mcp_log_guard()  # Identity passes now: nothing is stacked.
        assert logging.getLogRecordFactory() is wrapped
    finally:
        logging.setLogRecordFactory(guard)


def test_the_guard_is_installed_once():
    from airedale.safe_log import install_mcp_log_guard

    factory = logging.getLogRecordFactory()
    install_mcp_log_guard()
    install_mcp_log_guard()
    assert logging.getLogRecordFactory() is factory


# --------------------------------------------------------------------------- #
# A bytecode-only copy of the SDK keeps the path where it was built in ``record.pathname``
# --------------------------------------------------------------------------- #
PROBE_SCRIPT = textwrap.dedent(
    """
    import asyncio, io, logging, sys
    sys.path.insert(0, sys.argv[1])
    import mcp
    assert mcp.__file__.startswith(sys.argv[1]), mcp.__file__
    from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    server = McpServerSpec(name="fake", command=sys.executable, args=("-c", sys.argv[2], sys.argv[3], sys.argv[4]))
    asyncio.run(_mcp_tool_metadata_catalog([server], {}))
    print("LOGGED-BEGIN")
    print(stream.getvalue())
    print("LOGGED-END")
    """
)


def _bytecode_copy(tmp_path, as_zip: bool):
    """Copy the installed SDK as ``.pyc`` files only, built as if they came from another directory."""
    import py_compile
    import zipfile
    from pathlib import Path

    import mcp

    source = Path(mcp.__file__).parent
    target = tmp_path / ("sdk.zip" if as_zip else "sdk")
    entries: dict[str, Path] = {}
    build = tmp_path / "compiled"
    for py_file in source.rglob("*.py"):
        relative = py_file.relative_to(source.parent)
        compiled = build / relative.with_suffix(".pyc")
        compiled.parent.mkdir(parents=True, exist_ok=True)
        # (The code objects keep this build path, which does not exist: it is not under the place the SDK is loaded from.)
        py_compile.compile(
            str(py_file), cfile=str(compiled), dfile=f"/nonexistent-build-dir/{relative.as_posix()}", doraise=True
        )
        entries[relative.with_suffix(".pyc").as_posix()] = compiled
    if as_zip:
        with zipfile.ZipFile(target, "w") as archive:
            for name, compiled in entries.items():
                archive.write(compiled, name)
    else:
        for name, compiled in entries.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(compiled.read_bytes())
    return target


@pytest.mark.parametrize("as_zip", [False, True], ids=["pyc-directory", "pyc-zip"])
@pytest.mark.parametrize("mode", ["invalid-output", "unsupported-request"])
def test_a_bytecode_only_copy_of_the_sdk_is_recognized(tmp_path, mode, as_zip):
    import subprocess

    location = _bytecode_copy(tmp_path, as_zip)
    result = subprocess.run(
        [sys.executable, "-c", PROBE_SCRIPT, str(location), SERVER, MARKER, mode],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    logged = result.stdout.split("LOGGED-BEGIN", 1)[1].split("LOGGED-END", 1)[0]
    assert "MCP SDK" in logged and "(details hidden)" in logged, logged  # The SDK did log, and the guard hid it.
    assert MARKER not in logged + result.stderr
