# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Log records made inside the MCP SDK carry no data that an MCP server sent.

The tests use a real stdio transport: a small Python MCP server, started by the metadata probe.
"""

from __future__ import annotations

import asyncio
import io
import itertools
import logging
import os
import sys
import textwrap
import threading
import types

import pytest

from airedale import safe_log
from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog
from airedale.safe_log import ensure_mcp_log_guard

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
        elif method == "tools/call":
            send({"jsonrpc": "2.0", "id": message["id"], "result": {"content": [], "isError": False}})
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


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG], ids=["warning", "debug"])
def test_an_sdk_logger_with_another_name_is_hidden_after_a_factory_change(caplog, level):
    """``mcp/client/session.py`` logs a tool name that the server controls, on a logger called ``client``."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from airedale.safe_log import ensure_mcp_log_guard

    ensure_mcp_log_guard()
    previous = logging.getLogRecordFactory()

    def app_factory(*args, **kwargs):
        return logging.LogRecord(*args, **kwargs)  # (it does not call the factory before it: layer 1 is gone)

    async def call_an_unlisted_tool():
        params = StdioServerParameters(command=sys.executable, args=["-c", SERVER, MARKER, "plain"])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool(MARKER, {})  # (Not in the list of the server: the SDK logs its name.)

    logging.setLogRecordFactory(app_factory)
    try:
        with caplog.at_level(level):
            asyncio.run(call_an_unlisted_tool())
    finally:
        logging.setLogRecordFactory(previous)
    sdk = [record for record in caplog.records if record.name == "client"]
    assert sdk, "the SDK made no record on its 'client' logger: the test does not cover it"
    assert MARKER not in caplog.text and all("(details hidden)" in record.getMessage() for record in sdk)


def test_the_filter_on_every_logger_is_cheap_and_changes_no_other_record(caplog):
    import time

    from airedale.safe_log import ensure_mcp_log_guard

    names = [f"many.loggers.{index}" for index in range(300)]
    loggers = [logging.getLogger(name) for name in names]
    ensure_mcp_log_guard()  # (The filter is put on the loggers that exist.)
    assert all(any(f.__class__.__name__ == "_SdkRecordFilter" for f in logger.filters) for logger in loggers)
    with caplog.at_level(logging.DEBUG):
        started = time.perf_counter()
        for _ in range(20):
            for logger in loggers:
                logger.info("record of the application: %s", MARKER)
        elapsed = time.perf_counter() - started
    assert sum(MARKER in record.getMessage() for record in caplog.records) == 20 * 300
    assert elapsed < 5, elapsed  # (6000 records; the origin of a file name is looked up once.)


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

    from airedale.safe_log import _is_guard, ensure_mcp_log_guard

    ensure_mcp_log_guard()
    guard = logging.getLogRecordFactory()
    assert _is_guard(guard)

    def copy_of_the_attributes(*args, **kwargs):
        return logging.LogRecord(*args, **kwargs)

    copy_of_the_attributes = functools.wraps(guard)(copy_of_the_attributes)
    assert getattr(copy_of_the_attributes, "_airedale_mcp_log_guard", False)  # (The mark was copied.)
    logging.setLogRecordFactory(copy_of_the_attributes)
    try:
        ensure_mcp_log_guard()
        wrapped = logging.getLogRecordFactory()
        assert wrapped is not copy_of_the_attributes and _is_guard(wrapped)
        ensure_mcp_log_guard()  # Identity passes now: nothing is stacked.
        assert logging.getLogRecordFactory() is wrapped
    finally:
        logging.setLogRecordFactory(guard)


def test_an_unhashable_factory_does_not_crash_and_is_wrapped():
    import dataclasses

    from airedale.safe_log import _is_guard, ensure_mcp_log_guard

    ensure_mcp_log_guard()
    guard = logging.getLogRecordFactory()

    @dataclasses.dataclass  # (``eq=True`` without ``frozen`` sets ``__hash__`` to ``None``.)
    class UnhashableFactory:
        calls: int = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return logging.LogRecord(*args, **kwargs)

    app_factory = UnhashableFactory()
    with pytest.raises(TypeError):
        hash(app_factory)
    logging.setLogRecordFactory(app_factory)
    try:
        ensure_mcp_log_guard()  # (Used to raise ``TypeError`` from the ``WeakSet``.)
        wrapped = logging.getLogRecordFactory()
        assert wrapped is not app_factory and _is_guard(wrapped)
        logging.getLogger("x").warning("a record")
        assert app_factory.calls >= 1  # The factory of the application still runs.
    finally:
        logging.setLogRecordFactory(guard)


def test_a_proxy_that_is_equal_to_a_guard_is_wrapped():
    from airedale.safe_log import _is_guard, ensure_mcp_log_guard

    ensure_mcp_log_guard()
    guard = logging.getLogRecordFactory()

    class Proxy:
        """Like ``wrapt.FunctionWrapper``: it looks like the factory it wraps (equal, same hash, same attributes)."""

        def __init__(self, wrapped):
            self.__wrapped__ = wrapped

        def __call__(self, *args, **kwargs):
            return logging.LogRecord(*args, **kwargs)  # (It does not call the factory it wraps.)

        def __eq__(self, other):
            return self.__wrapped__ == getattr(other, "__wrapped__", other)

        def __hash__(self):
            return hash(self.__wrapped__)

        def __getattr__(self, name):
            return getattr(self.__wrapped__, name)

    proxy = Proxy(guard)
    assert proxy == guard and getattr(proxy, "_airedale_mcp_log_guard", False)
    logging.setLogRecordFactory(proxy)
    try:
        ensure_mcp_log_guard()
        wrapped = logging.getLogRecordFactory()
        assert wrapped is not proxy and _is_guard(wrapped) and not _is_guard(proxy)
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


# --------------------------------------------------------------------------- #
# SDK modules that are imported late make their loggers late
# --------------------------------------------------------------------------- #
LATE_MODULE_SCRIPT = textwrap.dedent(
    """
    import io, logging, sys
    marker, route, level = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert "mcp.client.auth.utils" not in sys.modules  # (Only imported when the OAuth code is used.)
    import airedale.mcp  # installs the guard
    assert "mcp.client.auth.utils" in sys.modules, "the guard must import the client-side SDK modules"
    import mcp.client.auth.utils as utils
    logging.setLogRecordFactory(lambda *args, **kwargs: logging.LogRecord(*args, **kwargs))  # layer 1 is gone
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = utils.logger
    logger.setLevel(level)
    if route == "root":
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
    elif route == "sdk-handler":
        logger.addHandler(handler)
    elif route == "no-propagate":
        logger.propagate = False
        logger.addHandler(handler)
    elif route == "last-resort":
        logger.propagate = False
        logging.lastResort = logging.StreamHandler(stream)
    code = "logger.warning('data of a server: %s', marker)\\nlogger.debug('more data: %s', marker)"
    exec(compile(code, utils.__file__, "exec"), {"logger": logger, "marker": marker})
    print("LOGGED-BEGIN")
    print(stream.getvalue())
    print("LOGGED-END")
    """
)


@pytest.mark.parametrize("level", [logging.WARNING, logging.DEBUG], ids=["warning", "debug"])
@pytest.mark.parametrize("route", ["root", "sdk-handler", "no-propagate", "last-resort"])
def test_an_sdk_module_that_is_only_imported_on_use_has_the_filter_too(route, level):
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", LATE_MODULE_SCRIPT, MARKER, route, str(level)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    logged = result.stdout.split("LOGGED-BEGIN", 1)[1].split("LOGGED-END", 1)[0]
    assert "MCP SDK WARNING" in logged, logged  # The record was handled (not lost) and was hidden.
    assert MARKER not in logged + result.stderr


# --------------------------------------------------------------------------- #
# An ambiguous file name (``<string>``) is decided by the code that made the record, not by the name
# --------------------------------------------------------------------------- #
_counter = itertools.count()


@pytest.fixture
def restored_factory():
    """Give back the record factory that the test found."""
    factory = logging.getLogRecordFactory()
    try:
        yield
    finally:
        logging.setLogRecordFactory(factory)


@pytest.fixture
def app_logger(restored_factory):
    """``(logger, handled)``: a logger of the application and the messages of the records that it handled."""
    logger = logging.getLogger(f"airedale.test.app{next(_counter)}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handled: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            handled.append(record)

    logger.addHandler(_Collect())
    ensure_mcp_log_guard()  # (After the logger exists: the filter goes on every logger that exists.)
    return logger, handled


def _messages(handled) -> str:
    return "\n".join(record.getMessage() for record in handled)


def _run_string_code(code: str, **names):
    """Run ``code`` as code of the application: ``compile(..., "<string>")`` and ``exec``."""
    exec(compile(code, "<string>", "exec"), {"__name__": "__main__", **names})


@pytest.fixture
def sdk_string_module(monkeypatch):
    """An ``mcp`` module that defines code made by ``compile(..., "<string>")``: like the methods that dataclasses write.

    ``module.run(source, **names)`` runs ``source`` as code of that module (its globals are the ones of the module).
    """
    module = types.ModuleType(f"mcp._airedale_test{next(_counter)}")
    exec(
        compile("def marker_function():\n    return 1\n", "<string>", "exec"),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def run(source: str, **names):
        module.__dict__.update(names)
        exec(compile(source, "<string>", "exec"), module.__dict__)

    module.run = run  # pyright: ignore[reportAttributeAccessIssue]
    return module


def test_app_code_made_by_exec_keeps_its_log_text(app_logger, sdk_string_module):
    logger, handled = app_logger
    _run_string_code("logger.warning('app data %s', marker)", logger=logger, marker=MARKER)
    assert "app data " + MARKER in _messages(handled)


def test_app_code_of_python_dash_c_keeps_its_log_text():
    import subprocess

    script = textwrap.dedent(
        """
        import io, logging, sys
        import airedale.mcp  # loads the SDK and installs the guard
        from airedale.safe_log import ensure_mcp_log_guard
        stream = io.StringIO()
        logger = logging.getLogger("app")
        logger.addHandler(logging.StreamHandler(stream))
        logger.propagate = False
        ensure_mcp_log_guard()
        logger.warning("app data %s", sys.argv[1])
        print(stream.getvalue())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, MARKER], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "app data " + MARKER in result.stdout


def test_a_definition_that_the_sdk_imports_is_not_sdk_code(app_logger):
    import mcp.server.sse  # noqa: F401  (Imports ``sse_starlette``: the module then holds definitions that it does not own.)
    import sse_starlette.sse as sse

    logger, handled = app_logger
    code = "logger.warning('app data %s', marker)"
    exec(compile(code, sse.__file__, "exec"), {"__name__": "__main__", "logger": logger, "marker": MARKER})
    assert "app data " + MARKER in _messages(handled)


def test_sdk_code_made_by_exec_is_hidden(app_logger, sdk_string_module):
    logger, handled = app_logger
    sdk_string_module.run("logger.warning('sdk data %s', marker)", logger=logger, marker=MARKER)
    assert MARKER not in _messages(handled)
    assert "MCP SDK WARNING" in _messages(handled)


def test_two_nested_functions_with_one_file_line_and_name_are_hidden(app_logger, sdk_string_module, restored_factory):
    """``findCaller`` picks the SDK ``emit`` (``stacklevel=2``), and the app ``emit`` has the same file, line and name."""
    logger, handled = app_logger
    namespace = {"__name__": "__main__"}
    exec(
        compile("def emit(logger, marker):\n    logger.warning(marker, stacklevel=2)\n", "<string>", "exec"), namespace
    )
    sdk_string_module.run("def emit(logger, marker, app_emit):\n    app_emit(logger, marker)\n")
    results: list[str] = []
    guard = logging.getLogRecordFactory()

    def probe(name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs):
        results.append(safe_log._frame_origin(fn, lno, func))
        return guard(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    logging.setLogRecordFactory(probe)
    sdk_string_module.emit(logger, "data " + MARKER, namespace["emit"])
    assert results == ["unknown"]  # Two frames with the same triple and different origins.
    assert MARKER not in _messages(handled) and len(handled) == 1


def test_a_record_that_another_thread_handles_inside_app_code_with_the_same_triple_is_hidden(
    app_logger, sdk_string_module
):
    """The thread that handles a record can run app code with the same file, line and function: that is no proof."""
    logger, handled = app_logger
    record = logging.LogRecord(
        "airedale.test.sdk", logging.WARNING, "<string>", 2, "data %s", (MARKER,), None, func="<module>"
    )
    errors: list[BaseException] = []

    def handle_in_app_code():
        try:
            _run_string_code("x = 1\nlogger.handle(record)", logger=logger, record=record)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    thread = threading.Thread(target=handle_in_app_code)
    thread.start()
    thread.join(timeout=10)
    assert not errors
    assert MARKER not in _messages(handled) and len(handled) == 1
    logger.handle(record)  # The same record, handled later by the thread that made it.
    assert MARKER not in _messages(handled)


def test_an_unhashable_record_class_is_kept_for_app_code_and_hidden_for_sdk_code(app_logger, sdk_string_module):
    class Unhashable(logging.LogRecord):
        __hash__ = None  # pyright: ignore[reportAssignmentType]

    logger, handled = app_logger
    base = logging.getLogRecordFactory()
    logging.setLogRecordFactory(lambda *args, **kwargs: Unhashable(*args, **kwargs) if args else base(*args, **kwargs))
    ensure_mcp_log_guard()
    _run_string_code("logger.warning('app data %s', marker)", logger=logger, marker=MARKER)
    sdk_string_module.run("logger.warning('sdk data %s', marker)", logger=logger, marker=MARKER)
    texts = [record.getMessage() for record in handled]
    assert texts[0] == "app data " + MARKER
    assert MARKER not in texts[1] and texts[1].endswith("(details hidden)")
    assert all(isinstance(record, Unhashable) for record in handled)


def test_an_equal_record_that_the_guard_did_not_make_does_not_get_the_evidence_of_another(
    app_logger, sdk_string_module
):
    class EqualByPlace(logging.LogRecord):
        def _key(self):
            return (self.pathname, self.lineno, self.funcName)

        def __eq__(self, other):
            return isinstance(other, EqualByPlace) and self._key() == other._key()

        def __hash__(self):
            return hash(self._key())

    logger, handled = app_logger
    base = logging.getLogRecordFactory()
    logging.setLogRecordFactory(lambda *args, **kwargs: EqualByPlace(*args, **kwargs))
    ensure_mcp_log_guard()
    _run_string_code("x = 1\nlogger.warning('app data %s', marker)", logger=logger, marker=MARKER)
    first = handled[0]  # (Kept alive: its evidence is in the store.)
    assert first.getMessage() == "app data " + MARKER and first.lineno == 2 and first.funcName == "<module>"
    logging.setLogRecordFactory(base)
    second = EqualByPlace(
        "airedale.test.sdk", logging.WARNING, "<string>", 2, "sdk data %s", (MARKER,), None, func="<module>"
    )
    assert second == first and hash(second) == hash(first)
    logger.handle(second)
    assert MARKER not in handled[1].getMessage()
    assert first is not second


def test_a_record_without_weak_reference_has_no_evidence_and_is_hidden(app_logger, sdk_string_module, monkeypatch):
    logger, handled = app_logger

    def no_reference(*args, **kwargs):
        raise TypeError("cannot create weak reference")

    monkeypatch.setattr(safe_log.weakref, "ref", no_reference)
    _run_string_code("logger.warning('app data %s', marker)", logger=logger, marker=MARKER)  # Does not raise.
    monkeypatch.undo()
    assert len(handled) == 1
    assert MARKER not in handled[0].getMessage() and handled[0].getMessage().endswith("(details hidden)")


def test_the_origins_of_the_real_sdk_are_its_own_files():
    import gc

    ensure_mcp_log_guard()
    origins = safe_log._origins
    with origins._lock:
        origins._last = None  # Collect again.
        origins._refresh()
    prefixes = origins._prefixes
    assert prefixes
    not_in_the_sdk = sorted(name for name in origins._filenames if not name.startswith(prefixes) and name != "<string>")
    assert not_in_the_sdk == []  # (No file of ``contextlib``, ``dataclasses``, ``sse_starlette``, ``pydantic`` ...)
    namespaces = {id(vars(module)) for name, module in list(sys.modules.items()) if safe_log._is_mcp_name(name)}
    missing = set()
    for item in gc.get_objects():
        if isinstance(item, types.FunctionType) and id(item.__globals__) in namespaces:
            if "<locals>" in item.__qualname__:
                continue
            name = item.__code__.co_filename
            if not safe_log._is_ambiguous(name) and not (
                os.path.normcase(name) in origins._filenames or os.path.normcase(name).startswith(prefixes)
            ):
                missing.add(name)
    assert missing == set()


# --------------------------------------------------------------------------- #
# The origins are collected again when ``mcp`` modules change or finish their import
# --------------------------------------------------------------------------- #
def _fake_sdk_module(monkeypatch, name: str, filename: str | None = None, *, initializing: bool = False):
    """An ``mcp`` module in ``sys.modules``. With ``filename``, ``sdk_function(logger, marker)`` is code of that file."""
    import importlib.machinery

    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    assert module.__spec__ is not None
    module.__spec__._initializing = initializing  # pyright: ignore[reportAttributeAccessIssue]
    if filename is not None:
        exec(
            compile("def sdk_function(logger, marker):\n    logger.warning('sdk data %s', marker)\n", filename, "exec"),
            module.__dict__,
        )
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _unique_file(label: str) -> str:
    return f"/nonexistent-build-dir/{label}{next(_counter)}/module.py"


def _app_log(logger, filename: str, text: str = "app data") -> None:
    exec(compile(f"logger.warning('{text} %s', marker)", filename, "exec"), {"logger": logger, "marker": MARKER})


def test_a_module_that_is_still_importing_is_not_a_negative_answer(app_logger, monkeypatch):
    logger, handled = app_logger
    path = _unique_file("partial")
    module = _fake_sdk_module(monkeypatch, f"mcp._airedale_test_partial{next(_counter)}", initializing=True)
    origins = safe_log._origins
    assert origins.classify("some.logger", path) == "unknown"  # The module has no function yet: not "not the SDK".
    assert os.path.normcase(path) not in origins._answers
    _app_log(logger, path)  # The factory found that this code is not the SDK: kept, and nothing is cached.
    assert "app data " + MARKER in _messages(handled)
    assert os.path.normcase(path) not in origins._answers
    # The import goes on: the module defines a function of this file, then it is complete.
    exec(
        compile("def late(logger, marker):\n    logger.warning('sdk data %s', marker)\n", path, "exec"), module.__dict__
    )
    assert module.__spec__ is not None
    module.__spec__._initializing = False  # pyright: ignore[reportAttributeAccessIssue]
    module.late(logger, MARKER)  # pyright: ignore[reportAttributeAccessIssue]
    assert MARKER not in handled[-1].getMessage() and handled[-1].getMessage().endswith("(details hidden)")
    assert origins.classify("some.logger", path) == "sdk"


def test_a_module_that_replaces_another_one_with_the_same_count_is_collected(app_logger, monkeypatch):
    """``len(sys.modules)`` does not change, but a module of the last collection is another object (or gone)."""
    logger, handled = app_logger
    name = f"mcp._airedale_test_replaced{next(_counter)}"
    first_file, second_file, third_file, fourth_file = (_unique_file(label) for label in "abcd")
    _fake_sdk_module(monkeypatch, name, first_file)
    _app_log(logger, second_file)  # The answer "not the SDK" for the second file is cached.
    _app_log(logger, third_file)
    count = len(sys.modules)
    # (a) Another module object under the same name.
    second = _fake_sdk_module(monkeypatch, name, second_file)
    assert len(sys.modules) == count
    second.sdk_function(logger, MARKER)  # pyright: ignore[reportAttributeAccessIssue]
    assert MARKER not in handled[-1].getMessage() and handled[-1].getMessage().endswith("(details hidden)")
    # (b) One module is removed and another one comes, in one step.
    other = f"mcp._airedale_test_other{next(_counter)}"
    monkeypatch.delitem(sys.modules, name)
    third = _fake_sdk_module(monkeypatch, other, third_file)
    assert len(sys.modules) == count
    third.sdk_function(logger, MARKER)  # pyright: ignore[reportAttributeAccessIssue]
    assert MARKER not in handled[-1].getMessage() and handled[-1].getMessage().endswith("(details hidden)")
    _app_log(logger, fourth_file)
    assert handled[-1].getMessage() == "app data " + MARKER


LATE_MODULE_BODY = textwrap.dedent(
    """
    import sys
    sys.modules.pop("airedale_test_leaving_module", None)  # (A module that is not an MCP module leaves.)

    def log_it(logger, marker):
        logger.warning("sdk data %s", marker)
    """
)


def test_an_mcp_module_that_comes_in_while_another_module_leaves_is_collected(
    app_logger, tmp_path, monkeypatch, request
):
    """The import watcher runs before the module is in ``sys.modules``: its signal is kept until the import is over.

    A guard that took the signal at the first record (nothing new in ``sys.modules`` yet) would trust its cached "not
    the SDK" for this file after the import, because the module count is the same.
    """
    import importlib
    import importlib.machinery
    import py_compile

    import mcp

    logger, handled = app_logger
    name = "mcp._airedale_test_late"
    build_path = "/nonexistent-build-dir/late/_airedale_test_late.py"
    source = tmp_path / "_airedale_test_late.py"
    source.write_text(LATE_MODULE_BODY)
    py_compile.compile(str(source), cfile=str(tmp_path / "_airedale_test_late.pyc"), dfile=build_path, doraise=True)
    source.unlink()  # Only the bytecode: its code objects keep the build path.
    monkeypatch.setattr(mcp, "__path__", [*mcp.__path__, str(tmp_path)])
    monkeypatch.setitem(sys.modules, "airedale_test_leaving_module", types.ModuleType("airedale_test_leaving_module"))
    request.addfinalizer(lambda: (sys.modules.pop(name, None), mcp.__dict__.pop("_airedale_test_late", None)))
    paused, release = threading.Event(), threading.Event()

    class PauseBeforeInsertion:
        """A finder after the watcher: it waits before the module goes in ``sys.modules``, then finds it."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != name:
                return None
            paused.set()
            assert release.wait(30)
            return importlib.machinery.PathFinder.find_spec(fullname, path)

    pause = PauseBeforeInsertion()
    request.addfinalizer(lambda: sys.meta_path.remove(pause) if pause in sys.meta_path else None)
    sys.meta_path.insert(1, pause)
    origins = safe_log._origins
    _app_log(logger, build_path)  # 1. The answer "not the SDK" for this file is cached.
    assert handled[-1].getMessage() == "app data " + MARKER
    assert origins._answers.get(os.path.normcase(build_path)) is False
    count = len(sys.modules)
    errors: list[BaseException] = []

    def import_module():
        try:
            importlib.import_module(name)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    importer = threading.Thread(target=import_module)
    importer.start()
    try:
        assert paused.wait(30)  # 2. The watcher has seen the import. The module is not in ``sys.modules`` yet.
        assert name not in sys.modules
        _app_log(logger, build_path)  # 3. A record from the cached path: the snapshot is the same as the last one.
        assert handled[-1].getMessage() == "app data " + MARKER
        assert name in origins._unresolved  # The signal is not taken: the import is not over.
    finally:
        release.set()
        importer.join(30)
    assert not errors and not importer.is_alive()
    assert "airedale_test_leaving_module" not in sys.modules and len(sys.modules) == count  # 4. Same count.
    sys.modules[name].log_it(logger, MARKER)  # pyright: ignore[reportAttributeAccessIssue]
    assert MARKER not in handled[-1].getMessage() and handled[-1].getMessage().endswith("(details hidden)")
    assert name not in origins._unresolved


def test_cached_app_records_make_no_snapshot(app_logger, monkeypatch):
    logger, handled = app_logger
    for _ in range(3):  # (The first record can collect, and a new thread or import can change the counts.)
        logger.warning("settle")
    calls: list[int] = []
    real = safe_log._mcp_snapshot

    def spy():
        calls.append(1)
        return real()

    monkeypatch.setattr(safe_log, "_mcp_snapshot", spy)
    for _ in range(1000):
        logger.warning("app data %s", MARKER)
    assert calls == []
    assert len(handled) == 1003 and all(record.getMessage() != "" for record in handled)


IMPORT_IN_A_THREAD = textwrap.dedent(
    """
    import importlib, io, logging, os, py_compile, sys, threading, types
    import airedale.mcp  # loads the SDK and installs the guard
    import mcp
    from airedale.safe_log import ensure_mcp_log_guard

    directory, marker = sys.argv[1], sys.argv[2]
    name = "mcp._airedale_test_import"
    build_path = "/nonexistent-build-dir/import/_airedale_test_import.py"
    body = '''
    import logging, sys
    import airedale_test_shared as shared
    assert __spec__._initializing is True
    logger = logging.getLogger("airedale.test.sdk")
    logger.warning("sdk data during the import %s", shared.marker)
    shared.kept.append(logger.makeRecord(
        logger.name, logging.WARNING, sys._getframe().f_code.co_filename, 3, "sdk data kept %s", (shared.marker,), None))
    shared.entered.set()
    assert shared.release.wait(30)

    def log_it(logger, marker):
        logger.warning("sdk data after the import %s", marker)
    '''
    with open(os.path.join(directory, "_airedale_test_import.py"), "w") as handle:
        handle.write(body)
    py_compile.compile(
        os.path.join(directory, "_airedale_test_import.py"),
        cfile=os.path.join(directory, "_airedale_test_import.pyc"), dfile=build_path, doraise=True)
    os.unlink(os.path.join(directory, "_airedale_test_import.py"))  # Only the bytecode.
    mcp.__path__.append(directory)
    shared = types.ModuleType("airedale_test_shared")
    shared.marker, shared.kept = marker, []
    shared.entered, shared.release = threading.Event(), threading.Event()
    sys.modules["airedale_test_shared"] = shared

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    loggers = [logging.getLogger("airedale.test.app"), logging.getLogger("airedale.test.sdk")]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    ensure_mcp_log_guard()

    errors = []
    def import_it():
        try:
            importlib.import_module(name)
        except BaseException as error:
            errors.append(error)
    importer = threading.Thread(target=import_it)
    importer.start()
    assert shared.entered.wait(30), errors
    loggers[0].warning("app data while the import waits %s", marker)  # (From this file, which the SDK does not know.)
    shared.release.set()
    importer.join(30)
    assert not errors and not importer.is_alive(), errors
    sys.modules[name].log_it(loggers[1], marker)
    loggers[1].handle(shared.kept[0])  # The record that was made during the import.
    print("LOGGED-BEGIN")
    print(stream.getvalue())
    print("LOGGED-END")
    """
)


def test_an_sdk_module_that_is_importing_in_another_thread_is_not_cached_as_not_the_sdk(tmp_path):
    import subprocess

    script = tmp_path / "import_in_a_thread.py"
    script.write_text(IMPORT_IN_A_THREAD)
    modules = tmp_path / "modules"  # (Not the directory of the script: the SDK would own the script.)
    modules.mkdir()
    result = subprocess.run(
        [sys.executable, str(script), str(modules), MARKER], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]
    logged = result.stdout.split("LOGGED-BEGIN", 1)[1].split("LOGGED-END", 1)[0]
    assert logged.count("(details hidden)") == 3, logged  # During the import, the kept record, after the import.
    assert logged.count("app data while the import waits " + MARKER) == 1, logged
    assert logged.count(MARKER) == 1, logged  # Only the app record has the marker.
