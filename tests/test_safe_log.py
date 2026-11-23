# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Log lines about an exception have no message and no traceback values (data of MCP servers and of pi)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import traceback

import pytest

import airedale.mcp as mcp_module
from airedale.harness.pi_spans import PiSpanRecorder
from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog
from airedale.safe_log import exception_summary, safe_error
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


@pytest.mark.skipif(os.name == "nt", reason="needs a POSIX shell script")
def test_stdio_servers_of_the_probe_start_in_the_workspace(tmp_path, monkeypatch):
    """A relative command exists only in the workspace of the run, not in the directory of Airedale."""

    workspace, elsewhere = tmp_path / "ws", tmp_path / "airedale"
    workspace.mkdir()
    elsewhere.mkdir()
    script = workspace / "server.sh"
    script.write_text('#!/bin/sh\necho started-in-$(basename "$PWD") >&2\nexit 1\n')
    script.chmod(0o755)
    monkeypatch.chdir(elsewhere)
    server = McpServerSpec(name="rel", command="./server.sh")
    notes: list[str] = []
    asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes, workspace))
    assert len(notes) == 1 and "started-in-ws" in notes[0]
    # Without the workspace (the old behaviour) the command is not found.
    notes = []
    asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes))
    assert notes == []


def test_the_runner_probes_in_its_workspace(tmp_path, monkeypatch):
    import airedale.harness.base as base
    from airedale.harness.pi import PiRunner

    seen = {}

    async def catalog(servers, headers, notes=None, cwd=None):
        seen["cwd"] = cwd
        return {}

    class _Enabled:
        enabled = True

    monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", catalog)
    monkeypatch.setattr(base, "LLMObs", _Enabled)
    runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="s", command="x")])
    asyncio.run(runner._span_context())
    assert seen["cwd"] == str(tmp_path)


def test_a_noisy_server_does_not_use_memory_or_disk_in_proportion_to_its_stderr(monkeypatch):
    """64 MiB on stderr: only the last 16 KiB are kept, and no file is made for it."""
    import sys
    import tempfile
    import threading
    import tracemalloc

    from airedale.mcp import MCP_PROBE_STDERR_TAIL_BYTES

    def no_file(*args, **kwargs):
        raise AssertionError("the stderr of a probed server must not go to a file")

    monkeypatch.setattr(tempfile, "TemporaryFile", no_file)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_file)
    script = (
        "import sys; block = b'x' * 65535 + bytes([10]); "
        "[sys.stderr.buffer.write(block) for _ in range(1024)]; "
        "sys.stderr.buffer.write(b'THE-END'); sys.stderr.flush(); sys.exit(1)"
    )
    server = McpServerSpec(name="loud", command=sys.executable, args=("-c", script))
    notes: list[str] = []
    threads_before = threading.active_count()
    tracemalloc.start()
    try:
        asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(notes) == 1 and notes[0].endswith("THE-END")
    assert len(notes[0].encode()) < MCP_PROBE_STDERR_TAIL_BYTES + 200
    assert peak < 4 * 1024 * 1024, peak  # (64 MiB of output; the buffer is 16 KiB, a read is 64 KiB.)
    assert threading.active_count() <= threads_before  # The reader thread ended with the output.


# --------------------------------------------------------------------------- #
# The reader of the stderr of a probed server stops, also when a child keeps the pipe open
# --------------------------------------------------------------------------- #
def _probe_threads() -> list:
    import threading

    return [thread for thread in threading.enumerate() if thread.name == "mcp-probe-stderr" and thread.is_alive()]


def _is_closed(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


def test_a_write_handle_that_stays_open_does_not_keep_the_reader_or_the_descriptor():
    import time

    from airedale.mcp import _StderrTail

    async def scenario():
        tail = _StderrTail(1024)
        held = os.dup(tail.errlog.fileno())  # (A grandchild that keeps the write end open.)
        try:
            os.write(held, b"first words\n")
            started = time.monotonic()
            text = await tail.finish()
            elapsed = time.monotonic() - started
            return tail, text, elapsed
        finally:
            os.close(held)

    tail, text, elapsed = asyncio.run(scenario())
    assert text == "first words"
    assert elapsed < 1.5
    tail._thread.join(timeout=1.5)
    assert not tail._thread.is_alive()
    assert _is_closed(tail._read_fd)


def test_a_writer_that_never_stops_is_cut_off_at_the_deadline():
    import threading
    import time

    from airedale.mcp import _StderrTail

    async def scenario():
        tail = _StderrTail(1024)
        held = os.dup(tail.errlog.fileno())
        failure: list[BaseException] = []
        stop = threading.Event()

        def write_without_end():
            try:
                while not stop.is_set():
                    os.write(held, b"x" * 4096)
            except (
                OSError
            ) as error:  # The read end is closed: ``EPIPE`` (a ``BrokenPipeError``), or another error code.
                failure.append(error)
            finally:
                os.close(held)

        writer = threading.Thread(target=write_without_end, daemon=True)
        writer.start()
        try:
            started = time.monotonic()
            await tail.finish()
            elapsed = time.monotonic() - started
            writer.join(timeout=5)
            return tail, elapsed, failure, writer.is_alive()
        finally:
            stop.set()

    tail, elapsed, failure, writer_alive = asyncio.run(scenario())
    assert elapsed < 1.0 + 0.5
    assert not tail._thread.is_alive() and _is_closed(tail._read_fd)
    assert not writer_alive and len(failure) == 1  # The writer saw the closed pipe.


def test_a_cancelled_finish_still_ends_the_reader_and_closes_the_descriptor():
    from airedale.mcp import _StderrTail

    async def scenario():
        tail = _StderrTail(1024)
        held = os.dup(tail.errlog.fileno())
        try:
            task = asyncio.create_task(tail.finish())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(2.0)  # The deadline (1 s) passes with nobody waiting for the reader.
            return tail
        finally:
            os.close(held)

    tail = asyncio.run(scenario())
    assert not tail._thread.is_alive()
    assert _is_closed(tail._read_fd)


def test_finish_does_not_block_the_event_loop():
    import time

    from airedale.mcp import _StderrTail

    async def scenario():
        tail = _StderrTail(1024)
        held = os.dup(tail.errlog.fileno())
        gaps: list[float] = []
        running = True

        async def ticks():
            last = time.monotonic()
            while running:
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        ticker = asyncio.create_task(ticks())
        try:
            await tail.finish()  # (About 1 s: the write end is held.)
        finally:
            running = False
            await ticker
            os.close(held)
        return gaps

    gaps = asyncio.run(scenario())
    assert len(gaps) > 20 and max(gaps) < 0.2, max(gaps)


GRANDCHILD_SERVER = """
import subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=sys.stderr,
)
with open(sys.argv[1], "w") as handle:
    handle.write(str(child.pid))
sys.exit(0)
"""


def test_a_child_of_a_server_that_keeps_stderr_open_does_not_keep_the_reader_alive(tmp_path, monkeypatch):
    """The server exits; its child holds the stderr pipe for 60 s. After the probe, no reader thread is left."""
    import signal
    import sys
    import time

    if os.name == "nt":  # No Job Object: Windows lets the child of the server live on.
        import mcp.os.win32.utilities as win32_utilities

        monkeypatch.setattr(win32_utilities, "_create_job_object", lambda: None)
    pid_file = tmp_path / "child.pid"
    server = McpServerSpec(name="leaky", command=sys.executable, args=("-c", GRANDCHILD_SERVER, str(pid_file)))
    notes: list[str] = []
    child_pid = None
    try:
        started = time.monotonic()
        assert asyncio.run(_mcp_tool_metadata_catalog([server], {}, notes)) == {}
        elapsed = time.monotonic() - started
        child_pid = int(pid_file.read_text())
        assert elapsed < 10  # (Not the 60 s of the child.)
        deadline = time.monotonic() + 3
        while _probe_threads() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _probe_threads() == []
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None:
            with contextlib.suppress(OSError):
                os.kill(child_pid, signal.SIGTERM)


# --------------------------------------------------------------------------- #
# safe_error: the error that a guard raises in place of a raw one (ROMAIN-65, O7)
# --------------------------------------------------------------------------- #
class _WrapperError(RuntimeError):
    pass


def _raised(error: BaseException) -> BaseException:
    """Give ``error`` a traceback (a frame, and a source line without the marker)."""
    try:
        _raise_in_helper(error)
    except BaseException as raised:  # noqa: BLE001
        return raised
    raise AssertionError  # pragma: no cover


def _text(error: BaseException) -> str:
    """Everything that a library can copy from ``error``: ``str``, ``repr``, ``args``, notes, and the whole chain."""
    return "\n".join(
        [
            str(error),
            repr(error),
            repr(error.args),
            repr(getattr(error, "__notes__", None)),
            "".join(traceback.format_exception(error)),
        ]
    )


def test_safe_error_wraps_an_exception_with_its_notes_and_chain():
    secret = SECRET
    cause = ValueError(secret)
    error = _raised(KeyError(secret))
    error.__cause__ = cause
    error.add_note(secret)
    safe = safe_error(error, _WrapperError)
    assert type(safe) is _WrapperError
    assert str(safe) == "KeyError at test_safe_log.py:" + str(safe).rsplit(":", 1)[1]
    assert (safe.__cause__, safe.__context__, getattr(safe, "__notes__", None)) == (None, None, None)
    assert SECRET not in _text(safe)
    assert "raise error" in _text(safe)  # The frames stay.


def test_safe_error_wraps_an_exception_group_of_exceptions():
    secret = SECRET
    group = _raised(ExceptionGroup(secret, [ValueError(secret), ExceptionGroup("h", [KeyError(secret)])]))
    safe = safe_error(group, _WrapperError)
    assert type(safe) is _WrapperError
    assert str(safe).startswith("KeyError, ValueError at ")
    assert SECRET not in _text(safe)


def test_safe_error_keeps_the_shape_of_a_mixed_group():
    secret = SECRET
    leaf = ValueError(secret)
    leaf.add_note(secret)
    cancelled = asyncio.CancelledError()
    inner = BaseExceptionGroup(secret, [KeyError(secret), KeyboardInterrupt()])
    inner.add_note(secret)
    group = _raised(BaseExceptionGroup(secret, [leaf, cancelled, inner]))
    group.add_note(secret)
    group.__cause__ = ValueError(secret)

    safe = safe_error(group, _WrapperError)

    assert type(safe) is BaseExceptionGroup
    assert not isinstance(safe, Exception)  # (Still not retried, still not caught by ``except Exception``.)
    assert str(safe) == "CancelledError, KeyError, KeyboardInterrupt, ValueError at " + str(safe).split(" at ")[1]
    wrapped, kept, safe_inner = safe.exceptions
    assert type(wrapped) is _WrapperError
    assert kept is cancelled  # The same object.
    assert type(safe_inner) is BaseExceptionGroup
    assert [type(item) for item in safe_inner.exceptions] == [_WrapperError, KeyboardInterrupt]
    assert safe.__cause__ is None and safe.__context__ is None and getattr(safe, "__notes__", None) is None
    assert SECRET not in _text(safe)
    assert SECRET not in _text(safe_inner)


def test_safe_error_keeps_a_cancellation_and_cuts_its_raw_chain():
    secret = SECRET
    error = _raised(asyncio.CancelledError("Cancelled via cancel scope 0x1"))
    error.__cause__ = ValueError(secret)
    error.__context__ = ValueError(secret)
    error.add_note(secret)
    safe = safe_error(error, _WrapperError)
    assert safe is error  # The same object: asyncio, anyio and ddtrace see a cancellation.
    assert safe.args == ("Cancelled via cancel scope 0x1",)  # The message is not changed.
    assert safe.__cause__ is None and safe.__context__ is None and getattr(safe, "__notes__", None) is None
    assert SECRET not in _text(safe)


def test_safe_error_keeps_a_cancellation_context_for_anyio_and_cuts_its_raw_context():
    secret = SECRET
    inner = asyncio.CancelledError("Cancelled via cancel scope 0x2")
    inner.__context__ = ValueError(secret)
    inner.add_note(secret)
    error = asyncio.CancelledError("Cancelled via cancel scope 0x1")
    error.__context__ = inner
    safe = safe_error(error, _WrapperError)
    assert safe is error and safe.__context__ is inner
    assert inner.args == ("Cancelled via cancel scope 0x2",)
    assert inner.__context__ is None and getattr(inner, "__notes__", None) is None
    assert SECRET not in _text(safe)


def test_safe_error_does_not_loop_on_a_cycle_of_cancellation_contexts():
    first, second = asyncio.CancelledError("a"), asyncio.CancelledError("b")
    first.__context__, second.__context__ = second, first
    assert safe_error(first, _WrapperError) is first


@pytest.mark.parametrize("error", [SystemExit(3), KeyboardInterrupt()], ids=["system-exit", "keyboard-interrupt"])
def test_safe_error_keeps_the_other_base_exceptions(error):
    error.__context__ = ValueError(SECRET)
    safe = safe_error(error, _WrapperError)
    assert safe is error
    assert safe.__context__ is None
    if isinstance(safe, SystemExit):
        assert safe.code == 3


class _CustomGroup(BaseExceptionGroup):
    """A legal group that is not an ``Exception``, though all its leaves are."""


def test_safe_error_keeps_a_custom_group_of_exceptions_out_of_exception():
    secret = SECRET
    custom = _CustomGroup(secret, [ValueError(secret), KeyError(secret)])
    assert not isinstance(custom, Exception)  # (``BaseExceptionGroup(...)`` alone would make an ``ExceptionGroup``.)
    custom.add_note(secret)
    custom.extra = secret  # type: ignore[attr-defined]
    nested = _CustomGroup(secret, [_CustomGroup(secret, [OSError(secret)])])
    outer = _raised(BaseExceptionGroup(secret, [custom, ValueError(secret), nested]))

    for original in (_raised(custom), nested, outer):
        safe = safe_error(original, _WrapperError)
        assert isinstance(safe, BaseExceptionGroup) and not isinstance(safe, Exception)
        assert SECRET not in _text(safe)
        assert not hasattr(safe, "extra") and getattr(safe, "__notes__", None) is None
    safe = safe_error(custom, _WrapperError)
    assert [type(leaf) for leaf in safe.exceptions] == [_WrapperError, _WrapperError]  # type: ignore[attr-defined]
    assert str(safe).startswith("KeyError, ValueError") and str(safe).endswith("(2 sub-exceptions)")
    # ``subgroup`` and ``split`` make groups that are not an ``Exception`` either.
    part = safe.subgroup(_WrapperError)  # type: ignore[attr-defined]
    assert isinstance(part, BaseExceptionGroup) and not isinstance(part, Exception)
    matching, rest = safe.split(_WrapperError)  # type: ignore[attr-defined]
    assert not isinstance(matching, Exception) and rest is None
    # The outer group keeps the shape of its leaves: a safe group, a wrapper, and a safe group.
    safe_outer = safe_error(outer, _WrapperError)
    kinds = [isinstance(leaf, Exception) for leaf in safe_outer.exceptions]  # type: ignore[attr-defined]
    assert kinds == [False, True, False]


def test_the_summary_names_at_most_8_leaves_in_order():
    types = [ValueError, KeyError, OSError, TypeError, RuntimeError, IndexError, LookupError, ArithmeticError, EOFError]
    group = BaseExceptionGroup(
        "g", [BaseExceptionGroup("h", [t("x") for t in types[:5]]), *[t("x") for t in types[5:]]]
    )
    summary = exception_summary(group)
    assert summary == "ArithmeticError, IndexError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError"
    assert "EOFError" not in summary  # The 9th leaf.
    safe = safe_error(BaseExceptionGroup("g", [group, asyncio.CancelledError()]), _WrapperError)
    assert str(safe).startswith("ArithmeticError, IndexError, KeyError, LookupError, OSError, RuntimeError, TypeError")


def test_the_summary_skips_a_group_that_contains_itself():
    class OddError(Exception):
        """Not a real group: its ``exceptions`` list holds itself and a leaf."""

    odd = OddError("x")
    odd.exceptions = [odd, ValueError(SECRET)]  # type: ignore[attr-defined]
    assert exception_summary(odd) == "ValueError"
