# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Describe an exception for a log line without its message or its traceback values.

The message of an exception can hold data that came from an MCP server or from pi: the validation error of the MCP
SDK for a ``tools/list`` result holds the whole tool object, and a ``ValueError`` from ``int()`` holds the value. Logs
go to terminals and CI systems, so a log line about such an exception gets only the type names and the place in the
code, never ``exc_info`` and never ``str(exception)``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_MAX_LEAVES = 8


def exception_summary(error: BaseException) -> str:
    """Return the type name(s) of ``error`` and where it was raised, for example ``ValueError at pi_spans.py:512``.

    An exception group is described by the types of its leaves (at most 8). No message, no argument, no local value.
    """
    names = sorted({type(leaf).__name__ for leaf in _leaves(error)[:_MAX_LEAVES]})
    place = ""
    traceback = error.__traceback__
    if traceback is not None:
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        place = f" at {os.path.basename(traceback.tb_frame.f_code.co_filename)}:{traceback.tb_lineno}"
    return ", ".join(names) + place


def _leaves(error: BaseException) -> list[BaseException]:
    group = getattr(error, "exceptions", None)
    if isinstance(group, tuple | list) and group and all(isinstance(item, BaseException) for item in group):
        return [leaf for item in group for leaf in _leaves(item)]
    return [error]


# --------------------------------------------------------------------------- #
# Log records made inside the MCP SDK
# --------------------------------------------------------------------------- #
_mcp_guard_lock = threading.Lock()
_mcp_guard_installed = False


def install_mcp_log_guard() -> None:
    """Rewrite every log record that the MCP SDK makes to a fixed text. Idempotent; permanent; thread-safe.

    The SDK logs data that an MCP server sent, before any code of Airedale can catch it: a line of invalid JSON on
    stdout is logged with its text and a traceback (``mcp.client.stdio``), and a request that the client does not
    support is logged with its parameters on the *root* logger (``mcp.shared.session``). Logs go to terminals and CI
    systems, and a successful run has no private report.

    A filter on the ``mcp`` or the root logger would miss the records of child loggers, since the filter of a logger
    does not run for the records that its children pass up. So the log record factory is wrapped: it runs for every
    record, knows the source file, and replaces the message of a record whose source file is inside the installed
    ``mcp`` package with ``MCP SDK <level> at <module>:<line> (details hidden)``, without arguments, exception,
    traceback or stack. The level and the logger name are kept; records from other files are not touched.
    The wrapper keeps the factory that was there before it, and is installed once, so concurrent probes are safe.
    """
    global _mcp_guard_installed
    with _mcp_guard_lock:
        if _mcp_guard_installed:
            return
        import mcp

        package_file = getattr(mcp, "__file__", None)
        if not package_file:  # pragma: no cover - a namespace package: nothing to recognize
            return
        directory = os.path.dirname(os.path.abspath(package_file))
        prefixes = tuple({os.path.normcase(directory) + os.sep, os.path.normcase(os.path.realpath(directory)) + os.sep})
        previous = logging.getLogRecordFactory()

        def factory(
            name: str,
            level: int,
            fn: str,
            lno: int,
            msg: Any,
            args: Any,
            exc_info: Any,
            func: str | None = None,
            sinfo: str | None = None,
            **kwargs: Any,
        ) -> logging.LogRecord:
            if os.path.normcase(fn).startswith(prefixes):
                module = os.path.splitext(os.path.basename(fn))[0]
                msg = f"MCP SDK {logging.getLevelName(level)} at {module}:{lno} (details hidden)"
                args, exc_info, sinfo = (), None, None
            return previous(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

        logging.setLogRecordFactory(factory)
        _mcp_guard_installed = True
