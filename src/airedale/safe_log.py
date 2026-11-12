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

import importlib
import logging
import os
import pkgutil
import sys
import threading
import weakref
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
_GUARD_MARK = "_airedale_mcp_log_guard"  # (a debug aid, never proof: ``functools.wraps`` copies attributes)
_guard_lock = threading.Lock()
#: The guard factories that this module made. A factory is a guard only if it is one of these objects.
_guards: weakref.WeakSet[Any] = weakref.WeakSet()


class _McpOrigins:
    """Tell whether a log record was made by code of the installed ``mcp`` package.

    ``record.pathname`` is the ``co_filename`` of the calling code, which is not always inside the directory where the
    package is loaded from: a bytecode-only install (a relocated ``.pyc`` directory, a ``.pyc`` in a ZIP) keeps the
    path where the code was built. So the origins are collected from the loaded ``mcp`` modules themselves: the
    directories of their ``__file__`` and spec origin, and the ``co_filename`` of the functions they define. They are
    collected again when a file name is not known (one answer for each file name is kept).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modules_seen = -1  # How many ``mcp`` modules were loaded when the origins were last collected.
        self._filenames: set[str] = set()
        self._prefixes: set[str] = set()
        self._answers: dict[str, bool] = {}

    def is_sdk(self, logger_name: str, filename: str) -> bool:
        if logger_name == "mcp" or logger_name.startswith("mcp."):
            return True
        key = os.path.normcase(filename)
        answer = self._answers.get(key)
        if answer is None:
            with self._lock:
                self._collect()
                answer = key in self._filenames or key.startswith(tuple(self._prefixes))
                self._answers[key] = answer
        return answer

    def _collect(self) -> None:
        count = sum(1 for name in list(sys.modules) if name == "mcp" or name.startswith("mcp."))
        if count == self._modules_seen:
            return  # Nothing new was loaded since the last time: the answer for a new file name is "not the SDK".
        self._modules_seen = count
        for name, module in list(sys.modules.items()):
            if name != "mcp" and not name.startswith("mcp."):
                continue
            try:
                self._collect_module(module)
            except Exception:  # noqa: BLE001 - a module with odd attributes must not break logging
                continue

    def _collect_module(self, module: Any) -> None:
        origins = [getattr(module, "__file__", None), getattr(getattr(module, "__spec__", None), "origin", None)]
        origins.extend(getattr(module, "__path__", None) or ())
        for origin in origins:
            if isinstance(origin, str) and origin:
                directory = origin if os.path.isdir(origin) else os.path.dirname(origin)
                for candidate in {os.path.abspath(directory), os.path.realpath(directory)}:
                    self._prefixes.add(os.path.normcase(candidate) + os.sep)
        for value in list(vars(module).values()):
            for function in _functions_of(value):
                self._filenames.add(os.path.normcase(function.__code__.co_filename))


def _functions_of(value: Any) -> list[Any]:
    """Return the functions that ``value`` is, or defines if it is a class (one level of classes)."""
    found = []
    candidates = list(vars(value).values()) if isinstance(value, type) else [value]
    for candidate in candidates:
        candidate = getattr(candidate, "__func__", candidate)  # static and class methods
        if hasattr(candidate, "__code__") and hasattr(candidate.__code__, "co_filename"):
            found.append(candidate)
    return found


_origins = _McpOrigins()


def _hidden_message(level: int, filename: str, lineno: int) -> str:
    module = os.path.splitext(os.path.basename(filename))[0]
    return f"MCP SDK {logging.getLevelName(level)} at {module}:{lineno} (details hidden)"


def _guard(previous: Any) -> Any:
    """Layer 1: a record factory that makes the record of the SDK with a fixed text, then calls ``previous``."""

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
        if _origins.is_sdk(name, fn):
            msg = _hidden_message(level, fn, lno)
            args, exc_info, sinfo = (), None, None
        return previous(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    setattr(factory, _GUARD_MARK, True)
    _guards.add(factory)
    return factory


class _SdkRecordFilter(logging.Filter):
    """Layer 2: a logger filter that gives a record of the SDK a fixed text, whatever factory made the record.

    ``Logger.handle`` runs the filters of the logger on which the record was made, when the record is handled, so this
    does not depend on the record factory (which the application can replace, also from another thread during a probe).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _origins.is_sdk(record.name, record.pathname):
            record.msg = _hidden_message(record.levelno, record.pathname, record.lineno)
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


_sdk_filter = _SdkRecordFilter()


def _install_filters() -> None:
    """Put the filter on the root logger (the SDK logs on it) and on every logger that exists.

    Not only on ``mcp`` / ``mcp.*``: some SDK loggers have other names (``client`` in ``mcp/client/session.py``,
    ``client.stdio.win32``, ``server``). The filter acts only on records made by SDK code (the origin of a file name is
    looked up once, then cached), so the records of other loggers are not changed.
    """
    logging.getLogger().addFilter(_sdk_filter)  # (``addFilter`` does nothing when the filter is already there.)
    for candidate in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger):  # (Not a ``PlaceHolder``.)
            candidate.addFilter(_sdk_filter)


_client_side_imported = False


def _import_client_side_sdk() -> None:
    """Import the client-side parts of the SDK once, so that their loggers exist before the filters are installed.

    The SDK makes a logger when a module is imported. A module that is imported later (the OAuth parser of
    ``mcp.client.auth`` is only imported when it is used) would make a logger without the filter. So every submodule of
    ``mcp.client``, ``mcp.shared``, ``mcp.types`` and the ``mcp.os`` package of this platform is imported now. A
    submodule that cannot be imported (an optional dependency is missing) is skipped, and so are ``__main__`` modules
    (they run code). The server-side parts (``mcp.server``, ``mcp.cli``) are not used by Airedale and not imported here.
    """
    global _client_side_imported
    if _client_side_imported:
        return
    _client_side_imported = True
    platform_package = "mcp.os.win32" if os.name == "nt" else "mcp.os.posix"
    for package in ("mcp.client", "mcp.shared", "mcp.types", platform_package):
        try:
            module = importlib.import_module(package)
        except Exception:  # noqa: BLE001 - an optional part must not stop the guard
            continue
        search_path = getattr(module, "__path__", None)
        if not search_path:
            continue
        for info in pkgutil.walk_packages(search_path, package + ".", onerror=lambda _name: None):
            if info.name.rsplit(".", 1)[-1] == "__main__":
                continue
            try:
                importlib.import_module(info.name)
            except Exception:  # noqa: BLE001 - for example a missing optional dependency
                continue


def ensure_mcp_log_guard() -> None:
    """Make sure that every log record made by the MCP SDK has a fixed text. Cheap; call it before each use of the SDK.

    The SDK logs data that an MCP server sent, before any code of Airedale can catch it: a line of invalid JSON on
    stdout is logged with its text and a traceback (``mcp.client.stdio``), and a request that the client does not
    support is logged with its parameters on the *root* logger (``mcp.shared.session``). Logs go to terminals and CI
    systems, and a successful run has no private report.

    A record of the SDK (see :class:`_McpOrigins`, or a logger called ``mcp`` or ``mcp.*``) gets the message
    ``MCP SDK <level> at <module>:<line> (details hidden)``, without arguments, exception, traceback or stack. The level
    and the logger name are kept; records from other code are not touched. Two layers do it:

    1. The log record factory is wrapped (a filter on the ``mcp`` logger alone would miss the records of its children,
       which pass up without its filter). The application can replace the factory later, and one that does not call the
       factory before it removes this layer, so the current factory is wrapped again when it is not one of the guards
       made here (checked by identity: ``functools.wraps`` copies attributes).
    2. A filter on the root logger and on every logger that exists (:class:`_SdkRecordFilter`). It runs when the
       record is handled, whatever factory made it, so a factory that another thread sets during a probe cannot remove it.

    Not covered: code in this process that copies the raw message into its own field inside its own record factory.
    Code in the process is trusted (it can read the data directly), and so are the loggers it replaces or creates during
    a probe (the SDK makes its loggers when its modules are imported). Thread-safe.
    """
    with _guard_lock:
        _import_client_side_sdk()
        current = logging.getLogRecordFactory()
        if current not in _guards:  # (By identity: an attribute can be copied to another factory.)
            logging.setLogRecordFactory(_guard(current))
        _install_filters()


#: The name that the first version of the guard had.
install_mcp_log_guard = ensure_mcp_log_guard
