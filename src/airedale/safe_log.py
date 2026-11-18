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
import inspect
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
#: Weak references to the guard factories that this module made. A factory is a guard only if it *is* one of these
#: objects (compared with ``is``): a ``WeakSet`` would hash the factory and compare it with ``==``, and the application
#: factory can be unhashable (``TypeError``) or equal to a guard (a proxy such as ``wrapt.FunctionWrapper``).
_guards: list[weakref.ref[Any]] = []


def _is_guard(factory: Any) -> bool:
    _guards[:] = [reference for reference in _guards if reference() is not None]  # (Forget the ones that are gone.)
    return any(reference() is factory for reference in _guards)


#: What is known about the origin of a log record: made by code of the ``mcp`` package, by other code, or not decided.
_SDK, _APP, _UNKNOWN = "sdk", "app", "unknown"


def _is_mcp_name(name: Any) -> bool:
    return isinstance(name, str) and (name == "mcp" or name.startswith("mcp."))


def _is_ambiguous(filename: str) -> bool:
    """Tell whether a file name can be the name of code of many origins: empty, or ``<string>``, ``<stdin>`` and so on."""
    return not filename or (filename.startswith("<") and filename.endswith(">"))


class _McpOrigins:
    """Tell whether a log record was made by code of the installed ``mcp`` package.

    ``record.pathname`` is the ``co_filename`` of the calling code, which is not always inside the directory where the
    package is loaded from: a bytecode-only install (a relocated ``.pyc`` directory, a ``.pyc`` in a ZIP) keeps the
    path where the code was built. So the origins are collected from the loaded ``mcp`` modules themselves: the
    directories of their ``__file__`` and spec origin, and the ``co_filename`` of the functions that they **define**
    (not the ones that they import: a function belongs to a module when its ``__globals__`` is the dictionary of that
    module, after :func:`inspect.unwrap`). They are collected again when a file name is not known.

    A file name is not always enough. ``<string>`` is the name of any code made by ``exec`` or ``compile``, and the SDK
    has such code too (the methods that ``dataclasses`` writes), so a record from ``<string>`` can come from the
    application or from the SDK. :meth:`classify` gives ``_UNKNOWN`` for these names, and the caller decides with other
    evidence (see :func:`_frame_origin`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modules_seen = -1  # How many ``mcp`` modules were loaded when the origins were last collected.
        self._filenames: frozenset[str] = frozenset()
        self._prefixes: tuple[str, ...] = ()
        self._answers: dict[str, bool] = {}

    def classify(self, logger_name: str, filename: str) -> str:
        """Return ``_SDK`` or ``_APP`` for the file name, or ``_UNKNOWN`` when it is an ambiguous name of the SDK."""
        if _is_mcp_name(logger_name):
            return _SDK
        key = os.path.normcase(filename)
        if _is_ambiguous(key):
            with self._lock:  # (Not cached: the same name can be the name of other code the next time.)
                self._collect()
                return _UNKNOWN if key in self._filenames else _APP
        answer = self._answers.get(key)
        if answer is None:
            with self._lock:
                self._collect()
                answer = key in self._filenames or key.startswith(self._prefixes)
                self._answers[key] = answer
        return _SDK if answer else _APP

    def is_sdk(self, logger_name: str, filename: str) -> bool:
        return self.classify(logger_name, filename) == _SDK

    def _collect(self) -> None:
        count = sum(1 for name in list(sys.modules) if _is_mcp_name(name))
        if count == self._modules_seen:
            return  # Nothing new was loaded since the last time: the answer for a new file name is "not the SDK".
        filenames: set[str] = set()
        prefixes: set[str] = set()
        for name, module in list(sys.modules.items()):
            if not _is_mcp_name(name):
                continue
            try:
                _collect_module(module, filenames, prefixes)
            except Exception:  # noqa: BLE001 - a module with odd attributes must not break logging
                continue
        self._filenames, self._prefixes = frozenset(filenames), tuple(prefixes)
        self._answers.clear()
        self._modules_seen = count


def _collect_module(module: Any, filenames: set[str], prefixes: set[str]) -> None:
    origins = [getattr(module, "__file__", None), getattr(getattr(module, "__spec__", None), "origin", None)]
    origins.extend(getattr(module, "__path__", None) or ())
    for origin in origins:
        if isinstance(origin, str) and origin:
            directory = origin if os.path.isdir(origin) else os.path.dirname(origin)
            for candidate in {os.path.abspath(directory), os.path.realpath(directory)}:
                prefixes.add(os.path.normcase(candidate) + os.sep)
    namespace = vars(module)
    for value in list(namespace.values()):
        for function in _functions_defined_by(value, module, namespace):
            filenames.add(os.path.normcase(function.__code__.co_filename))


def _functions_defined_by(value: Any, module: Any, namespace: dict[str, Any]) -> list[Any]:
    """Return the functions that ``module`` defines in ``value``: ``value`` itself, or the methods of a class it defines.

    A function is defined by the module when its ``__globals__`` is the dictionary of the module (after the
    ``__wrapped__`` chain: ``@asynccontextmanager`` gives a ``contextlib`` function, and the SDK code is the wrapped
    one). A class is looked into only when it is defined by the module (``__module__``), one level deep: the class
    attributes can be functions, ``staticmethod`` / ``classmethod`` objects and properties.
    """
    if isinstance(value, type):
        if getattr(value, "__module__", None) != getattr(module, "__name__", object()):
            return []
        candidates = []
        for attribute in list(vars(value).values()):
            if isinstance(attribute, property):
                candidates.extend((attribute.fget, attribute.fset, attribute.fdel))
            else:
                candidates.append(getattr(attribute, "__func__", attribute))  # static and class methods
    else:
        candidates = [value]
    found = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            function = inspect.unwrap(candidate)
        except ValueError:  # A cycle of ``__wrapped__``.
            continue
        code = getattr(function, "__code__", None)
        if hasattr(code, "co_filename") and getattr(function, "__globals__", None) is namespace:
            found.append(function)
    return found


_origins = _McpOrigins()


def _hidden_message(level: int, filename: str, lineno: int) -> str:
    module = os.path.splitext(os.path.basename(filename))[0]
    return f"MCP SDK {logging.getLevelName(level)} at {module}:{lineno} (details hidden)"


# --------------------------------------------------------------------------- #
# The origin of a record that has an ambiguous file name: the frame that ``Logger.findCaller`` picks
# --------------------------------------------------------------------------- #
def _frame_state(frame: Any) -> str:
    name = frame.f_globals.get("__name__")
    if not isinstance(name, str):
        return _UNKNOWN
    return _SDK if _is_mcp_name(name) else _APP


def _frame_origin(fn: str, lno: int, func: str | None) -> str:
    """Return ``_SDK`` or ``_APP`` for the code that made a record, or ``_UNKNOWN`` when that is not sure.

    It runs in the record factory, on the thread that logs, while ``Logger._log`` is on the stack. The file name, the
    line and the function name of the record come from the frame that ``Logger.findCaller`` picks: this repeats its
    loop, with the ``stacklevel`` of ``_log``. Every step that fails gives ``_UNKNOWN`` (the record is then hidden):

    1. A ``Logger._log`` frame whose ``fn``, ``lno`` and ``func`` are the arguments of the factory.
    2. The logger of that frame uses ``Logger.findCaller`` itself, and ``logging._srcfile`` is set.
    3. The frame of the loop is the one with the same file, line and function name as the record.
    4. Every frame above ``_log`` with that same triple has the same origin (two nested functions of different origin
       can have the same file name, line and name: for example two ``emit`` of code made with ``compile(...,
       "<string>")``, called with ``stacklevel=2``).

    The origin is the ``__name__`` in the globals of the frame: ``mcp`` or ``mcp.*`` is the SDK.
    """
    try:
        internal = logging._is_internal_frame  # pyright: ignore[reportAttributeAccessIssue]
        log_code = logging.Logger._log.__code__
        if not logging._srcfile:
            return _UNKNOWN
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code is log_code:
                locals_ = frame.f_locals
                if locals_.get("fn") == fn and locals_.get("lno") == lno and locals_.get("func") == func:
                    break
            frame = frame.f_back
        if frame is None:
            return _UNKNOWN
        logger = frame.f_locals.get("self")
        if not isinstance(logger, logging.Logger) or type(logger).findCaller is not logging.Logger.findCaller:
            return _UNKNOWN
        stacklevel = frame.f_locals.get("stacklevel")
        if not isinstance(stacklevel, int) or isinstance(stacklevel, bool):
            return _UNKNOWN
        log_frame = frame
        selected = log_frame
        while stacklevel > 0:  # The loop of ``Logger.findCaller``.
            following = selected.f_back
            if following is None:
                break
            selected = following
            if not internal(selected):
                stacklevel -= 1
        code = selected.f_code
        if (code.co_filename, selected.f_lineno, code.co_name) != (fn, lno, func):
            return _UNKNOWN
        state = _frame_state(selected)
        if state == _UNKNOWN:
            return _UNKNOWN
        other = log_frame.f_back
        while other is not None:
            if (other.f_code.co_filename, other.f_lineno, other.f_code.co_name) == (fn, lno, func) and (
                _frame_state(other) != state
            ):
                return _UNKNOWN
            other = other.f_back
        return state
    except Exception:  # noqa: BLE001 - logging must not fail: no evidence
        return _UNKNOWN


# --------------------------------------------------------------------------- #
# The evidence of a record: kept by the identity of the record, never by its hash or its equality
# --------------------------------------------------------------------------- #
_EVIDENCE_PURGE_AT = 256


class _Evidence:
    __slots__ = ("funcName", "lineno", "pathname", "ref")

    def __init__(self, ref: weakref.ref[Any], pathname: Any, lineno: Any, func_name: Any) -> None:
        self.ref, self.pathname, self.lineno, self.funcName = ref, pathname, lineno, func_name


#: ``id(record)`` -> the evidence that the factory found. A ``WeakKeyDictionary`` would hash the record and compare it
#: with ``==``: a record class can be unhashable, and two equal records are not one record. The lock is an ``RLock``,
#: because a ``__del__`` that logs can run while the lock is held, on the same thread.
_evidence: dict[int, _Evidence] = {}
_evidence_lock = threading.RLock()


def _remember_app_record(record: Any) -> None:
    """Keep the evidence that ``record`` was made by code that is not the SDK. No evidence is kept for any other state."""
    try:
        entry = _Evidence(
            weakref.ref(record),  # (No callback: it could run in any thread, during garbage collection.)
            getattr(record, "pathname", None),
            getattr(record, "lineno", None),
            getattr(record, "funcName", None),
        )
    except TypeError:  # No weak reference is possible: the record has no evidence, and it is hidden.
        return
    with _evidence_lock:
        _evidence[id(record)] = entry
        if len(_evidence) > _EVIDENCE_PURGE_AT:
            for key in [key for key, item in list(_evidence.items()) if item.ref() is None]:
                _evidence.pop(key, None)


def _has_app_evidence(record: Any) -> bool:
    """Tell whether the factory found that this very record was made by code that is not the SDK."""
    with _evidence_lock:
        entry = _evidence.get(id(record))
    if entry is None or entry.ref() is not record:
        return False
    return (
        entry.pathname == getattr(record, "pathname", None)
        and entry.lineno == getattr(record, "lineno", None)
        and entry.funcName == getattr(record, "funcName", None)
    )


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
        state = _origins.classify(name, fn)
        if state == _UNKNOWN:  # An ambiguous file name: the frame of the caller tells.
            state = _frame_origin(fn, lno, func)
        if state != _APP:  # The SDK, or not sure: hide.
            msg = _hidden_message(level, fn, lno)
            args, exc_info, sinfo = (), None, None
        record = previous(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)
        if state == _APP:
            _remember_app_record(record)
        return record

    setattr(factory, _GUARD_MARK, True)
    _guards.append(weakref.ref(factory))
    return factory


class _SdkRecordFilter(logging.Filter):
    """Layer 2: a logger filter that gives a record of the SDK a fixed text, whatever factory made the record.

    ``Logger.handle`` runs the filters of the logger on which the record was made, when the record is handled, so this
    does not depend on the record factory (which the application can replace, also from another thread during a probe).

    A record with an ambiguous file name is kept only when the guard factory found, when the record was made, that its
    code is not the SDK (:func:`_has_app_evidence`). This never looks at the current stack: the thread that handles a
    record can run any code.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        state = _origins.classify(record.name, record.pathname)
        if state == _UNKNOWN and _has_app_evidence(record):
            state = _APP
        if state != _APP:
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
        if not _is_guard(current):  # (By identity: an attribute can be copied, and a proxy can be equal to a guard.)
            logging.setLogRecordFactory(_guard(current))
        _install_filters()


#: The name that the first version of the guard had.
install_mcp_log_guard = ensure_mcp_log_guard
