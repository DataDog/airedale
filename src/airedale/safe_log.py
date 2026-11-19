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

import contextlib
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


def _is_claude_name(name: Any) -> bool:
    """Tell whether a logger belongs to the Claude Agent SDK (its loggers are made with ``__name__``)."""
    return isinstance(name, str) and (name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."))


def _is_ambiguous(filename: str) -> bool:
    """Tell whether a file name can be the name of code of many origins: empty, or ``<string>``, ``<stdin>`` and so on."""
    return not filename or (filename.startswith("<") and filename.endswith(">"))


class _McpImportWatcher:
    """A finder at the start of ``sys.meta_path`` that notes the name of every import of an ``mcp`` module.

    It never finds anything (``find_spec`` returns ``None``): it only tells :class:`_McpOrigins` that an ``mcp`` import has
    started. The importer calls it **before** the new module is put in ``sys.modules``, so the module itself is not
    visible yet: :class:`_McpOrigins` keeps the name until the import is over (see ``_refresh``).
    """

    def __init__(self) -> None:
        self.names: set[str] = set()  # (``add`` and ``pop`` of a ``set`` are atomic.)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if _is_mcp_name(fullname):
            self.names.add(fullname)

    def pending(self) -> list[str]:
        """Return a copy of the noted names. It removes nothing: see :meth:`commit`."""
        return list(self.names)

    def commit(self, names: list[str]) -> None:
        """Forget these names, once the caller has recorded them. A name is never in neither place."""
        for name in names:
            self.names.discard(name)


_import_watcher = _McpImportWatcher()


def _gil_disabled() -> bool:
    """Tell whether this is a free-threaded build that runs without the GIL (``sys._is_gil_enabled()`` is false)."""
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return callable(is_gil_enabled) and not is_gil_enabled()


def _install_import_watcher() -> None:
    """Put the watcher first in ``sys.meta_path`` (again, when another hook was put before it)."""
    finders = sys.meta_path
    if finders and finders[0] is _import_watcher:
        return
    with contextlib.suppress(ValueError):
        finders.remove(_import_watcher)
    finders.insert(0, _import_watcher)


def _mcp_snapshot() -> dict[str, tuple[Any, bool]]:
    """Return ``{name: (module, initializing)}`` for every ``mcp`` / ``mcp.*`` entry of ``sys.modules``.

    ``initializing`` is the ``_initializing`` flag of the spec: CPython sets it while the module runs its code.
    """
    snapshot = {}
    for name, module in list(sys.modules.items()):
        if _is_mcp_name(name):
            snapshot[name] = (module, bool(getattr(getattr(module, "__spec__", None), "_initializing", False)))
    return snapshot


def _same_snapshot(first: dict[str, tuple[Any, bool]], second: dict[str, tuple[Any, bool]]) -> bool:
    return first.keys() == second.keys() and all(
        first[name][0] is second[name][0] and first[name][1] == second[name][1] for name in first
    )


def _importing_mcp_names() -> set[str] | None:
    """Return the ``mcp`` names whose import is running now (a module lock exists), or ``None`` when that is not known."""
    locks = getattr(sys.modules.get("importlib._bootstrap"), "_module_locks", None)
    if locks is None:
        return None
    try:
        return {name for name in list(locks) if _is_mcp_name(name)}
    except Exception:  # noqa: BLE001
        return None


class _McpOrigins:
    """Tell whether a log record was made by code of the installed ``mcp`` package.

    ``record.pathname`` is the ``co_filename`` of the calling code, which is not always inside the directory where the
    package is loaded from: a bytecode-only install (a relocated ``.pyc`` directory, a ``.pyc`` in a ZIP) keeps the
    path where the code was built. So the origins are collected from the loaded ``mcp`` modules themselves: the
    directories of their ``__file__`` and spec origin, and the ``co_filename`` of the functions that they **define**
    (not the ones that they import: a function belongs to a module when its ``__globals__`` is the dictionary of that
    module, after :func:`inspect.unwrap`).

    A file name is not always enough. ``<string>`` is the name of any code made by ``exec`` or ``compile``, and the SDK
    has such code too (the methods that ``dataclasses`` writes), so a record from ``<string>`` can come from the
    application or from the SDK. :meth:`classify` gives ``_UNKNOWN`` for these names, and the caller decides with other
    evidence (see :func:`_frame_origin`).

    **When the origins are collected again.** The collection is not a count of modules: it is a snapshot of the identity
    and the import state of every ``mcp`` module (:func:`_mcp_snapshot`), and it is made again when the snapshot is
    not the same as the one of the last collection.

    * A module that is still importing (``_initializing``) has not defined all its functions. While one is, the
      collection is **not complete**: a path that is not known is ``_UNKNOWN``, and that answer is never cached.
    * A cached answer "not the SDK" (``_APP``) is valid for the snapshot that it was made with. A full snapshot costs
      about 200 µs (4,000 loaded modules), about 100 times a logging call, so it is made only when a cheap trigger says
      that something can have changed: an ``mcp`` import has started (:class:`_McpImportWatcher`, which sees the
      import of a name also when another module leaves ``sys.modules`` in the same step), ``len(sys.modules)``, a module
      of the last snapshot that is not the same object in ``sys.modules`` any more, or an incomplete collection.
    * The import watcher is called **before** the new module is in ``sys.modules``. A name that it noted stays
      *unresolved* until the import is over (the module is in the snapshot, or its import lock is gone: it failed, or it
      was not an import). It is not given up when a snapshot that is made before the module exists is the same as the
      last one: the module can come after that, when another module leaves ``sys.modules`` and ``len`` is the same.

    It never waits for an import and never imports, and it never waits for its own lock (another thread that holds it
    while it reads a module attribute could wait for an import that this thread has started): the answer is then
    ``_UNKNOWN``. Nothing is logged while the lock is held.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None
        #: Odd while ``_refresh`` runs, even otherwise; it grows by 2 for each collection (only ``_refresh`` writes it).
        self._sequence = 0
        self._last: dict[str, tuple[Any, bool]] | None = None  # The snapshot of the last collection.
        self._last_len = -1  # ``len(sys.modules)`` before that snapshot.
        self._complete = False  # No module of that snapshot was importing.
        self._unresolved: set[str] = set()  # Names of imports that started and are not over (see above).
        self._filenames: frozenset[str] = frozenset()
        self._prefixes: tuple[str, ...] = ()
        self._answers: dict[str, bool] = {}  # Normalized file name -> is the SDK. Valid for ``self._last``.

    def classify(self, logger_name: str, filename: str) -> str:
        """Return ``_SDK`` or ``_APP`` for the file name, or ``_UNKNOWN`` when it is not known (yet) who owns it.

        Every ambiguous file name (``<string>``, ``<stdin>``, empty, ...) is ``_UNKNOWN``, whatever the ``mcp`` modules
        define: only the frame of the code can tell (see :func:`_frame_origin`).
        """
        if _is_mcp_name(logger_name):
            return _SDK
        key = os.path.normcase(filename)
        ambiguous = _is_ambiguous(key)
        if not ambiguous:
            # A cached "not the SDK" answer is trusted only when it, the snapshot and the watcher are of one generation:
            # ``_sequence`` is read first, and read again after the stale check. A collection (``_refresh``) makes it odd
            # while it runs, and even and higher when it is over, so an answer that was read before or during a
            # collection is never used. Nothing here waits: when in doubt, the answer is ``_UNKNOWN`` or ``_decide``.
            sequence = self._sequence
            answer = self._answers.get(key)
            if answer is True:
                return _SDK
            if answer is False:
                if _gil_disabled():
                    # Free-threaded build: a reader that has no lock has no guarantee to see the writes of a collection
                    # (reading the size of a ``set`` is a relaxed load). The cached answer is checked under the lock.
                    return self._decide(key, ambiguous)
                if sequence & 1:
                    return _UNKNOWN  # A collection is running: its watcher names are in no state that can be read.
                if not self._stale() and self._sequence == sequence:
                    return _APP
        return self._decide(key, ambiguous)

    def is_sdk(self, logger_name: str, filename: str) -> bool:
        return self.classify(logger_name, filename) == _SDK

    def _stale(self) -> bool:
        """Cheap triggers: tell whether the last collection can be out of date (it never makes a snapshot)."""
        last = self._last
        if last is None or not self._complete or _import_watcher.names or self._unresolved:
            return True
        if len(sys.modules) != self._last_len:
            return True
        modules = sys.modules
        return any(modules.get(name) is not module for name, (module, _initializing) in last.items())

    def _decide(self, key: str, ambiguous: bool) -> str:
        if ambiguous:
            # ``<string>``, ``<stdin>`` and the like name code of any origin, also code that the SDK runs with ``exec``
            # at run time under a name that no ``mcp`` module has used yet: the file name never decides. The caller
            # decides with the frame of the code (the record factory), or hides the record (the filter, with no evidence).
            return _UNKNOWN
        me = threading.get_ident()
        if self._owner == me or not self._lock.acquire(blocking=False):
            return _UNKNOWN  # (A log record inside the collection, or another thread is collecting: do not wait.)
        self._owner = me
        try:
            if self._stale():
                try:
                    self._refresh()
                except Exception:  # noqa: BLE001 - a failed collection must not break logging, nor be trusted
                    return _UNKNOWN
            if key in self._filenames or key.startswith(self._prefixes):
                self._answers[key] = True
                return _SDK
            if not self._complete:
                return _UNKNOWN  # A module that is importing can define code in this file: not cached.
            self._answers[key] = False
            return _APP
        finally:
            self._owner = None
            self._lock.release()

    def _refresh(self) -> None:
        """Collect again when the snapshot is not the one of the last collection, and settle the imports that are over.

        Called with ``_lock`` held. The sequence number is odd for the whole call, so that a thread that has no lock
        (the cached answers of :meth:`classify`) cannot use a cache, a snapshot or a watcher that is half changed.
        """
        self._sequence += 1
        try:
            self._refresh_locked()
        except BaseException:
            # The collection did not end: no cached answer is trusted until another one does (``_stale`` is true while
            # ``_complete`` is false). The names of the imports are still in the watcher or in ``_unresolved``: they
            # are moved, not removed, before they are recorded.
            self._complete = False
            raise
        finally:
            self._sequence += 1

    def _refresh_locked(self) -> None:
        # Before the snapshot: a later import is in the next collection. The names are copied, recorded, and only then
        # forgotten by the watcher: if this is interrupted between two of these steps, a name is still in one place.
        noted = _import_watcher.pending()
        self._unresolved.update(noted)
        _import_watcher.commit(noted)
        importing = (
            _importing_mcp_names()
        )  # (Before the snapshot: a module that is not in it, and whose import is over, has failed.)
        length = len(
            sys.modules
        )  # (Before the snapshot: a module that comes in after it makes the length differ later.)
        snapshot = _mcp_snapshot()
        if self._last is None or not _same_snapshot(snapshot, self._last):
            filenames: set[str] = set()
            prefixes: set[str] = set()
            for module, _initializing in snapshot.values():
                try:
                    _collect_module(module, filenames, prefixes)
                except Exception:  # noqa: BLE001 - a module with odd attributes must not break logging
                    continue
            self._filenames, self._prefixes = frozenset(filenames), tuple(prefixes)
            self._answers = {}
            self._last = snapshot
        self._last_len = length
        self._complete = not any(initializing for _module, initializing in snapshot.values())
        for name in list(self._unresolved):
            if name in snapshot or importing is None or name not in importing:
                self._unresolved.discard(name)


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


def _hidden_message(level: int, filename: str, lineno: int, name: str = "") -> str:
    module = os.path.splitext(os.path.basename(filename))[0]
    sdk = "Claude Agent SDK" if _is_claude_name(name) else "MCP SDK"
    return f"{sdk} {logging.getLevelName(level)} at {module}:{lineno} (details hidden)"


# --------------------------------------------------------------------------- #
# The origin of a record that has an ambiguous file name: the frame that ``Logger.findCaller`` picks
# --------------------------------------------------------------------------- #
def _frame_state(frame: Any) -> str:
    return _SDK if _is_mcp_name(frame.f_globals.get("__name__")) else _APP


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
_evidence_purge_at = _EVIDENCE_PURGE_AT


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
    global _evidence_purge_at
    with _evidence_lock:
        _evidence[id(record)] = entry
        if len(_evidence) > _evidence_purge_at:
            for key in [key for key, item in list(_evidence.items()) if item.ref() is None]:
                _evidence.pop(key, None)
            # (Amortized: the next purge is when the store has doubled. Records that live on, for example in a test
            # handler, would else make each new record scan the whole store.)
            _evidence_purge_at = max(_EVIDENCE_PURGE_AT, 2 * len(_evidence))


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
        state = _SDK if _is_claude_name(name) else _origins.classify(name, fn)
        decided_by_frame = state == _UNKNOWN
        if decided_by_frame:  # An ambiguous file name, or an import is running: the frame of the caller tells.
            state = _frame_origin(fn, lno, func)
        if state != _APP:  # The SDK, or not sure: hide.
            msg = _hidden_message(level, fn, lno, name)
            args, exc_info, sinfo = (), None, None
        record = previous(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)
        if decided_by_frame and state == _APP:  # (The filter looks for this only when it cannot decide by the file.)
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
        state = _SDK if _is_claude_name(record.name) else _origins.classify(record.name, record.pathname)
        if state == _UNKNOWN and _has_app_evidence(record):
            state = _APP
        if state != _APP:
            record.msg = _hidden_message(record.levelno, record.pathname, record.lineno, record.name)
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
        _install_import_watcher()
        _import_client_side_sdk()
        current = logging.getLogRecordFactory()
        if not _is_guard(current):  # (By identity: an attribute can be copied, and a proxy can be equal to a guard.)
            logging.setLogRecordFactory(_guard(current))
        _install_filters()


#: The name that the first version of the guard had.
install_mcp_log_guard = ensure_mcp_log_guard
