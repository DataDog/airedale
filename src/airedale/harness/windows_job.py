# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""A Windows Job Object for the process tree of one pi run.

``taskkill /T`` finds the children of a process through the process itself: when pi has already exited (a child
keeps its stdout open, for example) the children are lost and keep running. A Job Object holds the whole tree, whatever
the parent does, and ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` kills it when the last handle of the job closes, also when
Airedale itself dies.

The job is created and pi is assigned to it right after the spawn. Children that pi creates before the assignment are
not in the job. That window is the time that Python needs to go from the return of ``CreateProcess`` to
``AssignProcessToJobObject`` (a few calls, no ``await`` in between except the one of the spawn itself); pi needs far
longer than that (it is a Node program that loads its extensions) before it starts its first child. The process is
not created suspended: asyncio gives no handle on the main thread to resume it. Assigning a process that is already
in a job works on Windows 8 and later (nested jobs); where it fails, the caller falls back to ``taskkill``.

Only ``ctypes`` and the Windows API are used. On another system, :func:`attach` returns ``None``.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9  # JobObjectExtendedLimitInformation
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100
#: Exit code of the processes that ``TerminateJobObject`` stops.
TERMINATED_EXIT_CODE = 1


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class Kernel32(Protocol):
    """The calls of the Windows API that this module makes (a fake implements them in the tests)."""

    def CreateJobObjectW(self, attributes: Any, name: Any) -> Any: ...  # noqa: N802
    def SetInformationJobObject(self, job: Any, information_class: int, information: Any, size: int) -> int: ...  # noqa: N802
    def OpenProcess(self, access: int, inherit: bool, pid: int) -> Any: ...  # noqa: N802
    def AssignProcessToJobObject(self, job: Any, process: Any) -> int: ...  # noqa: N802
    def TerminateJobObject(self, job: Any, exit_code: int) -> int: ...  # noqa: N802
    def CloseHandle(self, handle: Any) -> int: ...  # noqa: N802


def load_kernel32() -> Kernel32 | None:
    """Return the Windows API with the right argument types, or ``None`` when this is not Windows."""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    kernel32 = win_dll("kernel32", use_last_error=True)
    handle = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = handle
    kernel32.SetInformationJobObject.argtypes = [handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = handle
    kernel32.AssignProcessToJobObject.argtypes = [handle, handle]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = [handle, ctypes.c_uint32]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [handle]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32  # type: ignore[no-any-return]


class WindowsJob:
    """A Job Object that kills every process in it when it is terminated or when its last handle closes."""

    def __init__(self, api: Kernel32, handle: Any) -> None:
        self._api = api
        self._handle: Any = handle

    @classmethod
    def attach(cls, pid: int, api: Kernel32 | None = None) -> WindowsJob | None:
        """Create a job with ``KILL_ON_JOB_CLOSE`` and assign the process ``pid`` to it.

        Return ``None`` (after a warning) when any step fails: the caller then stops the run with ``taskkill`` as
        before. Nothing is left open on failure.
        """
        api = api or load_kernel32()
        if api is None:
            return None
        job = api.CreateJobObjectW(None, None)
        if not job:
            logger.warning("Unable to create a Windows Job Object for pi; its children are stopped with taskkill")
            return None
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(
            job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(information), ctypes.sizeof(information)
        ):
            api.CloseHandle(job)
            logger.warning("Unable to set the limits of the Windows Job Object; pi is stopped with taskkill")
            return None
        process = api.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not process:
            api.CloseHandle(job)
            logger.warning("Unable to open pi to assign it to a Job Object; its children are stopped with taskkill")
            return None
        try:
            assigned = api.AssignProcessToJobObject(job, process)
        finally:
            api.CloseHandle(process)
        if not assigned:
            api.CloseHandle(job)
            logger.warning("Unable to assign pi to a Job Object; its children are stopped with taskkill")
            return None
        return cls(api, job)

    def terminate(self) -> bool:
        """Stop every process of the job now (pi already gone or not). Return False when the call failed."""
        if self._handle is None:
            return False
        return bool(self._api.TerminateJobObject(self._handle, TERMINATED_EXIT_CODE))

    def close(self) -> None:
        """Close the handle (idempotent). With ``KILL_ON_JOB_CLOSE``, a process still in the job is killed."""
        handle, self._handle = self._handle, None
        if handle is not None:
            self._api.CloseHandle(handle)
