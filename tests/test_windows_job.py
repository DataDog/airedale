# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""The Windows Job Object of a pi run. The Windows API is replaced by a fake: CI runs on Linux only."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import sys
from typing import Any

import pytest

import airedale.harness.pi as pi_module
from airedale.harness import windows_job
from airedale.harness.windows_job import (
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    PROCESS_SET_QUOTA,
    PROCESS_TERMINATE,
    WindowsJob,
)


class FakeKernel32:
    """Records the calls; ``fail`` names the call that returns failure."""

    def __init__(self, fail: str | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fail = fail
        self.flags: int | None = None
        self.size: int | None = None
        self.open_handles: set[str] = set()

    def CreateJobObjectW(self, attributes, name):  # noqa: N802
        self.calls.append(("CreateJobObjectW", attributes, name))
        if self.fail == "CreateJobObjectW":
            return None
        self.open_handles.add("job")
        return "job"

    def SetInformationJobObject(self, job, information_class, information, size):  # noqa: N802
        self.calls.append(("SetInformationJobObject", job, information_class))
        self.flags = information._obj.BasicLimitInformation.LimitFlags  # `information` is a ctypes.byref(...)
        self.size = size
        return 0 if self.fail == "SetInformationJobObject" else 1

    def OpenProcess(self, access, inherit, pid):  # noqa: N802
        self.calls.append(("OpenProcess", access, inherit, pid))
        if self.fail == "OpenProcess":
            return None
        self.open_handles.add("process")
        return "process"

    def AssignProcessToJobObject(self, job, process):  # noqa: N802
        self.calls.append(("AssignProcessToJobObject", job, process))
        return 0 if self.fail == "AssignProcessToJobObject" else 1

    def TerminateJobObject(self, job, exit_code):  # noqa: N802
        self.calls.append(("TerminateJobObject", job, exit_code))
        return 0 if self.fail == "TerminateJobObject" else 1

    def CloseHandle(self, handle):  # noqa: N802
        self.calls.append(("CloseHandle", handle))
        self.open_handles.discard(handle)
        return 1


def names(api: FakeKernel32) -> list[str]:
    return [call[0] for call in api.calls]


class TestAttach:
    def test_job_has_kill_on_close_and_pi_is_assigned(self):
        api = FakeKernel32()
        job = WindowsJob.attach(4242, api)
        assert job is not None
        assert names(api) == [
            "CreateJobObjectW",
            "SetInformationJobObject",
            "OpenProcess",
            "AssignProcessToJobObject",
            "CloseHandle",  # The process handle only: the job stays open.
        ]
        assert api.flags == JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE == 0x2000
        assert ("SetInformationJobObject", "job", JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS) in api.calls
        assert JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS == 9
        assert ("OpenProcess", PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, 4242) in api.calls
        assert ("AssignProcessToJobObject", "job", "process") in api.calls
        assert api.open_handles == {"job"}

    def test_the_structure_has_the_size_of_the_windows_one(self):
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION: 144 bytes on 64-bit Windows, 112 on 32-bit.
        expected = 144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112
        api = FakeKernel32()
        WindowsJob.attach(1, api)
        # (c_size_t is 8 bytes here on Linux 64-bit, as on Windows 64-bit; LONG is 32-bit on Windows, c_uint32 here.)
        assert api.size == expected

    @pytest.mark.parametrize(
        ("failing", "leaves_open"),
        [
            ("CreateJobObjectW", set()),
            ("SetInformationJobObject", set()),
            ("OpenProcess", set()),
            ("AssignProcessToJobObject", set()),
        ],
    )
    def test_a_failing_step_returns_none_and_leaves_nothing_open(self, failing, leaves_open, caplog):
        api = FakeKernel32(fail=failing)
        with caplog.at_level(logging.WARNING):
            assert WindowsJob.attach(4242, api) is None
        assert api.open_handles == leaves_open
        assert "taskkill" in caplog.text

    def test_nothing_on_a_system_that_is_not_windows(self):
        assert windows_job.load_kernel32() is None or sys.platform == "win32"
        assert WindowsJob.attach(1) is None or sys.platform == "win32"


class TestJob:
    def test_terminate_calls_terminatejobobject(self):
        api = FakeKernel32()
        job = WindowsJob.attach(1, api)
        assert job is not None
        assert job.terminate() is True
        assert ("TerminateJobObject", "job", 1) in api.calls

    def test_terminate_failure_is_reported(self):
        api = FakeKernel32()
        job = WindowsJob.attach(1, api)
        assert job is not None
        api.fail = "TerminateJobObject"
        assert job.terminate() is False

    def test_close_is_idempotent_and_closes_the_job(self):
        api = FakeKernel32()
        job = WindowsJob.attach(1, api)
        assert job is not None
        job.close()
        job.close()
        assert names(api).count("CloseHandle") == 2  # The process handle, then the job once.
        assert api.open_handles == set()
        assert job.terminate() is False


class _Process:
    def __init__(self, returncode=None):
        self.pid = 4242
        self.returncode = returncode
        self.killed = False

    def kill(self):
        self.killed = True
        self.returncode = 1


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(pi_module, "_is_windows", lambda: True)
    monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(
        pi_module.os, "killpg", lambda *a: (_ for _ in ()).throw(AssertionError("killpg")), raising=False
    )


def _stop(process, job):
    asyncio.run(asyncio.wait_for(pi_module._stop_process_group(process, job), timeout=10))


def _no_taskkill(monkeypatch):
    async def exec_(*argv, **kwargs):
        raise AssertionError(f"taskkill must not run with a job: {argv}")

    monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", exec_)


class TestStop:
    def test_job_stops_the_tree_even_when_pi_already_exited(self, windows, monkeypatch):
        _no_taskkill(monkeypatch)
        api = FakeKernel32()
        job = WindowsJob.attach(4242, api)
        _stop(_Process(returncode=0), job)  # Pi is gone, a child may still run.
        assert ("TerminateJobObject", "job", 1) in api.calls

    def test_job_is_used_and_pi_is_waited_for(self, windows, monkeypatch):
        _no_taskkill(monkeypatch)
        api = FakeKernel32()
        job = WindowsJob.attach(4242, api)
        process = _Process()
        _stop(process, job)
        assert ("TerminateJobObject", "job", 1) in api.calls
        assert process.killed  # The fake does not exit by itself: pi is killed as the last resort.

    def test_failed_terminate_falls_back_to_taskkill(self, windows, monkeypatch):
        api = FakeKernel32()
        job = WindowsJob.attach(4242, api)
        api.fail = "TerminateJobObject"
        seen: list[tuple[str, ...]] = []

        class _Helper:
            returncode = 0

            async def wait(self):
                return 0

        async def exec_(*argv, **kwargs):
            seen.append(argv)
            return _Helper()

        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", exec_)
        process = _Process()
        _stop(process, job)
        assert seen == [("taskkill", "/PID", "4242", "/T", "/F")]
        assert process.killed

    def test_without_a_job_taskkill_is_used_as_before(self, windows, monkeypatch):
        seen: list[tuple[str, ...]] = []

        class _Helper:
            returncode = 0

            async def wait(self):
                return 0

        async def exec_(*argv, **kwargs):
            seen.append(argv)
            return _Helper()

        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", exec_)
        _stop(_Process(), None)
        assert seen == [("taskkill", "/PID", "4242", "/T", "/F")]

    def test_cleanup_closes_the_job_also_when_the_stop_fails(self, windows, monkeypatch):
        api = FakeKernel32()
        job = WindowsJob.attach(4242, api)
        _no_taskkill(monkeypatch)

        async def broken_stop(process, job=None):
            raise RuntimeError("stop failed")

        monkeypatch.setattr(pi_module, "_stop_process_group", broken_stop)
        with pytest.raises(RuntimeError, match="stop failed"):
            asyncio.run(pi_module._cleanup_process(_Process(), [], job))  # type: ignore[arg-type]
        assert api.open_handles == set()


class TestSpawn:
    def test_pi_is_assigned_to_a_job_right_after_the_spawn_and_the_job_is_closed_after_the_run(
        self, windows, monkeypatch, tmp_path
    ):
        api = FakeKernel32()
        order: list[str] = []
        real_attach = WindowsJob.attach

        def attach(pid, api_=None):
            order.append(f"attach {pid}")
            return real_attach(pid, api)

        monkeypatch.setattr(pi_module.WindowsJob, "attach", staticmethod(attach))

        class _Stream:
            async def read(self, n=-1):
                return b""

            async def readline(self):
                return b""

            def write(self, data):
                pass

            async def drain(self):
                pass

            def close(self):
                pass

            def is_closing(self):
                return True

            async def wait_closed(self):
                pass

        class _Proc(_Process):
            def __init__(self):
                super().__init__(returncode=0)
                self.stdin, self.stdout, self.stderr = _Stream(), _Stream(), _Stream()

            async def wait(self):
                return 0

        async def fake_exec(*argv, **kwargs):
            order.append("spawn")
            return _Proc()

        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", fake_exec)
        (tmp_path / "pi.cmd").write_text("x")  # The Windows lookup needs a program to find.
        runner = pi_module.PiRunner(cwd=tmp_path, pi_command=["pi"])
        with contextlib.suppress(Exception):  # The fake process is minimal: only the order of the calls matters.
            asyncio.run(
                asyncio.wait_for(
                    runner._run_process(
                        ["pi"],
                        user_prompt="Q",
                        env={"PATH": str(tmp_path), "PATHEXT": ".cmd"},
                        collector=pi_module._PiEventCollector(),
                        model=pi_module.ModelSpec.parse("anthropic/claude-sonnet-4-6"),
                        progress=None,
                    ),
                    timeout=10,
                )
            )
        assert order[:2] == ["spawn", "attach 4242"]  # Nothing in between.
        assert ("TerminateJobObject", "job", 1) in api.calls
        assert api.open_handles == set()
