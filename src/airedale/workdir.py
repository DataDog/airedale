# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Per-run working-directory provisioning for evaluation cells.

A :class:`WorkspaceManager` owns a clone cache and per-repo locks for an entire
matrix run. Each distinct repo source is cloned **once, lazily** into a cache
directory; every workspace handed to a run is a fresh ``git worktree`` of that
clone with the scenario's setup steps applied. A workdir with no ``repo`` yields
a fresh empty directory (today's sandbox behaviour) with any remove/write steps
applied.

Safety
------
The manager only ever deletes paths under its own cache root. ``repo="self"`` is
materialised as an independent ``git clone --no-hardlinks`` under the cache root,
so tearing the cache down can never touch the user's real checkout — the source
repo is only ever read from.

This module uses plain ``git`` purely as an internal mechanism for creating
isolated eval workspaces; it never mutates the user's own checkout. The project's
"use jj, never git" rule governs how *we* version this repo, not the harness's
runtime behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from airedale.config.experiment import (
    RemoveStep,
    RestoreStep,
    WriteStep,
    _git_toplevel,
    _looks_like_git_url,
)
from airedale.types import slugify

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from airedale.config.experiment import WorkdirConfig, WorkdirStep

logger = logging.getLogger(__name__)


@dataclass
class _ClonedRepo:
    """A single cached clone plus the worktrees created from it."""

    source: str
    is_local: bool
    lock: asyncio.Lock
    path: Path | None = None
    worktrees: set[Path] = field(default_factory=set)


class WorkspaceManager:
    """Own the clone cache + per-repo locks for a whole matrix run.

    Use it as an async context manager wrapping the matrix; ``workspace()``
    yields a fresh per-run working directory.
    """

    def __init__(self, *, config_dir: Path) -> None:
        self._config_dir = Path(config_dir)
        self._cache_root: Path | None = None
        self._clones: dict[str, _ClonedRepo] = {}
        self._registry_lock = asyncio.Lock()

    async def __aenter__(self) -> WorkspaceManager:
        self._cache_root = Path(tempfile.mkdtemp(prefix="dd-ai-devx-clones-"))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Remove every worktree we created, then delete the entire cache root.

        Idempotent. Only ever deletes paths under the cache root.
        """
        if self._cache_root is None:
            return
        for entry in self._clones.values():
            if entry.path is None:
                continue
            for worktree in list(entry.worktrees):
                with contextlib.suppress(Exception):
                    await self._run_git("worktree", "remove", "--force", str(worktree), cwd=entry.path)
        self._assert_within_cache(self._cache_root)
        shutil.rmtree(self._cache_root, ignore_errors=True)
        self._clones.clear()
        self._cache_root = None

    @contextlib.asynccontextmanager
    async def workspace(self, workdir: WorkdirConfig | None) -> AsyncIterator[Path]:
        """Yield a fresh per-run working directory for one run/repetition."""
        if self._cache_root is None:
            raise RuntimeError("WorkspaceManager must be entered before creating workspaces")

        if workdir is None or workdir.repo is None:
            path = self._fresh_dir("empty")
            try:
                self._apply_steps(path, workdir.steps if workdir is not None else ())
                yield path
            finally:
                self._remove_path(path)
            return

        source_key, source, is_local = self._resolve_source(workdir)
        entry = await self._get_clone_entry(source_key, source, is_local)
        ref = workdir.ref or "HEAD"

        async with entry.lock:
            if entry.path is None:
                entry.path = await self._clone(entry)
            worktree = await self._add_worktree(entry, ref)
        try:
            self._apply_steps(worktree, workdir.steps)
            yield worktree
        finally:
            async with entry.lock:
                with contextlib.suppress(Exception):
                    if entry.path is not None:
                        await self._run_git("worktree", "remove", "--force", str(worktree), cwd=entry.path)
                entry.worktrees.discard(worktree)
            self._remove_path(worktree)

    # ----------------------------------------------------------------- #
    # Source resolution + cloning
    # ----------------------------------------------------------------- #
    def _resolve_source(self, workdir: WorkdirConfig) -> tuple[str, str, bool]:
        """Return (cache key, source, is_local) for a workdir's repo."""
        repo = workdir.repo
        assert repo is not None
        if repo == "self":
            toplevel = _git_toplevel(self._config_dir)
            if toplevel is None:
                raise RuntimeError('repo="self" but the config directory is not inside a git repository')
            resolved = str(Path(toplevel).resolve())
            return resolved, resolved, True
        if _looks_like_git_url(repo):
            return repo, repo, False
        resolved = str(Path(repo).resolve())
        return resolved, resolved, True

    async def _get_clone_entry(self, source_key: str, source: str, is_local: bool) -> _ClonedRepo:
        async with self._registry_lock:
            entry = self._clones.get(source_key)
            if entry is None:
                entry = _ClonedRepo(source=source, is_local=is_local, lock=asyncio.Lock())
                self._clones[source_key] = entry
            return entry

    async def _clone(self, entry: _ClonedRepo) -> Path:
        assert self._cache_root is not None
        clone_path = self._cache_root / f"clone-{slugify(Path(entry.source).name) or 'repo'}-{uuid.uuid4().hex[:8]}"
        args = ["clone", "--no-checkout"]
        if entry.is_local:
            # An independent copy of the objects so deleting the clone never
            # touches the source repo's object store.
            args.append("--no-hardlinks")
        args.extend([entry.source, str(clone_path)])
        await self._run_git(*args)
        logger.debug("Cloned %s into %s", entry.source, clone_path)
        return clone_path

    async def _add_worktree(self, entry: _ClonedRepo, ref: str) -> Path:
        assert self._cache_root is not None
        assert entry.path is not None
        worktree = self._cache_root / f"wt-{uuid.uuid4().hex}"
        await self._run_git("worktree", "add", "--detach", str(worktree), ref, cwd=entry.path)
        entry.worktrees.add(worktree)
        return worktree

    # ----------------------------------------------------------------- #
    # Step application
    # ----------------------------------------------------------------- #
    def _apply_steps(self, workspace: Path, steps: tuple[WorkdirStep, ...]) -> None:
        for step in steps:
            if isinstance(step, RestoreStep):
                self._apply_restore(workspace, step)
            elif isinstance(step, RemoveStep):
                self._apply_remove(workspace, step)
            elif isinstance(step, WriteStep):
                self._apply_write(workspace, step)
            else:  # pragma: no cover - exhaustive by construction
                raise RuntimeError(f"unknown workdir step: {step!r}")

    def _apply_restore(self, workspace: Path, step: RestoreStep) -> None:
        self._run_git_sync("restore", f"--source={step.from_ref}", "--", *step.paths, cwd=workspace)

    def _apply_remove(self, workspace: Path, step: RemoveStep) -> None:
        root = workspace.resolve()
        for pattern in step.paths:
            matches = glob.glob(str(workspace / pattern), recursive=True)
            if not matches:
                logger.debug("remove pattern matched nothing: %s", pattern)
                continue
            for match in matches:
                target = Path(match)
                self._assert_within(target, root)
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    with contextlib.suppress(FileNotFoundError):
                        target.unlink()

    def _apply_write(self, workspace: Path, step: WriteStep) -> None:
        root = workspace.resolve()
        target = (workspace / step.path).resolve()
        self._assert_within(target, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if step.source_path is not None:
            shutil.copyfile(step.source_path, target)
        else:
            target.write_text(step.content or "")

    # ----------------------------------------------------------------- #
    # Git + filesystem helpers
    # ----------------------------------------------------------------- #
    async def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd) if cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (exit {proc.returncode}): {stderr.decode(errors='replace').strip()}"
            )
        return stdout.decode(errors="replace")

    def _run_git_sync(self, *args: str, cwd: Path | None = None) -> str:
        import subprocess

        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}")
        return result.stdout

    def _fresh_dir(self, prefix: str) -> Path:
        assert self._cache_root is not None
        path = self._cache_root / f"{prefix}-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _remove_path(self, path: Path) -> None:
        if not path.exists():
            return
        self._assert_within_cache(path)
        shutil.rmtree(path, ignore_errors=True)

    def _assert_within_cache(self, path: Path) -> None:
        assert self._cache_root is not None
        root = self._cache_root.resolve()
        resolved = path.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise RuntimeError(f"refusing to delete {resolved} outside cache root {root}")

    @staticmethod
    def _assert_within(path: Path, root: Path) -> None:
        resolved = path.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise RuntimeError(f"path {resolved} escapes workspace {root}")
