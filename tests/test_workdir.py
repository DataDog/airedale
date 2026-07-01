"""Tests for airedale.workdir — WorkspaceManager.

These run against a small local throwaway git repo created per test. Plain git
is used here because the workspace manager *is* the harness's own git-based
mechanism; everything is fully offline.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from airedale.config.experiment import RemoveStep, RestoreStep, WorkdirConfig, WriteStep
from airedale.workdir import WorkspaceManager

if TYPE_CHECKING:
    from pathlib import Path

GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={
            **GIT_ENV,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@e",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@e",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _make_repo(path: Path) -> Path:
    """Create a repo with two commits and tags v1 / v2."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    (path / "README.md").write_text("v1 readme\n")
    (path / "keep.txt").write_text("keep\n")
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", "first", cwd=path)
    _git("tag", "v1", cwd=path)
    (path / "README.md").write_text("v2 readme\n")
    (path / "new.txt").write_text("new in v2\n")
    _git("add", "-A", cwd=path)
    _git("commit", "-q", "-m", "second", cwd=path)
    _git("tag", "v2", cwd=path)
    return path


class TestWorkspaceManager:
    def test_no_repo_yields_empty_dir_with_steps(self, tmp_path):
        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                workdir = WorkdirConfig(
                    repo=None,
                    steps=(WriteStep(path="main.py", content="print('hi')\n"),),
                )
                async with mgr.workspace(workdir) as cwd:
                    assert (cwd / "main.py").read_text() == "print('hi')\n"
                    # No git repo in an empty workspace.
                    assert not (cwd / ".git").exists()
                    return cwd

        cwd = asyncio.run(run())
        # Cleaned up on exit.
        assert not cwd.exists()

    def test_worktree_at_ref(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                workdir = WorkdirConfig(repo=str(repo), ref="v1")
                async with mgr.workspace(workdir) as cwd:
                    assert (cwd / "README.md").read_text() == "v1 readme\n"
                    assert not (cwd / "new.txt").exists()

        asyncio.run(run())

    def test_default_ref_is_head(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            async with (
                WorkspaceManager(config_dir=tmp_path) as mgr,
                mgr.workspace(WorkdirConfig(repo=str(repo))) as cwd,
            ):
                assert (cwd / "README.md").read_text() == "v2 readme\n"
                assert (cwd / "new.txt").exists()

        asyncio.run(run())

    def test_clone_once_caching(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                wd = WorkdirConfig(repo=str(repo))
                async with mgr.workspace(wd):
                    pass
                async with mgr.workspace(wd):
                    pass
                # One clone entry, reused for both worktrees.
                assert len(mgr._clones) == 1
                entry = next(iter(mgr._clones.values()))
                assert entry.path is not None

        asyncio.run(run())

    def test_restore_remove_write_steps(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                workdir = WorkdirConfig(
                    repo=str(repo),
                    ref="v2",
                    steps=(
                        RestoreStep(from_ref="v1", paths=("README.md",)),
                        RemoveStep(paths=("new.txt",)),
                        WriteStep(path="NOTES.md", content="evaluate\n"),
                    ),
                )
                async with mgr.workspace(workdir) as cwd:
                    # restored from v1
                    assert (cwd / "README.md").read_text() == "v1 readme\n"
                    # removed
                    assert not (cwd / "new.txt").exists()
                    # written
                    assert (cwd / "NOTES.md").read_text() == "evaluate\n"
                    # untouched
                    assert (cwd / "keep.txt").read_text() == "keep\n"

        asyncio.run(run())

    def test_write_from_source_file(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")
        source = tmp_path / "input.json"
        source.write_text('{"k": 1}\n')

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                workdir = WorkdirConfig(
                    repo=str(repo),
                    steps=(WriteStep(path="fixtures/input.json", source_path=str(source)),),
                )
                async with mgr.workspace(workdir) as cwd:
                    assert (cwd / "fixtures" / "input.json").read_text() == '{"k": 1}\n'

        asyncio.run(run())

    def test_aclose_removes_worktrees_and_cache(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            mgr = WorkspaceManager(config_dir=tmp_path)
            await mgr.__aenter__()
            cache_root = mgr._cache_root
            async with mgr.workspace(WorkdirConfig(repo=str(repo))):
                pass
            assert cache_root.exists()
            await mgr.aclose()
            return cache_root

        cache_root = asyncio.run(run())
        assert not cache_root.exists()

    def test_self_clone_leaves_source_intact(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")
        readme_before = (repo / "README.md").read_text()

        async def run():
            # local source -> --no-hardlinks clone under the cache root.
            async with (
                WorkspaceManager(config_dir=tmp_path) as mgr,
                mgr.workspace(WorkdirConfig(repo=str(repo))) as cwd,
            ):
                # Mutating/deleting the clone must not affect the source.
                (cwd / "README.md").write_text("mutated\n")

        asyncio.run(run())
        assert repo.exists()
        assert (repo / ".git").exists()
        assert (repo / "README.md").read_text() == readme_before

    def test_refuses_to_delete_outside_cache(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                with pytest.raises(RuntimeError, match="outside cache root"):
                    mgr._remove_path(outside)

        asyncio.run(run())

    def test_concurrent_worktrees_serialized(self, tmp_path):
        repo = _make_repo(tmp_path / "repo")

        async def run():
            async with WorkspaceManager(config_dir=tmp_path) as mgr:
                wd = WorkdirConfig(repo=str(repo))

                async def one():
                    async with mgr.workspace(wd) as cwd:
                        await asyncio.sleep(0.01)
                        assert (cwd / "README.md").exists()

                await asyncio.gather(*(one() for _ in range(4)))
                assert len(mgr._clones) == 1

        asyncio.run(run())
