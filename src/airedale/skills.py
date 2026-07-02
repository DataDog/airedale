# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Skills staging for provider agent SDKs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from airedale.types import slugify

logger = logging.getLogger(__name__)

# Marker dropped inside each skill we stage so re-staging into the same cwd is
# idempotent while a *repo*-provided skill of the same name (no marker) is
# treated as a collision rather than silently clobbered.
STAGED_MARKER = ".dd-ai-devx-staged"


def stage_skills_for_claude(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Claude Agent SDK discovery.

    Copies each skill directory to ``<cwd>/.claude/skills/<name>`` and returns
    the staged skill names. Re-staging our own skills is idempotent; a name
    collision with a repo-provided skill raises.

    Args:
        skill_dirs: Paths to skill directories (each containing SKILL.md).
        cwd: Working directory where skills will be staged.

    Returns:
        List of skill names (derived from directory basenames).

    Raises:
        ValueError: If a skill directory doesn't exist or collides with a repo skill.
    """
    names = _stage_skill_dirs(skill_dirs, Path(cwd) / ".claude" / "skills")
    if names:
        logger.info("Staged %d skills for Claude: %s", len(names), ", ".join(names))
    return names


def stage_skills_for_codex(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Codex discovery.

    Copies each skill directory to ``<cwd>/.codex/skills/<name>`` and returns
    the staged skill names. Re-staging our own skills is idempotent; a name
    collision with a repo-provided skill raises.

    Args:
        skill_dirs: Paths to skill directories (each containing SKILL.md).
        cwd: Working directory where skills will be staged.

    Returns:
        List of skill names (derived from directory basenames).

    Raises:
        ValueError: If a skill directory doesn't exist or collides with a repo skill.
    """
    names = _stage_skill_dirs(skill_dirs, Path(cwd) / ".codex" / "skills")
    if names:
        logger.info("Staged %d skills for Codex: %s", len(names), ", ".join(names))
    return names


def exclude_staged_skills_from_git(cwd: str, skill_names: list[str], *, subdir: str) -> None:
    """Append staged skill paths to the worktree's git exclude file.

    When the run ``cwd`` is a git worktree (e.g. a cloned repo workspace), the
    scenario skills we stage under ``<cwd>/<subdir>/<name>`` would otherwise show
    up as untracked changes to the agent (``git status``). See
    :func:`exclude_paths_from_git`.
    """
    exclude_paths_from_git(cwd, [f"{subdir}/{name}" for name in skill_names])


def exclude_paths_from_git(cwd: str, relative_paths: list[str]) -> None:
    """Add ``relative_paths`` (workspace-relative dirs) to the local git excludes.

    Adds anchored patterns to the repo's local exclude file
    (``$GIT_DIR/info/exclude``, resolved via ``git rev-parse --git-path`` so it
    works for both plain repos and worktrees), hiding harness-created artifacts
    from the agent's untracked-changes view without touching the tracked
    ``.gitignore``. No-op when ``cwd`` is not a git repository, git is
    unavailable, or ``relative_paths`` is empty.
    """
    if not relative_paths:
        return
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--git-path", "info/exclude"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if result.returncode != 0:
        return
    raw = result.stdout.strip()
    if not raw:
        return

    exclude_path = Path(raw)
    if not exclude_path.is_absolute():
        exclude_path = Path(cwd) / exclude_path

    # Anchored directory patterns relative to the worktree root.
    patterns = [f"/{path.strip('/')}/" for path in relative_paths]
    try:
        existing = exclude_path.read_text().splitlines() if exclude_path.exists() else []
        new = [pattern for pattern in patterns if pattern not in existing]
        if not new:
            return
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "" if not existing or existing[-1] == "" else "\n"
        with exclude_path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + "\n".join(new) + "\n")
    except OSError:
        logger.debug("Unable to update git exclude file at %s", exclude_path, exc_info=True)


def discover_claude_skill_names(cwd: str) -> list[str]:
    """Return all skill names discoverable under ``<cwd>/.claude/skills``.

    This is the scenario ∪ repo skill set (staged scenario skills plus any the
    cloned repo ships), which forms the Claude ``skills`` allow-list.
    """
    return _discover_skill_names(Path(cwd) / ".claude" / "skills")


def _discover_skill_names(skills_base: Path) -> list[str]:
    if not skills_base.is_dir():
        return []
    return sorted(d.name for d in skills_base.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())


def _stage_skill_dirs(skill_dirs: list[str], skills_base: Path) -> list[str]:
    if not skill_dirs:
        return []

    skills_base.mkdir(parents=True, exist_ok=True)
    skill_names: list[str] = []
    for skill_dir in skill_dirs:
        source_path = Path(skill_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Skill directory does not exist: {skill_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {skill_dir}")

        skill_name = _derive_skill_name(source_path.name)
        skill_names.append(skill_name)

        target_path = skills_base / skill_name
        if target_path.exists():
            if (target_path / STAGED_MARKER).exists():
                logger.debug("Replacing previously staged skill: %s", skill_name)
                shutil.rmtree(target_path)
            else:
                raise ValueError(
                    f"Skill '{skill_name}' (from {source_path}) collides with an existing project "
                    f"skill at {target_path}; scenario and repo skill names must be unique"
                )

        logger.debug("Staging skill %s from %s to %s", skill_name, source_path, target_path)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        (target_path / STAGED_MARKER).write_text("")

    return skill_names


def _derive_skill_name(directory_name: str) -> str:
    """Derive a skill name from a directory basename.

    Uses slugify to ensure the name is valid for both SDKs.
    """
    return slugify(directory_name)
