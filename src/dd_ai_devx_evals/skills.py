"""Skills staging for provider agent SDKs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from dd_ai_devx_evals.types import slugify

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
