"""Skills staging for provider agent SDKs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from dd_ai_devx_evals.types import slugify

logger = logging.getLogger(__name__)


def stage_skills_for_claude(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Claude Agent SDK discovery.

    Copies each skill directory to <cwd>/.claude/skills/<name> and returns
    the list of skill names. Idempotent - existing directories are replaced.

    Args:
        skill_dirs: List of paths to skill directories (each containing SKILL.md)
        cwd: Working directory where skills will be staged

    Returns:
        List of skill names (derived from directory basenames)

    Raises:
        ValueError: If a skill directory doesn't exist
    """
    if not skill_dirs:
        return []

    skill_names = []
    skills_base = Path(cwd) / ".claude" / "skills"
    skills_base.mkdir(parents=True, exist_ok=True)

    for skill_dir in skill_dirs:
        source_path = Path(skill_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Skill directory does not exist: {skill_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {skill_dir}")

        # Derive skill name from directory basename
        skill_name = _derive_skill_name(source_path.name)
        skill_names.append(skill_name)

        # Stage the skill
        target_path = skills_base / skill_name
        if target_path.exists():
            logger.debug("Replacing existing staged skill: %s", skill_name)
            shutil.rmtree(target_path)

        logger.debug("Staging skill %s from %s to %s", skill_name, source_path, target_path)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)

    logger.info("Staged %d skills for Claude: %s", len(skill_names), ", ".join(skill_names))
    return skill_names


def stage_skills_for_codex(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Codex discovery.

    Copies each skill directory to <cwd>/.codex/skills/<name> and returns
    the list of skill names. Idempotent - existing directories are replaced.

    Args:
        skill_dirs: List of paths to skill directories (each containing SKILL.md)
        cwd: Working directory where skills will be staged

    Returns:
        List of skill names (derived from directory basenames)

    Raises:
        ValueError: If a skill directory doesn't exist
    """
    if not skill_dirs:
        return []

    skill_names = []
    skills_base = Path(cwd) / ".codex" / "skills"
    skills_base.mkdir(parents=True, exist_ok=True)

    for skill_dir in skill_dirs:
        source_path = Path(skill_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Skill directory does not exist: {skill_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {skill_dir}")

        # Derive skill name from directory basename
        skill_name = _derive_skill_name(source_path.name)
        skill_names.append(skill_name)

        # Stage the skill
        target_path = skills_base / skill_name
        if target_path.exists():
            logger.debug("Replacing existing staged skill: %s", skill_name)
            shutil.rmtree(target_path)

        logger.debug("Staging skill %s from %s to %s", skill_name, source_path, target_path)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)

    logger.info("Staged %d skills for Codex: %s", len(skill_names), ", ".join(skill_names))
    return skill_names


def _derive_skill_name(directory_name: str) -> str:
    """Derive a skill name from a directory basename.

    Uses slugify to ensure the name is valid for both SDKs.
    """
    return slugify(directory_name)
