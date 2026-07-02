# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Tests for airedale.skills — stage_skills_for_claude/codex."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from airedale.skills import (
    exclude_staged_skills_from_git,
    stage_skills_for_claude,
    stage_skills_for_codex,
)


def _make_skill_dir(tmp_path: Path, name: str) -> Path:
    """Create a minimal skill directory with a SKILL.md."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\n\nA test skill.\n")
    return skill_dir


class TestStageSkillsForClaude:
    def test_stages_single_skill(self, tmp_path):
        skill = _make_skill_dir(tmp_path / "sources", "my-skill")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names = stage_skills_for_claude([str(skill)], str(cwd))
        assert len(names) == 1
        assert names[0] == "my-skill"
        staged = cwd / ".claude" / "skills" / "my-skill" / "SKILL.md"
        assert staged.exists()

    def test_stages_multiple_skills(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        skills = [_make_skill_dir(sources, f"skill-{i}") for i in range(3)]
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names = stage_skills_for_claude([str(s) for s in skills], str(cwd))
        assert len(names) == 3
        for i in range(3):
            assert (cwd / ".claude" / "skills" / f"skill-{i}" / "SKILL.md").exists()

    def test_idempotent_second_call(self, tmp_path):
        skill = _make_skill_dir(tmp_path / "sources", "my-skill")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names1 = stage_skills_for_claude([str(skill)], str(cwd))
        names2 = stage_skills_for_claude([str(skill)], str(cwd))
        assert names1 == names2
        staged = cwd / ".claude" / "skills" / "my-skill" / "SKILL.md"
        assert staged.exists()

    def test_empty_list_returns_empty(self, tmp_path):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        assert stage_skills_for_claude([], str(cwd)) == []

    def test_missing_source_raises(self, tmp_path):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        with pytest.raises(ValueError, match="does not exist"):
            stage_skills_for_claude([str(tmp_path / "nonexistent")], str(cwd))

    def test_collision_with_repo_skill_raises(self, tmp_path):
        # A repo ships .claude/skills/my-skill (no staged marker); a scenario
        # skill of the same name must not silently clobber it.
        skill = _make_skill_dir(tmp_path / "sources", "my-skill")
        cwd = tmp_path / "cwd"
        repo_skill = cwd / ".claude" / "skills" / "my-skill"
        repo_skill.mkdir(parents=True)
        (repo_skill / "SKILL.md").write_text("# repo skill\n")
        with pytest.raises(ValueError, match="collides with an existing project skill"):
            stage_skills_for_claude([str(skill)], str(cwd))

    def test_skill_name_slugified(self, tmp_path):
        skill = _make_skill_dir(tmp_path / "sources", "My Skill With Spaces")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names = stage_skills_for_claude([str(skill)], str(cwd))
        # slugify turns spaces into dashes
        assert names[0] == "my-skill-with-spaces"
        assert (cwd / ".claude" / "skills" / "my-skill-with-spaces").is_dir()


class TestStageSkillsForCodex:
    def test_stages_to_codex_path(self, tmp_path):
        skill = _make_skill_dir(tmp_path / "sources", "apm-skill")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names = stage_skills_for_codex([str(skill)], str(cwd))
        assert len(names) == 1
        assert names[0] == "apm-skill"
        staged = cwd / ".codex" / "skills" / "apm-skill" / "SKILL.md"
        assert staged.exists()

    def test_idempotent_second_call(self, tmp_path):
        skill = _make_skill_dir(tmp_path / "sources", "apm-skill")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        names1 = stage_skills_for_codex([str(skill)], str(cwd))
        names2 = stage_skills_for_codex([str(skill)], str(cwd))
        assert names1 == names2

    def test_empty_list_returns_empty(self, tmp_path):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        assert stage_skills_for_codex([], str(cwd)) == []

    def test_missing_source_raises(self, tmp_path):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        with pytest.raises(ValueError, match="does not exist"):
            stage_skills_for_codex([str(tmp_path / "no-such-dir")], str(cwd))


class TestExcludeStagedSkillsFromGit:
    def _init_repo(self, path):
        import os
        import subprocess

        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
        subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)

    def test_appends_anchored_patterns_in_git_repo(self, tmp_path):
        self._init_repo(tmp_path)
        exclude_staged_skills_from_git(str(tmp_path), ["apm", "k8s"], subdir=".claude/skills")
        content = (tmp_path / ".git" / "info" / "exclude").read_text()
        assert "/.claude/skills/apm/" in content.splitlines()
        assert "/.claude/skills/k8s/" in content.splitlines()

    def test_idempotent(self, tmp_path):
        self._init_repo(tmp_path)
        exclude_staged_skills_from_git(str(tmp_path), ["apm"], subdir=".claude/skills")
        exclude_staged_skills_from_git(str(tmp_path), ["apm"], subdir=".claude/skills")
        content = (tmp_path / ".git" / "info" / "exclude").read_text()
        assert content.count("/.claude/skills/apm/") == 1

    def test_noop_outside_git_repo(self, tmp_path):
        # Must not raise and must not create a .git directory.
        exclude_staged_skills_from_git(str(tmp_path), ["apm"], subdir=".claude/skills")
        assert not (tmp_path / ".git").exists()

    def test_empty_names_noop(self, tmp_path):
        self._init_repo(tmp_path)
        exclude_staged_skills_from_git(str(tmp_path), [], subdir=".claude/skills")
        # No exclude content written for an empty name list.
        exclude_file = tmp_path / ".git" / "info" / "exclude"
        if exclude_file.exists():
            assert "/.claude/skills/" not in exclude_file.read_text()
