"""Tests for dd_ai_devx_evals.skills — stage_skills_for_claude/codex."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dd_ai_devx_evals.skills import stage_skills_for_claude, stage_skills_for_codex


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
