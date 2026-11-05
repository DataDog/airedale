# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the skill frontmatter parser: it must read what pi reads."""

from __future__ import annotations

import pytest

from airedale.skills import _frontmatter, _pi_skill_name, pi_skill_dirs

#: (label, file content) of a project skill ``project/SKILL.md``. Pi's name for each is the reference.
FILES = [
    ("plain", "---\nname: alpha\ndescription: d\n---\n"),
    ("flow mapping", "---\n{name: alpha, description: d}\n---\n"),
    ("indented mapping", "---\n  name: alpha\n  description: d\n---\n"),
    ("double quoted key", '---\n"name": alpha\ndescription: d\n---\n'),
    ("single quoted key", "---\n'name': alpha\n'description': d\n---\n"),
    ("key with escape", '---\n"na\\x6de": alpha\ndescription: d\n---\n'),
    ("key with unicode escape", '---\n"na\\u006de": alpha\ndescription: d\n---\n'),
    ("key with spaces", '---\n" name ": alpha\ndescription: d\n---\n'),
    ("explicit key", "---\n? name\n: alpha\ndescription: d\n---\n"),
    ("quoted value", "---\nname: 'alpha'\ndescription: d\n---\n"),
    ("tagged string", "---\nname: !!str alpha\ndescription: d\n---\n"),
    ("multiline double quote", '---\nname: "al\\\n  pha"\ndescription: d\n---\n'),
    ("multiline plain", "---\nname: al\n  pha\ndescription: d\n---\n"),
    ("block scalar", "---\nname: >-\n  alpha\ndescription: |\n  d\n---\n"),
    ("comment", "---\nname: alpha # note\ndescription: d\n---\n"),
    ("duplicate keys", "---\nname: beta\nname: alpha\ndescription: d\n---\n"),
    ("duplicate with null", "---\nname: alpha\nname: null\ndescription: d\n---\n"),
    ("no space after colon", "---\nname:alpha\ndescription: d\n---\n"),
    ("hex number", "---\nname: 0xff\ndescription: d\n---\n"),
    ("octal number", "---\nname: 0o17\ndescription: d\n---\n"),
    ("legacy octal", "---\nname: 017\ndescription: d\n---\n"),
    ("underscore number", "---\nname: 1_000\ndescription: d\n---\n"),
    ("binary number", "---\nname: 0b11\ndescription: d\n---\n"),
    ("exponent", "---\nname: 1e3\ndescription: d\n---\n"),
    ("float", "---\nname: 1.5\ndescription: d\n---\n"),
    ("infinity", "---\nname: .inf\ndescription: d\n---\n"),
    ("yaml 1.1 bool yes", "---\nname: yes\ndescription: d\n---\n"),
    ("yaml 1.1 bool on", "---\nname: On\ndescription: d\n---\n"),
    ("yaml 1.2 bool", "---\nname: true\ndescription: d\n---\n"),
    ("null", "---\nname: ~\ndescription: d\n---\n"),
    ("empty name", "---\nname:\ndescription: d\n---\n"),
    ("empty quoted name", "---\nname: ''\ndescription: d\n---\n"),
    ("sexagesimal", "---\nname: 1:30\ndescription: d\n---\n"),
    ("date", "---\nname: 2001-01-01\ndescription: d\n---\n"),
    ("timestamp", "---\nname: 2001-01-01T10:00:00Z\ndescription: d\n---\n"),
    ("list name", "---\nname: [alpha]\ndescription: d\n---\n"),
    ("mapping name", "---\nname: {a: b}\ndescription: d\n---\n"),
    ("blank description", "---\nname: alpha\ndescription: '  '\n---\n"),
    ("number description", "---\nname: alpha\ndescription: 5\n---\n"),
    ("no description", "---\nname: alpha\n---\n"),
    ("BOM and CRLF", "\ufeff---\r\nname: alpha\r\ndescription: d\r\n---\r\n"),
    ("CR only", "---\rname: alpha\rdescription: d\r---\r"),
    ("end delimiter suffix", "---\nname: alpha\ndescription: d\n---body\n"),
    ("ellipsis end", "---\nname: alpha\ndescription: d\n...\n"),
    ("no end", "---\nname: alpha\ndescription: d\n"),
    ("start suffix", "---x\nname: alpha\ndescription: d\n---\n"),
    ("empty frontmatter", "---\n---\ntext\n"),
    ("not a mapping", "---\njust text\n---\n"),
    ("list", "---\n- name\n- alpha\n---\n"),
    ("invalid yaml", "---\nname: [alpha\ndescription: d\n---\n"),
    ("tab indent", "---\nname:\talpha\ndescription: d\n---\n"),
    ("anchor and alias", "---\nname: &n alpha\ndescription: *n\n---\n"),
    ("merge key", "---\n<<: {name: alpha}\ndescription: d\n---\n"),
    ("two documents", "---\nname: alpha\ndescription: d\n---\n"),
    ("directive", "---\n%YAML 1.2\nname: alpha\ndescription: d\n---\n"),
    ("unknown tag", "---\nname: !foo alpha\ndescription: d\n---\n"),
    ("explicit int", "---\nname: !!int 5\ndescription: d\n---\n"),
    ("explicit int, text", "---\nname: !!int x\ndescription: d\n---\n"),
    ("explicit float, int text", "---\nname: !!float 1\ndescription: d\n---\n"),
    ("explicit bool, text", "---\nname: !!bool yes\ndescription: d\n---\n"),
    ("explicit null, text", "---\nname: !!null x\ndescription: d\n---\n"),
    ("explicit str, number", "---\nname: !!str 5\ndescription: d\n---\n"),
    ("quoted with int tag", '---\nname: !!int "5"\ndescription: d\n---\n'),
    ("binary tag", "---\nname: !!binary aGk=\ndescription: d\n---\n"),
    ("empty tag suffix", "---\nname: !! x\ndescription: d\n---\n"),
    ("alias without anchor", "---\nname: *x\ndescription: d\n---\n"),
    ("alias as key without anchor", "---\n*a : x\ndescription: d\n---\n"),
    ("tab after quoted key", '---\n"name":\talpha\ndescription: d\n---\n'),
    ("tab before colon", "---\nname\t: alpha\ndescription: d\n---\n"),
    ("trailing tab line", "---\nname: alpha\ndescription: d\n\t\n---\n"),
    ("tab indent", "---\nname:\n\talpha\ndescription: d\n---\n"),
    ("number keys", "---\n1: a\n01: b\nname: alpha\ndescription: d\n---\n"),
    ("nan keys", "---\n.nan: a\n.nan: b\nname: alpha\ndescription: d\n---\n"),
    ("null keys", "---\nnull: a\n~: b\nname: alpha\ndescription: d\n---\n"),
    ("string and number keys", "---\n'1': a\n1: b\nname: alpha\ndescription: d\n---\n"),
    ("int and float keys", "---\n1: a\n1.0: b\nname: alpha\ndescription: d\n---\n"),
    ("non-ascii", "---\nname: alphä\ndescription: d\n---\n"),
    ("long key", "---\n? " + "k" * 2000 + "\n: v\nname: alpha\ndescription: d\n---\n"),
]


def _write(tmp_path, content):
    skill = tmp_path / "project" / "SKILL.md"
    skill.parent.mkdir(exist_ok=True)
    skill.write_bytes(content.encode())
    return skill


@pytest.mark.parametrize(("label", "content"), FILES, ids=[label for label, _ in FILES])
def test_name_matches_pi(pi_reference, tmp_path, label, content):
    skill = _write(tmp_path, content)
    answers = pi_reference({"action": "skills", "cwd": str(tmp_path), "paths": [str(skill)]})
    ours = _pi_skill_name(skill)
    for version, answer in answers.items():
        assert answer["names"] == ([ours] if ours else []), (version, answer)


@pytest.mark.parametrize(
    ("content", "name"),
    [
        ("---\n{name: alpha, description: d}\n---\n", "alpha"),
        ('---\n"name": alpha\ndescription: d\n---\n', "alpha"),
        ('---\n"na\\x6de": alpha\ndescription: d\n---\n', "alpha"),
        ('---\n" name ": alpha\ndescription: d\n---\n', "project"),  # the key is " name ": not the name
        ("---\nname: 0xff\ndescription: d\n---\n", "project"),  # a number: pi uses the directory name
        ("---\nname: 1_000\ndescription: d\n---\n", "1_000"),  # not a number in YAML 1.2
        ("---\nname: yes\ndescription: d\n---\n", "yes"),  # not a boolean in YAML 1.2
        ("---\nname: 2001-01-01\ndescription: d\n---\n", "2001-01-01"),  # a string in the core schema
        ("---\nname: alpha\nname: beta\ndescription: d\n---\n", None),  # duplicate key: pi skips the skill
        ("---body\n", None),
    ],
)
def test_names(tmp_path, content, name):
    assert _pi_skill_name(_write(tmp_path, content)) == name


def test_frontmatter_keeps_string_values_only(tmp_path):
    skill = _write(tmp_path, "---\nname: alpha\ndescription: d\nn: 1\nb: true\nl: [x]\n---\n")
    assert _frontmatter(skill) == {"name": "alpha", "description": "d"}


@pytest.mark.parametrize(
    "content",
    [
        "---\n{name: alpha, description: d}\n---\n",
        '---\n"name": alpha\ndescription: d\n---\n',
        "---\nname: !!str alpha\ndescription: d\n---\n",
    ],
)
def test_collision_with_a_project_skill_in_another_yaml_form(tmp_path, content):
    scenario = tmp_path / "src" / "alpha"
    scenario.mkdir(parents=True)
    (scenario / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n")
    cwd = tmp_path / "cwd"
    project = cwd / ".pi" / "skills" / "other"
    project.mkdir(parents=True)
    (project / "SKILL.md").write_text(content)
    with pytest.raises(ValueError, match="'alpha'"):
        pi_skill_dirs([str(scenario)], str(cwd))


def test_duplicate_key_skill_is_not_a_collision(tmp_path):
    scenario = tmp_path / "src" / "alpha"
    scenario.mkdir(parents=True)
    (scenario / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n")
    cwd = tmp_path / "cwd"
    project = cwd / ".pi" / "skills" / "other"
    project.mkdir(parents=True)
    # Pi rejects the YAML (duplicate key) and does not load this skill.
    (project / "SKILL.md").write_text("---\nname: beta\nname: alpha\ndescription: d\n---\n")
    assert pi_skill_dirs([str(scenario)], str(cwd))[0] == str(scenario.resolve())
