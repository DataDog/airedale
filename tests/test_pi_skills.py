# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the skill frontmatter parser: it must read what pi reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from airedale.pi_frontmatter import UnsupportedFrontmatterError
from airedale.skills import _frontmatter, _pi_skill_name, _pi_skills, pi_skill_dirs

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
    ("tab as indentation", "---\nname:\n\talpha\ndescription: d\n---\n"),
    ("number keys", "---\n1: a\n01: b\nname: alpha\ndescription: d\n---\n"),
    ("nan keys", "---\n.nan: a\n.nan: b\nname: alpha\ndescription: d\n---\n"),
    ("null keys", "---\nnull: a\n~: b\nname: alpha\ndescription: d\n---\n"),
    ("string and number keys", "---\n'1': a\n1: b\nname: alpha\ndescription: d\n---\n"),
    ("int and float keys", "---\n1: a\n1.0: b\nname: alpha\ndescription: d\n---\n"),
    ("plain value with a tab", "---\nname: alpha\tbeta\ndescription: d\n---\n"),
    ("quoted name with a tab, tab after colon", '---\nname: "alpha\tbeta"\ndescription:\td\n---\n'),
    ("quoted name with an escaped tab", '---\nname: "alpha\\tbeta"\ndescription: d\n---\n'),
    ("tab before a comment", "---\nname: alpha\t# c\ndescription: d\n---\n"),
    ("tab at the end of a line", "---\nname: alpha\t\ndescription: d\n---\n"),
    ("tab in a flow mapping", "---\n{name:\talpha, description:\td}\n---\n"),
    ("tab after a sequence indicator", "---\nname: alpha\ndescription: d\nx:\n-\ta\n---\n"),
    ("nested duplicate key", "---\nname: alpha\ndescription: d\nmetadata: {x: 1, x: 2}\n---\n"),
    ("nested duplicate key, block", "---\nname: alpha\ndescription: d\nmetadata:\n  x: 1\n  x: 2\n---\n"),
    ("duplicate key in a list", "---\nname: alpha\ndescription: d\nl:\n- {x: 1, x: 2}\n---\n"),
    ("nested key that is not a duplicate", "---\nname: alpha\ndescription: d\nm: {x: 1}\nn: {x: 2}\n---\n"),
    ("root set with values", "---\n!!set\nname: alpha\ndescription: d\n---\n"),
    ("root set with null values", "---\n!!set\nname:\ndescription:\n---\n"),
    ("root anchor", "---\n&a {name: alpha, description: d}\n---\n"),
    ("set with values, nested", "---\nname: alpha\ndescription: d\ns: !!set {a: 1}\n---\n"),
    ("set with null values, nested", "---\nname: alpha\ndescription: d\ns: !!set {a, b: ~}\n---\n"),
    ("set on a sequence", "---\nname: alpha\ndescription: d\ns: !!set [a]\n---\n"),
    ("omap with a repeated key", "---\nname: alpha\ndescription: d\no: !!omap [a: 1, a: 2]\n---\n"),
    ("omap with repeated scalars", "---\nname: alpha\ndescription: d\no: !!omap [x, x]\n---\n"),
    ("omap with distinct keys", "---\nname: alpha\ndescription: d\no: !!omap [a: 1, b: 2]\n---\n"),
    ("pairs with two keys in an item", "---\nname: alpha\ndescription: d\np: !!pairs [{a: 1, b: 2}]\n---\n"),
    ("pairs with a repeated key", "---\nname: alpha\ndescription: d\np: !!pairs [a: 1, a: 2]\n---\n"),
    ("map tag on a sequence", "---\nname: alpha\ndescription: d\nm: !!map [a]\n---\n"),
    ("large number keys", "---\nname: alpha\ndescription: d\n9007199254740992: a\n9007199254740993: b\n---\n"),
    ("hex and decimal keys", "---\nname: alpha\ndescription: d\n0x10: a\n16: b\n---\n"),
    ("overflow keys", "---\nname: alpha\ndescription: d\n1e400: a\n1e401: b\n---\n"),
    ("zero keys", "---\nname: alpha\ndescription: d\n-0: a\n0: b\n---\n"),
    ("alias as a repeated key", "---\nname: alpha\ndescription: d\nk: &a [1]\nm: {? *a : 1, ? *a : 2}\n---\n"),
    ("collection keys that look the same", "---\nname: alpha\ndescription: d\nm: {? [a] : 1, ? [a] : 2}\n---\n"),
    ("tab in a continuation line, after a dash", "---\nname: alpha\n  -\tbeta\ndescription: d\n---\n"),
    ("tab in a continuation line, after a question mark", "---\nname: alpha\n  ?\tbeta\ndescription: d\n---\n"),
    ("tab in a continuation line, inside", "---\nname: alpha\n  be\tta\ndescription: d\n---\n"),
    ("tab in a continuation line, after a colon", "---\nname: alpha\n  x:\tbeta\ndescription: d\n---\n"),
    ("tab in a continuation line, before a comment", "---\nname: alpha\n  beta\t# c\ndescription: d\n---\n"),
    ("tab after a dash in a value", "---\nname: a -\tb\ndescription: d\n---\n"),
    ("tab after a question mark in a value", "---\nname: a ?\tb\ndescription: d\n---\n"),
    ("tab after a sequence indicator, indented", "---\nname: alpha\ndescription: d\nx:\n  -\ta\n---\n"),
    ("tab after an explicit key indicator", "---\nname: alpha\ndescription: d\n?\tk\n: v\n---\n"),
    ("tab between two sequence indicators", "---\nname: alpha\ndescription: d\nl:\n  -\t- a\n---\n"),
    ("tab after an anchor", "---\nname: &x\talpha\ndescription: d\n---\n"),
    ("tab after a tag", "---\nname: !!str\talpha\ndescription: d\n---\n"),
    ("tab in a flow sequence", "---\nname: alpha\ndescription: d\nm: [a,\tb]\n---\n"),
    ("recursive sequence alias", "---\nname: alpha\ndescription: d\nmetadata: &m [*m]\n---\n"),
    ("recursive mapping alias", "---\nname: alpha\ndescription: d\nmetadata: &m {self: *m}\n---\n"),
    ("recursive alias in a block mapping", "---\nname: alpha\ndescription: d\nm: &m\n  self: *m\n---\n"),
    ("mutually recursive aliases", "---\nname: alpha\ndescription: d\na: &a {b: &b {a: *a, b: *b}}\n---\n"),
    ("alias to a collection of its own parent", "---\n&r {name: alpha, description: d, me: *r}\n---\n"),
    ("alias used before its anchor ends", "---\nname: alpha\ndescription: d\nm: &m [*m, *m]\n---\n"),
    ("alias key replaces the description with null", "---\nname: alpha\n&k description: d\n*k : null\n---\n"),
    ("alias key replaces the name with null", "---\n&k name: alpha\n*k : null\ndescription: d\n---\n"),
    ("alias key replaces the name with a list", "---\n&k name: alpha\n*k : [x]\ndescription: d\n---\n"),
    ("alias key replaces the name with a string", "---\n&k name: alpha\n*k : beta\ndescription: d\n---\n"),
    ("alias key replaces the name with a number", "---\n&k name: alpha\n*k : 5\ndescription: d\n---\n"),
    ("alias key with the same string value", "---\n&k name: alpha\n*k : alpha\ndescription: d\n---\n"),
    ("alias key of another key", "---\n&k other: x\n*k : y\nname: alpha\ndescription: d\n---\n"),
    (
        "omap of aliases to open mappings",
        "---\nname: alpha\ndescription: d\nmetadata: &a {b: &b {o: !!omap [*a, *b]}}\n---\n",
    ),
    ("omap of an alias to an open mapping", "---\nname: alpha\ndescription: d\nm: &a {o: !!omap [*a]}\n---\n"),
    ("pairs of an alias to an open mapping", "---\nname: alpha\ndescription: d\nm: &a {o: !!pairs [*a]}\n---\n"),
    ("omap of an alias to a finished mapping", "---\nname: alpha\ndescription: d\nm: &a {x: 1}\no: !!omap [*a]\n---\n"),
    ("omap of an alias to an open sequence", "---\nname: alpha\ndescription: d\nm: &a [!!omap [*a]]\n---\n"),
    ("integer of 5000 digits as the name", "---\nname: " + "9" * 5000 + "\ndescription: d\n---\n"),
    ("negative integer of 5000 digits as the name", "---\nname: -" + "9" * 5000 + "\ndescription: d\n---\n"),
    ("integer of 5000 zeros", "---\nname: " + "0" * 5000 + "\ndescription: d\n---\n"),
    ("integer of 5000 digits as a value", "---\nname: alpha\ndescription: d\nx: " + "9" * 5000 + "\n---\n"),
    (
        "keys that are integers of 5000 digits",
        "---\nname: alpha\ndescription: d\n" + "9" * 5000 + ": a\n" + "9" * 4999 + "8: b\n---\n",
    ),
    ("float of 5000 digits", "---\nname: alpha\ndescription: d\nx: " + "9" * 5000 + ".5\n---\n"),
    ("integer of 400 digits as a value", "---\nname: alpha\ndescription: d\nx: " + "9" * 400 + "\n---\n"),
    ("hex integer of 5000 digits", "---\nname: alpha\ndescription: d\nx: 0x" + "f" * 5000 + "\n---\n"),
    ("non-ascii", "---\nname: alphä\ndescription: d\n---\n"),
    ("long key", "---\n? " + "k" * 2000 + "\n: v\nname: alpha\ndescription: d\n---\n"),
]


def _alias_cases() -> list[tuple[str, str]]:
    """``!!omap`` and ``!!pairs`` with aliases to a finished mapping: 0, 1 and 2 pairs, one or two aliases."""
    cases = []
    for tag in ("omap", "pairs"):
        for mapping, label in (("{}", "0 pairs"), ("{x: 1}", "1 pair"), ("{x: 1, y: 2}", "2 pairs")):
            for aliases, count in (("*a", "one alias"), ("*a, *a", "two aliases of one mapping")):
                body = f"m: &a {mapping}\no: !!{tag} [{aliases}]"
                cases.append(
                    (f"{tag} of {count} to a mapping with {label}", f"---\nname: alpha\ndescription: d\n{body}\n---\n")
                )
        # Two different finished mappings with the same key, and an alias next to a literal item.
        body = f"m: &a {{x: 1}}\nn: &b {{x: 2}}\no: !!{tag} [*a, *b]"
        cases.append((f"{tag} of two aliases with the same key", f"---\nname: alpha\ndescription: d\n{body}\n---\n"))
        body = f"m: &a {{x: 1}}\no: !!{tag} [*a, {{y: 2}}]"
        cases.append((f"{tag} of an alias and a literal item", f"---\nname: alpha\ndescription: d\n{body}\n---\n"))
        body = f"s: &k x\no: !!{tag} [*k]"
        cases.append((f"{tag} of an alias to a scalar", f"---\nname: alpha\ndescription: d\n{body}\n---\n"))
        body = f"o: !!{tag} [{{x: 1}}, {{x: 2}}]"
        cases.append((f"{tag} of literal items", f"---\nname: alpha\ndescription: d\n{body}\n---\n"))
    return cases


ALIAS_CASES = _alias_cases()
FILES += ALIAS_CASES


def _write(tmp_path, content):
    skill = tmp_path / "project" / "SKILL.md"
    skill.parent.mkdir(exist_ok=True)
    skill.write_bytes(content.encode())
    return skill


#: Files with a tab that the reader can not place: it refuses to read them (see ``UnsupportedFrontmatterError``).
UNSUPPORTED = {
    *(label for label, _ in ALIAS_CASES if "alias" in label),
    "omap of an alias to a finished mapping",
    "omap of an alias to an open sequence",
    "tab as indentation",
    "plain value with a tab",
    "tab in a continuation line, after a dash",
    "tab in a continuation line, after a question mark",
    "tab in a continuation line, inside",
    "tab after a dash in a value",
    "tab after a question mark in a value",
    "tab between two sequence indicators",
    "omap of aliases to open mappings",
    "omap of an alias to an open mapping",
    "pairs of an alias to an open mapping",
    "tab after an anchor",
    "tab after a tag",
}


@pytest.mark.parametrize(("label", "content"), FILES, ids=[label for label, _ in FILES])
def test_name_matches_pi(pi_reference, tmp_path, label, content):
    skill = _write(tmp_path, content)
    answers = pi_reference({"action": "skills", "content": content})
    if label in UNSUPPORTED:
        with pytest.raises(UnsupportedFrontmatterError):
            _pi_skill_name(skill)
        return
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


def _collision(tmp_path, scenario_content, project_content):
    scenario = tmp_path / "src" / "scenario"
    scenario.mkdir(parents=True)
    (scenario / "SKILL.md").write_text(scenario_content)
    cwd = tmp_path / "cwd"
    project = cwd / ".pi" / "skills" / "other"
    project.mkdir(parents=True)
    (project / "SKILL.md").write_text(project_content)
    return pi_skill_dirs([str(scenario)], str(cwd))


def test_a_tab_in_a_plain_name_is_not_read_as_a_space(tmp_path):
    # Pi reads the same name "alpha<TAB>beta" in both skills and hides the project skill.
    # The reader can not place the tab of the plain name: the run fails, it does not go on.
    with pytest.raises(UnsupportedFrontmatterError, match="tab"):
        _collision(
            tmp_path,
            '---\nname: "alpha\\tbeta"\ndescription: d\n---\n',
            "---\nname: alpha\tbeta\ndescription: d\n---\n",
        )


def test_a_separator_tab_does_not_change_a_quoted_name(tmp_path):
    # The tab after ``description:`` is white space; the tab in the name is part of the name.
    with pytest.raises(ValueError, match="'alpha\\tbeta'"):
        _collision(
            tmp_path,
            '---\nname: "alpha\\tbeta"\ndescription: d\n---\n',
            '---\nname: "alpha\tbeta"\ndescription:\td\n---\n',
        )


def test_a_name_with_a_space_is_not_a_name_with_a_tab(tmp_path):
    assert _collision(
        tmp_path,
        '---\nname: "alpha beta"\ndescription: d\n---\n',
        '---\nname: "alpha\tbeta"\ndescription:\td\n---\n',
    )


def test_nested_duplicate_key_skill_is_not_a_collision(tmp_path):
    # Pi skips the file (the nested mapping has a repeated key), so the names do not collide.
    assert _collision(
        tmp_path,
        "---\nname: alpha\ndescription: d\n---\n",
        "---\nname: alpha\ndescription: d\nmetadata: {x: 1, x: 2}\n---\n",
    )


def test_tagged_root_set_is_not_a_collision(tmp_path):
    assert _collision(
        tmp_path,
        "---\nname: alpha\ndescription: d\n---\n",
        "---\n!!set\nname: alpha\ndescription: d\n---\n",
    )


def test_numbers_are_the_same_key_when_javascript_says_so(tmp_path):
    assert _collision(
        tmp_path,
        "---\nname: alpha\ndescription: d\n---\n",
        "---\nname: alpha\ndescription: d\n9007199254740992: a\n9007199254740993: b\n---\n",
    )


def test_a_tab_in_a_continuation_line_is_not_read_as_a_space(tmp_path):
    # Pi reads "alpha -<TAB>beta" for the project skill (the dash is part of the value), the same name as the scenario
    # skill. A space would hide the collision: the run fails because the tab can not be placed.
    with pytest.raises(UnsupportedFrontmatterError, match="tab"):
        _collision(
            tmp_path,
            '---\nname: "alpha -\\tbeta"\ndescription: d\n---\n',
            "---\nname: alpha\n  -\tbeta\ndescription: d\n---\n",
        )


@pytest.mark.parametrize("metadata", ["&m [*m]", "&m {self: *m}"])
def test_recursive_alias_in_a_project_skill_still_collides(tmp_path, metadata):
    # Pi loads the skill (the alias is valid), so the name collides with the scenario skill.
    with pytest.raises(ValueError, match="'alpha'"):
        _collision(
            tmp_path,
            "---\nname: alpha\ndescription: d\n---\n",
            f"---\nname: alpha\ndescription: d\nmetadata: {metadata}\n---\n",
        )


def test_alias_key_that_nulls_the_description_is_not_a_collision(tmp_path):
    # Pi reads description null and skips the skill, so there is no collision.
    assert _collision(
        tmp_path,
        "---\nname: alpha\ndescription: d\n---\n",
        "---\nname: alpha\n&k description: d\n*k : null\n---\n",
    )


def test_alias_key_that_nulls_the_name_gives_the_directory_name(tmp_path):
    # Pi names the skill "other" (its directory), so it collides with a scenario skill of that name.
    with pytest.raises(ValueError, match="'other'"):
        _collision(
            tmp_path,
            "---\nname: other\ndescription: d\n---\n",
            "---\n&k name: alpha\n*k : null\ndescription: d\n---\n",
        )


def test_omap_of_open_mappings_stops_the_run(tmp_path):
    # Pi loads this skill as "alpha". The reader can not tell if the ordered map is valid, so it does not skip the skill.
    with pytest.raises(UnsupportedFrontmatterError, match="alias"):
        _collision(
            tmp_path,
            "---\nname: alpha\ndescription: d\n---\n",
            "---\nname: alpha\ndescription: d\nmetadata: &a {b: &b {o: !!omap [*a, *b]}}\n---\n",
        )


@pytest.mark.parametrize("tag", ["omap", "pairs"])
@pytest.mark.parametrize("pairs", ["{}", "{x: 1}", "{x: 1, y: 2}"])
def test_alias_item_in_an_ordered_map_stops_the_run(tmp_path, tag, pairs):
    # Pi loads this skill as "alpha" (a collision with the scenario skill). The reader does not copy how the package
    # turns the pairs of an aliased node, so it stops: it never skips the skill.
    with pytest.raises(UnsupportedFrontmatterError, match="alias"):
        _collision(
            tmp_path,
            "---\nname: alpha\ndescription: d\n---\n",
            f"---\nname: alpha\ndescription: d\nm: &a {pairs}\no: !!{tag} [*a]\n---\n",
        )


def test_literal_items_of_an_ordered_map_keep_the_rules(tmp_path):
    # No alias: pi loads the skill, so the collision is found.
    with pytest.raises(ValueError, match="'alpha'"):
        _collision(
            tmp_path,
            "---\nname: alpha\ndescription: d\n---\n",
            "---\nname: alpha\ndescription: d\no: !!omap [{x: 1}, {y: 2}]\n---\n",
        )


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_settings_json_with_a_byte_order_mark_lists_skills(tmp_path, bom):
    # Pi removes a BOM from .pi/settings.json before it parses it.
    cwd = tmp_path / "cwd"
    skill = cwd / "tools" / "extra"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: extra\ndescription: d\n---\n")
    (cwd / ".pi").mkdir()
    (cwd / ".pi" / "settings.json").write_bytes(bom + json.dumps({"skills": ["../tools/extra"]}).encode())
    assert pi_skill_dirs([], str(cwd)) == [str(cwd / ".pi" / ".." / "tools" / "extra")]


def test_settings_json_with_a_huge_integer_still_lists_skills(tmp_path):
    # JSON.parse reads an integer of 5000 digits as Infinity: the settings file is valid for pi.
    cwd = tmp_path / "cwd"
    skill = cwd / "tools" / "extra"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: extra\ndescription: d\n---\n")
    (cwd / ".pi").mkdir()
    (cwd / ".pi" / "settings.json").write_text('{"other": ' + "9" * 5000 + ', "skills": ["../tools/extra"]}')
    assert pi_skill_dirs([], str(cwd)) == [str(cwd / ".pi" / ".." / "tools" / "extra")]


def test_a_long_integer_in_a_project_skill_still_collides(tmp_path):
    # Pi reads the number as Infinity and loads the skill: the name collides. No exception, no skipped skill.
    with pytest.raises(ValueError, match="'alpha'"):
        _collision(
            tmp_path,
            "---\nname: alpha\ndescription: d\n---\n",
            "---\nname: alpha\ndescription: d\nx: " + "9" * 5000 + "\n---\n",
        )


def _settings(tmp_path, entries):
    cwd = tmp_path / "cwd"
    (cwd / ".pi").mkdir(parents=True)
    (cwd / ".pi" / "settings.json").write_text(json.dumps({"skills": entries}))
    return cwd


def test_tilde_user_in_settings_is_an_ordinary_relative_path(tmp_path):
    # Pi expands only "~" and "~/...": "~name" is a path below .pi. Path.expanduser raised RuntimeError for it.
    from airedale.skills import _pi_settings_skill_paths

    cwd = _settings(tmp_path, ["~user-that-does-not-exist/skills", "~root/skills", "~x"])
    assert _pi_settings_skill_paths(cwd) == [
        cwd / ".pi" / "~user-that-does-not-exist" / "skills",
        cwd / ".pi" / "~root" / "skills",
        cwd / ".pi" / "~x",
    ]


def test_tilde_and_tilde_slash_are_the_home_directory(tmp_path, monkeypatch):
    from airedale.skills import _pi_settings_skill_paths

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    cwd = _settings(tmp_path, ["~", "~/skills/extra"])
    assert _pi_settings_skill_paths(cwd) == [home, home / "skills" / "extra"]


def test_skill_dirs_do_not_raise_for_a_tilde_user_entry(tmp_path):
    cwd = _settings(tmp_path, ["~user-that-does-not-exist/skills"])
    assert pi_skill_dirs([], str(cwd)) == []  # The path does not exist: it is left out.


def _skill(path, name, body="d"):
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {body}\n---\n")
    return path


def test_symlink_to_the_scenario_skill_is_the_same_file_not_a_collision(tmp_path):
    # Pi follows directory symlinks and loads a file once (by its real path): no collision diagnostic.
    scenario = _skill(tmp_path / "src" / "alpha", "alpha")
    cwd = tmp_path / "cwd"
    (cwd / ".pi" / "skills").mkdir(parents=True)
    (cwd / ".pi" / "skills" / "linked").symlink_to(scenario, target_is_directory=True)
    assert pi_skill_dirs([str(scenario)], str(cwd))[0] == str(scenario.resolve())


def test_different_skill_with_the_same_name_is_still_a_collision(tmp_path):
    scenario = _skill(tmp_path / "src" / "alpha", "alpha")
    cwd = tmp_path / "cwd"
    _skill(cwd / ".pi" / "skills" / "other", "alpha", "another file")
    with pytest.raises(ValueError, match="collides"):
        pi_skill_dirs([str(scenario)], str(cwd))


def test_two_links_to_one_project_skill_load_it_once(tmp_path, caplog):
    cwd = tmp_path / "cwd"
    target = _skill(tmp_path / "shared" / "beta", "beta")
    root = cwd / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / "one").symlink_to(target, target_is_directory=True)
    (root / "two").symlink_to(target, target_is_directory=True)
    with caplog.at_level("WARNING"):
        assert pi_skill_dirs([], str(cwd)) == [str(root)]
    assert "shadowed" not in caplog.text


def test_symlink_to_a_directory_above_is_refused(tmp_path):
    cwd = tmp_path / "cwd"
    root = cwd / ".pi" / "skills"
    _skill(root / "team" / "gamma", "gamma")
    link = root / "team" / "up"
    link.symlink_to(root, target_is_directory=True)  # A link to an ancestor.
    with pytest.raises(ValueError, match="symbolic link cycle") as error:
        pi_skill_dirs([], str(cwd))
    assert str(root) in str(error.value)
    assert str(link) in str(error.value)


def test_link_to_the_project_is_not_a_cycle_because_hidden_directories_are_skipped(tmp_path):
    # The walk of the project does not enter ``.agents`` or ``.pi``, so it never comes back: pi loads the same files.
    cwd = tmp_path / "cwd"
    agents = cwd / ".agents" / "skills"
    _skill(agents / "team" / "delta", "delta")
    (agents / "team" / "up").symlink_to(cwd, target_is_directory=True)
    assert pi_skill_dirs([], str(cwd)) == [str(agents / "team" / "delta" / "SKILL.md")]


def test_cycle_in_a_scenario_skill_directory_is_refused(tmp_path):
    scenario = tmp_path / "scenario"
    _skill(scenario / "one", "one")
    (scenario / "back").symlink_to(scenario, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link cycle") as error:
        pi_skill_dirs([str(scenario)], str(tmp_path / "cwd"))
    assert str(scenario) in str(error.value)


def test_cycle_below_a_skill_directory_is_not_walked_and_not_refused(tmp_path):
    # Pi does not go below a directory with SKILL.md, so the link is never followed.
    cwd = tmp_path / "cwd"
    root = cwd / ".pi" / "skills"
    skill = _skill(root / "gamma", "gamma")
    (skill / "back").symlink_to(root, target_is_directory=True)
    assert pi_skill_dirs([], str(cwd)) == [str(root)]


def test_the_error_of_a_cycle_has_no_file_content(tmp_path):
    cwd = tmp_path / "cwd"
    root = cwd / ".agents" / "skills"
    root.mkdir(parents=True)
    (root / "loose.md").write_text("---\nname: x\ndescription: SECRET-BODY\n---\n")
    (root / "loop").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError) as error:
        pi_skill_dirs([], str(cwd))
    assert "SECRET-BODY" not in str(error.value)


def test_broken_symlink_is_skipped(tmp_path):
    cwd = tmp_path / "cwd"
    root = cwd / ".pi" / "skills"
    root.mkdir(parents=True)
    (root / "broken").symlink_to(tmp_path / "missing", target_is_directory=True)
    assert pi_skill_dirs([], str(cwd)) == [str(root)]


HOME_ENTRIES = [
    "~", "~/", "~//", "~/extra", "~//extra", "~///a//b", "~/./extra", "~/a/../extra", "~/../extra", "~/a/./b/", "~/a//",
    "~x", "~x/skills", "~user/skills", "/abs/path", "rel/path", "./rel", "../up", "~.", "~~", " ~/extra", "~ /extra",
]  # fmt: skip


def test_home_paths_are_those_of_pi(pi_reference, tmp_path, monkeypatch):
    """The skill paths of .pi/settings.json: same result as pi's normalizePath (with a fixed home)."""
    from airedale.skills import _expand_pi_home

    home = "/home/review"
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    for version, native in pi_reference({"action": "home_paths", "entries": HOME_ENTRIES, "home": home}).items():
        for entry, expected in zip(HOME_ENTRIES, native, strict=True):
            # (Pi gives a path string: compared as paths, so "./rel" and "rel" are the same.)
            assert _expand_pi_home(entry) == Path(expected), (version, entry, expected)


def test_tilde_slash_slash_keeps_the_home_directory(tmp_path, monkeypatch):
    from airedale.skills import _pi_settings_skill_paths

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    cwd = _settings(tmp_path, ["~//extra", "~///a//b", "~/a/../c"])
    assert _pi_settings_skill_paths(cwd) == [home / "extra", home / "a" / "b", home / "c"]


def test_a_collision_below_tilde_slash_slash_is_found(tmp_path, monkeypatch):
    # Pi reads <home>/extra for "~//extra", finds the skill there, and reports the collision with the scenario skill.
    home = tmp_path / "home"
    _skill(home / "extra" / "alpha", "alpha")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    scenario = _skill(tmp_path / "src" / "alpha", "alpha")
    cwd = _settings(tmp_path, ["~//extra"])
    with pytest.raises(ValueError, match="collides"):
        pi_skill_dirs([str(scenario)], str(cwd))


def _md(name, body="d"):
    return f"---\nname: {name}\ndescription: {body}\n---\n"


#: Trees with links but no cycle (path -> content, path -> target): the skills that pi loads from them, as pairs of a
#: name and a real file. Trees with a cycle are refused (see above): pi follows them until the system refuses, and
#: the aliases that it meets first depend on the system.
SKILL_TREES = {
    "link to a markdown file": (
        {"notes/n.md": _md("n")},
        {".agents/skills/d/link.md": "notes/n.md", ".pi/skills/link.md": "notes/n.md"},
    ),
    "two links to one skill": (
        {"shared/beta/SKILL.md": _md("beta")},
        {".pi/skills/one": "shared/beta", ".pi/skills/two": "shared/beta"},
    ),
    "two links to one directory in two roots": (
        {"shared/d/x.md": _md("x"), "shared/d/y/SKILL.md": _md("y")},
        {".agents/skills/one": "shared/d", ".pi/skills/two": "shared/d"},
    ),
    "link to the project (hidden directories are skipped)": (
        {".agents/skills/s/x.md": _md("x"), ".pi/skills/loose.md": _md("loose"), "docs/d/SKILL.md": _md("d")},
        {".agents/skills/s/home": "", ".pi/skills/home": ""},
    ),
    "nameless skill reached by links with the same name": (
        {"shared/B/zzzz.md": "---\ndescription: d\n---\n"},
        {".agents/skills/p/foo": "shared/B", ".agents/skills/q/foo": "shared/B"},
    ),
    "same name in two directories, one linked to the other (no cycle)": (
        {".agents/skills/a/zz-after.md": _md("shared", "A"), ".agents/skills/b/zz-after.md": _md("shared", "B")},
        {".agents/skills/a/to-b": ".agents/skills/b"},
    ),
    "same name, links that come before and after the real directory": (
        {"shared/m/x.md": _md("shared", "M"), ".agents/skills/zz/x.md": _md("shared", "Z")},
        {".agents/skills/aa-link": "shared/m", ".agents/skills/zz-link": "shared/m"},
    ),
    "same name, two linked directories in the two roots": (
        {"shared/p/SKILL.md": _md("shared", "P"), "shared/q/SKILL.md": _md("shared", "Q")},
        {".agents/skills/one": "shared/q", ".pi/skills/two": "shared/p", ".pi/skills/three": "shared/q"},
    ),
    "same name in two files": (
        {".agents/skills/e/loose.md": _md("same"), ".agents/skills/d/SKILL.md": _md("same")},
        {},
    ),
    "link to a directory with a collision": (
        {"shared/one/SKILL.md": _md("same"), ".pi/skills/two/SKILL.md": _md("same")},
        {".agents/skills/one": "shared/one"},
    ),
}

#: Trees where a skill with no ``name`` is reached through links whose parent directories have different names: pi
#: takes the name of the link that it reads first (the order of the listing of the system), so the run is refused.
ALIAS_TREES = {
    "link to a directory of a skill with no name, by two aliases": (
        {"shared/B/zzzz.md": "---\ndescription: d\n---\n"},
        {".agents/skills/ax": "shared/B", ".agents/skills/by": "shared/B"},
    ),
    "link to a directory of a skill with no name, by two hops": (
        {"shared/B/zzzz.md": "---\ndescription: d\n---\n"},
        {"shared/b-alias": "shared/B", ".agents/skills/ax": "shared/b-alias", ".agents/skills/by": "shared/B"},
    ),
    "links in a chain (no cycle), names from the directories": (
        {"shared/z/n.md": "---\ndescription: d\n---\n"},
        {
            "shared/y": "shared/z",
            "shared/x": "shared/y",
            ".agents/skills/b": "shared/x",
            ".agents/skills/a": "shared/y",
        },
    ),
}


@pytest.mark.parametrize("label", ALIAS_TREES)
def test_skill_without_name_reached_by_links_with_different_names_is_refused(tmp_path, label):
    files, links = ALIAS_TREES[label]
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    with pytest.raises(ValueError, match="no 'name'") as error:
        pi_skill_dirs([], str(cwd))
    assert "ax" in str(error.value) or "a'" in str(error.value) or "b'" in str(error.value)


def test_skill_without_name_in_the_scenario_and_in_the_project_under_two_names_is_refused(tmp_path):
    scenario = tmp_path / "scenario" / "one"
    scenario.mkdir(parents=True)
    (scenario / "n.md").write_text("---\ndescription: d\n---\n")
    cwd = tmp_path / "cwd"
    (cwd / ".agents" / "skills").mkdir(parents=True)
    (cwd / ".agents" / "skills" / "other").symlink_to(scenario, target_is_directory=True)
    with pytest.raises(ValueError, match="no 'name'"):
        pi_skill_dirs([str(scenario)], str(cwd))


def test_skill_without_name_with_the_same_name_through_each_link_is_kept(tmp_path):
    # Both links are called ``foo``: the skill is ``foo`` whichever link pi reads first.
    files = {"shared/B/zzzz.md": "---\ndescription: d\n---\n"}
    links = {".agents/skills/p/foo": "shared/B", ".agents/skills/q/foo": "shared/B"}
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    assert pi_skill_dirs([], str(cwd)) == [
        str(cwd / ".agents/skills/p/foo/zzzz.md"),
        str(cwd / ".agents/skills/q/foo/zzzz.md"),
    ]


#: Trees with a cycle: refused before pi starts.
CYCLIC_TREES = {
    "agents loose file and a link to the root": (
        {".agents/skills/loose.md": _md("loose")},
        {".agents/skills/loop": ".agents/skills"},
    ),
    "pi loose file and a link to the root": (
        {".pi/skills/loose.md": _md("loose")},
        {".pi/skills/loop": ".pi/skills"},
    ),
    "link below a directory": (
        {".agents/skills/a/x.md": _md("x")},
        {".agents/skills/a/back": ".agents/skills"},
    ),
    "two links that point to each other": (
        {".agents/skills/a/one.md": _md("one"), ".agents/skills/b/two.md": _md("two")},
        {".agents/skills/a/to-b": ".agents/skills/b", ".agents/skills/b/to-a": ".agents/skills/a"},
    ),
    "root skill and a link": (
        {".agents/skills/team/y.md": _md("y")},
        {".agents/skills/team/loop": ".agents/skills"},
    ),
    "two directories with links to each other, same name (round 16)": (
        {".agents/skills/a/zz-after.md": _md("shared", "A"), ".agents/skills/b/zz-after.md": _md("shared", "B")},
        {".agents/skills/a/to-b": ".agents/skills/b", ".agents/skills/b/to-a": ".agents/skills/a"},
    ),
    "two directories with links to each other, no names (round 16)": (
        {
            ".agents/skills/a/zz-after.md": "---\ndescription: A\n---\n",
            ".agents/skills/b/zz-after.md": "---\ndescription: B\n---\n",
            ".agents/skills/a/before.md": "---\ndescription: A\n---\n",
            ".agents/skills/b/before.md": "---\ndescription: B\n---\n",
        },
        {".agents/skills/a/to-b": ".agents/skills/b", ".agents/skills/b/to-a": ".agents/skills/a"},
    ),
    "cycle by two hops (the tree of the review)": (
        {"shared/A/00.md": _md("alpha"), "shared/B/zzzz.md": "---\ndescription: d\n---\n"},
        {
            "shared/b-alias": "shared/B",
            "shared/A/ax": "shared/b-alias",
            "shared/A/by": "shared/B",
            "shared/B/aa": "shared/A",
            ".agents/skills/start": "shared/A",
        },
    ),
}


def _write_tree(cwd, files, links):
    cwd.mkdir(exist_ok=True)
    for path, content in files.items():
        (cwd / path).parent.mkdir(parents=True, exist_ok=True)
        (cwd / path).write_text(content)
    for path in links:
        (cwd / path).parent.mkdir(parents=True, exist_ok=True)
    for path, target in links.items():
        (cwd / path).symlink_to(cwd / target, target_is_directory=not target.endswith(".md"))


@pytest.mark.parametrize("label", CYCLIC_TREES)
def test_trees_with_a_link_cycle_are_refused(tmp_path, label):
    files, links = CYCLIC_TREES[label]
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    with pytest.raises(ValueError, match="symbolic link cycle"):
        pi_skill_dirs([], str(cwd))


def test_the_collision_of_the_review_with_a_cycle_is_refused_not_missed(tmp_path):
    # Pi names a skill with no ``name`` from the parent directory of the alias that it meets first: ``ax`` or ``by``
    # depending on the system. A scenario skill ``by`` must not make the result depend on it: refuse.
    files, links = CYCLIC_TREES["cycle by two hops (the tree of the review)"]
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    scenario = _skill(tmp_path / "scenario" / "by", "by")
    with pytest.raises(ValueError, match="symbolic link cycle"):
        pi_skill_dirs([str(scenario)], str(cwd))


def test_the_graph_of_round_16_is_refused_with_a_scenario_skill_too(tmp_path):
    # ``a/to-b -> b`` and ``b/to-a -> a``: which of the two same-name files wins depends on where the system stops
    # the cycle. Refused, whatever the scenario skills are.
    files, links = CYCLIC_TREES["two directories with links to each other, same name (round 16)"]
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    scenario = _skill(tmp_path / "scenario" / "shared", "shared")
    with pytest.raises(ValueError, match="symbolic link cycle") as error:
        pi_skill_dirs([str(scenario)], str(cwd))
    assert str(cwd / ".agents" / "skills") in str(error.value)
    with pytest.raises(ValueError, match="symbolic link cycle"):
        pi_skill_dirs([], str(cwd))


@pytest.mark.parametrize("label", SKILL_TREES)
def test_skills_of_a_tree_with_links_are_those_of_pi(pi_reference, tmp_path, label):
    """Same ``(name, real file)`` pairs as pi 1.4.0 and 1.5.0 (or both fail on a name collision)."""
    files, links = SKILL_TREES[label]
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    real_cwd = os.path.realpath(cwd)
    loaded: set[tuple[str, str]] = set()
    ours_collides = False
    seen_names: set[str] = set()
    seen_files: set[str] = set()
    try:
        for path in pi_skill_dirs([], str(cwd)):
            for name, skill_file in _pi_skills(Path(path)):
                real = os.path.realpath(skill_file)
                if real in seen_files or name in seen_names:  # Pi keeps the first file, and the first name.
                    continue
                seen_files.add(real)
                seen_names.add(name)
                loaded.add((name, os.path.relpath(real, real_cwd)))
    except ValueError:
        ours_collides = True
    request = {"action": "skill_tree", "files": files, "links": links}
    for version, answer in pi_reference(request).items():
        if ours_collides:
            assert answer["collisions"], version
            continue
        assert loaded == {tuple(pair) for pair in answer["skills"]}, version


def _alias_tree(tmp_path, *, shadowing_name, aliases):
    root = tmp_path / "cwd"
    for path in ("stable", "shared/B", ".agents/skills", ".pi"):
        (root / path).mkdir(parents=True)
    (root / "stable/SKILL.md").write_text(_md(shadowing_name))
    (root / "shared/B/zzzz.md").write_text("---\ndescription: d\n---\n")
    (root / ".pi/settings.json").write_text(json.dumps({"skills": ["../stable"]}))
    for alias in aliases:
        (root / ".agents/skills" / alias).symlink_to(root / "shared/B", target_is_directory=True)
    return root


def test_alias_that_a_project_skill_shadows_still_counts_for_the_names_of_the_file(tmp_path):
    # ``ax`` is the name of a project skill: the alias ``ax`` of the file is dropped as a shadowed skill. The file
    # also has the alias ``by``: two derived names for one file, whatever pi reads first.
    root = _alias_tree(tmp_path, shadowing_name="ax", aliases=("ax", "by"))
    with pytest.raises(ValueError, match="no 'name'"):
        pi_skill_dirs([], str(root))


def test_alias_that_a_project_skill_shadows_counts_whatever_the_order_of_the_links(tmp_path):
    root = _alias_tree(tmp_path, shadowing_name="by", aliases=("ax", "by"))
    with pytest.raises(ValueError, match="no 'name'"):
        pi_skill_dirs([], str(root))


def test_aliases_with_the_same_name_one_of_them_shadowed_are_kept(tmp_path):
    # The two links are called ``foo``: one derived name, so no refusal, even when a project skill shadows it.
    root = _alias_tree(tmp_path, shadowing_name="foo", aliases=())
    for parent in ("p", "q"):
        (root / ".agents/skills" / parent).mkdir()
        (root / ".agents/skills" / parent / "foo").symlink_to(root / "shared/B", target_is_directory=True)
    assert pi_skill_dirs([], str(root)) == [
        str(root / ".pi/../stable"),
        str(root / ".agents/skills/p/foo/zzzz.md"),
        str(root / ".agents/skills/q/foo/zzzz.md"),
    ]


def test_scenario_alias_dropped_as_a_duplicate_still_counts(tmp_path):
    scenario = tmp_path / "scenario" / "one"
    scenario.mkdir(parents=True)
    (scenario / "n.md").write_text("---\ndescription: d\n---\n")
    cwd = tmp_path / "cwd"
    (cwd / ".agents" / "skills").mkdir(parents=True)
    (cwd / ".agents" / "skills" / "other").symlink_to(scenario, target_is_directory=True)
    with pytest.raises(ValueError, match="no 'name'"):
        pi_skill_dirs([str(scenario)], str(cwd))


@pytest.mark.parametrize("root", [".pi/skills", ".agents/skills"])
def test_shadowed_alias_fixture_of_the_review_is_refused_and_pi_selects_only_the_first_skill(
    pi_reference, tmp_path, root
):
    """``00-first`` is called ``a``; the links ``a`` and ``b`` lead to one skill file with no ``name``.

    Pi 1.4.0 and 1.5.0 load only ``a``/``00-first`` (the alias ``a`` loses the name collision, and the file is not
    loaded again as ``b``). The paths that airedale would pass to pi (and the names of the file through its links
    differ, ``a`` and ``b``) are refused before the probe, so pi's choice does not depend on the order of the system.
    """
    files = {
        f"{root}/00-first/SKILL.md": _md("a"),
        "shared/SKILL.md": "---\ndescription: d\n---\n",
    }
    links = {f"{root}/a": "shared", f"{root}/b": "shared"}
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    with pytest.raises(ValueError, match="no 'name'"):
        pi_skill_dirs([], str(cwd))
    request = {"action": "skill_tree", "files": files, "links": links}
    for version, answer in pi_reference(request).items():
        assert answer["skills"] == [["a", f"{root}/00-first/SKILL.md"]], version
        assert answer["collisions"] == 1, version


@pytest.mark.parametrize("root", [".pi/skills", ".agents/skills"])
def test_aliases_with_an_explicit_target_name_are_kept(pi_reference, tmp_path, root):
    # Control: the file has a ``name``, so every link gives it the same name and the links are not refused.
    files = {
        f"{root}/00-first/SKILL.md": _md("first"),
        "shared/SKILL.md": _md("target"),
    }
    links = {f"{root}/a": "shared", f"{root}/b": "shared"}
    cwd = tmp_path / "cwd"
    _write_tree(cwd, files, links)
    loaded = set()
    for path in pi_skill_dirs([], str(cwd)):
        for name, skill_file in _pi_skills(Path(path)):
            loaded.add((name, os.path.relpath(os.path.realpath(skill_file), os.path.realpath(cwd))))
    for version, answer in pi_reference({"action": "skill_tree", "files": files, "links": links}).items():
        assert loaded == {tuple(pair) for pair in answer["skills"]}, version
