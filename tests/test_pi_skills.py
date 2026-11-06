# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the skill frontmatter parser: it must read what pi reads."""

from __future__ import annotations

import json

import pytest

from airedale.pi_frontmatter import UnsupportedFrontmatterError
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
