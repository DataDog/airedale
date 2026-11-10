# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the pi names of MCP tools, and the collisions between servers."""

from __future__ import annotations

import asyncio
import itertools
import logging

import pytest

from airedale.harness.base import McpToolMetadata, _mcp_tool_definitions
from airedale.harness.pi import PiRunner
from airedale.mcp import (
    McpServerSpec,
    PiToolNameOrderError,
    pi_mcp_tool_name,
    pi_mcp_tool_names,
    pi_possible_tool_name_collisions,
    pi_registered_tool_names,
)
from airedale.types import ModelSpec

#: Complete catalogs: (server, tools) in a registration order.
CATALOGS = [
    pytest.param([("a", ["b__c"]), ("a__b", ["c"])], id="sanitized-separators"),
    pytest.param([("a", ["b😀c"]), ("a__b", ["c"])], id="emoji-two-underscores"),
    pytest.param([("a", ["b😀c"]), ("a__b", ["c"]), ("a_b", ["c"])], id="three-servers"),
    pytest.param([("a", ["x"]), ("b", ["x"])], id="no-collision"),
    pytest.param([("a", ["b__c", "b--c"]), ("a__b", ["c"])], id="hidden-tool-removes-the-collision"),
    pytest.param([("a", ["get-doc", "get_doc"]), ("b", ["get_doc"])], id="collision-inside-one-server"),
    pytest.param([("s", ["😀" * 40]), ("t", ["x"])], id="long-emoji-name"),
    pytest.param([("s", ["t" * 80]), ("u", ["t" * 80])], id="long-names"),
    pytest.param([("a😀", ["x"]), ("a__", ["x"])], id="emoji-in-server-name"),
    pytest.param([("a", ["é", "_"]), ("b", ["é"])], id="non-ascii"),
    pytest.param([("a", ["b__c", "b--c"]), ("a__b", ["c_01b8a75b"])], id="forced-hash-names-collide"),
    pytest.param([("a", ["x-y", "x_y"]), ("a__b", ["c"]), ("a_", ["b__c"])], id="three-servers-with-hashes"),
]


@pytest.mark.parametrize("catalog", CATALOGS)
def test_names_match_pi(pi_reference, catalog):
    """One server: the names are the ones of pi 1.4.0 and 1.5.0."""
    for server, tools in catalog:
        answers = pi_reference({"action": "names", "servers": [[server, tools]]})
        ours = pi_mcp_tool_names(server, tools)
        for version, answer in answers.items():
            assert {item["tool"]: item["name"] for item in answer} == ours, version


@pytest.mark.parametrize("catalog", CATALOGS)
def test_possible_collisions_match_pi(pi_reference, catalog):
    """With complete catalogs, the listed collisions are those that change a name in pi, in some order."""
    expected = {(server, tool): pi_mcp_tool_names(server, tools)[tool] for server, tools in catalog for tool in tools}
    changed = False
    for order in itertools.permutations(catalog):
        for answer in pi_reference({"action": "names", "servers": [[s, t] for s, t in order]}).values():
            for item in answer:
                if item["name"] != expected[(item["server"], item["tool"])]:
                    changed = True
    collisions = pi_possible_tool_name_collisions(dict(catalog))
    assert bool(collisions) == changed


@pytest.mark.parametrize("catalog", CATALOGS)
def test_registered_names_match_pi_in_every_order(pi_reference, catalog):
    """The names that pi gives, with each connection order: same as the registration loop of pi."""
    for order in itertools.permutations(catalog):
        ours = pi_registered_tool_names(list(order))
        for version, answer in pi_reference({"action": "names", "servers": [[s, t] for s, t in order]}).items():
            assert {(item["server"], item["tool"]): item["name"] for item in answer} == ours, (version, order)


def test_names_that_hash_suffixes_make_collide():
    # Pi hashes the two tools of ``a`` (same name after sanitizing); the hashed name of the first is the name of the tool
    # of ``a__b``. If ``a__b`` connects first, two tools get the same name.
    catalog = {"a": ["b__c", "b--c"], "a__b": ["c_01b8a75b"]}
    first = pi_registered_tool_names([("a__b", catalog["a__b"]), ("a", catalog["a"])])
    assert first[("a__b", "c_01b8a75b")] == first[("a", "b__c")] == "mcp__a__b__c_01b8a75b"
    collisions = pi_possible_tool_name_collisions(catalog)
    assert "mcp__a__b__c_01b8a75b" in collisions
    assert {pair for pairs in collisions.values() for pair in pairs} >= {("a", "b__c"), ("a__b", "c_01b8a75b")}


def test_many_independent_servers_do_not_change_the_result():
    catalog = {f"s{index}": [f"t{index}"] for index in range(8)}
    catalog["a"] = ["b__c"]
    catalog["a__b"] = ["c"]
    assert "mcp__a__b__c" in pi_possible_tool_name_collisions(catalog)
    assert pi_possible_tool_name_collisions({f"s{index}": [f"t{index}"] for index in range(8)}) == {}


def test_a_hash_suffix_of_a_first_collision_can_collide_with_a_third_server():
    # ``a`` and ``a__b`` collide on ``mcp__a__b__c``; when ``a`` loses, its name has a hash suffix, and the tool of
    # the third server has exactly that name. Only a check of the three servers together sees it.
    hashed = pi_registered_tool_names([("a__b", ["c"]), ("a", ["b__c"])])[("a", "b__c")]
    assert hashed != "mcp__a__b__c"
    third_tool = hashed.removeprefix("mcp__a__b__")  # (the server "a_-b" is written "a__b" by pi)
    assert hashed == "mcp__a__b__" + third_tool
    catalog = {"a": ["b__c"], "a__b": ["c"], "a_-b": [third_tool]}
    fillers = {f"s{index}": [f"t{index}"] for index in range(6)}  # 9 servers in all: more than 6.
    collisions = pi_possible_tool_name_collisions({**fillers, **catalog})
    assert hashed in collisions
    assert {("a", "b__c"), ("a_-b", third_tool)} <= set(collisions[hashed])


def _seven_servers_that_share_a_name():
    # Every ``("a" + "_" * i, "_" * (6 - i) + "x")`` is called ``mcp__a________x``.
    return {"a" + "_" * i: ["_" * (6 - i) + "x"] for i in range(7)}


def test_more_than_six_servers_that_can_share_a_name_are_rejected():
    catalog = _seven_servers_that_share_a_name()
    assert len({pi_mcp_tool_name(server, tools[0]) for server, tools in catalog.items()}) == 1
    with pytest.raises(PiToolNameOrderError, match="7 MCP servers"):
        pi_possible_tool_name_collisions(catalog)
    # Six of them are checked with every order.
    six = dict(list(catalog.items())[:6])
    assert pi_possible_tool_name_collisions(six)


def test_the_runner_refuses_a_scenario_with_too_many_servers_that_share_a_name():
    from airedale.harness.pi import PiRunner

    servers = [
        McpServerSpec(name=name, command="x", tool_names=tools)
        for name, tools in _seven_servers_that_share_a_name().items()
    ]
    with pytest.raises(ValueError, match="7 MCP servers"):
        PiRunner(cwd=".", mcp_servers=servers)._check_tool_name_collisions()


def test_the_warning_after_the_probe_says_that_the_order_cannot_be_checked(caplog):
    servers = [McpServerSpec(name=name, command="x") for name in _seven_servers_that_share_a_name()]
    metadata = {(name, tools[0]): McpToolMetadata() for name, tools in _seven_servers_that_share_a_name().items()}
    with caplog.at_level(logging.WARNING, logger="airedale.harness.base"):
        _mcp_tool_definitions(servers, sdk_name="pi", mcp_tool_metadata=metadata)  # (A warning, not an error.)
    assert "7 MCP servers can take the same pi tool names" in caplog.text


def test_non_bmp_characters_count_twice():
    # Pi works with UTF-16: an emoji is two code units, so two underscores.
    assert pi_mcp_tool_name("a", "b😀c") == "mcp__a__b__c"
    # 33 emoji are 66 code units: the name is longer than 64 and gets a hash.
    assert len(pi_mcp_tool_name("s", "😀" * 40)) == 64
    assert pi_mcp_tool_name("s", "😀" * 20).endswith("_" * 40)


def test_collision_between_servers_is_listed():
    collisions = pi_possible_tool_name_collisions({"a": ["b__c"], "a__b": ["c"]})
    assert collisions == {"mcp__a__b__c": [("a", "b__c"), ("a__b", "c")]}


def test_collision_inside_a_server_is_not_listed():
    # Both tools of ``a`` get a suffix, whatever the order: the name of ``b`` is free.
    assert pi_possible_tool_name_collisions({"a": ["x-y", "x_y"], "b": ["x_y"]}) == {}


class TestPreflight:
    def test_allow_lists_that_can_collide_raise(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("b__c",)),
            McpServerSpec(name="a__b", command="y", tool_names=("c",)),
        ]
        with pytest.raises(
            ValueError, match=r"(?s)Possible pi tool name collision.*mcp__a__b__c.*a/b__c.*a__b/c.*allow-list"
        ):
            PiRunner(cwd=tmp_path, mcp_servers=servers)._check_tool_name_collisions()

    def test_emoji_allow_lists_raise(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("b😀c",)),
            McpServerSpec(name="a__b", command="y", tool_names=("c",)),
        ]
        with pytest.raises(ValueError, match="Possible"):
            PiRunner(cwd=tmp_path, mcp_servers=servers)._check_tool_name_collisions()

    def test_allow_lists_whose_hash_names_collide_raise(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("b__c", "b--c")),
            McpServerSpec(name="a__b", command="y", tool_names=("c_01b8a75b",)),
        ]
        with pytest.raises(ValueError, match=r"Possible pi tool name collision.*mcp__a__b__c_01b8a75b"):
            PiRunner(cwd=tmp_path, mcp_servers=servers)._check_tool_name_collisions()

    def test_servers_without_an_allow_list_are_not_checked(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("b__c",)),
            McpServerSpec(name="a__b", command="y"),
        ]
        PiRunner(cwd=tmp_path, mcp_servers=servers)._check_tool_name_collisions()

    def test_names_that_differ_do_not_raise(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("one",)),
            McpServerSpec(name="b", command="y", tool_names=("one",)),
        ]
        PiRunner(cwd=tmp_path, mcp_servers=servers)._check_tool_name_collisions()

    def test_run_agent_stops_before_pi_starts(self, tmp_path):
        servers = [
            McpServerSpec(name="a", command="x", tool_names=("b__c",)),
            McpServerSpec(name="a__b", command="y", tool_names=("c",)),
        ]
        runner = PiRunner(cwd=tmp_path, mcp_servers=servers)
        with pytest.raises(ValueError, match="Possible"):
            asyncio.run(
                runner._run_agent(
                    model=ModelSpec.parse("anthropic/claude-x"),
                    system_prompt="s",
                    user_prompt="u",
                    trace_headers={},
                    progress=None,
                )
            )


def test_probed_tools_that_can_collide_log_a_warning(caplog):
    servers = [McpServerSpec(name="a", command="x"), McpServerSpec(name="a__b", command="y")]
    metadata = {("a", "b__c"): McpToolMetadata(), ("a__b", "c"): McpToolMetadata()}
    with caplog.at_level(logging.WARNING, logger="airedale.harness.base"):
        definitions = _mcp_tool_definitions(servers, sdk_name="pi", mcp_tool_metadata=metadata)
    # Fixed text: the configured server names and the number of collisions, no tool name.
    assert "MCP servers a, a__b (possible collisions: 1)" in caplog.text
    assert "mcp__a__b__c" not in caplog.text and "b__c" not in caplog.text
    assert "connects second" in caplog.text
    # The names are not changed.
    assert [definition["name"] for definition in definitions] == ["mcp__a__b__c", "mcp__a__b__c"]


def test_the_collision_warning_has_no_tool_name_that_a_server_advertised(caplog):
    secret = "sk-ADVERTISED-SECRET-0123456789"
    servers = [McpServerSpec(name="a", command="x"), McpServerSpec(name="a__b", command="y")]
    metadata = {("a", f"b__{secret}"): McpToolMetadata(), ("a__b", secret): McpToolMetadata()}
    with caplog.at_level(logging.DEBUG, logger="airedale.harness.base"):
        _mcp_tool_definitions(servers, sdk_name="pi", mcp_tool_metadata=metadata)
    assert "connects second" in caplog.text  # The collision was found, so the warning ran.
    assert "SECRET" not in caplog.text and "ADVERTISED" not in caplog.text


def test_no_warning_without_a_collision(caplog):
    servers = [McpServerSpec(name="a", command="x"), McpServerSpec(name="b", command="y")]
    metadata = {("a", "one"): McpToolMetadata(), ("b", "one"): McpToolMetadata()}
    with caplog.at_level(logging.WARNING, logger="airedale.harness.base"):
        _mcp_tool_definitions(servers, sdk_name="pi", mcp_tool_metadata=metadata)
    assert "connects second" not in caplog.text
