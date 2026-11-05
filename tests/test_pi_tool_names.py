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
from airedale.mcp import McpServerSpec, pi_mcp_tool_name, pi_mcp_tool_names, pi_possible_tool_name_collisions
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
    assert "mcp__a__b__c" in caplog.text
    assert "connects second" in caplog.text
    # The names are not changed.
    assert [definition["name"] for definition in definitions] == ["mcp__a__b__c", "mcp__a__b__c"]


def test_no_warning_without_a_collision(caplog):
    servers = [McpServerSpec(name="a", command="x"), McpServerSpec(name="b", command="y")]
    metadata = {("a", "one"): McpToolMetadata(), ("b", "one"): McpToolMetadata()}
    with caplog.at_level(logging.WARNING, logger="airedale.harness.base"):
        _mcp_tool_definitions(servers, sdk_name="pi", mcp_tool_metadata=metadata)
    assert "connects second" not in caplog.text
