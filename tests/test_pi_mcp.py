# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the copy of pi's MCP configuration merge (``airedale.pi_mcp``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from airedale.harness.pi import PiRunner
from airedale.mcp import McpServerSpec
from airedale.pi_mcp import discover_pi_project_mcp_servers, merge_project_mcp_config, validate_server_config


class _Raw(str):
    """File content that is written as it is (not an entry)."""


SCENARIO = {"same": {"type": "stdio", "command": "scenario", "args": [], "exposure": "direct"}}
SCENARIO_UNDERSCORE = {"same_name": SCENARIO["same"]}

#: (label, scenario servers, name in the repo file, entry or raw text, BOM, scenario servers that change)
CASES: list[tuple[str, dict, str, Any, bool, list[str]]] = [
    ("same name, other command", SCENARIO, "same", {"command": "project"}, False, ["same"]),
    ("BOM", SCENARIO, "same", {"command": "project"}, True, ["same"]),
    ("http, unused env", SCENARIO, "same", {"url": "https://example.test/mcp", "env": {"unused": 1}}, False, ["same"]),
    ("stdio, unused headers", SCENARIO, "same", {"command": "project", "headers": {"unused": 1}}, False, ["same"]),
    ("disabled override", SCENARIO, "same", {"enabled": False}, False, ["same"]),
    ("disabled with command", SCENARIO, "same", {"command": "project", "enabled": False}, False, ["same"]),
    ("exposure override", SCENARIO, "same", {"exposure": "hidden"}, False, ["same"]),
    ("exposure alias", SCENARIO, "same", {"exposure": "codemode-deferred"}, False, ["same"]),
    ("empty override is a no-op", SCENARIO, "same", {}, False, []),
    ("override of the same value", SCENARIO, "same", {"exposure": "direct"}, False, []),
    ("override with an invalid value", SCENARIO, "same", {"exposure": "bogus"}, False, []),
    ("override with another key", SCENARIO, "same", {"args": []}, False, []),
    ("unknown type", SCENARIO, "same", {"type": "bogus"}, False, []),
    ("type without command", SCENARIO, "same", {"type": "stdio"}, False, []),
    ("sse", SCENARIO, "same", {"type": "sse", "url": "https://example.test"}, False, []),
    ("command is not a string", SCENARIO, "same", {"command": 5}, False, []),
    ("args are not strings", SCENARIO, "same", {"command": "x", "args": [1]}, False, []),
    ("url is not http", SCENARIO, "same", {"url": "ftp://example.test"}, False, []),
    ("null command", SCENARIO, "same", {"command": None}, False, []),
    ("entry is not an object", SCENARIO, "same", "text", False, []),
    ("normalized name", SCENARIO_UNDERSCORE, "same-name", {"command": "project"}, False, []),
    ("other server", SCENARIO, "other", {"command": "project"}, False, []),
    ("invalid json", SCENARIO, "same", _Raw("{not json"), False, []),
    ("timeout 0", SCENARIO, "same", {"command": "x", "timeout": 0}, False, []),
    ("timeout true", SCENARIO, "same", {"command": "x", "timeout": True}, False, []),
    ("timeout 5", SCENARIO, "same", {"command": "x", "timeout": 5}, False, ["same"]),
    ("auth in project file", SCENARIO, "same", {"url": "https://example.test", "auth": {"provider": "p"}}, False, []),
    (
        "oauth port as float",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"callbackPort": 8080.0}},
        False,
        ["same"],
    ),
    ("oauth port bad", SCENARIO, "same", {"url": "https://e.test", "oauth": {"callbackPort": 0}}, False, []),
    (
        "oauth cimd with client id",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"clientRegistration": "cimd", "clientId": "x"}},
        False,
        [],
    ),
    ("url with a tab in the scheme", SCENARIO, "same", {"url": "ht\ttps://example.invalid/mcp"}, False, ["same"]),
    ("url with a leading NUL", SCENARIO, "same", {"url": "\x00https://example.invalid/mcp"}, False, ["same"]),
    ("url with a newline", SCENARIO, "same", {"url": "http://exa\nmple.test/"}, False, ["same"]),
    (
        "oauth metadata url 127.1",
        SCENARIO,
        "same",
        {"url": "https://example.invalid/mcp", "oauth": {"authServerMetadataUrl": "http://127.1/meta"}},
        False,
        ["same"],
    ),
    (
        "oauth metadata url, not loopback",
        SCENARIO,
        "same",
        {"url": "https://example.invalid/mcp", "oauth": {"authServerMetadataUrl": "http://example.test/meta"}},
        False,
        [],
    ),
    (
        "oauth callback url, hex loopback",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"callbackUrl": "http://0x7f.1:8080/callback"}},
        False,
        ["same"],
    ),
    (
        "oauth cimd callback with dot segments",
        SCENARIO,
        "same",
        {
            "url": "https://e.test",
            "oauth": {"clientRegistration": "cimd", "callbackUrl": "http://localhost/x/../callback"},
        },
        False,
        ["same"],
    ),
    (
        "auth with http and a host that is not loopback",
        SCENARIO,
        "same",
        {"url": "http://e.test", "auth": {"provider": "p"}},
        False,
        [],
    ),
    ("url without a host", SCENARIO, "same", {"url": "http:///"}, False, []),
    ("url with a bad port", SCENARIO, "same", {"url": "http://e.test:99999/"}, False, []),
    ("type null with command", SCENARIO, "same", {"command": "project", "type": None}, False, []),
    ("type null with url", SCENARIO, "same", {"url": "https://e.test/mcp", "type": None}, False, []),
    ("type null alone", SCENARIO, "same", {"type": None}, False, []),
    ("type http with url", SCENARIO, "same", {"url": "https://e.test/mcp", "type": "http"}, False, ["same"]),
    (
        "type streamable-http with url",
        SCENARIO,
        "same",
        {"url": "https://e.test/mcp", "type": "streamable-http"},
        False,
        ["same"],
    ),
    ("type stdio with url", SCENARIO, "same", {"url": "https://e.test/mcp", "type": "stdio"}, False, []),
    ("identical definition", SCENARIO, "same", SCENARIO["same"], False, ["same"]),
]


def _write_project(cwd, name, entry, bom):
    (cwd / ".pi").mkdir(parents=True, exist_ok=True)
    text = str(entry) if isinstance(entry, _Raw) else json.dumps({"mcpServers": {name: entry}})
    (cwd / ".pi" / "mcp.json").write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode())


@pytest.mark.parametrize(("label", "scenario", "name", "entry", "bom", "changed"), CASES, ids=[c[0] for c in CASES])
def test_scenario_servers_that_the_project_file_changes(tmp_path, label, scenario, name, entry, bom, changed):
    _write_project(tmp_path, name, entry, bom)
    merge = merge_project_mcp_config(tmp_path / ".pi" / "mcp.json", scenario)
    assert merge.changed == changed


@pytest.mark.parametrize(("label", "scenario", "name", "entry", "bom", "changed"), CASES, ids=[c[0] for c in CASES])
def test_merge_matches_pi(pi_reference, tmp_path, label, scenario, name, entry, bom, changed):
    """The servers (and their config) after the merge are the ones that pi 1.4.0 and 1.5.0 load."""
    cwd, agent = tmp_path / "cwd", tmp_path / "agent"
    agent.mkdir()
    (agent / "mcp.json").write_text(json.dumps({"mcpServers": scenario}))
    _write_project(cwd, name, entry, bom)
    merge = merge_project_mcp_config(cwd / ".pi" / "mcp.json", scenario)
    ours = {name: (entry.scope, entry.config) for name, entry in merge.servers.items()}
    for version, answer in pi_reference({"action": "mcp", "agentDir": str(agent), "cwd": str(cwd)}).items():
        theirs = {server["name"]: (server["scope"], server["config"]) for server in answer["servers"]}
        assert ours == theirs, version
        assert bool(merge.errors) == bool(answer["errors"]), version


class TestRunnerCheck:
    def _runner(self, tmp_path, servers, name, entry, bom=False):
        _write_project(tmp_path, name, entry, bom)
        return PiRunner(cwd=tmp_path, mcp_servers=servers)

    def test_disabled_entry_with_the_name_of_a_scenario_server_raises(self, tmp_path):
        runner = self._runner(tmp_path, [McpServerSpec(name="same", command="scenario")], "same", {"enabled": False})
        with pytest.raises(ValueError, match="same"):
            runner._check_project_mcp_collisions()

    def test_bom_entry_raises(self, tmp_path):
        runner = self._runner(
            tmp_path, [McpServerSpec(name="same", command="scenario")], "same", {"command": "project"}, bom=True
        )
        with pytest.raises(ValueError, match="same"):
            runner._check_project_mcp_collisions()

    def test_entry_that_pi_rejects_does_not_raise(self, tmp_path):
        runner = self._runner(tmp_path, [McpServerSpec(name="same", command="scenario")], "same", {"type": "stdio"})
        runner._check_project_mcp_collisions()

    def test_normalized_name_does_not_raise_and_is_not_reported(self, tmp_path):
        servers = [McpServerSpec(name="same_name", command="scenario")]
        runner = self._runner(tmp_path, servers, "same-name", {"command": "project"})
        runner._check_project_mcp_collisions()
        # Pi rejects the repo entry: the scenario server stays, the repo one is not loaded.
        assert [server.name for server in runner._reported_mcp_servers()] == ["same_name"]

    def test_no_op_override_does_not_raise(self, tmp_path):
        runner = self._runner(tmp_path, [McpServerSpec(name="same", command="scenario")], "same", {})
        runner._check_project_mcp_collisions()


class TestTraceHeaders:
    """A repo entry that repeats a scenario server replaces it, and the replacement has no trace header."""

    TRACE = {"traceparent": "00-11111111111111111111111111111111-2222222222222222-01"}

    def _runner(self, tmp_path, entry):
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="same", url="https://example.invalid/mcp")])
        (tmp_path / ".pi").mkdir(exist_ok=True)
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"same": entry}}))
        return runner

    def _copy_of_untraced_config(self, tmp_path):
        runner = self._runner(tmp_path, {})
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": runner._mcp_config({})["mcpServers"]}))
        return runner

    def test_copy_of_the_untraced_config_raises_when_the_run_is_traced(self, tmp_path):
        runner = self._copy_of_untraced_config(tmp_path)
        with pytest.raises(ValueError, match="same"):
            runner._check_project_mcp_collisions(self.TRACE)

    def test_copy_of_the_untraced_config_raises_without_trace(self, tmp_path):
        # The entry is still a repo entry for pi, whatever the headers are.
        runner = self._copy_of_untraced_config(tmp_path)
        with pytest.raises(ValueError, match="same"):
            runner._check_project_mcp_collisions({})

    def test_no_op_override_is_allowed_when_the_run_is_traced(self, tmp_path):
        self._runner(tmp_path, {})._check_project_mcp_collisions(self.TRACE)

    def test_pi_drops_the_trace_header_of_a_replaced_server(self, pi_reference, tmp_path):
        """The real merge: the project entry wins and the global trace header is gone."""
        runner = self._copy_of_untraced_config(tmp_path)
        agent = tmp_path / "agent"
        agent.mkdir()
        (agent / "mcp.json").write_text(json.dumps(runner._mcp_config(self.TRACE)))
        answers = pi_reference({"action": "mcp", "agentDir": str(agent), "cwd": str(tmp_path)})
        for answer in answers.values():
            assert "headers" not in answer["servers"][0]["config"]


class TestReporting:
    def test_servers_with_fields_that_the_transport_ignores_are_reported(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "http": {"url": "https://example.test/mcp", "env": {"X": "secret"}},
                        "cli": {"command": "x", "headers": {"X": "secret"}},
                        "bogus": {"type": "bogus"},
                        "empty": {},
                        "off": {"command": "x", "enabled": False},
                    }
                }
            )
        )
        reported = discover_pi_project_mcp_servers(tmp_path)
        assert [(server.name, server.type) for server in reported] == [("http", "http"), ("cli", "stdio")]
        assert "secret" not in json.dumps([server.to_safe_dict() for server in reported])
        assert reported[0].env == {} and reported[1].headers == {}


class TestValidation:
    @pytest.mark.parametrize("name", ["a b", "a.b", "", "é", "a\n"])
    def test_invalid_names(self, name):
        assert isinstance(validate_server_config(name, {"command": "x"}), str)

    def test_exposure_aliases_are_resolved(self):
        config = validate_server_config("x", {"command": "x", "exposure": "codemode-deferred"})
        assert isinstance(config, dict)
        assert config["exposure"] == "codemode"

    def test_url_wins_over_command(self):
        config = validate_server_config("x", {"command": "x", "url": "https://example.test"})
        assert isinstance(config, dict)

    @pytest.mark.parametrize("url", ["http:", "http://", "https:///", "example.test", "http://[::1"])
    def test_urls_without_a_host(self, url):
        assert isinstance(validate_server_config("x", {"url": url}), str)
