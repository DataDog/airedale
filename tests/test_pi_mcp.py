# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the copy of pi's MCP configuration merge (``airedale.pi_mcp``)."""

from __future__ import annotations

import base64
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
    ("url, soft hyphen before IPv4", SCENARIO, "same", {"url": "https://\u00ad127.1/mcp"}, False, ["same"]),
    ("url, encoded soft hyphen before IPv4", SCENARIO, "same", {"url": "https://%C2%AD127.1/mcp"}, False, ["same"]),
    ("url, zero-width space in IPv4", SCENARIO, "same", {"url": "https://127\u200b.1/mcp"}, False, ["same"]),
    (
        "oauth metadata url, soft hyphen before 127.1",
        SCENARIO,
        "same",
        {"url": "https://example.invalid/mcp", "oauth": {"authServerMetadataUrl": "http://\u00ad127.1/meta"}},
        False,
        ["same"],
    ),
    (
        "oauth port above the range of a float",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"callbackPort": 10**400}},
        False,
        [],
    ),
    (
        "oauth port, a float that is infinity",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"callbackPort": 1e999}},
        False,
        [],
    ),
    (
        "oauth port, a negative huge integer",
        SCENARIO,
        "same",
        {"url": "https://e.test", "oauth": {"callbackPort": -(10**400)}},
        False,
        [],
    ),
    ("server name of 5000 digits", SCENARIO, "9" * 5000, {"command": "project"}, False, []),
    ("server name that is an array index", SCENARIO, "4294967294", {"command": "project"}, False, []),
    ("server name above the array indexes", SCENARIO, "4294967295", {"command": "project"}, False, []),
    (
        "file with an integer of 5000 digits",
        SCENARIO,
        "same",
        _Raw('{"other": ' + "9" * 5000 + ', "mcpServers": {"same": {"command": "project"}}}'),
        False,
        ["same"],
    ),
    (
        "file with a negative integer of 5000 digits",
        SCENARIO,
        "same",
        _Raw('{"other": [-' + "9" * 5000 + '], "mcpServers": {"same": {"command": "project"}}}'),
        False,
        ["same"],
    ),
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
    _write_project(tmp_path, name, entry, bom)
    project_file = tmp_path / ".pi" / "mcp.json"
    merge = merge_project_mcp_config(project_file, scenario)
    ours = {name: (entry.scope, entry.config) for name, entry in merge.servers.items()}
    request = {
        "action": "mcp",
        "global_file": {"mcpServers": scenario},
        "project_b64": base64.b64encode(project_file.read_bytes()).decode(),
    }
    for version, answer in pi_reference(request).items():
        theirs = {server["name"]: (server["scope"], server["config"]) for server in answer["servers"]}
        assert ours == theirs, version
        assert bool(merge.errors) == bool(answer["error_count"]), version


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

    @pytest.mark.parametrize("url", ["https://%C2%AD127.1/mcp", "https://127\u200b.1/mcp"])
    def test_uncertain_host_of_a_same_name_entry_stops_the_run(self, tmp_path, url):
        # Pi removes the soft hyphen and the zero-width space from the host: the entry replaces the server.
        runner = self._runner(tmp_path, [McpServerSpec(name="same", command="scenario")], "same", {"url": url})
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
        request = {
            "action": "mcp",
            "global_file": runner._mcp_config(self.TRACE),
            "project_b64": base64.b64encode((tmp_path / ".pi" / "mcp.json").read_bytes()).decode(),
        }
        for answer in pi_reference(request).values():
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


def test_server_names_are_ordered_like_object_entries():
    # Array indexes come first, in number order; the others keep their order. A name with more than 10 digits (or
    # of thousands of digits) is a plain name and does not need a conversion to an integer.
    from airedale.pi_mcp_config import _js_entries

    names = ["b", "10", "9" * 5000, "2", "4294967295", "4294967294", "a"]
    assert [name for name, _ in _js_entries(dict.fromkeys(names))] == [
        "2",
        "10",
        "4294967294",
        "b",
        "9" * 5000,
        "4294967295",
        "a",
    ]


#: URLs of a server and of its OAuth settings: pi 1.4.0 and 1.5.0 refuse the entry when one of them is not a valid URL.
URL_ENTRIES = [
    ("http://[::1]/mcp", True),
    ("http://[::1%lo0]/mcp", False),
    ("http://[::1%25lo0]/mcp", False),
    ("http://[fe80::1%eth0]:8080/mcp", False),
    ("http://[%]/mcp", False),
    ("http://[::1%]/mcp", False),
    ("https://example.test/mcp", True),
    ("http://[not-an-ip]/mcp", False),
    ("http://[1::2::3]/mcp", False),
    ("http://a b/mcp", False),
    ("ftp://example.test/mcp", False),
]


def _url_entries():
    for url, _ in URL_ENTRIES:
        yield f"server {url}", {"url": url}
        yield f"callbackUrl {url}", {"url": "https://e.test", "oauth": {"callbackUrl": url}}
        yield f"authServerMetadataUrl {url}", {"url": "https://e.test", "oauth": {"authServerMetadataUrl": url}}


@pytest.mark.parametrize("entries", [dict(_url_entries())], ids=["urls"])
def test_exact_validation_never_accepts_what_pi_refuses(pi_reference, entries):
    """An exact accept (strict check of the scenario) is an accept of pi; the guard never rejects what pi accepts."""
    from airedale.pi_mcp import validate_server_config

    names = {f"s{index}": label for index, label in enumerate(entries)}
    request = {
        "action": "mcp",
        "global_file": {"mcpServers": {name: entries[label] for name, label in names.items()}},
        "project_b64": None,
    }
    for version, answer in pi_reference(request).items():
        accepted_by_pi = {names[server["name"]] for server in answer["servers"]}
        for label, config in entries.items():
            strict = not isinstance(validate_server_config("x", config, exact=True), str)
            guard = not isinstance(validate_server_config("x", config), str)
            if strict:
                assert label in accepted_by_pi, (version, label, "strict check accepts what pi refuses")
            if label in accepted_by_pi:
                assert guard, (version, label, "guard refuses what pi accepts")
