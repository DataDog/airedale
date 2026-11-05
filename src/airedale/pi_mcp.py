# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""How pi reads and merges MCP server configuration.

Airedale gives its scenario MCP servers to pi in the ``mcp.json`` of a private
agent directory (the "global" servers of pi). Pi also reads the repo file
``<cwd>/.pi/mcp.json`` and merges it over them. This module copies that merge
and the validation of each entry (``extensions/mcp/config.js`` and
``core/mcp-servers.js`` of pi 1.4.0 and 1.5.0), so that Airedale knows:

* which scenario servers a repo file would replace or change (the run must fail:
  the scenario must win, as for the other harnesses, and pi cannot do that);
* which servers the repo file adds (for the span metadata and the agent
  manifest).

Where pi 1.4.0 and 1.5.0 differ, the more permissive rule is used (an entry is
valid when one of the two versions accepts it): a wrong "valid" only stops a run
that could have run, and a wrong "invalid" would let a repo file change a
scenario server.

Differences from pi that remain: URLs are checked with a simple parser, not the
WHATWG one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import SplitResult, urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from airedale.mcp import McpServerSpec

logger = logging.getLogger(__name__)

#: Keys that a repo entry without ``command``, ``url`` or ``type`` can set.
OVERRIDE_KEYS = ("enabled", "exposure", "toolExposure")
_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_EXPOSURES = ("codemode", "deferred", "direct", "hidden")
#: Older exposure names that pi accepts and replaces.
_EXPOSURE_ALIASES = {"codemode-deferred": "codemode"}
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
_ARRAY_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
_MAX_ARRAY_INDEX = 2**32 - 2


@dataclass
class PiMcpEntry:
    """One server of the merged configuration."""

    config: dict[str, Any]
    #: ``"global"`` (a scenario server) or ``"project"`` (defined by the repo file).
    scope: str


@dataclass
class PiMcpMerge:
    """Result of the merge of a repo ``.pi/mcp.json`` over the scenario servers."""

    servers: dict[str, PiMcpEntry] = field(default_factory=dict)
    #: Pi's diagnostics for the repo file (entries that pi does not load).
    errors: list[str] = field(default_factory=list)
    #: Scenario servers that the repo file replaces or changes.
    changed: list[str] = field(default_factory=list)

    @property
    def project_servers(self) -> dict[str, dict[str, Any]]:
        """Servers defined by the repo file (not the scenario servers), by name."""
        return {name: entry.config for name, entry in self.servers.items() if entry.scope == "project"}


def mcp_namespace(server: str) -> str:
    """Namespace of the tools of a server (pi: ``mcp__<server>`` with ``-`` as ``_``)."""
    return "mcp__" + server.replace("-", "_")


def merge_project_mcp_config(project_file: Path, scenario: Mapping[str, Mapping[str, Any]]) -> PiMcpMerge:
    """Merge the repo file ``project_file`` over the scenario servers, as pi does.

    ``scenario`` has the ``mcp.json`` entry of each scenario server. A missing
    file gives the scenario servers unchanged.
    """
    merge = PiMcpMerge(servers={name: PiMcpEntry(dict(config), "global") for name, config in scenario.items()})
    before = {name: dict(config) for name, config in scenario.items()}
    parsed = _read_json_object(project_file, merge.errors)
    if parsed is not None:
        _apply_project_entries(project_file, parsed, merge)
    merge.changed = [name for name, config in before.items() if merge.servers[name].config != config]
    return merge


def discover_pi_project_mcp_servers(
    cwd: str | Path, scenario: Mapping[str, Mapping[str, Any]] | None = None
) -> list[McpServerSpec]:
    """Return the servers that the repo file ``<cwd>/.pi/mcp.json`` adds, for reporting.

    Pi reads this file itself (the runner does not give these servers to pi).
    The specs describe the project servers in the span metadata and the agent
    manifest. Only the servers that pi loads and starts are listed: an entry
    that pi rejects (invalid, or a name that differs from a scenario server only
    in ``-``/``_``) and an entry with ``"enabled": false`` are left out. The repo
    file can hold credentials, and these specs go to telemetry, so ``env``
    values, ``args`` and everything in ``url`` after the host and port are
    redacted (headers are redacted by :meth:`McpServerSpec.to_safe_dict`).
    """
    from airedale.mcp import REDACTED, McpServerSpec, url_without_secrets

    merge = merge_project_mcp_config(Path(cwd) / ".pi" / "mcp.json", scenario or {})
    specs = []
    for name, config in merge.project_servers.items():
        if config.get("enabled") is False:
            continue
        if _is_http(config):
            specs.append(
                McpServerSpec(
                    name=name,
                    type="http",
                    url=url_without_secrets(config["url"]),
                    headers=dict.fromkeys(config.get("headers") or {}, REDACTED),
                )
            )
        else:
            specs.append(
                McpServerSpec(
                    name=name,
                    type="stdio",
                    command=config["command"],
                    args=tuple(REDACTED for _ in config.get("args") or []),
                    env=dict.fromkeys(config.get("env") or {}, REDACTED),
                )
            )
    return specs


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _read_json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    """Read a pi ``mcp.json`` (BOM allowed). ``None``: missing, or pi rejects the file."""
    if not path.is_file():
        return None
    try:
        # Node decodes invalid UTF-8 with U+FFFD and strips one BOM.
        text = path.read_bytes().decode("utf-8", errors="replace").removeprefix("\ufeff")
        parsed = json.loads(text, parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")
        logger.warning("Ignoring unreadable project MCP config at %s", path)
        return None
    servers = parsed.get("mcpServers") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or ("mcpServers" in parsed and not isinstance(servers, dict)):
        errors.append(f'{path}: expected an object with an "mcpServers" object')
        return None
    return parsed


def _reject_constant(name: str) -> Any:
    # JSON.parse rejects NaN and Infinity; Python's parser accepts them.
    raise ValueError(f"Unexpected token {name} in JSON")


def _js_entries(mapping: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Entries in the order of ``Object.entries``: array indexes first, in number order."""
    indexes = [key for key in mapping if _ARRAY_INDEX_RE.fullmatch(key) and int(key) <= _MAX_ARRAY_INDEX]
    indexes.sort(key=int)
    index_set = set(indexes)
    rest = [key for key in mapping if key not in index_set]
    return [(key, mapping[key]) for key in [*indexes, *rest]]


def _apply_project_entries(path: Path, parsed: dict[str, Any], merge: PiMcpMerge) -> None:
    """Apply the entries of a repo file (pi: ``readConfigFile`` with project scope)."""
    servers, errors = merge.servers, merge.errors
    for name, value in _js_entries(parsed.get("mcpServers") or {}):
        if isinstance(value, dict) and _is_override(value):
            base = servers.get(name)
            extra = [key for key in value if key not in OVERRIDE_KEYS]
            if base is None:
                errors.append(f'{path}: server "{name}" needs "command" or "url", or a global server to override')
            elif extra:
                errors.append(f'{path}: server "{name}": an override can only set {", ".join(OVERRIDE_KEYS)}')
            else:
                config = validate_server_config(name, {**base.config, **value})
                if isinstance(config, str):
                    errors.append(f"{path}: {config}")
                else:
                    servers[name] = PiMcpEntry(config, base.scope)
            continue
        config = validate_server_config(name, value)
        if isinstance(config, str):
            errors.append(f"{path}: {config}")
            continue
        # Names that differ only in "-" and "_" would share a namespace.
        clash = next(
            (other for other in servers if other != name and mcp_namespace(other) == mcp_namespace(name)), None
        )
        if clash is not None:
            errors.append(f'{path}: server "{name}" conflicts with "{clash}"')
            continue
        if _is_http(config) and config.get("auth"):
            errors.append(f'{path}: server "{name}": auth is only allowed in the global mcp.json')
            continue
        servers[name] = PiMcpEntry(config, "project")


def _is_override(value: Mapping[str, Any]) -> bool:
    """An entry without ``command``, ``url`` and ``type`` (a ``null`` value counts as set)."""
    return "command" not in value and "url" not in value and "type" not in value


# --------------------------------------------------------------------------- #
# Validation (pi: ``validateMcpServerConfig``)
# --------------------------------------------------------------------------- #
def validate_server_config(name: str, raw: Any) -> dict[str, Any] | str:
    """Validate one server entry. Return the entry with exposure aliases resolved, or an error."""
    if not _SERVER_NAME_RE.fullmatch(name):
        return f'invalid server name "{name}" (use letters, digits, "_" and "-")'
    if not isinstance(raw, dict):
        return f'server "{name}" must be an object'
    value = _resolve_exposure_aliases(raw)
    server_type = value.get("type")
    exposure = value.get("exposure")
    exposures = ", ".join(f'"{item}"' for item in _EXPOSURES)
    if "exposure" in value and not _is_exposure(exposure):
        return f'server "{name}": exposure must be one of {exposures}'
    if "toolExposure" in value:
        tool_exposure = value["toolExposure"]
        if not isinstance(tool_exposure, dict):
            return f'server "{name}": toolExposure must map tool names to exposures'
        for tool, entry in tool_exposure.items():
            if not _is_exposure(entry):
                return f'server "{name}": toolExposure "{tool}" must be one of {exposures}'
    if "enabled" in value and not isinstance(value["enabled"], bool):
        return f'server "{name}": enabled must be a boolean'
    if "description" in value and not isinstance(value["description"], str):
        return f'server "{name}": description must be a string'
    if "timeout" in value:
        timeout = value["timeout"]
        if not _is_number(timeout) or not timeout > 0:
            return f'server "{name}": timeout must be a positive number of seconds'
    if server_type == "sse":
        return f'server "{name}": legacy SSE transport is not supported; use the streamable HTTP URL'
    if isinstance(value.get("url"), str) and server_type in (None, "http", "streamable-http"):
        url = _parse_url(value["url"])
        if url is None or url.scheme not in ("http", "https"):
            return f'server "{name}": url must be an http or https URL'
        if "headers" in value and not _is_string_record(value["headers"]):
            return f'server "{name}": headers must map names to strings'
        oauth_error = _validate_oauth(value.get("oauth"), "oauth" in value)
        if oauth_error:
            return f'server "{name}": {oauth_error}'
        if "auth" in value:
            auth = value["auth"]
            if not isinstance(auth, dict) or not isinstance(auth.get("provider"), str) or not auth["provider"]:
                return f'server "{name}": auth.provider must be a provider name'
            if url.scheme != "https" and _host(url) not in _LOOPBACK_HOSTS:
                return f'server "{name}": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]'
        return value
    if isinstance(value.get("command"), str) and server_type in (None, "stdio"):
        args = value.get("args")
        if "args" in value and not (isinstance(args, list) and all(isinstance(arg, str) for arg in args)):
            return f'server "{name}": args must be an array of strings'
        if "env" in value and not _is_string_record(value["env"]):
            return f'server "{name}": env must map names to strings'
        if "cwd" in value and not isinstance(value["cwd"], str):
            return f'server "{name}": cwd must be a string'
        return value
    return f'server "{name}" needs either "command" (stdio) or "url" (streamable HTTP)'


def _is_http(config: Mapping[str, Any]) -> bool:
    """The branch of ``validate_server_config`` that a valid entry took (url first, as pi)."""
    return isinstance(config.get("url"), str) and config.get("type") in (None, "http", "streamable-http")


def _validate_oauth(value: Any, present: bool) -> str | None:
    if not present:
        return None
    if not isinstance(value, dict):
        return "oauth must be an object"
    if "clientId" in value and not isinstance(value["clientId"], str):
        return "oauth.clientId must be a string"
    if "clientSecret" in value and not isinstance(value["clientSecret"], str):
        return "oauth.clientSecret must be a string"
    port: Any = value.get("callbackPort")
    if "callbackPort" in value and not (_is_integer(port) and 1 <= float(port) <= 65535):
        return "oauth.callbackPort must be a port number"
    if "callbackUrl" in value:
        callback = value["callbackUrl"]
        if not isinstance(callback, str) or not _is_loopback_redirect_uri(callback):
            return "oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
        url_port = _port(_parse_url(callback))
        if url_port and "callbackPort" in value and int(url_port) != port:
            return "oauth.callbackUrl and oauth.callbackPort name different ports"
    if "scope" in value and not isinstance(value["scope"], str):
        return "oauth.scope must be a string"
    if "clientName" in value and (not isinstance(value["clientName"], str) or not value["clientName"].strip()):
        return "oauth.clientName must be a non-empty string"
    registration = value.get("clientRegistration")
    if "clientRegistration" in value and registration not in ("auto", "dcr"):
        if registration != "cimd":
            return 'oauth.clientRegistration must be "auto", "dcr", or "cimd"'
        if "clientId" in value or "clientName" in value:
            return 'oauth.clientRegistration "cimd" cannot be combined with oauth.clientId or oauth.clientName'
        callback_url = value.get("callbackUrl")
        if isinstance(callback_url, str):
            parsed = _parse_url(callback_url)
            if parsed is None or _host(parsed) == "[::1]" or parsed.path != "/callback":
                return (
                    'oauth.clientRegistration "cimd" requires oauth.callbackUrl on localhost or 127.0.0.1 '
                    "with path /callback"
                )
    if "authServerMetadataUrl" in value:
        metadata_url = value["authServerMetadataUrl"]
        url = _parse_url(metadata_url) if isinstance(metadata_url, str) else None
        if url is None or not (url.scheme == "https" or (url.scheme == "http" and _host(url) in _LOOPBACK_HOSTS)):
            return "oauth.authServerMetadataUrl must be an https URL, or http on localhost, 127.0.0.1, or [::1]"
    return None


def _resolve_exposure_aliases(value: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(value)
    if "exposure" in resolved:
        resolved["exposure"] = _alias(resolved["exposure"])
    tool_exposure = resolved.get("toolExposure")
    if isinstance(tool_exposure, dict):
        resolved["toolExposure"] = {tool: _alias(entry) for tool, entry in tool_exposure.items()}
    return resolved


def _alias(value: Any) -> Any:
    return _EXPOSURE_ALIASES.get(value, value) if isinstance(value, str) else value


def _is_exposure(value: Any) -> bool:
    return isinstance(value, str) and value in _EXPOSURES


def _is_string_record(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(entry, str) for entry in value.values())


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    """``Number.isInteger``: ``1.0`` is an integer, ``true`` is not a number."""
    return _is_number(value) and float(value).is_integer()


_URL_RE = re.compile(r"^[\x00-\x20]*([A-Za-z][A-Za-z0-9+.-]*):[/\\]*([^/\\?#]+)")


def _parse_url(value: str) -> SplitResult | None:
    """Parse ``value`` like ``URL.canParse`` (simplified): ``None`` when it is not a URL with a host."""
    match = _URL_RE.match(value)
    if match is None:
        return None
    scheme, authority = match.group(1).lower(), match.group(2)
    if scheme not in ("http", "https"):
        return SplitResult(scheme, authority, "", "", "")
    try:
        # Repeat the parse with a normal prefix, so that "http:host" and "http:\\host" work.
        rest = value.strip()[len(match.group(1)) + 1 :].lstrip("/\\")
        parts = urlsplit(f"{scheme}://{rest.replace(chr(92), '/')}")
        _ = parts.port  # Raises ValueError for an invalid port.
    except ValueError:
        return None
    return parts if parts.hostname else None


def _host(url: SplitResult | None) -> str:
    """``URL.hostname``: IPv6 addresses are in brackets."""
    host = (url.hostname or "") if url is not None else ""
    return f"[{host}]" if ":" in host else host


def _port(url: SplitResult | None) -> str:
    """``URL.port``: empty for the default port of the scheme."""
    if url is None or url.port is None or (url.scheme, url.port) in (("http", 80), ("https", 443)):
        return ""
    return str(url.port)


def _is_loopback_redirect_uri(value: str) -> bool:
    url = _parse_url(value)
    return (
        url is not None
        and url.scheme == "http"
        and _host(url) in _LOOPBACK_HOSTS
        and not url.query
        and not url.fragment
    )
