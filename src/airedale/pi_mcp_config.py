# Python translation of parts of pi (https://github.com/earendil-works/pi):
#   packages/coding-agent/src/extensions/mcp/config.ts  (readConfigFile: merge of the project mcp.json)
#   packages/coding-agent/src/core/mcp-servers.ts       (validateMcpServerConfig, mcpNamespace)
# as published in @earendil-works/pi-coding-agent 1.4.0 and 1.5.0.
#
# MIT License
#
# Copyright (c) 2025 Mario Zechner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""How pi reads and validates MCP server configuration (a translation of pi's code).

Only code that comes from pi is in this file (see the notice above); the code
that uses it for Airedale is in :mod:`airedale.pi_mcp`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from airedale import js_json
from airedale.whatwg_url import is_loopback_host, parse_url

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


def mcp_namespace(server: str) -> str:
    """Namespace of the tools of a server (pi: ``mcp__<server>`` with ``-`` as ``_``)."""
    return "mcp__" + server.replace("-", "_")


def apply_project_config(project_file: Path, servers: dict[str, PiMcpEntry], errors: list[str]) -> None:
    """Merge the repo file ``project_file`` into ``servers`` (the global servers), as pi does.

    ``servers`` is changed in place; the diagnostics of pi for the entries that
    it does not load are added to ``errors``. A missing file changes nothing.
    """
    parsed = _read_json_object(project_file, errors)
    if parsed is not None:
        _apply_project_entries(project_file, parsed, servers, errors)


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
        parsed = js_json.loads(text)
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")
        logger.warning("Ignoring unreadable project MCP config at %s", path)
        return None
    servers = parsed.get("mcpServers") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or ("mcpServers" in parsed and not isinstance(servers, dict)):
        errors.append(f'{path}: expected an object with an "mcpServers" object')
        return None
    return parsed


def _js_entries(mapping: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Entries in the order of ``Object.entries``: array indexes first, in number order."""
    # A key with more than 10 digits is above 2**32 - 2: it is a plain property (and no ``int()`` of a long string).
    indexes = [
        key for key in mapping if len(key) <= 10 and _ARRAY_INDEX_RE.fullmatch(key) and int(key) <= _MAX_ARRAY_INDEX
    ]
    indexes.sort(key=int)
    index_set = set(indexes)
    rest = [key for key in mapping if key not in index_set]
    return [(key, mapping[key]) for key in [*indexes, *rest]]


def _apply_project_entries(
    path: Path, parsed: dict[str, Any], servers: dict[str, PiMcpEntry], errors: list[str]
) -> None:
    """Apply the entries of a repo file (pi: ``readConfigFile`` with project scope)."""
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
        if is_http_config(config) and config.get("auth"):
            errors.append(f'{path}: server "{name}": auth is only allowed in the global mcp.json')
            continue
        servers[name] = PiMcpEntry(config, "project")


def _is_override(value: Mapping[str, Any]) -> bool:
    """An entry without ``command``, ``url`` and ``type`` (a ``null`` value counts as set)."""
    return "command" not in value and "url" not in value and "type" not in value


# --------------------------------------------------------------------------- #
# Validation (pi: ``validateMcpServerConfig``)
# --------------------------------------------------------------------------- #
def validate_server_config(name: str, raw: Any, *, exact: bool = False) -> dict[str, Any] | str:
    """Validate one server entry. Return the entry with exposure aliases resolved, or an error.

    The URL reader of Airedale does not decide every host exactly (an odd IPv6, an IDNA host): by default such a URL
    counts as accepted, which is the fail-closed answer for the entries of a repo file (a replacement is not missed).
    ``exact=True`` asks for the exact answer, for a config that Airedale itself writes for pi: a URL that is rejected
    **or** that cannot be decided is an error, so a server that pi may reject is not started or probed.
    """
    problem = _validate_server_config(name, raw)
    if exact and not isinstance(problem, str):
        return _undecided_url_problem(name, problem) or problem
    return problem


def _undecided_url_problem(name: str, config: dict[str, Any]) -> str | None:
    """Return an error for a URL of ``config`` that the reader cannot decide exactly (``None``: all are exact)."""
    oauth_value = config.get("oauth")
    oauth: dict[str, Any] = oauth_value if isinstance(oauth_value, dict) else {}
    for label, value in (
        ("url", config.get("url")),
        ("oauth.callbackUrl", oauth.get("callbackUrl")),
        ("oauth.authServerMetadataUrl", oauth.get("authServerMetadataUrl")),
    ):
        if isinstance(value, str):
            url = parse_url(value)
            if url is None or not url.exact:
                return f'server "{name}": {label} cannot be checked exactly (pi may reject it): {value!r}'
    return None


def _validate_server_config(name: str, raw: Any) -> dict[str, Any] | str:
    if not _SERVER_NAME_RE.fullmatch(name):
        return f'invalid server name "{name}" (use letters, digits, "_" and "-")'
    if not isinstance(raw, dict):
        return f'server "{name}" must be an object'
    value = _resolve_exposure_aliases(raw)
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
    if value.get("type") == "sse":
        return f'server "{name}": legacy SSE transport is not supported; use the streamable HTTP URL'
    if isinstance(value.get("url"), str) and _has_type(value, "http", "streamable-http"):
        url = parse_url(value["url"])
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
            if url.scheme != "https" and not is_loopback_host(url, _LOOPBACK_HOSTS):
                return f'server "{name}": auth requires an https URL, or http on localhost, 127.0.0.1, or [::1]'
        return value
    if isinstance(value.get("command"), str) and _has_type(value, "stdio"):
        args = value.get("args")
        if "args" in value and not (isinstance(args, list) and all(isinstance(arg, str) for arg in args)):
            return f'server "{name}": args must be an array of strings'
        if "env" in value and not _is_string_record(value["env"]):
            return f'server "{name}": env must map names to strings'
        if "cwd" in value and not isinstance(value["cwd"], str):
            return f'server "{name}": cwd must be a string'
        return value
    return f'server "{name}" needs either "command" (stdio) or "url" (streamable HTTP)'


def is_http_config(config: Mapping[str, Any]) -> bool:
    """The branch of ``validate_server_config`` that a valid entry took (url first, as pi)."""
    return isinstance(config.get("url"), str) and _has_type(config, "http", "streamable-http")


def _has_type(value: Mapping[str, Any], *types: str) -> bool:
    """JavaScript ``type === undefined || type === "..."``: a ``type`` that is ``null`` is not undefined."""
    return "type" not in value or value["type"] in types


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
    if "callbackPort" in value and not (_is_integer(port) and 1 <= port <= 65535):
        return "oauth.callbackPort must be a port number"
    if "callbackUrl" in value:
        callback = value["callbackUrl"]
        if not isinstance(callback, str) or not _is_loopback_redirect_uri(callback):
            return "oauth.callbackUrl must be an http URI on localhost, 127.0.0.1, or [::1] without query or fragment"
        url = parse_url(callback)
        if url is not None and url.port and "callbackPort" in value and int(url.port) != port:
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
            parsed = parse_url(callback_url)
            if parsed is None or (parsed.exact and parsed.hostname == "[::1]") or parsed.pathname != "/callback":
                return (
                    'oauth.clientRegistration "cimd" requires oauth.callbackUrl on localhost or 127.0.0.1 '
                    "with path /callback"
                )
    if "authServerMetadataUrl" in value:
        metadata_url = value["authServerMetadataUrl"]
        url = parse_url(metadata_url) if isinstance(metadata_url, str) else None
        if url is None or not (
            url.scheme == "https" or (url.scheme == "http" and is_loopback_host(url, _LOOPBACK_HOSTS))
        ):
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
    # (An integer that a JavaScript number cannot hold is Infinity there, which is not an integer for
    # ``Number.isInteger``; here a Python integer is compared as it is, and is never converted to a float.)
    return _is_number(value) and (isinstance(value, int) or value.is_integer())


def _is_loopback_redirect_uri(value: str) -> bool:
    url = parse_url(value)
    return (
        url is not None
        and url.scheme == "http"
        and is_loopback_host(url, _LOOPBACK_HOSTS)
        and not url.search
        and not url.hash
    )
