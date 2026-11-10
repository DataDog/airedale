# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""MCP server specifications and runtime metadata for provider agent SDKs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import os
import re
import signal
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

if TYPE_CHECKING:
    from pathlib import Path

    from claude_agent_sdk.types import McpServerConfig

    from airedale.config.experiment import McpServerConfig as McpServerConfigExperiment

logger = logging.getLogger(__name__)

# Constants for managed MCP server start/health
MANAGED_MCP_HEALTH_TIMEOUT_SECONDS = 30.0
MANAGED_MCP_HEALTH_POLL_SECONDS = 0.5
MANAGED_MCP_HEALTH_PROBE_TIMEOUT_SECONDS = 2.0
# Grace period for a managed server to exit after SIGTERM before we SIGKILL it.
MANAGED_MCP_TERMINATE_TIMEOUT_SECONDS = 3.0
MCP_TOOL_METADATA_TIMEOUT_SECONDS = 10.0
# How much of a managed server's stdout/stderr we retain for diagnostics. The
# pipes are drained continuously (see ``ManagedMcpServer._drain``) so the OS
# pipe buffer can never fill and wedge the child; we keep only a bounded tail
# of recent output for error reporting.
MANAGED_MCP_OUTPUT_TAIL_BYTES = 64 * 1024
MANAGED_MCP_DRAIN_CHUNK_BYTES = 4096


# Pi accepts only these characters in MCP server names (see pi docs/mcp.md).
PI_MCP_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_PI_TOOL_NAME_SAFE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
PI_MAX_TOOL_NAME_LENGTH = 64


@dataclass(frozen=True)
class McpServerSpec:
    """MCP server configuration passed through to provider agent SDKs."""

    name: str
    type: str | None = None
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Infer the transport type from the fields when not given explicitly."""
        if not self.type:
            object.__setattr__(self, "type", "stdio" if self.command and not self.url else "http")

    @classmethod
    def from_config(cls, config: McpServerConfigExperiment) -> McpServerSpec:
        """Build a server spec from an experiment config."""
        return cls(
            name=config.name,
            type=config.type,
            url=config.url,
            command=config.command,
            args=config.args,
            env=config.env or {},
            headers=config.headers or {},
            tool_names=config.tool_names,
        )

    @property
    def is_managed(self) -> bool:
        """True for http servers that carry a command used to auto-start them."""
        return self.type == "http" and self.command is not None

    def _require_url(self) -> str:
        """Return the ``url`` of an http server, or raise when it is missing."""
        if self.url is None:
            raise ValueError(f"MCP server {self.name!r} of type 'http' has no 'url'")
        return self.url

    def _require_command(self) -> str:
        """Return the ``command`` of a stdio server, or raise when it is missing."""
        if self.command is None:
            raise ValueError(f"MCP server {self.name!r} of type 'stdio' has no 'command'")
        return self.command

    def merged_headers(self, trace_headers: Mapping[str, str]) -> dict[str, str]:
        """Merge static MCP headers with per-run distributed-tracing headers."""
        headers = dict(self.headers)
        headers.update(trace_headers)
        return headers

    def to_claude_config(self, trace_headers: Mapping[str, str]) -> McpServerConfig:
        """Render this spec as a Claude Agent SDK MCP server config."""
        if self.type == "http":
            from claude_agent_sdk.types import McpHttpServerConfig

            return McpHttpServerConfig(type="http", url=self._require_url(), headers=self.merged_headers(trace_headers))
        if self.type == "stdio":
            from claude_agent_sdk.types import McpStdioServerConfig

            return McpStdioServerConfig(
                type="stdio", command=self._require_command(), args=list[str](self.args), env=dict[str, str](self.env)
            )
        raise ValueError(f"MCP server {self.name!r} has unsupported type (expected 'stdio' or 'http')")

    def to_codex_config_overrides(self, trace_headers: Mapping[str, str]) -> list[str]:
        """Render this spec as Codex CLI config override strings."""
        key_prefix = _toml_key_path("mcp_servers", self.name)
        overrides: list[str] = []
        if self.type == "http":
            overrides.append(f"{key_prefix}.url={_toml_value(self.url)}")
            headers = self.merged_headers(trace_headers)
            if headers:
                overrides.append(f"{key_prefix}.http_headers={_toml_value(headers)}")
            return overrides
        if self.type == "stdio":
            overrides.append(f"{key_prefix}.command={_toml_value(self.command)}")
            if self.args:
                overrides.append(f"{key_prefix}.args={_toml_value(list(self.args))}")
            if self.env:
                overrides.append(f"{key_prefix}.env={_toml_value(dict(self.env))}")
            return overrides
        raise ValueError(f"MCP server {self.name!r} has unsupported type (expected 'stdio' or 'http')")

    def to_pi_config(self, trace_headers: Mapping[str, str]) -> dict[str, Any]:
        """Render this spec as a pi ``mcp.json`` ``mcpServers`` entry.

        Every tool gets ``direct`` exposure, so the model sees MCP tools as
        declared tools (the same as with Claude and Codex), not through pi's
        ``codemode`` scripts. When ``tool_names`` is set, all other tools of the
        server are ``hidden``. Pi resolves ``$NAME`` and ``!command`` in header
        and env values; :func:`pi_literal` escapes the values so they stay
        literal.
        """
        if not PI_MCP_SERVER_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"MCP server name {self.name!r} is not valid for pi; use only letters, digits, '_' and '-'"
            )
        entry: dict[str, Any]
        if self.type == "http":
            entry = {"type": "http", "url": self.url}
            headers = self.merged_headers(trace_headers)
            if headers:
                entry["headers"] = {name: pi_literal(value) for name, value in headers.items()}
        elif self.type == "stdio":
            entry = {"type": "stdio", "command": self.command, "args": list(self.args)}
            if self.env:
                entry["env"] = {name: pi_literal(value) for name, value in self.env.items()}
        else:
            raise ValueError(f"MCP server {self.name!r} has unsupported type (expected 'stdio' or 'http')")
        tool_names = configured_tool_names(self)
        if any("*" in tool_name for tool_name in tool_names):
            # Pi reads `*` in a `toolExposure` key as a pattern; `tool_names`
            # is a list of exact names.
            raise ValueError(f"MCP server {self.name!r} tool_names must be exact tool names; '*' is not allowed")
        if tool_names:
            entry["exposure"] = "hidden"
            entry["toolExposure"] = dict.fromkeys(tool_names, "direct")
        else:
            entry["exposure"] = "direct"
        return entry

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for runtime config."""
        return {
            "name": self.name,
            "type": self.type,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "headers": dict(self.headers),
            "tool_names": list(self.tool_names),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a non-secret representation for LLMObs metadata/config."""
        data = self.to_dict()
        data["headers"] = dict.fromkeys(self.headers, "<redacted>")
        # Environment values often hold credentials: keep the variable names only.
        data["env"] = dict.fromkeys(self.env, "<set>")
        return data


def discover_claude_project_mcp_servers(cwd: str | Path) -> list[McpServerSpec]:
    """Parse ``<cwd>/.mcp.json`` into MCP server specs (Claude project scope).

    Claude's project-level MCP convention is a ``.mcp.json`` file shaped like
    ``{"mcpServers": {<name>: {type/command/args/env/url/headers}}}``. Because we
    keep ``strict_mcp_config=True`` (hermetic), Claude won't read it itself, so we
    parse it and inject the servers as ordinary specs. A missing, empty, or
    malformed file yields an empty list.
    """
    from pathlib import Path

    return _parse_mcp_json(Path(cwd) / ".mcp.json")


#: Replaces a value that is not reported.
REDACTED = "<redacted>"


def url_without_secrets(url: str | None) -> str | None:
    """Keep only the scheme, host and port of ``url``: the user info, path, query
    and fragment can hold tokens (for example ``/s/<token>/mcp``).

    The URL is read like pi reads it (``airedale.whatwg_url``): ``\\`` ends the host as ``/`` does, and tabs and
    newlines inside the URL are ignored. A URL that cannot be read, or whose host the reader cannot decide in
    every detail, is replaced by ``<redacted>``.
    """
    if not url:
        return url
    from airedale.whatwg_url import parse_url

    try:
        parsed = parse_url(url)
    except Exception:  # noqa: BLE001 - a report must not abort a run: what cannot be read is not reported
        return REDACTED
    # A host that the reader is not sure about (``exact=False``: IDNA, odd IPv6) can hold what pi reads as a path
    # or as user info (full-width slashes, ``@``, brackets): it is not reported.
    if parsed is None or not parsed.exact or not parsed.hostname:
        return REDACTED
    return f"{parsed.scheme}://{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")


def _parse_mcp_json(path: Path) -> list[McpServerSpec]:
    """Parse an ``{"mcpServers": {...}}`` file; a missing or bad file gives ``[]``."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable project MCP config at %s", path, exc_info=True)
        return []

    servers_raw = data.get("mcpServers") if isinstance(data, Mapping) else None
    if not isinstance(servers_raw, Mapping):
        return []

    specs: list[McpServerSpec] = []
    for name, cfg in servers_raw.items():
        if not isinstance(cfg, Mapping):
            continue
        args, env, headers = cfg.get("args") or [], cfg.get("env") or {}, cfg.get("headers") or {}
        if not (
            all(isinstance(cfg.get(key), str | None) for key in ("type", "url", "command"))
            and isinstance(args, list)
            and all(isinstance(arg, str) for arg in args)
            and _is_str_mapping(env)
            and _is_str_mapping(headers)
        ):
            # The value is not logged: it can hold a credential.
            logger.warning("Ignoring MCP server %r in %s: a field has the wrong type", str(name), path)
            continue
        specs.append(
            McpServerSpec(
                name=str(name),
                type=cfg.get("type"),
                url=cfg.get("url"),
                command=cfg.get("command"),
                args=tuple(args),
                env=dict(env),
                headers=dict(headers),
            )
        )
    return specs


def _is_str_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())


def configured_tool_names(spec: McpServerSpec) -> list[str]:
    """Return the configured tool names for a server spec (empty means all)."""
    return list(spec.tool_names)


def provider_mcp_tool_name(server_name: str, tool_name: str, *, sdk_name: str) -> str:
    """Return the provider-specific MCP tool name format."""
    if sdk_name == "claude-agent-sdk":
        return f"mcp__{server_name}__{tool_name}"
    if sdk_name == "pi":
        return pi_mcp_tool_name(server_name, tool_name)
    return f"mcp.{server_name}.{tool_name}"


def pi_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Return the name pi gives an MCP tool: ``mcp__<server>__<tool>``.

    Same rule as pi's ``createMcpToolName``: every character other than letters,
    digits and ``_`` becomes ``_``; a name longer than 64 characters is cut and
    gets ``_`` + the first 8 hex digits of ``sha256("<server>\0<tool>")``. Pi
    also adds that suffix when two tools of the server get the same name after
    sanitizing: use :func:`pi_mcp_tool_names` with the tool list of the server
    for that case.
    """
    name = _pi_plain_tool_name(server_name, tool_name)
    if len(name) <= PI_MAX_TOOL_NAME_LENGTH:
        return name
    return _pi_hashed_tool_name(server_name, tool_name)


def pi_mcp_tool_names(server_name: str, tool_names: Iterable[str]) -> dict[str, str]:
    """Return the pi name of each tool of one server (keys: MCP tool names).

    Same as pi's MCP extension: when two or more tools of the server get the
    same name after sanitizing (for example ``get-doc`` and ``get_doc``), all
    of them get the hash suffix, whatever their order. ``tool_names`` must be
    the complete tool list of the server for the result to match pi. As in
    pi, the names compared are the names after shortening (a long name with
    its hash suffix). Not reproduced: a collision with a tool of another
    server (``mcp__a__b__c`` from server ``a`` and from server ``a__b``),
    where pi's result depends on which server connects first; see
    :func:`pi_possible_tool_name_collisions`.
    """
    unique = list(dict.fromkeys(tool_names))
    plain = {tool: pi_mcp_tool_name(server_name, tool) for tool in unique}
    counts = Counter(plain.values())
    return {
        tool: pi_mcp_tool_name(server_name, tool)
        if counts[plain[tool]] == 1
        else _pi_hashed_tool_name(server_name, tool)
        for tool in unique
    }


def _pi_plain_tool_name(server_name: str, tool_name: str) -> str:
    """Return ``mcp__<server>__<tool>`` with every unsafe character replaced by ``_``, as pi does.

    Pi works with JavaScript strings (UTF-16): its regular expression replaces
    each UTF-16 code unit, so a character outside the Basic Multilingual Plane
    (for example an emoji) becomes two ``_``, and the length that pi limits is
    the number of UTF-16 code units.
    """
    return "".join(
        char if char in _PI_TOOL_NAME_SAFE_CHARS else ("__" if ord(char) > 0xFFFF else "_")
        for char in f"mcp__{server_name}__{tool_name}"
    )


def _pi_hashed_tool_name(server_name: str, tool_name: str) -> str:
    name = _pi_plain_tool_name(server_name, tool_name)
    # Node encodes a lone surrogate of a JavaScript string as U+FFFD.
    text = f"{server_name}\0{tool_name}".encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{name[: PI_MAX_TOOL_NAME_LENGTH - len(digest) - 1]}_{digest}"


def pi_registered_tool_names(
    servers: Sequence[tuple[str, Iterable[str]]],
) -> dict[tuple[str, str], str]:
    """Return the name that pi gives each ``(server, tool)`` when the servers connect in the order of ``servers``.

    This is the registration loop of pi's MCP extension (1.4.0 and 1.5.0) with
    the one name map that all servers share: a tool takes the plain name
    ``mcp__<server>__<tool>`` unless it is longer than 64 characters, unless
    another tool of the same server has that name after shortening, or unless a
    tool of another server (or an earlier tool of this server) already has it.
    Then the tool gets the hash suffix. Pi does not check the name with the suffix
    again: it can be the name of another tool (that is rare, but it happens, for
    example for a tool called ``c_01b8a75b``).
    """
    owners: dict[str, tuple[str, str]] = {}
    names: dict[tuple[str, str], str] = {}
    for server, tools in servers:
        unique = list(dict.fromkeys(tools))
        plain = Counter(pi_mcp_tool_name(server, tool) for tool in unique)
        current: set[str] = set()
        for tool in unique:
            owner = (server, tool)
            sanitized = _pi_plain_tool_name(server, tool)
            taken = (sanitized in owners and owners[sanitized] != owner) or sanitized in current or plain[sanitized] > 1
            name = (
                sanitized
                if len(sanitized) <= PI_MAX_TOOL_NAME_LENGTH and not taken
                else _pi_hashed_tool_name(server, tool)
            )
            owners[name] = owner
            current.add(name)
            names[owner] = name
    return names


#: Most servers of one group (servers that can take the same tool name) for which every connection order is tried.
_PI_MAX_SERVERS_FOR_ALL_ORDERS = 6


class PiToolNameOrderError(ValueError):
    """Too many servers can take the same tool names for every connection order to be checked."""


def _pi_possible_names(server: str, tools: Iterable[str]) -> set[str]:
    """Return every name that a tool of ``server`` can get in pi: the plain name or the one with a hash suffix."""
    return {name for tool in tools for name in (_pi_plain_tool_name(server, tool), _pi_hashed_tool_name(server, tool))}


def _pi_server_groups(catalog: Mapping[str, list[str]]) -> list[list[str]]:
    """Group the servers so that two servers are in one group when a name can be the name of a tool of both.

    A tool takes a name away from another tool only when the two can get the same name, so the connection order of
    servers in different groups does not change any name: each group is checked alone.
    """
    names = {server: _pi_possible_names(server, tools) for server, tools in catalog.items()}
    groups: list[tuple[list[str], set[str]]] = []
    for server in catalog:
        merged = ([server], set(names[server]))
        rest = []
        for members, taken in groups:
            if taken & merged[1]:
                merged = ([*members, *merged[0]], taken | merged[1])
            else:
                rest.append((members, taken))
        groups = [*rest, merged]
    return [members for members, _ in groups]


def pi_possible_tool_name_collisions(tools_by_server: Mapping[str, Iterable[str]]) -> dict[str, list[tuple[str, str]]]:
    """Return the tool names that depend on the connection order of the servers, with the ``(server, tool)`` pairs.

    Pi registers the tools of each server when it connects. A tool whose name
    is taken by a tool of another server gets a hash suffix, so which tool keeps
    the plain name depends on which server connects first, and the runner
    cannot know that order. The names are computed with :func:`pi_registered_tool_names`
    for every order. A name is listed when, in some order, a tool does not get the name
    that it has alone with its server (``pi_mcp_tool_names``), or when two tools get
    the same name. The names that a hash suffix makes are included: a suffix is not
    checked again by pi.

    Servers are grouped: servers that cannot get a common name (plain or with a hash suffix) cannot
    change each other's names, so each group is checked alone, with all its orders. When a group has more than 6
    servers, not all the orders can be tried and no approximation is made: :class:`PiToolNameOrderError`
    is raised (the result would not be reliable).

    ``tools_by_server`` has the tools that are known for each server. For a
    server with a ``tool_names`` allow-list, the list is not the whole catalog:
    a tool that the list hides is also registered, can take a hash suffix, and
    so can remove a collision (or make another one). The result is the
    collisions that are possible from the tools that are known.
    """
    catalog = {server: list(dict.fromkeys(tools)) for server, tools in tools_by_server.items()}
    alone = {
        (server, tool): name
        for server, tools in catalog.items()
        for tool, name in pi_mcp_tool_names(server, tools).items()
    }
    position = {pair: index for index, pair in enumerate(alone)}
    claims: dict[str, set[tuple[str, str]]] = {}
    for group in _pi_server_groups(catalog):
        if len(group) > _PI_MAX_SERVERS_FOR_ALL_ORDERS:
            raise PiToolNameOrderError(
                f"{len(group)} MCP servers can take the same pi tool names (the limit is "
                f"{_PI_MAX_SERVERS_FOR_ALL_ORDERS}): the names that pi gives them depend on the order in which they "
                "connect, and not all the orders can be checked. Rename servers so that fewer of them share tool names."
            )
        for order in itertools.permutations(group):
            names = pi_registered_tool_names([(server, catalog[server]) for server in order])
            by_name: dict[str, list[tuple[str, str]]] = {}
            for pair, name in names.items():
                by_name.setdefault(name, []).append(pair)
            for name, pairs in by_name.items():
                if len(pairs) > 1:  # Two tools with one name.
                    claims.setdefault(name, set()).update(pairs)
            for pair, name in names.items():
                if name != alone[pair]:  # This tool lost its name to another one.
                    claim = claims.setdefault(alone[pair], set())
                    claim.add(pair)
                    claim.update(other for other, other_name in names.items() if other_name == alone[pair])
    return {
        name: sorted(pairs, key=lambda pair: position[pair]) for name, pairs in sorted(claims.items()) if len(pairs) > 1
    }


def pi_literal(value: str) -> str:
    """Escape a value so that pi config resolution keeps it literal.

    Pi treats a leading ``!`` as a shell command and ``$NAME`` / ``${NAME}`` as
    environment interpolation in ``models.json`` and ``mcp.json`` values. In
    those values, ``$$`` gives a literal ``$`` and ``$!`` a literal ``!``.
    """
    escaped = str(value).replace("$", "$$")
    if escaped.startswith("!"):
        escaped = "$!" + escaped[1:]
    return escaped


@dataclass(frozen=True)
class McpToolMetadata:
    """Tool metadata returned by an MCP server's standard tools/list endpoint."""

    description: str | None = None
    input_schema: dict[str, Any] | None = None


async def _mcp_tool_metadata_catalog(
    servers: list[McpServerSpec], trace_headers: Mapping[str, str]
) -> dict[tuple[str, str], McpToolMetadata]:
    """Return actual MCP tool metadata keyed by (server_name, tool_name).

    Metadata is fetched through the standard MCP tools/list protocol. Failures are
    logged at debug level and represented by missing catalog entries.
    """
    if not servers:
        return {}

    server_catalogs = await asyncio.gather(*[_mcp_server_tool_metadata(server, trace_headers) for server in servers])
    catalog: dict[tuple[str, str], McpToolMetadata] = {}
    for server_catalog in server_catalogs:
        catalog.update(server_catalog)
    return catalog


async def _mcp_server_tool_metadata(
    server: McpServerSpec, trace_headers: Mapping[str, str]
) -> dict[tuple[str, str], McpToolMetadata]:
    """Fetch the metadata of every tool of a single MCP server.

    All tools are kept, also the tools that ``tool_names`` does not list: pi
    names a tool from the complete tool list of the server (collision
    suffixes, see :func:`pi_mcp_tool_names`). The tool definitions use only
    the configured tools.
    """
    try:
        tools = await asyncio.wait_for(
            _list_mcp_tools(server, trace_headers, all_pages=True), timeout=MCP_TOOL_METADATA_TIMEOUT_SECONDS
        )
    except Exception:
        logger.debug("Unable to fetch MCP tool metadata for server %s", server.name, exc_info=True)
        return {}

    catalog: dict[tuple[str, str], McpToolMetadata] = {}
    for mcp_tool in tools:
        tool_name = _mcp_tool_field(mcp_tool, "name")
        if not isinstance(tool_name, str):
            continue
        catalog[(server.name, tool_name)] = _mcp_tool_metadata(mcp_tool)
    return catalog


async def _list_mcp_tools(
    server: McpServerSpec, trace_headers: Mapping[str, str], *, all_pages: bool = False
) -> list[Any]:
    """List tools from an MCP server using the standard protocol.

    With ``all_pages``, every ``tools/list`` page is read (the tool metadata
    needs the complete list). Without it, only the first page (enough for a
    health probe, which must stay fast).
    """
    if server.type == "http":
        async with (
            streamablehttp_client(server._require_url(), headers=server.merged_headers(trace_headers)) as (
                read_stream,
                write_stream,
                _get_session_id,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            return await _list_all_tools(session) if all_pages else list((await session.list_tools()).tools)
    if server.type == "stdio":
        stdio_server = StdioServerParameters(
            command=server._require_command(), args=list(server.args), env=dict(server.env) or None
        )
        async with (
            stdio_client(stdio_server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            return await _list_all_tools(session) if all_pages else list((await session.list_tools()).tools)
    raise ValueError(f"MCP server {server.name!r} has unsupported type (expected 'stdio' or 'http')")


#: Most ``tools/list`` pages read from one server (a bad server cannot loop).
MCP_TOOL_LIST_MAX_PAGES = 50


class McpToolListIncompleteError(RuntimeError):
    """The ``tools/list`` of a server cannot be read to its end (too many pages, or a repeated cursor)."""


async def _list_all_tools(session: ClientSession) -> list[Any]:
    """Return the tools of every ``tools/list`` page (follows ``nextCursor``).

    The complete list is necessary: pi names tools from all the tools of the
    server. It never returns a partial list: it raises :class:`McpToolListIncompleteError` when
    :data:`MCP_TOOL_LIST_MAX_PAGES` pages leave a next cursor, or when a cursor repeats. The caller's timeout
    also bounds this.
    """
    tools: list[Any] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(MCP_TOOL_LIST_MAX_PAGES):
        result = await session.list_tools(cursor=cursor) if cursor is not None else await session.list_tools()
        tools.extend(result.tools)
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            return tools
        if cursor in seen:
            # A server that repeats a cursor never ends its list: what was read is not the whole list.
            raise McpToolListIncompleteError("the MCP server repeated a tools/list cursor; the tool list is incomplete")
        seen.add(cursor)
    # A partial list would give wrong tool names (pi names a tool from the whole list) and a catalog without the later
    # tools: the listing fails, like any other failure of the probe.
    raise McpToolListIncompleteError(
        f"the MCP server sent more than {MCP_TOOL_LIST_MAX_PAGES} tools/list pages; the tool list is incomplete"
    )


def _mcp_tool_metadata(tool: Any) -> McpToolMetadata:
    """Extract metadata from an MCP tool definition."""
    description_value = _mcp_tool_field(tool, "description")
    description = description_value.strip() if isinstance(description_value, str) else None
    input_schema_value = _mcp_tool_field(tool, "inputSchema", "input_schema")
    input_schema = dict(input_schema_value) if isinstance(input_schema_value, Mapping) else None
    return McpToolMetadata(description=description or None, input_schema=input_schema)


def _mcp_tool_field(tool: Any, *field_names: str) -> Any:
    """Get a field value from a tool, trying multiple field names."""
    for field_name in field_names:
        value = tool.get(field_name) if isinstance(tool, Mapping) else getattr(tool, field_name, None)
        if value is not None:
            return value
    return None


class _BoundedByteBuffer:
    """Accumulate bytes while retaining only the most recent ``limit`` bytes.

    Used to capture a managed server's stdout/stderr tail for diagnostics
    without growing without bound over a long-lived run.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()

    def append(self, chunk: bytes) -> None:
        self._data.extend(chunk)
        if len(self._data) > self._limit:
            del self._data[: len(self._data) - self._limit]

    def text(self) -> str:
        return bytes(self._data).decode(errors="replace")


class ManagedMcpServer:
    """Async context manager that starts/stops an HTTP MCP server if needed.

    For http servers that carry a launch ``command`` (``spec.is_managed``):
    - Probes health via the MCP protocol itself (a tools/list call)
    - If unreachable, starts the subprocess and polls until reachable
    - On exit, terminates the subprocess gracefully then forcefully
    - If already reachable, reuses existing server without owning it

    The subprocess is launched and awaited through :mod:`asyncio` (never the
    blocking :class:`subprocess.Popen` API) so teardown never blocks the event
    loop; the stop path is additionally shielded so an in-flight cancellation
    (Ctrl+C) still reaps the child instead of leaking it.
    """

    def __init__(self, spec: McpServerSpec) -> None:
        self.spec = spec
        self.process: asyncio.subprocess.Process | None = None
        self.owns_process = False
        # Background tasks that continuously drain stdout/stderr into the
        # bounded buffers below. Draining is mandatory: an undrained PIPE fills
        # its ~64 KB OS buffer, blocks the child on its next write, and then
        # deadlocks ``process.wait()`` (per the asyncio subprocess docs).
        self._drain_tasks: list[asyncio.Task[None]] = []
        self._output: dict[str, _BoundedByteBuffer] = {
            "stdout": _BoundedByteBuffer(MANAGED_MCP_OUTPUT_TAIL_BYTES),
            "stderr": _BoundedByteBuffer(MANAGED_MCP_OUTPUT_TAIL_BYTES),
        }

    async def __aenter__(self) -> ManagedMcpServer:
        if not self.spec.is_managed:
            # Not an http server, or no auto-start command configured
            return self

        # Check if already reachable via the MCP protocol
        if await self._is_healthy():
            logger.debug("MCP server %s already reachable at %s", self.spec.name, self.spec.url)
            return self

        # Start the server
        argv = [self.spec.command, *self.spec.args]
        logger.info("Starting MCP server %s with command: %s", self.spec.name, argv)
        env = os.environ.copy()
        if self.spec.env:
            env.update({str(k): str(v) for k, v in self.spec.env.items()})

        self.process = await asyncio.create_subprocess_exec(
            *argv,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Start in a new process group so we can signal the whole tree.
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self.owns_process = True
        # Drain the pipes for the whole lifetime of the server so its buffers
        # never fill (which would block the child and deadlock teardown).
        self._start_draining()

        # Wait for health check
        deadline = time.monotonic() + MANAGED_MCP_HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                # Process died during startup; the drain tasks have captured
                # whatever it wrote before exiting.
                await self.process.wait()
                await self._stop_draining()
                raise RuntimeError(
                    f"MCP server {self.spec.name} process died (exit code {self.process.returncode})\n"
                    f"stdout: {self._output['stdout'].text()}\nstderr: {self._output['stderr'].text()}"
                )

            if await self._is_healthy():
                logger.info("MCP server %s started and reachable", self.spec.name)
                return self

            await asyncio.sleep(MANAGED_MCP_HEALTH_POLL_SECONDS)

        # Timed out
        raise RuntimeError(
            f"MCP server {self.spec.name} failed to become reachable within {MANAGED_MCP_HEALTH_TIMEOUT_SECONDS}s"
        )

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not (self.process and self.owns_process):
            return
        logger.info("Stopping MCP server %s", self.spec.name)
        # Shield the stop sequence so that, even when teardown runs while the
        # surrounding task is being cancelled, the child is still reaped.
        await asyncio.shield(self._terminate())

    def _start_draining(self) -> None:
        """Spawn background readers that empty stdout/stderr into bounded buffers."""
        process = self.process
        if process is None:
            return
        for key, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            if stream is not None:
                self._drain_tasks.append(asyncio.create_task(self._drain(stream, key)))

    async def _drain(self, stream: asyncio.StreamReader, key: str) -> None:
        """Continuously read ``stream`` into a bounded buffer until EOF/cancel."""
        buffer = self._output[key]
        with contextlib.suppress(Exception):
            while True:
                chunk = await stream.read(MANAGED_MCP_DRAIN_CHUNK_BYTES)
                if not chunk:
                    return
                buffer.append(chunk)

    async def _stop_draining(self) -> None:
        """Cancel and reap the drain tasks (they may be blocked on a read)."""
        tasks, self._drain_tasks = self._drain_tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _terminate(self) -> None:
        """Stop the owned subprocess gracefully (SIGTERM) then forcefully (SIGKILL)."""
        process = self.process
        if process is None:
            return
        try:
            self._signal_group(signal.SIGTERM, fallback=process.terminate)
            try:
                await asyncio.wait_for(process.wait(), timeout=MANAGED_MCP_TERMINATE_TIMEOUT_SECONDS)
                return
            except TimeoutError:
                logger.warning("MCP server %s did not terminate gracefully, forcing kill", self.spec.name)
            self._signal_group(signal.SIGKILL, fallback=process.kill)
            with contextlib.suppress(Exception):
                await process.wait()
        finally:
            # Stop reading the pipes regardless: a detached grandchild may still
            # hold the write end, so a drain task could otherwise block forever
            # and keep the event loop alive after every eval has finished.
            await self._stop_draining()

    def _signal_group(self, sig: int, *, fallback: Any) -> None:
        """Signal the child's process group, falling back to the lone child."""
        process = self.process
        if process is None or process.returncode is not None:
            return
        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), sig)
                return
        with contextlib.suppress(ProcessLookupError):
            fallback()

    async def _is_healthy(self) -> bool:
        """Check if the server is reachable by issuing an MCP tools/list call."""
        try:
            await asyncio.wait_for(_list_mcp_tools(self.spec, {}), timeout=MANAGED_MCP_HEALTH_PROBE_TIMEOUT_SECONDS)
            return True
        except Exception:
            return False


@contextlib.asynccontextmanager
async def managed_servers(specs: list[McpServerSpec]):
    """Start all auto-start MCP servers for a scenario."""
    async with contextlib.AsyncExitStack() as stack:
        for spec in specs:
            await stack.enter_async_context(ManagedMcpServer(spec))
        yield


# TOML rendering helpers
def _toml_key_path(*parts: str) -> str:
    """Build a TOML key path from parts."""
    return ".".join(_toml_key(part) for part in parts)


def _toml_key(value: str) -> str:
    """Format a single TOML key, quoting if needed."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return _toml_string(value)


def _toml_value(value: Any) -> str:
    """Format a value for TOML."""
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_toml_key(str(key))} = {_toml_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return _toml_string(str(value))


def _toml_string(value: str) -> str:
    """Quote a string value for TOML."""
    return json.dumps(value)
