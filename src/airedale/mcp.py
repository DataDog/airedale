# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""MCP server specifications and runtime metadata for provider agent SDKs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import time
from collections.abc import Mapping
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

    def merged_headers(self, trace_headers: Mapping[str, str]) -> dict[str, str]:
        """Merge static MCP headers with per-run distributed-tracing headers."""
        headers = dict(self.headers)
        headers.update(trace_headers)
        return headers

    def to_claude_config(self, trace_headers: Mapping[str, str]) -> McpServerConfig:
        """Render this spec as a Claude Agent SDK MCP server config."""
        if self.type == "http":
            from claude_agent_sdk.types import McpHttpServerConfig

            return McpHttpServerConfig(type="http", url=self.url, headers=self.merged_headers(trace_headers))
        if self.type == "stdio":
            from claude_agent_sdk.types import McpStdioServerConfig

            return McpStdioServerConfig(
                type="stdio", command=self.command, args=list[str](self.args), env=dict[str, str](self.env)
            )
        raise ValueError(f"MCP server {self.name!r} has unsupported type {self.type!r}")

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
        raise ValueError(f"MCP server {self.name!r} has unsupported type {self.type!r}")

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

    path = Path(cwd) / ".mcp.json"
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
        specs.append(
            McpServerSpec(
                name=str(name),
                type=cfg.get("type"),
                url=cfg.get("url"),
                command=cfg.get("command"),
                args=tuple(cfg.get("args", []) or []),
                env=dict(cfg.get("env", {}) or {}),
                headers=dict(cfg.get("headers", {}) or {}),
            )
        )
    return specs


def configured_tool_names(spec: McpServerSpec) -> list[str]:
    """Return the configured tool names for a server spec (empty means all)."""
    return list(spec.tool_names)


def provider_mcp_tool_name(server_name: str, tool_name: str, *, sdk_name: str) -> str:
    """Return the provider-specific MCP tool name format."""
    if sdk_name == "claude-agent-sdk":
        return f"mcp__{server_name}__{tool_name}"
    return f"mcp.{server_name}.{tool_name}"


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

    server_catalogs = await asyncio.gather(
        *[
            _mcp_server_tool_metadata(server, trace_headers, configured_tool_names=set(configured_tool_names(server)))
            for server in servers
        ]
    )
    catalog: dict[tuple[str, str], McpToolMetadata] = {}
    for server_catalog in server_catalogs:
        catalog.update(server_catalog)
    return catalog


async def _mcp_server_tool_metadata(
    server: McpServerSpec, trace_headers: Mapping[str, str], *, configured_tool_names: set[str]
) -> dict[tuple[str, str], McpToolMetadata]:
    """Fetch tool metadata from a single MCP server."""
    try:
        tools = await asyncio.wait_for(
            _list_mcp_tools(server, trace_headers), timeout=MCP_TOOL_METADATA_TIMEOUT_SECONDS
        )
    except Exception:
        logger.debug("Unable to fetch MCP tool metadata for server %s", server.name, exc_info=True)
        return {}

    catalog: dict[tuple[str, str], McpToolMetadata] = {}
    for mcp_tool in tools:
        tool_name = _mcp_tool_field(mcp_tool, "name")
        if not isinstance(tool_name, str):
            continue
        # If configured_tool_names is empty, it means all tools are allowed
        if configured_tool_names and tool_name not in configured_tool_names:
            continue
        catalog[(server.name, tool_name)] = _mcp_tool_metadata(mcp_tool)
    return catalog


async def _list_mcp_tools(server: McpServerSpec, trace_headers: Mapping[str, str]) -> list[Any]:
    """List tools from an MCP server using the standard protocol."""
    if server.type == "http":
        async with (
            streamablehttp_client(server.url, headers=server.merged_headers(trace_headers)) as (
                read_stream,
                write_stream,
                _get_session_id,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            return list((await session.list_tools()).tools)
    if server.type == "stdio":
        stdio_server = StdioServerParameters(
            command=server.command, args=list(server.args), env=dict(server.env) or None
        )
        async with (
            stdio_client(stdio_server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            return list((await session.list_tools()).tools)
    raise ValueError(f"MCP server {server.name!r} has unsupported type {server.type!r}")


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
