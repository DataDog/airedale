"""MCP server specifications and runtime metadata for provider agent SDKs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpServerConfig

    from dd_ai_devx_evals.config.experiment import McpServerConfig as McpServerConfigExperiment

logger = logging.getLogger(__name__)

# Constants for managed MCP server start/health
MANAGED_MCP_HEALTH_TIMEOUT_SECONDS = 30.0
MANAGED_MCP_HEALTH_POLL_SECONDS = 0.5
MANAGED_MCP_HEALTH_PROBE_TIMEOUT_SECONDS = 2.0
MCP_TOOL_METADATA_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class McpServerSpec:
    """MCP server configuration passed through to provider agent SDKs."""

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    tool_names: tuple[str, ...] = ()
    bearer_token_env_var: str | None = None
    # Managed start fields
    start_command: str | None = None
    start_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: McpServerConfigExperiment) -> McpServerSpec:
        """Build a server spec from an experiment config."""
        return cls(
            name=config.name,
            url=config.url,
            command=config.command,
            args=config.args,
            env=config.env or {},
            headers=config.headers or {},
            tool_names=config.tool_names,
            bearer_token_env_var=config.bearer_token_env_var,
            start_command=config.start_command,
            start_env=config.start_env or {},
        )

    def merged_headers(self, trace_headers: Mapping[str, str], *, include_bearer: bool = True) -> dict[str, str]:
        """Merge static MCP headers with per-run distributed-tracing headers."""
        headers = dict(self.headers)
        if include_bearer and self.bearer_token_env_var and (token := os.environ.get(self.bearer_token_env_var)):
            headers.setdefault("Authorization", f"Bearer {token}")
        headers.update(trace_headers)
        return headers

    def to_claude_config(self, trace_headers: Mapping[str, str]) -> McpServerConfig:
        """Render this spec as a Claude Agent SDK MCP server config."""
        if self.url:
            from claude_agent_sdk.types import McpHttpServerConfig

            return McpHttpServerConfig(type="http", url=self.url, headers=self.merged_headers(trace_headers))
        if self.command:
            from claude_agent_sdk.types import McpStdioServerConfig

            return McpStdioServerConfig(
                type="stdio", command=self.command, args=list[str](self.args), env=dict[str, str](self.env)
            )
        raise ValueError(f"MCP server {self.name!r} must define either url or command")

    def to_codex_config_overrides(self, trace_headers: Mapping[str, str]) -> list[str]:
        """Render this spec as Codex CLI config override strings."""
        key_prefix = _toml_key_path("mcp_servers", self.name)
        overrides: list[str] = []
        if self.url:
            overrides.append(f"{key_prefix}.url={_toml_value(self.url)}")
            headers = self.merged_headers(trace_headers, include_bearer=False)
            if headers:
                overrides.append(f"{key_prefix}.http_headers={_toml_value(headers)}")
            if self.bearer_token_env_var:
                overrides.append(f"{key_prefix}.bearer_token_env_var={_toml_value(self.bearer_token_env_var)}")
            return overrides
        if self.command:
            overrides.append(f"{key_prefix}.command={_toml_value(self.command)}")
            if self.args:
                overrides.append(f"{key_prefix}.args={_toml_value(list(self.args))}")
            if self.env:
                overrides.append(f"{key_prefix}.env={_toml_value(dict(self.env))}")
            return overrides
        raise ValueError(f"MCP server {self.name!r} must define either url or command")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for runtime config."""
        return {
            "name": self.name,
            "url": self.url,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "headers": dict(self.headers),
            "tool_names": list(self.tool_names),
            "bearer_token_env_var": self.bearer_token_env_var,
            "start_command": self.start_command,
            "start_env": dict(self.start_env),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a non-secret representation for LLMObs metadata/config."""
        data = self.to_dict()
        data["headers"] = dict.fromkeys(self.headers, "<redacted>")
        return data


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
    if server.url:
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
    if server.command:
        stdio_server = StdioServerParameters(
            command=server.command, args=list(server.args), env=dict(server.env) or None
        )
        async with (
            stdio_client(stdio_server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            return list((await session.list_tools()).tools)
    raise ValueError(f"MCP server {server.name!r} must define either url or command")


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


class ManagedMcpServer:
    """Async context manager that starts/stops an HTTP MCP server if needed.

    For HTTP servers with start_command configured:
    - Probes health via the MCP protocol itself (a tools/list call)
    - If unreachable, starts the subprocess and polls until reachable
    - On exit, terminates the subprocess gracefully then forcefully
    - If already reachable, reuses existing server without owning it
    """

    def __init__(self, spec: McpServerSpec) -> None:
        self.spec = spec
        self.process: subprocess.Popen[str] | None = None
        self.owns_process = False

    async def __aenter__(self) -> ManagedMcpServer:
        if not self.spec.url or not self.spec.start_command:
            # Not an HTTP server or no auto-start configured
            return self

        # Check if already reachable via the MCP protocol
        if await self._is_healthy():
            logger.debug("MCP server %s already reachable at %s", self.spec.name, self.spec.url)
            return self

        # Start the server
        logger.info("Starting MCP server %s with command: %s", self.spec.name, self.spec.start_command)
        env = os.environ.copy()
        if self.spec.start_env:
            env.update({str(k): str(v) for k, v in self.spec.start_env.items()})

        self.process = subprocess.Popen(
            self.spec.start_command,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Start in new process group for clean termination
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        self.owns_process = True

        # Wait for health check
        deadline = time.monotonic() + MANAGED_MCP_HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                # Process died
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    f"MCP server {self.spec.name} process died (exit code {self.process.returncode})\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
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
        if self.process and self.owns_process:
            logger.info("Stopping MCP server %s", self.spec.name)
            # Try graceful termination first
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning("MCP server %s did not terminate gracefully, forcing kill", self.spec.name)
                if hasattr(os, "killpg"):
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()

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
