"""Claude Agent SDK runner with native LLMObs instrumentation.

``anthropic`` models run through ``claude-agent-sdk`` (Claude Code). The SDK
ships a native ddtrace LLMObs integration that owns the agent/LLM/tool spans, so
this runner deliberately creates **no synthetic spans**. It only annotates the
active experiment span with the aggregated token usage it observed, ensuring the
experiment's ``token_count`` is complete even when the integration misses some
usage.

MCP servers are passed natively (``mcp_servers`` + an ``allowed_tools`` MCP
allow-list); skills are staged into ``<cwd>/.claude/skills`` and allow-listed via
``ClaudeAgentOptions(skills=[...])``. Distributed-tracing headers from
``current_trace_headers()`` are merged into the MCP HTTP headers so MCP-side
spans link back to the experiment span.

When a gateway base URL is resolved, Claude Code is pointed at it through
``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_CUSTOM_HEADERS`` and authenticated either
via the configured ``apiKeyHelper`` (run by Claude Code itself with a TTL) or a
static ``ANTHROPIC_API_KEY``. Nothing gateway-specific is set otherwise, so the
SDK falls back to its own auth.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk import query as claude_query
from ddtrace.llmobs import LLMObs

from dd_ai_devx_evals.harness.base import (
    AgentRunner,
    AgentRunResult,
    AgentToolCall,
    ProgressCallback,
    _notify,
    json_dumps_compact,
    json_safe,
)
from dd_ai_devx_evals.mcp import McpServerSpec, configured_tool_names
from dd_ai_devx_evals.skills import stage_skills_for_claude
from dd_ai_devx_evals.tracing import current_trace_headers
from dd_ai_devx_evals.types import HarnessResult, ModelSpec, UsageMetrics, coerce_int

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

SDK_NAME = "claude-agent-sdk"
CLAUDE_MCP_CONNECT_TIMEOUT_SECONDS = 15.0
CLAUDE_MCP_CONNECT_POLL_SECONDS = 0.25
CLAUDE_DIAGNOSTIC_TAIL_LINES = 20
CLAUDE_DIAGNOSTIC_VALUE_MAX_CHARS = 1_000
# Claude Code reads the apiKeyHelper output with this TTL (ms) before re-running.
CLAUDE_API_KEY_HELPER_TTL_MS = "7200000"

# Built-in (non-MCP) tools shipped by Claude Code. Under ``permission_mode=
# "dontAsk"`` only tools pre-approved via ``allowed_tools`` may run, so when a
# scenario allows "all" built-ins (``allowed_builtin_tools`` omitted) the harness
# pre-approves this full set. ``Skill`` is intentionally excluded: it is enabled
# through the ``skills`` option, not ``allowed_tools``.
CLAUDE_BUILTIN_TOOLS: tuple[str, ...] = (
    "Task",
    "Bash",
    "BashOutput",
    "KillShell",
    "Glob",
    "Grep",
    "Read",
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "NotebookRead",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "ExitPlanMode",
    "SlashCommand",
)


class ClaudeRunner(AgentRunner):
    """Run eval prompts through ``claude-agent-sdk`` with native LLMObs spans."""

    sdk_name = SDK_NAME

    def __init__(
        self,
        *,
        cwd: str | Path,
        mcp_servers: list[McpServerSpec] | None = None,
        allowed_builtin_tools: Iterable[str] | None = None,
        skills: Iterable[str] = (),
        max_turns: int | None = None,
        effort: str | None = None,
        gateway_base_url: str | None = None,
        gateway_headers: Mapping[str, str] | None = None,
        gateway_credentials_helper: str | None = None,
        gateway_api_key: str | None = None,
        claude_query_func: Callable[..., Any] | None = None,
        claude_client_factory: Callable[[ClaudeAgentOptions], Any] | None = None,
    ) -> None:
        super().__init__(
            cwd=cwd,
            mcp_servers=mcp_servers,
            allowed_builtin_tools=allowed_builtin_tools,
            skills=skills,
            max_turns=max_turns,
            effort=effort,
        )
        self.gateway_base_url = gateway_base_url
        self.gateway_headers = dict(gateway_headers or {})
        self.gateway_credentials_helper = gateway_credentials_helper
        self.gateway_api_key = gateway_api_key
        self._claude_query = claude_query_func or claude_query
        self._uses_custom_claude_query = claude_query_func is not None
        self._claude_client_factory = claude_client_factory or ClaudeSDKClient

    async def run(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        harness: str,
        progress: ProgressCallback | None = None,
    ) -> HarnessResult:
        await _notify(progress, f"{harness}/{model.label}: {SDK_NAME} run")
        trace_headers = current_trace_headers()
        run_result = await self._run_claude(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_headers=trace_headers,
            progress=progress,
        )
        missing_usage_error: RuntimeError | None = None
        if run_result.error is None:
            try:
                _raise_if_missing_usage(run_result.usage, model=model)
            except RuntimeError as exc:
                missing_usage_error = exc
        # Annotate the active experiment span so token totals are complete even
        # though the native integration owns the agent/LLM/tool spans.
        self._annotate_experiment_usage(run_result.usage)
        if run_result.error is not None:
            raise run_result.error
        if missing_usage_error is not None:
            raise missing_usage_error
        return HarnessResult(
            answer=run_result.answer,
            usage=run_result.usage,
            tool_calls=[tool_call.to_record() for tool_call in run_result.tool_calls],
            harness=harness,
        )

    def _annotate_experiment_usage(self, usage: UsageMetrics) -> None:
        if not LLMObs.enabled:
            return
        metrics = usage.to_llmobs_metrics()
        if not metrics:
            return
        with contextlib.suppress(Exception):
            LLMObs.annotate(metrics=metrics)

    async def _run_claude(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        trace_headers: Mapping[str, str],
        progress: ProgressCallback | None,
    ) -> AgentRunResult:
        skill_names = stage_skills_for_claude(self.skills, self.cwd)
        diagnostics = _ClaudeEvalDiagnostics()
        options = ClaudeAgentOptions(
            # Do not set ``tools`` here: Claude's CLI documents it as a filter for
            # built-in tools, which can hide MCP tools from the active surface.
            # MCP tools are exposed through ``mcp_servers`` and auto-approved via
            # ``allowed_tools``.
            system_prompt={"type": "preset", "preset": "claude_code", "append": system_prompt},
            mcp_servers={server.name: server.to_claude_config(trace_headers) for server in self.mcp_servers},
            strict_mcp_config=True,
            allowed_tools=self._claude_available_tools(),
            permission_mode="dontAsk",
            # Skills staged under ``<cwd>/.claude/skills`` are project-scoped, so
            # the project setting source must be enabled for discovery. With no
            # skills the harness keeps the surface minimal (no setting sources).
            setting_sources=["project"] if skill_names else [],
            skills=skill_names or None,
            settings=self._gateway_settings(),
            env=self._gateway_env(),
            max_turns=self.max_turns,
            model=model.name,
            effort=self.effort,
            cwd=self.cwd,
            include_partial_messages=True,
            stderr=diagnostics.record_stderr,
        )
        try:
            if self._uses_custom_claude_query:
                agent_stream = self._claude_query(prompt=user_prompt, options=options)
                try:
                    return await _collect_claude_stream(
                        agent_stream,
                        model=model,
                        max_turns=self.max_turns,
                        progress=progress,
                        diagnostics=diagnostics,
                    )
                finally:
                    close = getattr(agent_stream, "aclose", None)
                    if close is not None:
                        with contextlib.suppress(Exception):
                            await close()

            client = self._claude_client_factory(options)
            try:
                await client.connect()
                await _wait_for_claude_mcp_servers(client, self.mcp_servers, diagnostics)
                await client.query(user_prompt)
                return await _collect_claude_stream(
                    client.receive_response(),
                    model=model,
                    max_turns=self.max_turns,
                    progress=progress,
                    diagnostics=diagnostics,
                )
            finally:
                with contextlib.suppress(Exception):
                    await client.disconnect()
        except Exception:  # pragma: no cover - defensive runtime path
            if diagnostic_summary := diagnostics.summary():
                logger.exception("Claude Agent SDK eval run failed\n%s", diagnostic_summary)
            else:
                logger.exception("Claude Agent SDK eval run failed")
            raise

    def _claude_available_tools(self) -> list[str]:
        # ``None`` means "all built-in tools allowed": pre-approve the full
        # built-in set so ``dontAsk`` does not deny them. An explicit list
        # (including an empty list) is an exact allow-list.
        if self.allowed_builtin_tools is None:
            names: list[str] = list(CLAUDE_BUILTIN_TOOLS)
        else:
            names = [*self.allowed_builtin_tools]
        for server in self.mcp_servers:
            tool_names = configured_tool_names(server)
            if tool_names:
                names.extend(f"mcp__{server.name}__{tool_name}" for tool_name in tool_names)
            else:
                # No allow-list configured: expose every tool the server offers.
                names.append(f"mcp__{server.name}")
        return list(dict.fromkeys(names))

    def _gateway_env(self) -> dict[str, str]:
        if not self.gateway_base_url:
            return {}
        env: dict[str, str] = {"ANTHROPIC_BASE_URL": self.gateway_base_url}
        if self.gateway_headers:
            env["ANTHROPIC_CUSTOM_HEADERS"] = _format_claude_custom_headers(self.gateway_headers)
        if self.gateway_credentials_helper:
            env["CLAUDE_CODE_API_KEY_HELPER_TTL_MS"] = CLAUDE_API_KEY_HELPER_TTL_MS
        elif self.gateway_api_key:
            env["ANTHROPIC_API_KEY"] = self.gateway_api_key
        return env

    def _gateway_settings(self) -> str | None:
        if not self.gateway_base_url or not self.gateway_credentials_helper:
            return None
        return json.dumps({"apiKeyHelper": self.gateway_credentials_helper}, separators=(",", ":"))


def _raise_if_missing_usage(usage: UsageMetrics, *, model: ModelSpec) -> None:
    if usage.total_tokens or usage.input_tokens or usage.output_tokens:
        return
    raise RuntimeError(f"{SDK_NAME} completed {model.label} without reporting token usage")


def _format_claude_custom_headers(headers: Mapping[str, str]) -> str:
    """Render headers in the newline-separated format Claude Code parses."""
    return "\n".join(f"{name}: {value}" for name, value in headers.items())


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
class _ClaudeEvalDiagnostics:
    """Bounded diagnostics captured from a Claude Agent SDK eval run."""

    def __init__(self) -> None:
        self._stderr_tail: deque[str] = deque(maxlen=CLAUDE_DIAGNOSTIC_TAIL_LINES)
        self._system_tail: deque[str] = deque(maxlen=CLAUDE_DIAGNOSTIC_TAIL_LINES)

    def record_stderr(self, line: str) -> None:
        for item in str(line).splitlines() or [str(line)]:
            if item:
                self._stderr_tail.append(_bounded_claude_diagnostic_value(item))

    def record_message(self, message: Any) -> None:
        if diagnostic := _claude_system_diagnostic(message):
            self._system_tail.append(diagnostic)
        if diagnostic := _claude_result_error_diagnostic(message):
            self._system_tail.append(diagnostic)

    def record_system(self, diagnostic: str) -> None:
        self._system_tail.append(_bounded_claude_diagnostic_value(diagnostic))

    def summary(self) -> str:
        parts: list[str] = []
        if self._stderr_tail:
            parts.append("claude CLI stderr tail:\n" + "\n".join(self._stderr_tail))
        if self._system_tail:
            parts.append("claude system diagnostic tail:\n" + "\n".join(self._system_tail))
        return "\n".join(parts)


def _bounded_claude_diagnostic_value(value: Any) -> str:
    text = str(value).replace("\r", "\\r")
    if len(text) > CLAUDE_DIAGNOSTIC_VALUE_MAX_CHARS:
        return f"{text[:CLAUDE_DIAGNOSTIC_VALUE_MAX_CHARS]}…"
    return text


def _bounded_json_diagnostic(value: Any) -> str:
    return _bounded_claude_diagnostic_value(json_dumps_compact(json_safe(value)))


# --------------------------------------------------------------------------- #
# MCP readiness
# --------------------------------------------------------------------------- #
async def _wait_for_claude_mcp_servers(
    client: Any,
    servers: Iterable[McpServerSpec],
    diagnostics: _ClaudeEvalDiagnostics,
) -> None:
    """Wait until configured MCP servers are connected before sending the prompt."""
    server_list = list(servers)
    server_names = [server.name for server in server_list]
    if not server_names:
        return

    deadline = time.monotonic() + CLAUDE_MCP_CONNECT_TIMEOUT_SECONDS
    expected_tools_by_server = {server.name: set(configured_tool_names(server)) for server in server_list}
    status_text = "<not checked>"
    while True:
        status = await client.get_mcp_status()
        status_text = _format_claude_mcp_status(server_names, status)
        readiness = _claude_mcp_readiness(server_names, status, expected_tools_by_server=expected_tools_by_server)
        if readiness == "ready":
            diagnostics.record_system(f"mcp_status {status_text}")
            return
        if readiness == "failed":
            diagnostics.record_system(f"mcp_status {status_text}")
            raise RuntimeError(f"Claude MCP server unavailable before eval prompt: {status_text}")
        if time.monotonic() >= deadline:
            diagnostics.record_system(f"mcp_status {status_text}")
            raise RuntimeError(f"Timed out waiting for Claude MCP server before eval prompt: {status_text}")
        await asyncio.sleep(CLAUDE_MCP_CONNECT_POLL_SECONDS)


def _claude_mcp_readiness(
    server_names: Iterable[str],
    status: Any,
    *,
    expected_tools_by_server: Mapping[str, set[str]],
) -> str:
    """Return ready, pending, or failed for configured Claude MCP servers."""
    by_name = _claude_mcp_status_by_name(status)
    for name in server_names:
        server_status = by_name.get(name)
        if server_status is None:
            return "pending"
        state = str(server_status.get("status") or "pending")
        if state in {"failed", "needs-auth", "disabled"}:
            return "failed"
        if state != "connected":
            return "pending"
        expected_tools = expected_tools_by_server.get(name, set())
        if expected_tools and not expected_tools.issubset(_claude_mcp_tool_names(server_status)):
            return "pending"
    return "ready"


def _format_claude_mcp_status(server_names: Iterable[str], status: Any) -> str:
    by_name = _claude_mcp_status_by_name(status)
    parts: list[str] = []
    for name in server_names:
        server_status = by_name.get(name)
        if server_status is None:
            parts.append(f"{name}=missing")
            continue
        state = str(server_status.get("status") or "unknown")
        tools = sorted(_claude_mcp_tool_names(server_status))
        tool_text = f" tools={','.join(tools)}" if tools else ""
        error = server_status.get("error")
        error_text = f" error={_bounded_claude_diagnostic_value(error)}" if error else ""
        parts.append(f"{name}={state}{tool_text}{error_text}")
    return "; ".join(parts) if parts else "<no MCP servers configured>"


def _claude_mcp_status_by_name(status: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(status, Mapping):
        return {}
    servers = status.get("mcpServers")
    if not isinstance(servers, list):
        return {}
    return {
        str(server.get("name")): server
        for server in servers
        if isinstance(server, Mapping) and isinstance(server.get("name"), str)
    }


def _claude_mcp_tool_names(server_status: Mapping[str, Any]) -> set[str]:
    tools = server_status.get("tools")
    if not isinstance(tools, list):
        return set()
    return {str(tool.get("name")) for tool in tools if isinstance(tool, Mapping) and tool.get("name")}


# --------------------------------------------------------------------------- #
# Stream consumption
# --------------------------------------------------------------------------- #
async def _collect_claude_stream(
    agent_stream: Any,
    *,
    model: ModelSpec,
    max_turns: int | None,
    progress: ProgressCallback | None,
    diagnostics: _ClaudeEvalDiagnostics,
) -> AgentRunResult:
    """Consume Claude SDK messages into the normalized eval result."""
    result_usage = UsageMetrics()
    assistant_usage = UsageMetrics()
    text_parts: list[str] = []
    final_result = ""
    tool_uses: dict[str, tuple[str, Any]] = {}
    tool_calls: list[AgentToolCall] = []
    provider_error: RuntimeError | None = None
    async for message in agent_stream:
        diagnostics.record_message(message)
        if progress_message := _claude_progress_message(message, model):
            await _notify(progress, progress_message)
        final_result = (
            _consume_claude_message(message, result_usage, assistant_usage, text_parts, tool_uses, tool_calls)
            or final_result
        )
        provider_error = _claude_result_error(message, model=model, max_turns=max_turns) or provider_error

    answer = final_result or "".join(text_parts).strip() or "Claude did not produce a final response."
    usage = _effective_claude_usage(result_usage, assistant_usage)
    return AgentRunResult(answer=answer, usage=usage, tool_calls=tool_calls, error=provider_error)


def _consume_claude_message(
    message: Any,
    result_usage: UsageMetrics,
    assistant_usage: UsageMetrics,
    text_parts: list[str],
    tool_uses: dict[str, tuple[str, Any]],
    tool_calls: list[AgentToolCall],
) -> str:
    message_type = type(message).__name__
    if message_type == "ResultMessage":
        message_usage = getattr(message, "usage", None) or _claude_usage_from_model_usage(
            getattr(message, "model_usage", None)
        )
        result_usage.add(
            UsageMetrics.from_claude_sdk(message_usage, total_cost_usd=getattr(message, "total_cost_usd", None))
        )
        result = getattr(message, "result", None)
        return str(result or "")
    if message_type in {"TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage"}:
        _consume_claude_task_message(message, tool_calls)
        return ""

    if message_type == "AssistantMessage":
        assistant_usage.add(UsageMetrics.from_claude_sdk(getattr(message, "usage", None)))
    for block in getattr(message, "content", []) or []:
        _consume_claude_block(block, text_parts, tool_uses, tool_calls)
    return ""


def _claude_usage_from_model_usage(model_usage: Any) -> dict[str, int]:
    if not isinstance(model_usage, Mapping):
        return {}

    usage_entries = [value for value in model_usage.values() if isinstance(value, Mapping)] or [model_usage]
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_creation_input_tokens": ("cache_creation_input_tokens", "cacheCreationInputTokens"),
        "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens"),
    }
    for entry in usage_entries:
        for target_key, source_keys in aliases.items():
            totals[target_key] += max(coerce_int(entry.get(source_key)) for source_key in source_keys)
    return {key: value for key, value in totals.items() if value}


def _consume_claude_block(
    block: Any,
    text_parts: list[str],
    tool_uses: dict[str, tuple[str, Any]],
    tool_calls: list[AgentToolCall],
) -> None:
    block_type = type(block).__name__
    if block_type == "TextBlock" or hasattr(block, "text"):
        text_parts.append(str(getattr(block, "text", "")))
        return
    if block_type in {"ToolUseBlock", "ServerToolUseBlock"} or (hasattr(block, "id") and hasattr(block, "name")):
        tool_uses[str(getattr(block, "id", ""))] = (str(getattr(block, "name", "")), getattr(block, "input", {}) or {})
        return
    if block_type in {"ToolResultBlock", "ServerToolResultBlock"} or hasattr(block, "tool_use_id"):
        tool_use_id = str(getattr(block, "tool_use_id", ""))
        name, arguments = tool_uses.get(tool_use_id, ("claude_tool", {}))
        error_value = getattr(block, "is_error", None)
        tool_calls.append(
            AgentToolCall(
                name=_normalize_tool_name(name),
                arguments=arguments,
                result=getattr(block, "content", None),
                error="tool_error" if error_value else None,
                metadata={"tool_use_id": tool_use_id, "source": SDK_NAME},
            )
        )


def _consume_claude_task_message(message: Any, tool_calls: list[AgentToolCall]) -> None:
    usage = getattr(message, "usage", None)
    metadata = {
        "source": SDK_NAME,
        "message_type": type(message).__name__,
        "task_id": getattr(message, "task_id", None),
        "status": getattr(message, "status", None),
        "last_tool_name": getattr(message, "last_tool_name", None),
    }
    if usage is not None:
        metadata["usage"] = json_safe(usage)
    tool_calls.append(
        AgentToolCall(
            name="claude.nested_agent",
            arguments={"description": getattr(message, "description", None)},
            result=getattr(message, "summary", None) or getattr(message, "output_file", None) or "",
            error="task_failed" if getattr(message, "status", None) == "failed" else None,
            metadata=metadata,
        )
    )


def _effective_claude_usage(result_usage: UsageMetrics, assistant_usage: UsageMetrics) -> UsageMetrics:
    """Prefer aggregate ResultMessage usage, falling back to AssistantMessage usage."""
    if _usage_has_token_counts(result_usage):
        return result_usage
    if not _usage_has_token_counts(assistant_usage):
        return result_usage
    if result_usage.estimated_cost_usd and not assistant_usage.estimated_cost_usd:
        assistant_usage.estimated_cost_usd = result_usage.estimated_cost_usd
    return assistant_usage


def _usage_has_token_counts(usage: UsageMetrics) -> bool:
    return bool(usage.total_tokens or usage.input_tokens or usage.output_tokens)


# --------------------------------------------------------------------------- #
# Error detection
# --------------------------------------------------------------------------- #
def _claude_system_diagnostic(message: Any) -> str | None:
    if getattr(message, "subtype", None) != "api_retry":
        return None

    data = getattr(message, "data", None)
    if not isinstance(data, Mapping):
        return "api_retry"

    parts = ["api_retry"]
    status = data.get("error_status")
    if status is not None:
        parts.append(f"status={_bounded_claude_diagnostic_value(status)}")

    error = data.get("error")
    if isinstance(error, Mapping):
        error = error.get("type") or error.get("error") or error.get("message")
    if error is not None:
        parts.append(f"error={_bounded_claude_diagnostic_value(error)}")

    attempt = data.get("attempt")
    max_retries = data.get("max_retries")
    if attempt is not None and max_retries is not None:
        parts.append(
            f"attempt={_bounded_claude_diagnostic_value(attempt)}/{_bounded_claude_diagnostic_value(max_retries)}"
        )
    elif attempt is not None:
        parts.append(f"attempt={_bounded_claude_diagnostic_value(attempt)}")

    retry_delay_ms = data.get("retry_delay_ms")
    if retry_delay_ms is not None:
        parts.append(f"retry_delay_ms={_bounded_claude_diagnostic_value(retry_delay_ms)}")

    return " ".join(parts)


def _claude_result_error(message: Any, *, model: ModelSpec, max_turns: int | None) -> RuntimeError | None:
    if not _is_claude_error_result(message):
        return None
    summary = _claude_result_error_summary(message, max_turns=max_turns)
    return RuntimeError(f"{SDK_NAME} returned an error result for {model.label}: {summary}")


def _claude_result_error_diagnostic(message: Any) -> str | None:
    if not _is_claude_error_result(message):
        return None
    return f"result_error {_claude_result_error_summary(message, max_turns=None)}"


def _is_claude_error_result(message: Any) -> bool:
    return type(message).__name__ == "ResultMessage" and bool(getattr(message, "is_error", False))


def _claude_result_error_summary(message: Any, *, max_turns: int | None) -> str:
    parts: list[str] = []
    errors = getattr(message, "errors", None) or []
    if isinstance(errors, Iterable) and not isinstance(errors, str | bytes | Mapping):
        error_text = "; ".join(_bounded_claude_diagnostic_value(error) for error in errors if error)
        if error_text:
            parts.append(error_text)
    elif errors:
        parts.append(_bounded_claude_diagnostic_value(errors))

    for attr in ("subtype", "api_error_status", "stop_reason"):
        value = getattr(message, attr, None)
        if value not in (None, ""):
            parts.append(f"{attr}={_bounded_claude_diagnostic_value(value)}")

    num_turns = getattr(message, "num_turns", None)
    if num_turns is not None:
        turns = _bounded_claude_diagnostic_value(num_turns)
        if max_turns is not None:
            turns = f"{turns}/{_bounded_claude_diagnostic_value(max_turns)}"
        parts.append(f"turns={turns}")

    permission_denials = getattr(message, "permission_denials", None)
    if permission_denials:
        parts.append(f"permission_denials={_bounded_json_diagnostic(permission_denials)}")

    deferred_tool_use = getattr(message, "deferred_tool_use", None)
    if deferred_tool_use:
        parts.append(f"deferred_tool_use={_bounded_json_diagnostic(deferred_tool_use)}")

    return "; ".join(parts) if parts else "unknown Claude error result"


# --------------------------------------------------------------------------- #
# Progress rendering
# --------------------------------------------------------------------------- #
def _claude_progress_message(message: Any, model: ModelSpec) -> str | None:
    if diagnostic := _claude_system_diagnostic(message):
        return f"{model.label}: {diagnostic}"

    message_type = type(message).__name__
    if message_type == "ResultMessage":
        if diagnostic := _claude_result_error_diagnostic(message):
            return f"{model.label}: Claude error result - {diagnostic.removeprefix('result_error ')}"
        usage = UsageMetrics.from_claude_sdk(
            getattr(message, "usage", None) or _claude_usage_from_model_usage(getattr(message, "model_usage", None))
        )
        return f"{model.label}: final answer ready{_usage_suffix(usage.total_tokens)}"
    if message_type == "TaskStartedMessage":
        description = str(getattr(message, "description", "") or getattr(message, "task_id", "nested agent"))
        return f"{model.label}: nested agent started - {_truncate_progress_detail(description)}"
    if message_type == "TaskProgressMessage":
        description = str(getattr(message, "description", "") or getattr(message, "task_id", "nested agent"))
        last_tool = str(getattr(message, "last_tool_name", "") or "")
        last_tool_text = f"; last tool {_human_tool_name(last_tool)}" if last_tool else ""
        return (
            f"{model.label}: nested agent working - {_truncate_progress_detail(description)}"
            f"{last_tool_text}{_task_usage_suffix(getattr(message, 'usage', None))}"
        )
    if message_type == "TaskNotificationMessage":
        status = str(getattr(message, "status", "updated") or "updated")
        summary = str(getattr(message, "summary", "") or getattr(message, "task_id", "nested agent"))
        return (
            f"{model.label}: nested agent {status} - {_truncate_progress_detail(summary)}"
            f"{_task_usage_suffix(getattr(message, 'usage', None))}"
        )

    for block in getattr(message, "content", []) or []:
        tool_name = _claude_tool_name_from_block(block)
        if tool_name:
            detail = _tool_input_summary(tool_name, getattr(block, "input", {}) or {})
            detail_text = f" - {detail}" if detail else ""
            return f"{model.label}: calling {_human_tool_name(tool_name)}{detail_text}"
    return None


def _claude_tool_name_from_block(block: Any) -> str | None:
    block_type = type(block).__name__
    if block_type in {"ToolUseBlock", "ServerToolUseBlock"} or (hasattr(block, "id") and hasattr(block, "name")):
        return str(getattr(block, "name", "") or "") or None
    return None


def _human_tool_name(name: str) -> str:
    normalized = _normalize_tool_name(name)
    if normalized.startswith("mcp."):
        _, server, tool = normalized.split(".", 2)
        return f"MCP {server}.{tool}"
    return normalized


def _tool_input_summary(name: str, arguments: Any) -> str:
    if not isinstance(arguments, Mapping):
        return _truncate_progress_detail(str(arguments)) if arguments else ""

    normalized_name = _normalize_tool_name(name).lower()
    priority_keys = _tool_summary_priority_keys(normalized_name)
    for key in priority_keys:
        if key in arguments and arguments[key] not in (None, ""):
            return _format_tool_argument(key, arguments[key])

    for key, value in arguments.items():
        if isinstance(value, str | int | float | bool) and value not in (None, ""):
            return _format_tool_argument(str(key), value)
    if arguments:
        return _truncate_progress_detail(json_dumps_compact(arguments))
    return ""


def _tool_summary_priority_keys(normalized_name: str) -> tuple[str, ...]:
    if "websearch" in normalized_name or "web_search" in normalized_name:
        return ("query", "search", "prompt")
    if "webfetch" in normalized_name or "web_fetch" in normalized_name:
        return ("url", "uri", "prompt")
    if normalized_name.endswith("bash") or "bash" in normalized_name:
        return ("command", "cmd", "description")
    if "grep" in normalized_name:
        return ("pattern", "query", "path", "include")
    if "glob" in normalized_name:
        return ("pattern", "path")
    if "read" in normalized_name:
        return ("file_path", "path")
    if "task" in normalized_name or "agent" in normalized_name:
        return ("description", "prompt", "task", "subagent_type")
    return ("prompt", "query", "url", "command", "pattern", "path", "file_path", "description")


def _format_tool_argument(key: str, value: Any) -> str:
    label_by_key = {
        "cmd": "command",
        "command": "command",
        "description": "description",
        "file_path": "file",
        "include": "include",
        "path": "path",
        "pattern": "pattern",
        "prompt": "prompt",
        "query": "query",
        "search": "query",
        "subagent_type": "agent",
        "task": "task",
        "uri": "url",
        "url": "url",
    }
    label = label_by_key.get(key, key)
    return f"{label}: {_truncate_progress_detail(str(value))}"


def _task_usage_suffix(usage: Any) -> str:
    if usage is None:
        return ""
    tokens = coerce_int(getattr(usage, "total_tokens", 0))
    tool_uses = coerce_int(getattr(usage, "tool_uses", 0))
    duration_ms = coerce_int(getattr(usage, "duration_ms", 0))
    parts: list[str] = []
    if tokens:
        parts.append(f"{tokens:,} tokens")
    if tool_uses:
        parts.append(f"{tool_uses:,} tool uses")
    if duration_ms:
        parts.append(f"{duration_ms / 1000:.1f}s")
    return f" ({', '.join(parts)})" if parts else ""


def _usage_suffix(total_tokens: int) -> str:
    return f" ({total_tokens:,} tokens)" if total_tokens else ""


def _truncate_progress_detail(value: str, *, max_length: int = 180) -> str:
    single_line = " ".join(value.split())
    if len(single_line) <= max_length:
        return single_line
    return f"{single_line[: max_length - 1]}…"


def _normalize_tool_name(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return f"mcp.{parts[1]}.{parts[2]}"
    return name
