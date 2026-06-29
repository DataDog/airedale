"""Shared base for provider-native agent SDK runners.

This module hosts the provider-agnostic pieces of the harness layer:

- ``AgentToolCall`` / ``AgentRunResult`` value objects normalising SDK output.
- ``AgentRunner`` — the abstract base every provider runner implements.
- LLMObs span-annotation helpers shared by both providers. Claude relies on the
  native ``claude-agent-sdk`` ddtrace integration (so it only needs the
  experiment-span usage annotation), while Codex has no native integration and
  therefore drives the manual ``@agent`` / ``@llm`` decorator spans through
  these helpers.

The provider/model split is preserved on every span (``model.provider`` /
``model.name``) so the Datadog backend can compute estimated cost, while
``model.label`` is carried in tags/metadata for human-facing reporting.
"""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs

from dd_ai_devx_evals.mcp import (
    McpServerSpec,
    McpToolMetadata,
    configured_tool_names,
    provider_mcp_tool_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dd_ai_devx_evals.types import HarnessResult, ModelSpec, UsageMetrics

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None] | None]

# Identifier carried in the LLMObs prompt annotation; domain-neutral.
LLMOBS_PROMPT_ID = "dd-ai-devx-eval"


def json_safe(value: Any) -> Any:
    """Return a JSON-compatible value without stringifying structured objects."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        for kwargs in ({"mode": "json", "by_alias": True, "exclude_none": True}, {}):
            with contextlib.suppress(Exception):
                return json_safe(model_dump(**kwargs))

    if hasattr(value, "__dict__"):
        with contextlib.suppress(Exception):
            return json_safe(vars(value))

    return str(value)


def json_dumps_compact(value: Any) -> str:
    """Serialize values compactly for span input/output fields."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


async def _notify(progress: ProgressCallback | None, message: str) -> None:
    """Invoke a progress callback, awaiting it when it returns a coroutine."""
    if progress is None:
        return
    maybe_coro = progress(message)
    if maybe_coro is not None:
        await maybe_coro


@dataclass
class AgentToolCall:
    """Tool-call event observed from a provider agent SDK run."""

    name: str
    arguments: Any = field(default_factory=dict)
    result: Any = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable tool-call record for experiment output."""
        return {
            "name": self.name,
            "arguments": json_safe(self.arguments),
            "result": json_safe(self.result),
            "error": self.error,
            "metadata": json_safe(self.metadata),
        }


@dataclass
class AgentRunResult:
    """Normalized provider SDK run output plus deferred failure state."""

    answer: str
    usage: UsageMetrics
    tool_calls: list[AgentToolCall]
    error: RuntimeError | None = None


def _decorate_when_llmobs_enabled(
    func: Callable[..., Any],
    decorator: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> Callable[..., Any]:
    """Apply a LLMObs decorator only when it would create a span."""
    if not LLMObs.enabled:
        return func
    return decorator(func)


def _raise_if_missing_usage(usage: UsageMetrics, *, sdk_name: str, model: ModelSpec) -> None:
    """Fail loudly when an SDK reports zero usage (broken token accounting)."""
    if usage.total_tokens or usage.input_tokens or usage.output_tokens:
        return
    raise RuntimeError(f"{sdk_name} completed {model.label} without reporting token usage")


def _span_messages(value: Any) -> str:
    """Serialize structured span I/O payloads compactly."""
    return json_dumps_compact(value)


def _llm_output_message(answer: str, tool_calls: list[AgentToolCall]) -> dict[str, Any]:
    """Render the assistant message (plus tool calls) for an LLMObs llm span."""
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    formatted_tool_calls = []
    for tool_call in tool_calls:
        arguments = tool_call.arguments if isinstance(tool_call.arguments, Mapping) else {"value": tool_call.arguments}
        formatted_tool_call: dict[str, Any] = {"name": tool_call.name, "arguments": json_safe(arguments)}
        tool_id = tool_call.metadata.get("tool_use_id") if isinstance(tool_call.metadata, Mapping) else None
        if tool_id:
            formatted_tool_call["tool_id"] = str(tool_id)
        formatted_tool_calls.append(formatted_tool_call)
    if formatted_tool_calls:
        message["tool_calls"] = formatted_tool_calls
    return message


def _builtin_tool_definitions(allowed_builtin_tools: Iterable[str] | None) -> list[dict[str, Any]]:
    """Return name-only definitions for the provider's allow-listed builtins.

    ``None`` means "all built-in tools allowed"; since the full provider-native
    set is not enumerated here, no explicit builtin definitions are emitted in
    that case.
    """
    if allowed_builtin_tools is None:
        return []
    return [
        {"name": name, "schema": {"type": "object", "properties": {}}} for name in dict.fromkeys(allowed_builtin_tools)
    ]


def _mcp_tool_definitions(
    mcp_servers: list[McpServerSpec],
    *,
    sdk_name: str,
    mcp_tool_metadata: Mapping[tuple[str, str], McpToolMetadata] | None = None,
) -> list[dict[str, Any]]:
    """Return LLMObs tool definitions for the configured MCP tools."""
    definitions = []
    for server in mcp_servers:
        for tool_name in configured_tool_names(server):
            definition: dict[str, Any] = {"name": provider_mcp_tool_name(server.name, tool_name, sdk_name=sdk_name)}
            metadata = mcp_tool_metadata.get((server.name, tool_name)) if mcp_tool_metadata else None
            if metadata is not None:
                if metadata.description:
                    definition["description"] = metadata.description
                if metadata.input_schema is not None:
                    definition["schema"] = metadata.input_schema
            definitions.append(definition)
    return definitions


def _llmobs_tool_definitions(
    mcp_servers: list[McpServerSpec],
    *,
    sdk_name: str,
    allowed_builtin_tools: Iterable[str] | None = None,
    mcp_tool_metadata: Mapping[tuple[str, str], McpToolMetadata] | None = None,
) -> list[dict[str, Any]]:
    """Return all LLMObs tool definitions (builtins + MCP) for a run."""
    definitions: list[dict[str, Any]] = list(_builtin_tool_definitions(allowed_builtin_tools))
    definitions.extend(_mcp_tool_definitions(mcp_servers, sdk_name=sdk_name, mcp_tool_metadata=mcp_tool_metadata))
    return definitions


def _annotate_llm_span(
    *,
    model: ModelSpec,
    sdk_name: str,
    prompt_version: str,
    harness: str,
    system_prompt: str,
    user_prompt: str,
    answer: str,
    usage: UsageMetrics,
    mcp_servers: list[McpServerSpec],
    tool_calls: list[AgentToolCall],
    allowed_builtin_tools: Iterable[str] | None = None,
    mcp_tool_metadata: Mapping[tuple[str, str], McpToolMetadata] | None = None,
) -> None:
    """Annotate the manual ``llm`` span with I/O, usage, and tool definitions."""
    if not LLMObs.enabled:
        return
    LLMObs.annotate(
        prompt={
            "id": LLMOBS_PROMPT_ID,
            "version": prompt_version,
            "template": "{{prompt}}",
            "variables": {"prompt_version": prompt_version, "harness": harness, "model_name": model.label},
        },
        input_data=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        output_data=_llm_output_message(answer, tool_calls),
        metadata={
            "harness": harness,
            "reported_model_name": model.label,
            "sdk": sdk_name,
            "mcp_servers": [server.to_safe_dict() for server in mcp_servers],
            "tool_call_count": len(tool_calls),
        },
        metrics=usage.to_llmobs_metrics(),
        tags={"harness": harness, "prompt_version": prompt_version, "model_name": model.label, "sdk": sdk_name},
        tool_definitions=_llmobs_tool_definitions(
            mcp_servers,
            sdk_name=sdk_name,
            allowed_builtin_tools=allowed_builtin_tools,
            mcp_tool_metadata=mcp_tool_metadata,
        ),
    )


def _annotate_agent_loop_span(
    *,
    model: ModelSpec,
    sdk_name: str,
    prompt_version: str,
    harness: str,
    system_prompt: str,
    user_prompt: str,
    answer: str,
    usage: UsageMetrics,
    mcp_servers: list[McpServerSpec],
    tool_calls: list[AgentToolCall],
) -> None:
    """Annotate the outer ``agent`` span with structured I/O and usage."""
    if not LLMObs.enabled:
        return
    LLMObs.annotate(
        input_data=_span_messages(
            {"system_prompt": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}
        ),
        output_data=_span_messages(
            {"answer": answer, "tool_calls": [tool_call.to_record() for tool_call in tool_calls]}
        ),
        metadata={
            "harness": harness,
            "reported_model_name": model.label,
            "sdk": sdk_name,
            "mcp_servers": [server.to_safe_dict() for server in mcp_servers],
            "tool_call_count": len(tool_calls),
        },
        metrics=usage.to_llmobs_metrics(),
        tags={"harness": harness, "prompt_version": prompt_version, "model_name": model.label, "sdk": sdk_name},
    )


class AgentRunner(ABC):
    """Run one eval prompt through a provider-native agent SDK.

    Subclasses wire the provider SDK (MCP servers, skills, gateway auth) and
    decide how LLMObs spans are produced. Shared run state (MCP specs, allowed
    builtin tools, staged skills, turn budget, reasoning effort, working
    directory) is stored here; a temporary ``cwd`` is created and cleaned up
    automatically when none is supplied.
    """

    #: Name reported on spans/metadata; overridden per provider runner.
    sdk_name: str = "agent-sdk"

    def __init__(
        self,
        *,
        mcp_servers: list[McpServerSpec] | None = None,
        allowed_builtin_tools: Iterable[str] | None = None,
        skills: Iterable[str] = (),
        max_turns: int | None = None,
        effort: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.mcp_servers = list(mcp_servers or [])
        # ``None`` is the "all built-in tools allowed" sentinel; preserve it.
        self.allowed_builtin_tools = None if allowed_builtin_tools is None else tuple(allowed_builtin_tools)
        self.skills = list(skills)
        self.max_turns = max_turns
        self.effort = effort
        self._temp_cwd: tempfile.TemporaryDirectory[str] | None = None
        if cwd is None:
            self._temp_cwd = tempfile.TemporaryDirectory(prefix="dd-ai-devx-eval-")
            self.cwd = self._temp_cwd.name
        else:
            self.cwd = str(cwd)

    def __del__(self) -> None:
        if self._temp_cwd is not None:
            with contextlib.suppress(Exception):
                self._temp_cwd.cleanup()

    @abstractmethod
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
        """Run one eval prompt and return the structured harness result."""
