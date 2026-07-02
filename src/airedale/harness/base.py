# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

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
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs

from airedale.mcp import (
    McpServerSpec,
    McpToolMetadata,
    configured_tool_names,
    provider_mcp_tool_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from airedale.types import HarnessResult, ModelSpec, UsageMetrics

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
class AgentThought:
    """A non-tool assistant output segment (reasoning, plan, or intermediate text).

    Mirrors how the native ``claude_agent_sdk`` integration surfaces a
    ``ThinkingBlock``: as a plain assistant message interleaved in the output
    stream (see ``parse_content_blocks``). ``kind`` is carried for reporting and
    debugging but, like the native path, does not change the emitted message
    shape (both reasoning and text become a bare ``assistant`` message).
    """

    text: str
    kind: str = "reasoning"


#: An ordered assistant output segment: either a tool invocation or a
#: reasoning/plan/text ``AgentThought``. Providers that reconstruct spans by
#: hand (Codex) emit these in item order so the manual transcript interleaves
#: reasoning and tool calls the way the native Claude integration does.
AgentOutputSegment = AgentToolCall | AgentThought


@dataclass
class AgentRunResult:
    """Normalized provider SDK run output plus deferred failure state."""

    answer: str
    usage: UsageMetrics
    tool_calls: list[AgentToolCall]
    segments: list[AgentOutputSegment] = field(default_factory=list)
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


def _input_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    """Render the run's input as a role/content message list.

    Both the ``agent`` and ``llm`` spans share this shape so Codex's input mirrors
    the message-list structure the Claude integration emits (a serialized message
    list on the agent span, structured ``input.messages`` on the llm span). The
    system prompt is kept as a leading ``system`` message rather than dropped, so
    the actually-sent developer instructions stay visible.
    """
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _tool_call_id(tool_call: AgentToolCall) -> str | None:
    """Return a stable tool-call id from the normalized metadata, if any."""
    metadata = tool_call.metadata if isinstance(tool_call.metadata, Mapping) else {}
    tool_id = metadata.get("tool_use_id") or metadata.get("tool_id")
    return str(tool_id) if tool_id else None


def _tool_result_text(result: Any) -> str:
    """Stringify a tool result the way the Claude integration does.

    Plain strings pass through; structured results are JSON-serialized so a
    ``ToolResult.result`` always carries a string (matching ``safe_json`` in the
    native integration).
    """
    if isinstance(result, str):
        return result
    return json_dumps_compact(json_safe(result))


def _tool_call_entry(tool_call: AgentToolCall) -> dict[str, Any]:
    """Render one ``tool_use`` entry (LLMObs ``ToolCall`` shape)."""
    arguments = tool_call.arguments if isinstance(tool_call.arguments, Mapping) else {"value": tool_call.arguments}
    entry: dict[str, Any] = {"name": tool_call.name, "arguments": json_safe(arguments), "type": "tool_use"}
    tool_id = _tool_call_id(tool_call)
    if tool_id:
        entry["tool_id"] = tool_id
    return entry


def _tool_result_entry(tool_call: AgentToolCall) -> dict[str, Any]:
    """Render one ``tool_result`` entry (LLMObs ``ToolResult`` shape)."""
    entry: dict[str, Any] = {
        "name": tool_call.name,
        "result": _tool_result_text(tool_call.result),
        "type": "tool_result",
    }
    tool_id = _tool_call_id(tool_call)
    if tool_id:
        entry["tool_id"] = tool_id
    return entry


def _thought_message(thought: AgentThought) -> dict[str, Any]:
    """Render a reasoning/plan/text segment as a bare assistant message.

    Matches the native integration's handling of a ``ThinkingBlock`` (and of a
    plain ``TextBlock``): the content text becomes an ``assistant`` message. The
    ``kind`` is intentionally not encoded in the message so the shape stays
    identical to what Claude emits.
    """
    return {"role": "assistant", "content": thought.text}


def _tool_use_message(tool_call: AgentToolCall) -> dict[str, Any]:
    """Render one assistant ``tool_use`` message (LLMObs ``ToolCall`` shape)."""
    return {"role": "assistant", "content": "", "tool_calls": [_tool_call_entry(tool_call)]}


def _llm_output_messages(answer: str, segments: list[AgentOutputSegment]) -> list[dict[str, Any]]:
    """Render the llm span output as interleaved assistant messages.

    Mirrors the Claude integration's ``parse_content_blocks`` output: reasoning
    (and plan) segments become bare assistant messages and each tool invocation
    is its own assistant message carrying a single ``tool_calls`` entry, in the
    original item order, followed by a final assistant message holding the answer
    text. Tool *results* are intentionally not emitted here — they live on the
    dedicated tool spans and in the agent transcript, exactly as in the native
    path.
    """
    messages: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, AgentThought):
            messages.append(_thought_message(segment))
        else:
            messages.append(_tool_use_message(segment))
    messages.append({"role": "assistant", "content": answer})
    return messages


def _agent_transcript_messages(answer: str, segments: list[AgentOutputSegment]) -> list[dict[str, Any]]:
    """Render the agent span's full transcript as a role/content message list.

    This matches the Claude agent span's serialized output: an interleaved
    sequence of assistant reasoning messages, assistant ``tool_use`` messages and
    ``user`` ``tool_result`` messages (in item order), terminated by the final
    assistant answer. For a non-``llm`` span LLMObs serializes this list into
    ``output.value`` (a JSON string), just like the native integration.
    """
    messages: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, AgentThought):
            messages.append(_thought_message(segment))
            continue
        messages.append(_tool_use_message(segment))
        messages.append({"role": "user", "content": "", "tool_results": [_tool_result_entry(segment)]})
    messages.append({"role": "assistant", "content": answer})
    return messages


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
    segments: list[AgentOutputSegment],
    allowed_builtin_tools: Iterable[str] | None = None,
    mcp_tool_metadata: Mapping[tuple[str, str], McpToolMetadata] | None = None,
) -> None:
    """Annotate the manual ``llm`` span with I/O, usage, and tool definitions."""
    if not LLMObs.enabled:
        return
    tool_calls = [segment for segment in segments if isinstance(segment, AgentToolCall)]
    LLMObs.annotate(
        prompt={
            "id": LLMOBS_PROMPT_ID,
            "version": prompt_version,
            "template": "{{prompt}}",
            "variables": {"prompt_version": prompt_version, "harness": harness, "model_name": model.label},
        },
        input_data=_input_messages(system_prompt, user_prompt),
        output_data=_llm_output_messages(answer, segments),
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


def _agent_manifest(
    *,
    framework: str,
    model: ModelSpec,
    sdk_name: str,
    mcp_servers: list[McpServerSpec],
    allowed_builtin_tools: Iterable[str] | None,
    max_turns: int | None,
) -> dict[str, Any]:
    """Build the LLMObs ``agent_manifest`` mirroring the native Claude manifest.

    Shape parity with ``ClaudeAgentSdkIntegration._build_agent_manifest``:
    ``framework`` / ``model`` / ``tools`` (name-only) / ``dependencies.mcp_servers``
    / ``max_iterations``. Tool names are derived from the same builtin + MCP
    definitions used for the llm span (descriptions/schemas omitted — the manifest
    only lists names).
    """
    manifest: dict[str, Any] = {"framework": framework}
    if model.name:
        manifest["model"] = model.name
    tool_names = [
        definition["name"]
        for definition in _llmobs_tool_definitions(
            mcp_servers, sdk_name=sdk_name, allowed_builtin_tools=allowed_builtin_tools
        )
        if definition.get("name")
    ]
    if tool_names:
        manifest["tools"] = [{"name": name} for name in tool_names]
    if mcp_servers:
        manifest["dependencies"] = {"mcp_servers": [server.to_safe_dict() for server in mcp_servers]}
    if max_turns:
        manifest["max_iterations"] = max_turns
    return manifest


def _annotate_agent_loop_span(
    *,
    model: ModelSpec,
    sdk_name: str,
    framework: str,
    prompt_version: str,
    harness: str,
    system_prompt: str,
    user_prompt: str,
    answer: str,
    usage: UsageMetrics,
    mcp_servers: list[McpServerSpec],
    segments: list[AgentOutputSegment],
    allowed_builtin_tools: Iterable[str] | None = None,
    max_turns: int | None = None,
) -> None:
    """Annotate the outer ``agent`` span to match the native Claude agent span.

    Input/output are role/content message lists (serialized by LLMObs into
    ``input.value`` / ``output.value`` for this non-``llm`` span) and an
    ``agent_manifest`` is attached via ``metadata._dd.agent_manifest`` — the same
    location the native integration writes it.
    """
    if not LLMObs.enabled:
        return
    tool_calls = [segment for segment in segments if isinstance(segment, AgentToolCall)]
    metadata: dict[str, Any] = {
        "harness": harness,
        "reported_model_name": model.label,
        "sdk": sdk_name,
        "mcp_servers": [server.to_safe_dict() for server in mcp_servers],
        "tool_call_count": len(tool_calls),
    }
    manifest = _agent_manifest(
        framework=framework,
        model=model,
        sdk_name=sdk_name,
        mcp_servers=mcp_servers,
        allowed_builtin_tools=allowed_builtin_tools,
        max_turns=max_turns,
    )
    # `_dd.agent_manifest` is the location the native integration writes the
    # manifest; the public `annotate(metadata=...)` merges into the same dict.
    metadata["_dd"] = {"agent_manifest": manifest}
    LLMObs.annotate(
        input_data=_input_messages(system_prompt, user_prompt),
        output_data=_agent_transcript_messages(answer, segments),
        metadata=metadata,
        metrics=usage.to_llmobs_metrics(),
        tags={"harness": harness, "prompt_version": prompt_version, "model_name": model.label, "sdk": sdk_name},
    )


def _emit_tool_spans(tool_calls: list[AgentToolCall]) -> None:
    """Emit one ``tool``-kind child span per tool call.

    This reproduces the native Claude topology (``agent → llm`` + sibling ``tool``
    spans). Each tool span carries the raw arguments/result as ``input.value`` /
    ``output.value`` (LLMObs serializes them) and the normalized metadata,
    matching ``ClaudeAgentSdkIntegration._llmobs_set_tool_tags``. Call this while
    the parent ``agent`` span is active so the tool spans become its children.
    """
    if not LLMObs.enabled:
        return
    for tool_call in tool_calls:
        # Telemetry must never abort or mask a completed run: a failure here would
        # otherwise run *before* the caller re-raises the real provider error.
        with contextlib.suppress(Exception):
            _emit_tool_span(tool_call)


def _emit_tool_span(tool_call: AgentToolCall) -> None:
    """Emit a single ``tool``-kind span for one tool call."""
    metadata = dict(tool_call.metadata) if isinstance(tool_call.metadata, Mapping) else {}
    tool_id = _tool_call_id(tool_call)
    if tool_id:
        metadata.setdefault("tool_id", tool_id)
    if tool_call.error:
        metadata.setdefault("error", tool_call.error)
    with LLMObs.tool(name=tool_call.name) as span:
        LLMObs.annotate(
            span=span,
            input_data=json_safe(tool_call.arguments),
            output_data=json_safe(tool_call.result),
            metadata=metadata,
        )


class AgentRunner(ABC):
    """Run one eval prompt through a provider-native agent SDK.

    Subclasses wire the provider SDK (MCP servers, skills, gateway auth) and
    decide how LLMObs spans are produced. Shared run state (MCP specs, allowed
    builtin tools, staged skills, turn budget, reasoning effort, working
    directory) is stored here. ``cwd`` is required and owned by the caller (the
    ``WorkspaceManager`` in production, a ``tmp_path`` in tests); the runner
    never creates or cleans up working directories itself.
    """

    #: Name reported on spans/metadata; overridden per provider runner.
    sdk_name: str = "agent-sdk"

    def __init__(
        self,
        *,
        cwd: str | Path,
        mcp_servers: list[McpServerSpec] | None = None,
        allowed_builtin_tools: Iterable[str] | None = None,
        skills: Iterable[str] = (),
        max_turns: int | None = None,
        effort: str | None = None,
    ) -> None:
        self.mcp_servers = list(mcp_servers or [])
        # ``None`` is the "all built-in tools allowed" sentinel; preserve it.
        self.allowed_builtin_tools = None if allowed_builtin_tools is None else tuple(allowed_builtin_tools)
        self.skills = list(skills)
        self.max_turns = max_turns
        self.effort = effort
        self.cwd = str(cwd)

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
