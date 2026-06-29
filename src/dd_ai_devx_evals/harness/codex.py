"""OpenAI Codex runner with decorator-based LLMObs instrumentation.

``openai`` models run through ``openai-codex``. Codex has **no native ddtrace
LLMObs integration**, so this runner manufactures the spans manually: the run is
wrapped in an outer ``@agent`` span (the agentic loop) and an inner ``@llm`` span
(the model call). Both decorators are applied only when ``LLMObs.enabled`` so we
never emit empty spans, and both spans are annotated with I/O, usage, and tool
definitions. This mirrors the Claude path's span shape even though Claude's spans
come from its native integration.

MCP servers are wired through ``CodexConfig(config_overrides=...)`` rendered by
``McpServerSpec.to_codex_config_overrides``; distributed-tracing headers from
``current_trace_headers()`` are merged into the per-server ``http_headers``
override so MCP-side spans link back to the experiment span (HTTP transport
only; stdio servers cannot receive per-request headers).

Skills limitation
-----------------
Codex has no per-thread skills allow-list. Skill packages are staged into
``<cwd>/.codex/skills/<name>`` (repo scope) so Codex discovers them, and the
thread ``config`` carries a ``skill_approval`` flag so the non-interactive run is
permitted to invoke them. Unlike Claude, individual skills cannot be selectively
allow-listed per run; all staged skills are discoverable.

Gateway limitation
------------------
When an ``openai`` gateway is resolved, Codex is pointed at it via
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` in the runner env plus a custom
``model_providers`` config override. Static gateway headers are passed through
the config override where supported; per-request header injection beyond the MCP
boundary is not available through the Codex CLI, so gateways requiring custom
auth headers on the model API itself may not be fully supported.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent, llm
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

from dd_ai_devx_evals.harness.base import (
    AgentRunner,
    AgentRunResult,
    AgentToolCall,
    ProgressCallback,
    _annotate_agent_loop_span,
    _annotate_llm_span,
    _decorate_when_llmobs_enabled,
    _notify,
    _raise_if_missing_usage,
)
from dd_ai_devx_evals.mcp import McpServerSpec, _mcp_tool_metadata_catalog
from dd_ai_devx_evals.skills import stage_skills_for_codex
from dd_ai_devx_evals.tracing import current_trace_headers
from dd_ai_devx_evals.types import HarnessResult, ModelSpec, UsageMetrics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SDK_NAME = "openai-codex"
# Custom model-provider id registered through config overrides for gateway use.
CODEX_GATEWAY_PROVIDER_ID = "dd_ai_devx_gateway"


class CodexRunner(AgentRunner):
    """Run eval prompts through ``openai-codex`` with manual decorator spans."""

    sdk_name = SDK_NAME

    def __init__(
        self,
        *,
        mcp_servers: list[McpServerSpec] | None = None,
        allowed_builtin_tools: Iterable[str] | None = None,
        skills: Iterable[str] = (),
        max_turns: int | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        gateway_base_url: str | None = None,
        gateway_token: str | None = None,
        gateway_headers: Mapping[str, str] | None = None,
        codex_factory: Callable[[CodexConfig], Any] | None = None,
    ) -> None:
        super().__init__(
            mcp_servers=mcp_servers,
            allowed_builtin_tools=allowed_builtin_tools,
            skills=skills,
            max_turns=max_turns,
            effort=effort,
            cwd=cwd,
        )
        self.gateway_base_url = gateway_base_url
        self.gateway_token = gateway_token
        self.gateway_headers = dict(gateway_headers or {})
        self._codex_factory = codex_factory or AsyncCodex

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
        run_in_agent_span = _decorate_when_llmobs_enabled(
            self._run_agent_loop_span,
            agent(name=f"eval.{model.provider}_agent_loop", _automatic_io_annotation=False),
        )
        return await run_in_agent_span(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=prompt_version,
            harness=harness,
            progress=progress,
        )

    async def _run_agent_loop_span(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        harness: str,
        progress: ProgressCallback | None,
    ) -> HarnessResult:
        run_in_llm_span = _decorate_when_llmobs_enabled(
            self._run_llm_span,
            llm(model_name=model.name, model_provider=model.provider, name=f"eval.{model.provider}_agent"),
        )
        run_result, missing_usage_error = await run_in_llm_span(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=prompt_version,
            harness=harness,
            progress=progress,
        )
        _annotate_agent_loop_span(
            model=model,
            sdk_name=SDK_NAME,
            prompt_version=prompt_version,
            harness=harness,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=run_result.answer,
            usage=run_result.usage,
            mcp_servers=self.mcp_servers,
            tool_calls=run_result.tool_calls,
        )
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

    async def _run_llm_span(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        harness: str,
        progress: ProgressCallback | None,
    ) -> tuple[AgentRunResult, RuntimeError | None]:
        trace_headers = current_trace_headers()
        mcp_tool_metadata = await _mcp_tool_metadata_catalog(self.mcp_servers, trace_headers) if LLMObs.enabled else {}
        run_result = await self._run_codex(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_headers=trace_headers,
            progress=progress,
        )
        missing_usage_error: RuntimeError | None = None
        if run_result.error is None:
            try:
                _raise_if_missing_usage(run_result.usage, sdk_name=SDK_NAME, model=model)
            except RuntimeError as exc:
                missing_usage_error = exc
        _annotate_llm_span(
            model=model,
            sdk_name=SDK_NAME,
            prompt_version=prompt_version,
            harness=harness,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=run_result.answer,
            usage=run_result.usage,
            mcp_servers=self.mcp_servers,
            tool_calls=run_result.tool_calls,
            allowed_builtin_tools=self.allowed_builtin_tools,
            mcp_tool_metadata=mcp_tool_metadata,
        )
        return run_result, missing_usage_error

    async def _run_codex(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        trace_headers: Mapping[str, str],
        progress: ProgressCallback | None,
    ) -> AgentRunResult:
        skill_names = stage_skills_for_codex(self.skills, self.cwd)
        overrides = _codex_mcp_config_overrides(self.mcp_servers, trace_headers)
        overrides.extend(self._gateway_config_overrides())
        config = CodexConfig(config_overrides=tuple(overrides), cwd=self.cwd, env=self._codex_env())
        thread_config = {"skill_approval": True} if skill_names else None
        async with self._codex_factory(config) as codex:
            thread = await codex.thread_start(
                approval_mode=ApprovalMode.auto_review,
                config=thread_config,
                cwd=self.cwd,
                developer_instructions=system_prompt,
                ephemeral=True,
                model=model.name,
                sandbox=Sandbox.read_only,
            )
            await _notify(progress, f"{SDK_NAME}/{model.label}: thread {getattr(thread, 'id', '<unknown>')}")
            result = await thread.run(
                user_prompt,
                approval_mode=ApprovalMode.auto_review,
                cwd=self.cwd,
                model=model.name,
                sandbox=Sandbox.read_only,
            )
        answer = str(getattr(result, "final_response", None) or "Codex did not produce a final response.")
        usage = UsageMetrics.from_codex(getattr(result, "usage", None))
        tool_calls = _codex_tool_calls_from_items(getattr(result, "items", []) or [])
        return AgentRunResult(answer=answer, usage=usage, tool_calls=tool_calls)

    def _codex_env(self) -> dict[str, str] | None:
        if not self.gateway_base_url:
            return None
        env = dict(os.environ)
        env["OPENAI_BASE_URL"] = self.gateway_base_url
        if self.gateway_token:
            env["OPENAI_API_KEY"] = self.gateway_token
        return env

    def _gateway_config_overrides(self) -> list[str]:
        if not self.gateway_base_url:
            return []
        from dd_ai_devx_evals.mcp import _toml_key_path, _toml_value

        prefix = _toml_key_path("model_providers", CODEX_GATEWAY_PROVIDER_ID)
        overrides = [
            f"{prefix}.name={_toml_value(CODEX_GATEWAY_PROVIDER_ID)}",
            f"{prefix}.base_url={_toml_value(self.gateway_base_url)}",
            f"{prefix}.env_key={_toml_value('OPENAI_API_KEY')}",
            f"model_provider={_toml_value(CODEX_GATEWAY_PROVIDER_ID)}",
        ]
        if self.gateway_headers:
            overrides.append(f"{prefix}.http_headers={_toml_value(dict(self.gateway_headers))}")
        return overrides


def _codex_mcp_config_overrides(servers: Iterable[McpServerSpec], trace_headers: Mapping[str, str]) -> list[str]:
    overrides: list[str] = []
    for server in servers:
        overrides.extend(server.to_codex_config_overrides(trace_headers))
    return overrides


# --------------------------------------------------------------------------- #
# Tool-call extraction
# --------------------------------------------------------------------------- #
def _codex_tool_calls_from_items(items: Iterable[Any]) -> list[AgentToolCall]:
    tool_calls: list[AgentToolCall] = []
    for item in items:
        root = getattr(item, "root", item)
        item_type = _enum_value(getattr(root, "type", None))
        if item_type == "mcpToolCall":
            tool_calls.append(_codex_mcp_tool_call(root))
        elif item_type == "collabAgentToolCall":
            tool_calls.append(_codex_collab_agent_tool_call(root))
        elif item_type == "dynamicToolCall":
            tool_calls.append(_codex_dynamic_tool_call(root))
    return tool_calls


def _codex_mcp_tool_call(item: Any) -> AgentToolCall:
    server = str(getattr(item, "server", "unknown"))
    tool = str(getattr(item, "tool", "unknown"))
    error = getattr(item, "error", None)
    return AgentToolCall(
        name=f"mcp.{server}.{tool}",
        arguments=getattr(item, "arguments", {}) or {},
        result=getattr(item, "result", None),
        error=str(getattr(error, "message", error)) if error else None,
        metadata={
            "source": SDK_NAME,
            "server": server,
            "tool": tool,
            "status": _enum_value(getattr(item, "status", None)),
            "duration_ms": getattr(item, "duration_ms", None),
        },
    )


def _codex_collab_agent_tool_call(item: Any) -> AgentToolCall:
    return AgentToolCall(
        name=f"codex.collab_agent.{_enum_value(getattr(item, 'tool', 'unknown'))}",
        arguments={"prompt": getattr(item, "prompt", None)},
        result={"receiver_thread_ids": getattr(item, "receiver_thread_ids", [])},
        error=None,
        metadata={
            "source": SDK_NAME,
            "status": _enum_value(getattr(item, "status", None)),
            "model": getattr(item, "model", None),
            "sender_thread_id": getattr(item, "sender_thread_id", None),
        },
    )


def _codex_dynamic_tool_call(item: Any) -> AgentToolCall:
    return AgentToolCall(
        name=f"codex.dynamic_tool.{getattr(item, 'tool', 'unknown')}",
        arguments=getattr(item, "arguments", {}) or {},
        result=getattr(item, "content_items", None),
        error=None if getattr(item, "success", None) is not False else "dynamic_tool_failed",
        metadata={
            "source": SDK_NAME,
            "namespace": getattr(item, "namespace", None),
            "status": _enum_value(getattr(item, "status", None)),
            "duration_ms": getattr(item, "duration_ms", None),
        },
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))
