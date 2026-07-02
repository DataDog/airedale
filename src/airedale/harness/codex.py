# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""OpenAI Codex runner with decorator-based LLMObs instrumentation.

``openai`` models run through ``openai-codex``. Codex has **no native ddtrace
LLMObs integration**, so this runner manufactures the spans manually and shapes
them to match what the native ``claude_agent_sdk`` integration emits:

- an outer ``@agent`` span (the agentic loop) whose input/output are role/content
  message lists (a serialized transcript pairing ``tool_use`` and ``tool_result``
  messages) carrying an ``agent_manifest`` under ``metadata._dd.agent_manifest``;
- an inner ``@llm`` span (the model call) with structured ``input.messages`` /
  ``output.messages`` (each tool call its own assistant message, then the final
  answer) plus usage and tool definitions;
- one ``tool`` span per tool call, emitted as a child of the agent span so the
  trace topology mirrors the native ``agent -> llm + tool...`` tree.

The decorators are applied only when ``LLMObs.enabled`` so we never emit empty
spans. We only have aggregate Codex output (final answer + flat tool-call list),
not per-turn granularity, so the single ``llm`` span and the tool spans are
reconstructed from that aggregate rather than from per-model-call events.

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
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent, llm
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

from airedale.harness.base import (
    AgentOutputSegment,
    AgentRunner,
    AgentRunResult,
    AgentThought,
    AgentToolCall,
    ProgressCallback,
    _annotate_agent_loop_span,
    _annotate_llm_span,
    _decorate_when_llmobs_enabled,
    _emit_tool_spans,
    _notify,
    _raise_if_missing_usage,
)
from airedale.mcp import McpServerSpec, _mcp_tool_metadata_catalog
from airedale.skills import exclude_paths_from_git, stage_skills_for_codex
from airedale.tracing import current_trace_headers
from airedale.types import HarnessResult, ModelSpec, UsageMetrics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SDK_NAME = "openai-codex"
# Framework label surfaced in the agent-span manifest (mirrors the Claude path's
# "Claude Agent SDK").
CODEX_FRAMEWORK = "OpenAI Codex"
# Custom model-provider id registered through config overrides for gateway use.
CODEX_GATEWAY_PROVIDER_ID = "dd_ai_devx_gateway"
# Built-in (non-MCP) tools Codex ships and that this runner maps into tool/thought
# reporting. Codex does not gate built-ins on ``allowed_builtin_tools`` (that field
# is informational for Codex), so these are always available at runtime; the
# harness therefore lists them as the "available tools" when a scenario allows all
# built-ins (``allowed_builtin_tools`` omitted), mirroring how the Claude runner
# lists ``CLAUDE_BUILTIN_TOOLS`` for its own all-allowed case. ``update_plan`` is
# surfaced here for the manifest even though plan updates are reported as
# reasoning-style ``AgentThought`` segments rather than tool spans.
CODEX_BUILTIN_TOOLS: tuple[str, ...] = ("shell", "apply_patch", "update_plan", "web_search")
# Per-run isolated CODEX_HOME directory name (staged under the run cwd).
ISOLATED_CODEX_HOME_DIRNAME = ".dd-ai-devx-codex-home"


class CodexRunner(AgentRunner):
    """Run eval prompts through ``openai-codex`` with manual decorator spans."""

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
        gateway_token: str | None = None,
        gateway_headers: Mapping[str, str] | None = None,
        codex_factory: Callable[[CodexConfig], Any] | None = None,
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
        # Emit per-tool child spans first so they attach to this active agent
        # span (siblings of the inner llm span), reproducing the native Claude
        # `agent -> llm + tool...` topology.
        _emit_tool_spans(run_result.tool_calls)
        _annotate_agent_loop_span(
            model=model,
            sdk_name=SDK_NAME,
            framework=CODEX_FRAMEWORK,
            prompt_version=prompt_version,
            harness=harness,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=run_result.answer,
            usage=run_result.usage,
            mcp_servers=self.mcp_servers,
            segments=run_result.segments,
            allowed_builtin_tools=self._effective_builtin_tools(),
            max_turns=self.max_turns,
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
            segments=run_result.segments,
            allowed_builtin_tools=self._effective_builtin_tools(),
            mcp_tool_metadata=mcp_tool_metadata,
        )
        return run_result, missing_usage_error

    def _effective_builtin_tools(self) -> tuple[str, ...]:
        """Return the built-in tools to report as available for this run.

        Codex does **not** gate built-ins on ``allowed_builtin_tools`` (that field
        is informational for Codex — see ``AGENTS.md``), so every Codex built-in is
        always reachable regardless of the scenario's list. We therefore always
        report the full ``CODEX_BUILTIN_TOOLS`` set as the available built-ins,
        which also guarantees any observed built-in tool call (``shell`` /
        ``apply_patch`` / ``web_search``) appears in the manifest and llm-span
        tool definitions rather than looking like an undeclared tool in the UI.
        Contrast the Claude runner, which *does* gate (``permission_mode`` =
        ``dontAsk``) and so honors an explicit allow-list.
        """
        return CODEX_BUILTIN_TOOLS

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
        segments = _codex_output_segments(getattr(result, "items", []) or [])
        tool_calls = [segment for segment in segments if isinstance(segment, AgentToolCall)]
        return AgentRunResult(answer=answer, usage=usage, tool_calls=tool_calls, segments=segments)

    def _codex_env(self) -> dict[str, str]:
        """Build the Codex subprocess env (gateway wiring + CODEX_HOME isolation).

        Hermeticity: Codex reads MCP servers (and all other global config) from
        ``$CODEX_HOME/config.toml`` (default ``~/.codex``), so an operator's
        ambient servers would leak into runs — the Codex analog of the leak
        Claude's ``strict_mcp_config`` prevents. We therefore **always** isolate
        ``CODEX_HOME`` to a fresh, empty per-run dir under ``cwd`` so no global
        config ever loads.

        Auth is preserved across both paths:

        - **env auth** (a resolved gateway token or an ``OPENAI_API_KEY`` in the
          environment) is carried through and used directly.
        - otherwise we seed only ``auth.json`` into the isolated home, copied from
          the operator's real ``CODEX_HOME``, so ``codex login`` keeps working
          **without** inheriting the global ``config.toml``. We copy (not symlink)
          so token-refresh writes stay in the throwaway dir and never mutate the
          operator's auth state.

        Repo-scoped ``<cwd>/.codex/config.toml`` is **not** read by Codex for MCP
        servers (verified), so there is nothing to discover there.
        """
        env = dict(os.environ)
        if self.gateway_base_url:
            env["OPENAI_BASE_URL"] = self.gateway_base_url
            if self.gateway_token:
                env["OPENAI_API_KEY"] = self.gateway_token

        has_env_auth = bool(self.gateway_token) or bool(os.environ.get("OPENAI_API_KEY"))
        codex_home = Path(self.cwd) / ISOLATED_CODEX_HOME_DIRNAME
        codex_home.mkdir(parents=True, exist_ok=True)
        if not has_env_auth:
            operator_auth = self._operator_codex_home() / "auth.json"
            if operator_auth.is_file():
                shutil.copyfile(operator_auth, codex_home / "auth.json")
        env["CODEX_HOME"] = str(codex_home)
        # Keep the throwaway home out of the agent's untracked-changes view when
        # the workspace is a git worktree (no-op in a bare temp dir).
        exclude_paths_from_git(self.cwd, [ISOLATED_CODEX_HOME_DIRNAME])
        return env

    @staticmethod
    def _operator_codex_home() -> Path:
        """Resolve the operator's real CODEX_HOME (``$CODEX_HOME`` or ``~/.codex``)."""
        return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))

    def _gateway_config_overrides(self) -> list[str]:
        if not self.gateway_base_url:
            return []
        from airedale.mcp import _toml_key_path, _toml_value

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
# Output-segment extraction
# --------------------------------------------------------------------------- #
# Codex returns an ordered ``ThreadItem`` list covering every turn event: model
# reasoning, plan updates, built-in tool activity (shell command execution, file
# patches, web search), MCP/dynamic/collab tool calls, and the final assistant
# message. Only a subset was captured before, so runs that used only Codex's
# built-in tools (e.g. shelling into a checkout) reported no tool activity at
# all. We now walk the items **in order** and normalize each recognized type
# into either an ``AgentToolCall`` (tool spans + transcript) or an
# ``AgentThought`` (reasoning/plan, surfaced as assistant messages the way the
# native Claude integration surfaces ``ThinkingBlock`` content). Item types with
# no reporting analog (user message, hook prompt, agent message, image view,
# review-mode markers, compaction) are intentionally skipped: the final agent
# message is already carried as the run's ``answer``.
def _codex_output_segments(items: Iterable[Any]) -> list[AgentOutputSegment]:
    segments: list[AgentOutputSegment] = []
    for item in items:
        root = getattr(item, "root", item)
        builder = _CODEX_SEGMENT_BUILDERS.get(_enum_value(getattr(root, "type", None)))
        if builder is None:
            continue
        segment = builder(root)
        if segment is not None:
            segments.append(segment)
    return segments


def _codex_command_tool_call(item: Any) -> AgentToolCall:
    """Normalize a built-in shell (``commandExecution``) item into a tool call."""
    exit_code = getattr(item, "exit_code", None)
    status = _enum_value(getattr(item, "status", None))
    if status in {"failed", "declined"}:
        error = status
    elif isinstance(exit_code, int) and exit_code != 0:
        error = f"exit_code={exit_code}"
    else:
        error = None
    return AgentToolCall(
        name="shell",
        arguments={"command": str(getattr(item, "command", "") or ""), "cwd": str(getattr(item, "cwd", "") or "")},
        result=getattr(item, "aggregated_output", None),
        error=error,
        metadata={
            "source": SDK_NAME,
            "tool_id": getattr(item, "id", None),
            "status": status,
            "exit_code": exit_code,
            "duration_ms": getattr(item, "duration_ms", None),
            "command_source": _enum_value(getattr(item, "source", None)),
        },
    )


def _codex_file_change_tool_call(item: Any) -> AgentToolCall:
    """Normalize a built-in file-patch (``fileChange``) item into a tool call."""
    changes = getattr(item, "changes", []) or []
    rendered = [
        {
            "path": getattr(change, "path", None),
            "kind": _codex_patch_change_kind(getattr(change, "kind", None)),
            "diff": getattr(change, "diff", None),
        }
        for change in changes
    ]
    status = _enum_value(getattr(item, "status", None))
    return AgentToolCall(
        name="apply_patch",
        arguments={"changes": [{"path": change["path"], "kind": change["kind"]} for change in rendered]},
        result=rendered,
        error=(status if status in {"failed", "declined"} else None),
        metadata={
            "source": SDK_NAME,
            "tool_id": getattr(item, "id", None),
            "status": status,
            "file_count": len(rendered),
        },
    )


def _codex_web_search_tool_call(item: Any) -> AgentToolCall:
    """Normalize a built-in ``webSearch`` item into a tool call."""
    action = getattr(item, "action", None)
    return AgentToolCall(
        name="web_search",
        arguments={"query": str(getattr(item, "query", "") or "")},
        result=action if action is not None else None,
        error=None,
        metadata={"source": SDK_NAME, "tool_id": getattr(item, "id", None)},
    )


def _codex_reasoning_thought(item: Any) -> AgentThought | None:
    """Normalize a ``reasoning`` item into an ``AgentThought`` (or drop if empty).

    Codex exposes the human-readable reasoning ``summary``; the fuller ``content``
    is often omitted/encrypted, so we prefer ``summary`` and fall back to
    ``content``.
    """
    text = _codex_join_text(getattr(item, "summary", None) or getattr(item, "content", None))
    return AgentThought(text=text, kind="reasoning") if text else None


def _codex_plan_thought(item: Any) -> AgentThought | None:
    """Normalize a ``plan`` item into an ``AgentThought`` (or drop if empty)."""
    text = str(getattr(item, "text", "") or "").strip()
    return AgentThought(text=text, kind="plan") if text else None


def _codex_join_text(value: Any) -> str:
    """Join a Codex text field (``str`` or ``list[str]``) into a single string."""
    if value is None:
        return ""
    parts = [value] if isinstance(value, str) else list(value)
    return "\n\n".join(chunk for chunk in (str(part).strip() for part in parts) if chunk)


def _codex_patch_change_kind(kind: Any) -> str:
    """Extract the change kind (``add`` / ``delete`` / ``update``) from a ``FileUpdateChange``.

    ``FileUpdateChange.kind`` is a ``PatchChangeKind`` — a pydantic ``RootModel``
    whose ``root`` carries the discriminating ``type`` literal — not a plain enum,
    so ``_enum_value`` alone would stringify the whole model.
    """
    root = getattr(kind, "root", kind)
    return _enum_value(getattr(root, "type", root))


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
            "tool_id": getattr(item, "id", None),
            "server": server,
            "tool": tool,
            "status": _enum_value(getattr(item, "status", None)),
            "duration_ms": getattr(item, "duration_ms", None),
        },
    )


def _codex_collab_agent_tool_call(item: Any) -> AgentToolCall:
    status = _enum_value(getattr(item, "status", None))
    return AgentToolCall(
        name=f"codex.collab_agent.{_enum_value(getattr(item, 'tool', 'unknown'))}",
        arguments={"prompt": getattr(item, "prompt", None)},
        result={"receiver_thread_ids": getattr(item, "receiver_thread_ids", [])},
        error=("collab_agent_failed" if status == "failed" else None),
        metadata={
            "source": SDK_NAME,
            "tool_id": getattr(item, "id", None),
            "status": _enum_value(getattr(item, "status", None)),
            "model": getattr(item, "model", None),
            "sender_thread_id": getattr(item, "sender_thread_id", None),
        },
    )


def _codex_dynamic_tool_call(item: Any) -> AgentToolCall:
    status = _enum_value(getattr(item, "status", None))
    failed = getattr(item, "success", None) is False or status == "failed"
    return AgentToolCall(
        name=f"codex.dynamic_tool.{getattr(item, 'tool', 'unknown')}",
        arguments=getattr(item, "arguments", {}) or {},
        result=getattr(item, "content_items", None),
        error=("dynamic_tool_failed" if failed else None),
        metadata={
            "source": SDK_NAME,
            "tool_id": getattr(item, "id", None),
            "namespace": getattr(item, "namespace", None),
            "status": _enum_value(getattr(item, "status", None)),
            "duration_ms": getattr(item, "duration_ms", None),
        },
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


# Dispatch table mapping a ``ThreadItem`` ``type`` to its segment builder. Each
# builder returns an ``AgentOutputSegment`` (or ``None`` to skip an empty item).
_CODEX_SEGMENT_BUILDERS: dict[str, Callable[[Any], AgentOutputSegment | None]] = {
    "mcpToolCall": _codex_mcp_tool_call,
    "collabAgentToolCall": _codex_collab_agent_tool_call,
    "dynamicToolCall": _codex_dynamic_tool_call,
    "commandExecution": _codex_command_tool_call,
    "fileChange": _codex_file_change_tool_call,
    "webSearch": _codex_web_search_tool_call,
    "reasoning": _codex_reasoning_thought,
    "plan": _codex_plan_thought,
}
