"""Harness layer: provider-native agent SDK runners.

``create_runner`` selects the engine from the model provider (Claude Agent SDK
for ``anthropic``, OpenAI Codex for ``openai``), builds the MCP server specs and
gateway wiring from config, and returns a ready-to-run :class:`AgentRunner`.
``mcp_system_prompt`` returns a domain-neutral system prompt instructing the
model to prefer the available MCP tools/skills over answering from memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dd_ai_devx_evals.gateway import resolve_provider_config
from dd_ai_devx_evals.harness.base import (
    AgentRunner,
    AgentRunResult,
    AgentToolCall,
    ProgressCallback,
)
from dd_ai_devx_evals.harness.claude import ClaudeRunner
from dd_ai_devx_evals.harness.codex import CodexRunner
from dd_ai_devx_evals.mcp import McpServerSpec

if TYPE_CHECKING:
    from dd_ai_devx_evals.config.experiment import ScenarioConfig
    from dd_ai_devx_evals.config.gateway import GatewayConfig
    from dd_ai_devx_evals.types import ModelSpec

__all__ = [
    "AgentRunner",
    "AgentRunResult",
    "AgentToolCall",
    "ClaudeRunner",
    "CodexRunner",
    "ProgressCallback",
    "create_runner",
    "mcp_system_prompt",
]


def create_runner(
    model: ModelSpec,
    *,
    scenario: ScenarioConfig,
    gateway: GatewayConfig | None,
    cwd: str | None = None,
) -> AgentRunner:
    """Build the provider-appropriate :class:`AgentRunner` for one cell.

    MCP server specs are derived from ``scenario.mcp_servers``; gateway config is
    resolved per provider. When ``cwd`` is ``None`` the runner creates (and later
    cleans up) a temporary working directory.
    """
    mcp_servers = [McpServerSpec.from_config(server) for server in scenario.mcp_servers]
    skills = list(scenario.skills)
    allowed_builtin_tools = scenario.allowed_builtin_tools

    if model.provider == "anthropic":
        resolved = resolve_provider_config("anthropic", gateway)
        provider_config = gateway.get("anthropic") if gateway is not None else None
        credentials_helper = provider_config.credentials_helper if provider_config is not None else None
        return ClaudeRunner(
            mcp_servers=mcp_servers,
            allowed_builtin_tools=allowed_builtin_tools,
            skills=skills,
            max_turns=scenario.max_turns,
            effort=scenario.effort,
            cwd=cwd,
            gateway_base_url=resolved.base_url,
            gateway_headers=resolved.headers,
            gateway_credentials_helper=credentials_helper,
            gateway_api_key=resolved.api_key,
        )

    resolved = resolve_provider_config("openai", gateway)
    return CodexRunner(
        mcp_servers=mcp_servers,
        allowed_builtin_tools=allowed_builtin_tools,
        skills=skills,
        max_turns=scenario.max_turns,
        effort=scenario.effort,
        cwd=cwd,
        gateway_base_url=resolved.base_url,
        gateway_token=resolved.bearer_token or resolved.api_key,
        gateway_headers=resolved.headers,
    )


def mcp_system_prompt() -> str:
    """Return a domain-neutral system prompt for MCP/skill-aware runs.

    The prompt instructs the model to inspect the tools and skills exposed by its
    runtime and to use the available MCP server tools (and skills) to gather
    current, source-backed evidence rather than answering from memory whenever
    relevant tools exist.
    """
    return """
You are answering a question through an agentic runtime. Before answering,
inspect the capabilities exposed by your runtime and use the available skills,
MCP server tools, and other read-only tools to gather current, source-backed
evidence.

When the question concerns information that an available MCP server tool can
retrieve, you MUST call at least one relevant MCP server tool before producing a
final answer, even if you believe you already know the answer. Skills can help
you decide how to work, but they do not replace calling the relevant MCP server
tools when those tools cover the question. If no relevant MCP server tool is
available or all such tool calls fail, say so explicitly instead of answering
from memory.

Do not treat the local workspace as authoritative source material: the working
directory may be a blank temporary directory, so local Read, Grep, Glob, or LS
results are only runtime/workspace context. Prefer MCP-backed tools and relevant
skills over memory, generic web searches, or local workspace inspection. Choose
only tools or skills that are actually available in the current runtime; do not
call tools by names that are not exposed. Synthesize a concise final answer from
the gathered evidence.
""".strip()
