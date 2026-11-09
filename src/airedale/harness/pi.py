# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Pi coding-agent runner with decorator-based LLMObs instrumentation.

A scenario with ``harness = "pi"`` runs its models through the `pi` coding
agent. Pi has no Python API and no ddtrace integration. This runner starts
``pi --mode json`` as a subprocess for each run, reads its JSONL event stream
(see pi ``docs/json.md``), and makes manual LLMObs spans.

Spans
-----
The outer ``@agent`` span is the same as for Codex (see
:class:`DecoratorSpanAgentRunner`). Its children are more detailed than for
Codex, because the pi stream shows each step while it runs (see
:mod:`airedale.harness.pi_spans`):

- one ``llm`` span for each model call, with the input, the output, the usage
  and the real start and end times of that call;
- one ``tool`` span for each tool execution, with its real times. A tool call
  made by another tool (for example from a codemode script) is a child of
  that tool span;
- one ``task`` span for each compaction.

When LLMObs is enabled, pi also loads the airedale collector extension
(``pi_llmobs_extension.ts``, with ``-e``). It adds the request time, the time
to the first response byte, the HTTP status, the request ID and the messages of
each model request to the stream (see the limits in
:mod:`airedale.harness.pi_spans`). Without its data, the spans use the data of
the stream only. Pi stops at startup when it cannot load an extension (for
example an old pi): the runner then runs pi again without the collector.

Isolation
---------
Pi reads its user-level configuration from an agent directory (default
``~/.pi/agent``): settings, extensions, packages, MCP servers, models, skills,
context files and credentials. The runner always points ``PI_CODING_AGENT_DIR``
at a fresh, private temporary directory outside the workspace, so none of the
operator's configuration loads. That directory contains only the files the run
needs:

- ``settings.json``: an empty object, so pi defaults apply.
- ``models.json``: gateway wiring (see below), only when a gateway is set.
- ``mcp.json``: the scenario MCP servers, with ``direct`` exposure.
- ``auth.json``: a *copy* of the operator's pi credentials, only when no
  gateway credential and no provider API key env var is available (the same
  approach as the Codex runner).

Pi is also started with ``--no-session`` (no session file), ``--no-skills``
(user-global skills such as ``~/.agents/skills`` do not load; scenario and
project skills are passed with ``--skill``) and ``--approve`` (project ``.pi/``
files of a cloned repo load, like ``.claude/`` for Claude). ``PI_OFFLINE`` stops
model-catalog refreshes, so each run uses the catalog bundled with the
installed pi. ``PI_*`` session variables of a parent pi process are removed.

The pi process gets no Datadog variable from the operator: every ``DD_*``,
``DATADOG_*`` and ``_DD_*`` variable is removed, except the names in
``PI_DATADOG_ENV_ALLOWLIST`` (empty: pi needs none). The operator sets these
for the airedale process only (LLMObs). They hold credentials (``DD_API_KEY``,
``DD_APP_KEY``, keys inside values such as ``DD_ADDITIONAL_ENDPOINTS``, proxy
URLs with a password, ...) and settings that would make Datadog-instrumented
repo code under test report to the operator's account. A denylist of
credential names cannot be complete, so the filter is an allowlist. Without
it, pi, its ``bash`` tool, stdio MCP servers and repo code that pi loads
(``--approve``) get them in their environment. This is not a sandbox: repo
code runs as the same OS user, so only evaluate trusted repositories. Only
the variables above are removed: a Datadog key stored under another name
(for example in ``OTEL_EXPORTER_OTLP_HEADERS``) still reaches pi. A
gateway credential that ``gateway.toml`` names explicitly (``api_key_env``)
goes to pi in ``models.json`` only: the variable with that name, and the
``api_key_env`` variables of the other gateway providers, are removed from the
pi environment (see ``PiRunner._dropped_gateway_envs``). Of the provider key
variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``), pi gets only the one of
the evaluated provider, and only when ``models.json`` holds no token (see
``PiRunner._dropped_provider_key_envs``). A stdio MCP server that
used an inherited Datadog variable (for example ``DD_API_KEY``) does not get it any more.

Gateway and model details
-------------------------
Pi needs model details (API type, context window, output limit, cost) for each
model. The runner does not define models: it changes the pi *built-in* provider
of the model (``anthropic`` or ``openai``) with ``baseUrl``, ``headers`` and
``apiKey`` from ``gateway.toml``. The built-in catalog then supplies the model
details. For a model that is not in the catalog, ``[[providers.<p>.pi_models]]``
in ``gateway.toml`` adds pi model definitions as-is. A token from a
``credentials_helper`` is sent as ``Authorization: Bearer`` (``authHeader``),
the same as Claude Code's ``apiKeyHelper``; a key from ``api_key_env`` uses the
provider-native auth header. The token is written to ``models.json`` (mode
``0600``) and is not put in the pi process environment.

The runner checks that every assistant message reports the requested provider
and model, so a fuzzy ``--model`` match can never silently evaluate a different
model.

Limits
------
- ``max_turns``: pi has no turn limit. The runner counts ``turn_start`` events
  and stops pi (the run fails) when the count goes above ``max_turns``.
- ``effort`` is passed as ``--thinking`` when it is a pi thinking level.
- Repo ``.mcp.json`` is not read (pi does not read it). A repo
  ``.pi/mcp.json`` server with the same name as a scenario server would replace
  it in pi, so the runner raises in that case instead of running the wrong
  server.
- Distributed-trace headers reach HTTP MCP servers only (as for the other
  harnesses).
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import json
import logging
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from airedale import js_json
from airedale.harness import base as _base
from airedale.harness.base import (
    AgentOutputSegment,
    AgentRunResult,
    AgentThought,
    AgentToolCall,
    DecoratorSpanAgentRunner,
    ProgressCallback,
    _llmobs_tool_definitions,
    _notify,
)
from airedale.harness.pi_spans import (
    PI_LLMOBS_EXTENSION_PATH,
    PiSpanRecorder,
    message_text,
    tool_result_text,
)
from airedale.mcp import pi_literal, pi_possible_tool_name_collisions
from airedale.pi_mcp import discover_pi_project_mcp_servers, merge_project_mcp_config, validate_server_config
from airedale.skills import pi_skill_dirs
from airedale.types import ModelSpec, UsageMetrics, is_valid_pi_usage

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from airedale.mcp import McpServerSpec

logger = logging.getLogger(__name__)

SDK_NAME = "pi"
PI_FRAMEWORK = "pi"
#: Environment variable that overrides the pi command, for example
#: ``npx -y @earendil-works/pi-coding-agent`` or a JSON array of arguments
#: (see ``_default_pi_command``). Default: ``pi``.
PI_COMMAND_ENV = "AIREDALE_PI_COMMAND"
DEFAULT_PI_COMMAND = "pi"
#: Every built-in pi tool. Used when a scenario allows all built-in tools
#: (``allowed_builtin_tools`` omitted). On Windows, ``PI_WINDOWS_BUILTIN_TOOLS``
#: is added.
PI_BUILTIN_TOOLS: tuple[str, ...] = ("read", "bash", "edit", "write", "grep", "find", "ls")
#: Built-in pi tools that exist on Windows only.
PI_WINDOWS_BUILTIN_TOOLS: tuple[str, ...] = ("powershell",)
#: Claude Code built-in tool names mapped to the pi tool with the same job, so
#: one ``allowed_builtin_tools`` list (for example in ``[defaults]``) works for
#: both harnesses. Other names are passed to pi unchanged.
PI_TOOL_ALIASES: dict[str, str] = {
    "Read": "read",
    "Bash": "bash",
    "Edit": "edit",
    "MultiEdit": "edit",
    "Write": "write",
    "Grep": "grep",
    "Glob": "find",
    "LS": "ls",
}
#: MCP resource tools of pi. They count as MCP tools, but ``mcp__*`` does not
#: match them, so they are named in ``--tools`` when MCP servers are present.
PI_MCP_RESOURCE_TOOLS: tuple[str, ...] = ("list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource")
#: ``--tools`` pattern that matches every MCP tool.
PI_MCP_TOOLS_PATTERN = "mcp__*"
PI_THINKING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
#: Context file names that pi reads in a directory, in pi's order: the first
#: one that is a file is used.
PI_CONTEXT_FILE_NAMES: tuple[str, ...] = ("AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")
#: Provider API-key env vars that pi reads natively for the supported providers.
PI_PROVIDER_API_KEY_ENVS: dict[str, str] = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
#: Variables of a parent pi process (session state, config location) that must
#: not reach the child pi process.
PI_INHERITED_ENV_TO_DROP: tuple[str, ...] = (
    "PI_CODING_AGENT_DIR",
    "PI_CODING_AGENT_SESSION_DIR",
    "PI_SESSION_ID",
    "PI_SESSION_FILE",
    "PI_PROVIDER",
    "PI_MODEL",
    "PI_REASONING_LEVEL",
    "PI_CODING_AGENT",
    "PI_OFFLINE",
    "PI_TELEMETRY",
    "PI_SKIP_VERSION_CHECK",
)
#: Name prefixes of Datadog variables (``_DD_`` is used by Datadog tools for
#: internal settings). Variables with these prefixes do not reach pi, except the
#: names in ``PI_DATADOG_ENV_ALLOWLIST``. Case is ignored.
DATADOG_ENV_PREFIXES: tuple[str, ...] = ("DD_", "DATADOG_", "_DD_")
#: Datadog variables (upper case names) that reach pi. Empty: pi needs none.
#: Add a name here only when pi, or repo code run by pi, must read it, and its
#: value is not a secret.
PI_DATADOG_ENV_ALLOWLIST: frozenset[str] = frozenset()

#: Maximum size of one JSONL record on pi stdout (agent_end carries every message).
PI_STREAM_LIMIT_BYTES = 64 * 1024 * 1024
PI_STDERR_TAIL_BYTES = 16 * 1024
PI_TERMINATE_GRACE_SECONDS = 5.0
PI_EXIT_POLL_SECONDS = 0.1
#: The only stop reason of a complete final answer. Others (``length``,
#: ``toolUse``, ``error``, ``aborted``, ...) mean the run did not finish.
PI_COMPLETE_STOP_REASON = "stop"


class PiRunner(DecoratorSpanAgentRunner):
    """Run eval prompts through the ``pi`` coding agent (JSON mode)."""

    sdk_name = SDK_NAME
    framework = PI_FRAMEWORK

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
        gateway_bearer_token: str | None = None,
        gateway_api_key: str | None = None,
        gateway_api_key_env: str | None = None,
        other_gateway_api_key_envs: Iterable[str] = (),
        pi_models: Iterable[Mapping[str, Any]] = (),
        pi_command: Iterable[str] | None = None,
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
        self.gateway_bearer_token = gateway_bearer_token
        self.gateway_api_key = gateway_api_key
        #: Name of the variable that holds ``gateway_api_key`` (``api_key_env``
        #: in ``gateway.toml``). It does not reach pi: see ``_dropped_gateway_envs``.
        self.gateway_api_key_env = gateway_api_key_env
        #: ``api_key_env`` names of the other gateway providers. Pi does not use
        #: them, so they do not reach pi either.
        self.other_gateway_api_key_envs = list(other_gateway_api_key_envs)
        self.pi_models = [dict(model) for model in pi_models]
        #: The servers of the repo ``.pi/mcp.json`` as they were when the run started (see ``_project_servers``).
        self._project_snapshot: list[McpServerSpec] | None = None
        self.pi_command = list(pi_command) if pi_command is not None else _default_pi_command()

    def _effective_builtin_tools(self) -> tuple[str, ...]:
        if self.allowed_builtin_tools is None:
            return (*PI_BUILTIN_TOOLS, *PI_WINDOWS_BUILTIN_TOOLS) if _is_windows() else PI_BUILTIN_TOOLS
        return tuple(dict.fromkeys(PI_TOOL_ALIASES.get(name, name) for name in self.allowed_builtin_tools))

    async def _run_model_calls(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        harness: str,
        progress: ProgressCallback | None,
    ) -> tuple[AgentRunResult, RuntimeError | None]:
        """Run pi and make one span for each model call, tool call and compaction."""
        self._project_snapshot = discover_pi_project_mcp_servers(self.cwd, self._scenario_pi_configs())
        trace_headers, mcp_tool_metadata = await self._span_context()
        llmobs = _base.LLMObs
        spans: PiSpanRecorder | None = None
        if llmobs.enabled:
            spans = PiSpanRecorder(
                model=model,
                sdk_name=self.sdk_name,
                harness=harness,
                prompt_version=prompt_version,
                tool_definitions=_llmobs_tool_definitions(
                    self._reported_mcp_servers(),
                    sdk_name=self.sdk_name,
                    allowed_builtin_tools=self._effective_builtin_tools(),
                    mcp_tool_metadata=mcp_tool_metadata,
                ),
            )
        try:
            run_result = await self._run_agent(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                trace_headers=trace_headers,
                progress=progress,
                spans=spans,
            )
        finally:
            # Also after a failure or a cancellation: the spans show the steps
            # that ran. The agent span is still active, so it is their parent.
            if spans is not None:
                spans.close()
                spans.emit(llmobs)
        run_result.mcp_tool_metadata = mcp_tool_metadata
        return run_result, self._missing_usage_error(run_result, model)

    def _missing_usage_error(self, run_result: AgentRunResult, model: ModelSpec) -> RuntimeError | None:
        """No generic check for pi: the collector validates each usage record.

        The generic check fails a run whose total is zero. For pi, a record of
        zeros is data (a model that produced no tokens, or a price of zero), and
        a missing or damaged record already fails the run in the collector.
        """
        return None

    async def _run_agent(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        trace_headers: Mapping[str, str],
        progress: ProgressCallback | None,
        spans: PiSpanRecorder | None = None,
    ) -> AgentRunResult:
        self._check_builtin_tool_names()
        self._check_project_mcp_collisions(trace_headers)
        self._check_tool_name_collisions()
        skill_dirs = pi_skill_dirs(self.skills, self.cwd)
        agent_dir = _make_private_agent_dir(Path(self.cwd))
        try:
            system_prompt_path = self._write_agent_dir(
                agent_dir, model=model, system_prompt=system_prompt, trace_headers=trace_headers
            )
            collector_extension = spans is not None
            while True:
                command = self._command(
                    model=model,
                    system_prompt_path=system_prompt_path,
                    skill_dirs=skill_dirs,
                    collector_extension=collector_extension,
                )
                collector = _PiEventCollector(max_turns=self.max_turns, spans=spans)
                returncode, stderr_tail = await self._run_process(
                    command,
                    user_prompt=user_prompt,
                    env=self._env(agent_dir, model=model),
                    collector=collector,
                    model=model,
                    progress=progress,
                )
                if not (collector_extension and _collector_extension_failed(collector, returncode, stderr_tail)):
                    break
                # Pi stops at startup when it cannot load an extension, before
                # any model call: run it again without the collector, so that
                # telemetry never fails a run. The spans then use the stream.
                logger.warning("pi cannot load the airedale collector extension; running without it:\n%s", stderr_tail)
                collector_extension = False
        finally:
            shutil.rmtree(agent_dir, ignore_errors=True)
        return collector.result(model=model, returncode=returncode, stderr_tail=stderr_tail)

    # ------------------------------------------------------------------ #
    # Isolated agent directory
    # ------------------------------------------------------------------ #
    def _write_agent_dir(
        self, agent_dir: Path, *, model: ModelSpec, system_prompt: str, trace_headers: Mapping[str, str]
    ) -> Path:
        """Write the per-run pi agent directory; return the system-prompt file path."""
        agent_dir.chmod(0o700)
        _write_private_json(agent_dir / "settings.json", {})
        models_config = self._models_config(model)
        if models_config:
            _write_private_json(agent_dir / "models.json", models_config)
        mcp_config = self._mcp_config(trace_headers)
        if mcp_config:
            _write_private_json(agent_dir / "mcp.json", mcp_config)
        if not self._has_env_auth(model):
            operator_auth = _operator_agent_dir() / "auth.json"
            if operator_auth.is_file():
                shutil.copyfile(operator_auth, agent_dir / "auth.json")
                (agent_dir / "auth.json").chmod(0o600)
        # Pi reads ``--append-system-prompt`` as a file when the value is an
        # existing path, so always pass a file: the text is then never mistaken
        # for a path.
        system_prompt_path = agent_dir / "airedale-system-prompt.md"
        system_prompt_path.write_text(system_prompt + self._workspace_context_section(), encoding="utf-8")
        return system_prompt_path

    def _workspace_context_section(self) -> str:
        """Return the workspace's own context file as a pi ``project_context`` section.

        Pi runs with ``--no-context-files``, so it reads no ``AGENTS.md`` /
        ``CLAUDE.md`` from the parent directories of the workspace (they are
        not part of the evaluated repo). The context file in the workspace
        root belongs to the repo: it is added here, chosen as pi does (the
        first of ``PI_CONTEXT_FILE_NAMES`` that is a file) and in the format
        pi uses. Returns "" when there is none.
        """
        for name in PI_CONTEXT_FILE_NAMES:
            path = Path(self.cwd) / name
            try:
                if not path.is_file():
                    continue
                # As pi: invalid UTF-8 bytes become U+FFFD, a BOM is removed.
                content = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                logger.warning("Unable to read the workspace context file %s", path, exc_info=True)
                continue
            return (
                "\n\n<project_context>\nProject-specific instructions and guidelines:\n\n"
                f'<project_instructions path="{path}">\n{content}\n</project_instructions>\n</project_context>'
            )
        return ""

    def _models_config(self, model: ModelSpec) -> dict[str, Any]:
        """Build ``models.json``: change the built-in provider of ``model``.

        Only the provider of the evaluated model is changed. Without a gateway
        base URL, pi uses the provider's public API and its own auth.
        """
        entry: dict[str, Any] = {}
        if self.gateway_base_url:
            entry["baseUrl"] = self.gateway_base_url
            if self.gateway_headers:
                entry["headers"] = {name: pi_literal(value) for name, value in self.gateway_headers.items()}
            token = self._models_json_token()
            if token:
                entry["apiKey"] = pi_literal(token)
                if self.gateway_bearer_token:
                    entry["authHeader"] = True
        if self.pi_models:
            entry["models"] = self.pi_models
        if not entry:
            return {}
        return {"providers": {model.provider: entry}}

    def _mcp_config(self, trace_headers: Mapping[str, str]) -> dict[str, Any]:
        if not self.mcp_servers:
            return {}
        return {
            # MCP tools are declared directly; codemode is not turned on.
            "autoEnableCodemode": False,
            "mcpServers": {server.name: server.to_pi_config(trace_headers) for server in self.mcp_servers},
        }

    def _reported_mcp_servers(self) -> list[McpServerSpec]:
        """Return the scenario MCP servers, then the repo ``.pi/mcp.json`` servers.

        Pi reads the repo servers itself, so the runner does not configure
        them, but they are part of the runtime: the span metadata and the agent
        manifest list them. A repo server with the name of a scenario server
        makes the run fail (``_check_project_mcp_collisions``); here it is
        left out. Repo servers are not probed for their tools. They get no
        distributed-trace headers: pi reads the repo file as it is, and a
        project entry replaces a user-level entry with the same name, so the
        runner cannot add headers without changing the evaluated repo.
        """
        # Servers that pi rejects (a name that differs from a scenario server
        # only in "-"/"_") and replacements of a scenario server (the run fails
        # first) are not in the list.
        scenario_names = {server.name for server in self.mcp_servers}
        project = [server for server in self._project_servers() if server.name not in scenario_names]
        return [*self.mcp_servers, *project]

    def _check_project_mcp_collisions(self, trace_headers: Mapping[str, str] | None = None) -> None:
        """Raise when pi would not run every scenario MCP server.

        For pi, names that differ only in ``-``/``_`` are the same server. Two
        scenario servers with such names (for example ``docs-api`` and
        ``docs_api``): pi rejects the second one and runs without it. A repo
        ``.pi/mcp.json`` entry replaces the user-level entry with the same
        name. The scenario server must win, as for the other harnesses, and pi
        cannot do that. In both cases the run fails with a clear message.
        """
        if not self.mcp_servers:
            return
        self._check_scenario_server_names()
        project_file = Path(self.cwd) / ".pi" / "mcp.json"
        changed = merge_project_mcp_config(project_file, self._scenario_pi_configs(trace_headers, strict=True)).changed
        if changed:
            raise ValueError(
                f"Project MCP config {project_file} defines server(s) {', '.join(changed)} with the same name as "
                "scenario MCP server(s) and pi would use the project definition (or apply its changes, for example "
                '"enabled": false). Rename or remove the project entry.'
            )

    def _check_scenario_server_names(self) -> None:
        by_key: dict[str, list[str]] = {}
        for server in self.mcp_servers:
            by_key.setdefault(_pi_server_key(server.name), []).append(server.name)
        duplicates = sorted(", ".join(names) for names in by_key.values() if len(names) > 1)
        if duplicates:
            raise ValueError(
                f"Scenario MCP servers {'; '.join(duplicates)} have the same name for pi (pi ignores the "
                "difference between '-' and '_') and pi would run only one of them. Rename the server(s)."
            )

    def _check_builtin_tool_names(self) -> None:
        """Raise when an ``allowed_builtin_tools`` entry is not an exact tool name.

        The entries are joined into the value of ``--tools``. Pi splits that value at ``,``, trims each entry, reads
        ``*`` as a pattern, and reads a leading ``+`` or ``-`` as a change to its default tools (pi 1.4.0 and 1.5.0:
        ``cli/args.js``, ``getToolListError``, ``createToolNameMatcher``). ``"read,bash"`` would become two tools and
        ``"*"`` every tool. An entry with one of these is refused, before any server is probed.
        """
        if self.allowed_builtin_tools is None:
            return
        for entry in self.allowed_builtin_tools:
            reason = None
            if not isinstance(entry, str) or not entry.strip():
                reason = "it is empty"
            elif "," in entry:
                reason = "it has a ',' (pi would read several tools)"
            elif "*" in entry:
                reason = "it has a '*' (pi would read a pattern)"
            elif entry != entry.strip():
                reason = "it has leading or trailing whitespace (pi trims it)"
            elif entry[0] in "+-":
                reason = "it starts with '+' or '-' (pi would add or remove a default tool)"
            if reason is not None:
                raise ValueError(
                    f"allowed_builtin_tools entry {entry!r} is not an exact tool name: {reason}. "
                    "List one exact pi tool name per entry."
                )

    def check_scenario(self) -> None:
        """Raise :class:`ValueError` when the scenario alone (no workspace) makes the run impossible for pi.

        The checks of the run that need no repo: the names of the scenario servers, how they are written for pi
        (strict), the possible tool name collisions, and the scenario skill directories. The experiment calls this
        for every pi scenario **before** it starts a managed MCP server or an LLMObs experiment. The checks that
        need the repo (its ``.pi/mcp.json``, its skills) stay in the run (``_check_before_probe``).
        """
        self._check_builtin_tool_names()
        self._check_scenario_server_names()
        self._scenario_pi_configs({}, strict=True)
        self._check_tool_name_collisions()
        # (A directory without a project: only the scenario skills are read.)
        with tempfile.TemporaryDirectory(prefix="airedale-pi-check-") as empty:
            pi_skill_dirs(self.skills, empty)

    def _check_before_probe(self, trace_headers: Mapping[str, str]) -> None:
        """Run the local checks of the run before any MCP server is probed (the probe starts them)."""
        self._check_builtin_tool_names()
        self._check_project_mcp_collisions(trace_headers)
        self._check_tool_name_collisions()
        # A skill name that collides (scenario skill and project skill) refuses the run too.
        pi_skill_dirs(self.skills, self.cwd)

    def _check_tool_name_collisions(self) -> None:
        """Raise when tools of two servers with a ``tool_names`` list can get the same pi name.

        Pi gives a tool whose name another server's tool has a hash suffix, and
        the tool of the server that connects second gets it. The runner cannot
        know the order, so the tool names that it reports (LLMObs tool
        definitions, agent manifest) could be wrong. A ``tool_names`` list is
        an allow-list, not the catalog: the tools that it hides are registered
        too, and a hidden tool with a hash suffix can remove the collision, so
        the error says that the collision is possible. Servers without
        ``tool_names`` are not checked here (their tools are known only when
        the metadata probe runs; ``_mcp_tool_definitions`` logs a warning).
        """
        collisions = pi_possible_tool_name_collisions(
            {server.name: server.tool_names for server in self.mcp_servers if server.tool_names}
        )
        if not collisions:
            return
        details = "; ".join(
            f"{name} for " + " and ".join(f"{server}/{tool}" for server, tool in pairs)
            for name, pairs in sorted(collisions.items())
        )
        raise ValueError(
            f"Possible pi tool name collision between MCP servers: {details}. Pi adds a hash suffix to the tool of "
            "the server that connects second, so the tool names in the run report could be wrong. 'tool_names' is "
            "an allow-list, not the full tool list of the server: its hidden tools can change the result, so the "
            "collision is possible, not sure. Rename a server or change 'tool_names'."
        )

    def _scenario_pi_configs(
        self, trace_headers: Mapping[str, str] | None = None, *, strict: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Return the ``mcp.json`` entry of each scenario server, as it is written for pi (the global servers).

        The entries have the headers of the distributed trace of the run: a repo
        entry that repeats the static settings of a server, but not these
        headers, would replace the entry and remove them.

        ``strict``: raise :class:`ValueError` when a server cannot be written for pi (a name with a character that
        pi does not accept, a ``*`` in ``tool_names``) or when pi would reject the entry. The run is refused then,
        before any server is probed. Not strict (the reports): an empty result.
        """
        try:
            configs = {server.name: server.to_pi_config(trace_headers or {}) for server in self.mcp_servers}
        except ValueError:
            if strict:
                raise
            return {}
        if strict:
            for name, config in configs.items():
                problem = validate_server_config(name, config, exact=True)
                if isinstance(problem, str):
                    raise ValueError(f"Scenario MCP server {name!r} is not valid for pi: {problem}")
        return configs

    def _project_mcp_server_names(self) -> list[str]:
        """Return the names of the servers that pi loads and starts from the repo ``<cwd>/.pi/mcp.json``.

        Same list as the one in the span metadata: pi does not load an invalid
        entry, and does not start an entry with ``"enabled": false``.
        """
        return [server.name for server in self._project_servers()]

    def _project_servers(self) -> list[McpServerSpec]:
        """Return the servers that pi loads from the repo ``.pi/mcp.json``: the snapshot of the run, if there is one.

        Pi reads the file once, at startup. The task can change or delete the file while it runs, so everything
        that is reported about the run (tool definitions, agent manifest, dependencies) uses the list from the
        start of the run (``_run_model_calls`` takes it), not a new read after the run.
        """
        if self._project_snapshot is not None:
            return self._project_snapshot
        return discover_pi_project_mcp_servers(self.cwd, self._scenario_pi_configs())

    def _models_json_token(self) -> str | None:
        """Return the gateway token that ``models.json`` holds (``None`` when it holds no token)."""
        if not self.gateway_base_url:
            return None
        # An empty value (for example an empty ``api_key_env`` variable) is no token.
        return self.gateway_bearer_token or self.gateway_api_key or None

    def _dropped_gateway_envs(self, provider: str) -> list[str]:
        """Return the ``api_key_env`` variable names that must not reach pi.

        The gateway key goes to pi in the private ``models.json``. The variable
        that holds it would give the same key to pi, its ``bash`` tool, stdio
        MCP servers and repo extensions, so it is removed. The ``api_key_env``
        variables of the other gateway providers are removed too: pi does not
        use them. One exception: when ``models.json`` holds no token (no
        gateway ``base_url``), the key variable that pi reads itself for
        ``provider`` (the provider of the evaluated model, for example
        ``ANTHROPIC_API_KEY``) stays, because pi can need it to authenticate.
        """
        names = [name for name in (self.gateway_api_key_env, *self.other_gateway_api_key_envs) if name]
        native = PI_PROVIDER_API_KEY_ENVS.get(provider)
        if native and self._models_json_token() is None:
            names = [name for name in names if not _same_env_name(name, native)]
        return names

    def _dropped_provider_key_envs(self, provider: str) -> list[str]:
        """Return the provider key variables (``PI_PROVIDER_API_KEY_ENVS``) that must not reach pi.

        Pi needs at most the key variable of the evaluated provider, and only
        when ``models.json`` holds no token. The key variables of the other
        providers are always removed: pi, its tools, stdio MCP servers and
        repo extensions do not get a credential that the run does not use.
        """
        native = PI_PROVIDER_API_KEY_ENVS.get(provider)
        keep = native if native and self._models_json_token() is None else None
        return [name for name in PI_PROVIDER_API_KEY_ENVS.values() if keep is None or not _same_env_name(name, keep)]

    def _has_env_auth(self, model: ModelSpec) -> bool:
        if self._models_json_token():
            return True
        env_name = PI_PROVIDER_API_KEY_ENVS.get(model.provider)
        return bool(env_name and os.environ.get(env_name))

    def _env(self, agent_dir: Path, *, model: ModelSpec) -> dict[str, str]:
        dropped_envs = [*self._dropped_gateway_envs(model.provider), *self._dropped_provider_key_envs(model.provider)]
        env = {
            name: value
            for name, value in os.environ.items()
            if name not in PI_INHERITED_ENV_TO_DROP
            and not _is_blocked_datadog_env(name)
            and not any(_same_env_name(name, dropped) for dropped in dropped_envs)
        }
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
        env["PI_OFFLINE"] = "1"
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env["PI_TELEMETRY"] = "0"
        return env

    def _secret_values(self, env: Mapping[str, str]) -> list[str]:
        """Return the credentials that the run can hold: the gateway token and key, the provider key variables, and the
        value of any variable of the child environment that has a name like a secret (pi, an extension or a child
        process can print any of them to stderr)."""
        names = [*PI_PROVIDER_API_KEY_ENVS.values(), *self.other_gateway_api_key_envs]
        if self.gateway_api_key_env:
            names.append(self.gateway_api_key_env)
        # Known credentials: removed whatever their length (a short token is still a credential).
        values = [self.gateway_bearer_token, self.gateway_api_key]
        values.extend(value for name in names for value in [os.environ.get(name), env.get(name)])
        values.extend(self._auth_file_secrets(env))
        values.extend(self._configured_secrets())
        # Variables that only have a name like a secret: a short value (a flag such as ``1``) is more likely not one.
        values.extend(
            value
            for name, value in env.items()
            if _SECRET_ENV_NAME_RE.search(name) and len(value) >= _MIN_SECRET_LENGTH
        )
        return [value for value in dict.fromkeys(values) if value]

    def _configured_secrets(self) -> list[str]:
        """Return the credentials that Airedale writes into the private config of the run besides the token and the key.

        The gateway headers (``models.json``) and the headers and environment of the scenario MCP servers (``mcp.json``)
        can hold a credential (``X-Api-Key``, ``Authorization``, a ``*_TOKEN`` variable...), and pi, a server or a
        child process can print them. A value under a name like a secret is removed whatever its length; another value
        is removed from 8 characters (a short word such as ``evals`` is not a credential, and would hide itself in
        messages).
        """
        mappings: list[Mapping[str, str]] = [self.gateway_headers]
        for server in self.mcp_servers:
            mappings.extend([server.headers, server.env])
        values: list[str] = []
        for mapping in mappings:
            for name, value in mapping.items():
                text = str(value)
                if text and (_SECRET_ENV_NAME_RE.search(str(name)) or len(text) >= _MIN_CONFIG_VALUE_LENGTH):
                    values.append(text)
        return values

    def _auth_file_secrets(self, env: Mapping[str, str]) -> list[str]:
        """Return the secret values of the ``auth.json`` that was copied into the agent directory of the run.

        The copy holds the credentials of every provider of the operator (access and refresh tokens, keys). Pi, an
        extension or a tool can print the file. Pi can also write new tokens into the copy, so the copy and the file
        it came from are both read. The values are only used to clean the text of the run: they are kept nowhere.
        """
        agent_dir = env.get("PI_CODING_AGENT_DIR")
        if not agent_dir or not (Path(agent_dir) / "auth.json").is_file():
            return []  # No copy was made (the run has a key in the environment): nothing of the file is in the run.
        values: list[str] = []
        for path in (Path(agent_dir) / "auth.json", _operator_agent_dir() / "auth.json"):
            try:
                text = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            values.extend(_secret_strings_of_json(text))
        return values

    # ------------------------------------------------------------------ #
    # Command line
    # ------------------------------------------------------------------ #
    def _command(
        self,
        *,
        model: ModelSpec,
        system_prompt_path: Path,
        skill_dirs: list[str],
        collector_extension: bool = False,
    ) -> list[str]:
        """Build the pi command line.

        The user prompt is not on the command line: pi reads an ``@path``
        argument as a file to attach, even after ``--``. The prompt goes to
        stdin, which pi uses as the prompt text as-is.
        """
        command = [
            *self.pi_command,
            "--mode",
            "json",
            "--no-session",
            "--approve",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            # Pi loads AGENTS.md / CLAUDE.md from the working directory and
            # every parent directory: a file above the workspace (for example
            # /tmp/AGENTS.md) must not change the run. The workspace's own
            # file is added to the appended system prompt instead.
            "--no-context-files",
            "--provider",
            model.provider,
            "--model",
            model.name,
            "--append-system-prompt",
            str(system_prompt_path),
        ]
        if collector_extension:
            # Only when spans are made. Pi exits when it cannot load it.
            command.extend(["--extension", str(PI_LLMOBS_EXTENSION_PATH)])
        for skill_dir in skill_dirs:
            command.extend(["--skill", skill_dir])
        if self.effort:
            if self.effort in PI_THINKING_LEVELS:
                command.extend(["--thinking", self.effort])
            else:
                logger.warning(
                    "Effort %r is not a pi thinking level (%s); pi uses its default", self.effort, PI_THINKING_LEVELS
                )
        tool_entries = self._tool_entries()
        if tool_entries:
            command.extend(["--tools", ",".join(tool_entries)])
        else:
            command.append("--no-tools")
        return command

    def _tool_entries(self) -> list[str]:
        """Return the ``--tools`` allow-list: built-in tools, then the MCP tools.

        Plain names replace pi's default tool set. Pi declares a ``direct`` MCP
        tool only when a ``--tools`` entry names or matches it. ``mcp__*``
        matches every MCP tool whatever name pi gives it (pi shortens long
        names and adds a hash on collisions); the scenario ``tool_names``
        allow-list is enforced by ``toolExposure`` in ``mcp.json`` (other tools
        are ``hidden``). Tools of repo ``.pi/mcp.json`` servers keep the
        exposure the repo sets; for ``codemode`` exposure, the scenario must
        allow the ``codemode`` tool in ``allowed_builtin_tools``. It is never
        added implicitly, as it can call other models.

        A ``mcp__`` entry makes ``--tools`` filter the MCP tools, and the MCP
        resource tools (``PI_MCP_RESOURCE_TOOLS``) do not match ``mcp__*``: they
        are named, so that the model can read the resources of the servers.
        Pi adds them only when a server has resources, and only for servers
        that are not ``hidden``: a scenario server with a ``tool_names``
        allow-list is ``hidden`` (only the listed tools are ``direct``), so its
        resources stay unavailable, as everything that the list does not name.
        The LLMObs tool definitions do not list the resource tools: the runner
        does not know if pi adds them.
        """
        entries = list(self._effective_builtin_tools())
        if self.mcp_servers or self._project_mcp_server_names():
            entries.extend([PI_MCP_TOOLS_PATTERN, *PI_MCP_RESOURCE_TOOLS])
        return list(dict.fromkeys(entries))

    # ------------------------------------------------------------------ #
    # Subprocess
    # ------------------------------------------------------------------ #
    async def _run_process(
        self,
        command: list[str],
        *,
        user_prompt: str,
        env: dict[str, str],
        collector: _PiEventCollector,
        model: ModelSpec,
        progress: ProgressCallback | None,
    ) -> tuple[int | None, str]:
        """Run pi, feed its events to ``collector``; return (exit code, stderr tail).

        The process group of pi is always stopped and pi is always reaped before
        this returns, also on error or cancellation.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=PI_STREAM_LIMIT_BYTES,
                # Own process group, so that stopping pi also stops its children
                # (stdio MCP servers, bash tool commands). POSIX only: on
                # Windows, the process tree of pi is stopped instead.
                start_new_session=not _is_windows(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"pi executable not found ({command[0]!r}); install pi or set {PI_COMMAND_ENV}") from exc

        stderr_tail = _TailBuffer(PI_STDERR_TAIL_BYTES)
        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(_write_stdin(process, user_prompt)),
            asyncio.create_task(_drain(process.stderr, stderr_tail)),
        ]
        reader = asyncio.create_task(_read_events(process, collector, model=model, progress=progress))
        # Not ``process.wait()``: depending on the Python version it also waits
        # for the pipes to close, which a child that keeps stdout open prevents.
        waiter = asyncio.create_task(_wait_for_exit(process))
        tasks.extend([reader, waiter])
        try:
            await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if not reader.done():
                # Pi exited, but a child process can keep its stdout open, so
                # EOF may never come: read what is left for a bounded time.
                await asyncio.wait({reader}, timeout=PI_TERMINATE_GRACE_SECONDS)
            elif collector.settled and not collector.stop_requested and not waiter.done():
                await asyncio.wait({waiter}, timeout=PI_TERMINATE_GRACE_SECONDS)
                if not waiter.done():
                    # Pi settled but did not exit: stop it. Not a run failure.
                    logger.warning("pi did not exit after the run settled; stopping it")
                    collector.stopped_after_settle = True
            if reader.done() and not reader.cancelled() and reader.exception() is not None:
                raise reader.exception()  # type: ignore[misc]
        finally:
            await _finish_cleanup(_cleanup_process(process, tasks))
        # The tail goes into the error of the run, the logs and the spans: no credential in it.
        return process.returncode, stderr_tail.redacted_text(self._secret_values(env))


async def _cleanup_process(process: asyncio.subprocess.Process, tasks: list[asyncio.Task[Any]]) -> None:
    # Always stop the whole group: children of pi (stdio MCP servers,
    # background commands) can outlive pi itself.
    await _stop_process_group(process)
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _finish_cleanup(cleanup: Any) -> None:
    """Run ``cleanup`` to the end, also if the caller is cancelled meanwhile.

    A cancellation that arrives during cleanup is re-raised after it.
    """
    task = asyncio.ensure_future(cleanup)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Record it first: the cancellation must not be lost when it comes
            # at the same moment as the end of the cleanup.
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    task.result()


async def _wait_for_exit(process: asyncio.subprocess.Process) -> int:
    """Return when the pi process itself has exited (its pipes may stay open)."""
    while process.returncode is None:
        await asyncio.sleep(PI_EXIT_POLL_SECONDS)
    return process.returncode


async def _write_stdin(process: asyncio.subprocess.Process, text: str) -> None:
    """Send the prompt on stdin, then close it (pi reads stdin until EOF)."""
    stdin = process.stdin
    if stdin is None:
        return
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        stdin.write(text.encode("utf-8"))
        await stdin.drain()
    with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
        stdin.close()
        await stdin.wait_closed()


async def _read_events(
    process: asyncio.subprocess.Process,
    collector: _PiEventCollector,
    *,
    model: ModelSpec,
    progress: ProgressCallback | None,
) -> None:
    """Feed pi stdout records to ``collector`` until EOF, settle, or failure."""
    stdout = process.stdout
    if stdout is None:
        collector.fail("pi stdout is not available")
        return
    while True:
        try:
            line = await stdout.readline()
        except (ValueError, asyncio.LimitOverrunError):
            collector.fail(f"pi emitted a JSONL record larger than {PI_STREAM_LIMIT_BYTES} bytes")
            return
        if not line:
            return
        message = collector.feed_line(line, received_ns=time.time_ns())
        if message:
            await _notify(progress, f"{SDK_NAME}/{model.label}: {message}")
        if collector.stop_requested or collector.settled:
            return


#: Names of environment variables that hold a credential.
_SECRET_ENV_NAME_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)")
#: A header or environment value of the run config under another name is removed from this many characters.
_MIN_CONFIG_VALUE_LENGTH = 8
#: Keys of a credentials file (``auth.json``) whose value is a credential.
_SECRET_KEY_NAME_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|ACCESS|REFRESH)")
#: A value shorter than this is not taken for a credential (it would also hide ordinary words in a message).
_MIN_SECRET_LENGTH = 6
#: ``Authorization: Bearer <token>`` or ``Bearer <token>``: the token is removed even when it is not a known value.
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_REDACTED = "<redacted>"


def _secret_strings_of_json(text: str) -> list[str]:
    """Return the strings of a credentials file that can be a credential: long ones, and any under a secret-like key.

    Short ordinary values (``"oauth"``, ``"api_key"``) are left out so that they are not hidden in messages. A file
    that is not JSON gives its long tokens.
    """
    try:
        data = js_json.loads(text.removeprefix("\ufeff"))
    except ValueError:
        return re.findall(r"[A-Za-z0-9._~+/=-]{16,}", text)
    found: list[str] = []
    stack: list[tuple[Any, str]] = [(data, "")]
    while stack:
        item, key = stack.pop()
        if isinstance(item, dict):
            stack.extend((child, str(name)) for name, child in item.items())
        elif isinstance(item, list):
            stack.extend((child, key) for child in item)
        elif isinstance(item, str) and (len(item) >= _AUTH_LONG_VALUE or _SECRET_KEY_NAME_RE.search(key)):
            found.append(item)
    return found


#: A string of a credentials file with at least this many characters is taken for a credential, whatever its key.
_AUTH_LONG_VALUE = 20


def _redact_secrets(text: str, secrets: Iterable[str]) -> str:
    """Return ``text`` without the secrets (see :func:`_redact_bytes`) and without bearer tokens."""
    return _finish_redaction(_redact_bytes(text.encode(), secrets))


def _finish_redaction(data: bytes) -> str:
    """Decode redacted bytes (replacement characters for what is not UTF-8), strip them, remove bearer tokens."""
    return _BEARER_RE.sub(r"\1" + _REDACTED, data.decode("utf-8", errors="replace").strip())


def _redact_bytes(data: bytes, secrets: Iterable[str], *, truncated: bool = True) -> bytes:
    """Return ``data`` without the secrets (the longest first, whatever their length) and without a cut secret.

    The work is done on the bytes, before any decoding or stripping: a decoding with replacement characters would change
    a secret (a window that starts inside a UTF-8 character), and stripping would remove the newline that ends a
    secret (a PEM key). The stderr tail is a window on the end of the output: it can start in the middle of a secret,
    in the middle of one of its UTF-8 characters. The start is matched with the end of every secret *first*, before
    any secret is replaced: a shorter secret inside the cut part would otherwise change the start. This applies only
    when the window ``truncated`` the output, and not when a complete secret stands at the start.
    """
    encoded = sorted({form for secret in secrets if secret for form in _byte_forms(secret)}, key=len, reverse=True)
    # A cut start only exists when the window dropped the beginning of the output.
    if truncated:
        data = _drop_cut_secret(data, encoded)
    for secret in encoded:
        data = data.replace(secret, _REDACTED.encode())
    return data


def _byte_forms(secret: str) -> set[bytes]:
    """Return the byte strings in which ``secret`` can be printed. Never raises, never skips a value.

    A secret can hold characters that UTF-8 cannot encode: an unpaired surrogate (a JSON string such as
    ``"\\ud800"``), or the byte of an environment variable that the system could not decode (POSIX gives it as
    ``\\udcXX``, from the original byte). The forms: UTF-8 with the surrogates passed through, the original bytes of
    the environment (``os.fsencode``), what Node prints for an unpaired surrogate (U+FFFD), and the ASCII spelling of
    JSON (``\\uXXXX``).
    """
    forms: set[bytes] = set()
    encoders = (
        lambda text: text.encode("utf-8"),
        lambda text: text.encode("utf-8", "surrogatepass"),
        lambda text: os.fsencode(text),
        lambda text: _replace_surrogates(text).encode("utf-8"),
        lambda text: json.dumps(text, ensure_ascii=True)[1:-1].encode("ascii"),
    )
    for encode in encoders:
        try:
            form = encode(secret)
        except (UnicodeError, ValueError):
            continue
        if form:
            forms.add(form)
    return forms


def _replace_surrogates(text: str) -> str:
    """Replace each unpaired surrogate by U+FFFD, as Node does when it prints one."""
    return "".join("\ufffd" if "\ud800" <= char <= "\udfff" else char for char in text)


#: A start of the tail is taken for the cut end of a secret from this many bytes.
_MIN_CUT_SECRET = 4


def _drop_cut_secret(data: bytes, secrets: list[bytes]) -> bytes:
    """Remove the start of ``data`` when it is the end of a secret that the window cut.

    All the matches are looked for before any byte changes. A complete secret at the start wins over a cut that is not
    longer than it: the first bytes of a complete key can also be the end of the key (a repeated pattern), and removing
    only those would leave the rest of the key visible.
    """
    full = max((len(secret) for secret in secrets if data.startswith(secret)), default=0)
    best = 0
    for secret in secrets:
        for size in range(min(len(secret) - 1, len(data)), max(_MIN_CUT_SECRET, best + 1) - 1, -1):
            if data.startswith(secret[-size:]):
                best = size
                break
    return _REDACTED.encode() + data[best:] if best > full else data


def _collector_extension_failed(collector: _PiEventCollector, returncode: int | None, stderr_tail: str) -> bool:
    """Return ``True`` when pi stopped at startup because it cannot load the collector extension."""
    return (
        returncode not in (0, None)
        and collector.records == 0
        and "Failed to load extension" in stderr_tail
        and str(PI_LLMOBS_EXTENSION_PATH) in stderr_tail
    )


def _is_blocked_datadog_env(name: str) -> bool:
    """Return True when a Datadog variable must not reach pi (see ``PI_DATADOG_ENV_ALLOWLIST``)."""
    upper = name.upper()
    return upper.startswith(DATADOG_ENV_PREFIXES) and upper not in PI_DATADOG_ENV_ALLOWLIST


def _same_env_name(name: str, other: str) -> bool:
    """Return True when ``name`` and ``other`` are the same variable name.

    On Windows, variable names are not case-sensitive.
    """
    if os.name == "nt":
        return name.upper() == other.upper()
    return name == other


def _agent_dir_parent_candidates() -> list[Path]:
    """Return the parent directories to try for the private agent dir, in order.

    The first one is the process temporary directory (``TMPDIR``, ``TEMP``,
    ``TMP``, then the platform default). The others are the platform default
    temporary directories, which do not depend on these variables, then the
    home directory as a last option.
    """
    candidates = [Path(tempfile.gettempdir())]
    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT")
        if system_root:
            candidates.append(Path(system_root) / "Temp")
        candidates.extend([Path(r"C:\TEMP"), Path(r"C:\TMP")])
    else:
        candidates.extend([Path("/tmp"), Path("/var/tmp"), Path("/usr/tmp")])
    with contextlib.suppress(RuntimeError):  # no home directory is known
        candidates.append(Path.home())
    return candidates


def _is_inside(path: Path, root: Path) -> bool:
    """Return True when ``path`` is ``root`` or inside it.

    Symbolic links are resolved. The check also compares the file identity
    (device and inode) of ``root`` with ``path`` and each of its parents, so it
    is also correct on a case-insensitive file system (``/ws`` and ``/WS`` are
    the same directory). It cannot find a mount point outside ``root`` that
    shows a directory inside ``root`` (for example a bind mount of
    ``root/tmp``). When a parent cannot be examined, the result is True (the
    path is not used).
    """
    real_path = Path(os.path.realpath(path))
    real_root = Path(os.path.realpath(root))
    if real_path.is_relative_to(real_root):
        return True
    try:
        root_stat = real_root.stat()
    except OSError:
        # No workspace on disk: nothing can be inside it.
        return False
    for ancestor in (real_path, *real_path.parents):
        try:
            if os.path.samestat(ancestor.stat(), root_stat):
                return True
        except OSError:
            return True
    return False


def _make_private_agent_dir(workspace: Path) -> Path:
    """Create the private pi agent dir (mode ``0700``) outside ``workspace``.

    The agent dir holds ``models.json`` and can hold a copy of ``auth.json``
    (credentials). The agent works in ``workspace``, so the agent dir must not
    be in it. ``tempfile.mkdtemp()`` alone is not sufficient: ``TMPDIR``,
    ``TEMP`` or ``TMP`` can point into the workspace (directly or through a
    symbolic link). A parent inside the workspace is skipped, and the created
    directory is checked again after it exists.

    Raises:
        RuntimeError: No candidate parent is outside the workspace and writable.
    """
    for parent in _agent_dir_parent_candidates():
        try:
            if not parent.is_dir() or _is_inside(parent, workspace):
                continue
            agent_dir = Path(tempfile.mkdtemp(prefix="airedale-pi-agent-", dir=os.path.realpath(parent)))
        except OSError:
            continue
        if _is_inside(agent_dir, workspace):
            # The parent changed after the check (for example a symbolic link
            # replaced). Do not use it.
            shutil.rmtree(agent_dir, ignore_errors=True)
            continue
        return agent_dir
    raise RuntimeError(
        f"Cannot create the private pi agent directory outside the workspace {workspace}: every candidate "
        "parent directory (TMPDIR/TEMP/TMP, the system temporary directories, the home directory) is inside "
        "the workspace or is not writable."
    )


def _default_pi_command() -> list[str]:
    """Return the pi command from ``AIREDALE_PI_COMMAND`` (default ``pi``).

    A JSON array of strings gives the exact arguments, on every platform (the
    first one must not be empty). Any other value is split as a command
    line: with POSIX shell rules, or on Windows with Windows rules, where a
    backslash is a path separator (``C:\\tools\\pi.exe``), not an escape.
    """
    raw = os.environ.get(PI_COMMAND_ENV, "").strip()
    if not raw:
        return [DEFAULT_PI_COMMAND]
    if raw.startswith("["):
        try:
            command = json.loads(raw)
        except ValueError:
            command = None  # Not JSON (for example a path "[tools]/pi"): a command line.
        if command is not None:
            if not (
                isinstance(command, list) and command and all(isinstance(arg, str) for arg in command) and command[0]
            ):
                raise ValueError(
                    f"{PI_COMMAND_ENV} is a JSON value but not an array of strings with a non-empty first string"
                )
            return command
    return _split_windows_command(raw) if _is_windows() else shlex.split(raw)


def _split_windows_command(raw: str) -> list[str]:
    """Split a command line as Windows ``CommandLineToArgvW`` does.

    The program name (first argument) ends at the next ``"`` when it starts
    with one, else at the first space; its backslashes are literal. For the
    other arguments: spaces separate them outside double quotes; ``2n``
    backslashes before a ``"`` give ``n`` backslashes and the quote opens or
    closes a quoted part; ``2n+1`` give ``n`` and a literal ``"``; in a quoted
    part, ``""`` gives a literal ``"``; other backslashes are literal.
    """
    raw = raw.lstrip(" \t")
    if not raw:
        return []
    if raw[0] == '"':
        end = raw.find('"', 1)
        program, rest = (raw[1:], "") if end == -1 else (raw[1:end], raw[end + 1 :])
    else:
        cut = len(raw)
        for separator in (" ", "\t"):
            position = raw.find(separator)
            if position != -1:
                cut = min(cut, position)
        program, rest = raw[:cut], raw[cut:]
    args = [program]
    current: list[str] = []
    in_arg = in_quotes = False
    index = 0
    while index < len(rest):
        char = rest[index]
        if char == "\\":
            count = 0
            while index < len(rest) and rest[index] == "\\":
                count += 1
                index += 1
            if index < len(rest) and rest[index] == '"':
                current.append("\\" * (count // 2))
                if count % 2:
                    current.append('"')
                    index += 1
            else:
                current.append("\\" * count)
            in_arg = True
            continue
        if char == '"':
            in_arg = True
            if in_quotes and rest[index + 1 : index + 2] == '"':
                current.append('"')
                index += 2
                continue
            in_quotes = not in_quotes
        elif char in " \t" and not in_quotes:
            if in_arg:
                args.append("".join(current))
                current, in_arg = [], False
        else:
            current.append(char)
            in_arg = True
        index += 1
    if in_arg:
        args.append("".join(current))
    return args


def _operator_agent_dir() -> Path:
    """Resolve the operator's real pi agent dir (``$PI_CODING_AGENT_DIR`` or ``~/.pi/agent``)."""
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or (Path.home() / ".pi" / "agent")).expanduser()


def _pi_server_key(name: str) -> str:
    """Pi treats server names that differ only in ``-`` and ``_`` as the same server."""
    return str(name).replace("-", "_")


def _write_private_json(path: Path, data: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


class _TailBuffer:
    """Keep the last ``limit`` bytes written."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()
        #: True once the start of the output was dropped: only then can the tail start inside a secret.
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        self._data.extend(chunk)
        if len(self._data) > self._limit:
            del self._data[: len(self._data) - self._limit]
            self.truncated = True

    def text(self) -> str:
        return self._data.decode("utf-8", errors="replace").strip()

    def redacted_text(self, secrets: Iterable[str]) -> str:
        """Return the tail without the secrets: they are removed from the raw bytes, then the text is made."""
        return _finish_redaction(_redact_bytes(bytes(self._data), secrets, truncated=self.truncated))


async def _drain(stream: asyncio.StreamReader | None, buffer: _TailBuffer) -> None:
    if stream is None:
        return
    while chunk := await stream.read(4096):
        buffer.append(chunk)


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop the pi process group and reap pi.

    SIGTERM goes to the whole group (also when pi itself already exited, as its
    children can still run). If pi does not exit within the grace period, or a
    child is still in the group, SIGKILL follows. Pi is then reaped.

    Windows has no process groups and no ``os.killpg``: see
    ``_stop_process_tree_windows``.
    """
    if _is_windows():
        await _stop_process_tree_windows(process)
        return
    _signal_group(process, signal.SIGTERM)
    if process.returncode is None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_wait_for_exit(process), timeout=PI_TERMINATE_GRACE_SECONDS)
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_wait_for_exit(process), timeout=PI_TERMINATE_GRACE_SECONDS)


def _is_windows() -> bool:
    return os.name == "nt"


#: Windows command that stops a process and all its child processes.
WINDOWS_TASKKILL = "taskkill"


async def _stop_process_tree_windows(process: asyncio.subprocess.Process) -> None:
    """Stop pi and its child processes on Windows, then reap pi.

    ``taskkill /T /F`` stops the process tree of pi (stdio MCP servers, bash
    tool commands). Then pi itself is stopped (``TerminateProcess``) if it
    still runs, and reaped. A ``taskkill`` that does not end in time is
    stopped too.

    Limit: Windows finds the children through pi. When pi already exited
    (for example a child keeps stdout open), its children are not found and
    are not stopped. A Windows Job Object would be necessary for that.
    """
    if process.returncode is None:
        taskkill: asyncio.subprocess.Process | None = None
        try:
            taskkill = await asyncio.create_subprocess_exec(
                WINDOWS_TASKKILL,
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(taskkill.wait(), timeout=PI_TERMINATE_GRACE_SECONDS)
        except (TimeoutError, OSError):
            pass
        finally:
            if taskkill is not None and taskkill.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    taskkill.kill()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(_wait_for_exit(taskkill), timeout=PI_TERMINATE_GRACE_SECONDS)
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            process.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_wait_for_exit(process), timeout=PI_TERMINATE_GRACE_SECONDS)


def _signal_group(process: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the process group of pi (pi is the group leader: pgid == pid)."""
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return  # The group is empty: nothing to stop.
    except (PermissionError, OSError):
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.send_signal(sig)


# --------------------------------------------------------------------------- #
# Event stream
# --------------------------------------------------------------------------- #
@dataclass(eq=False)  # Compared by identity: pi can reuse a ``toolCallId``.
class _ToolExecution:
    tool_name: str
    #: ``toolCallId`` as pi sent it (can be "": pi can use it as an id).
    tool_id: str = ""
    args: Any = None
    result: Any = None
    is_error: bool = False
    duration_ms: Any = None
    finished: bool = False
    #: ``toolCallId`` of the tool that made this call (``codemode`` or an
    #: extension calling ``ctx.executeTool()``); ``None`` for a call of the model.
    parent_id: str | None = None
    #: The execution that ``parent_id`` named when this one started (the latest
    #: one with that id), once it is known. ``None``: not known (yet).
    parent: _ToolExecution | None = None
    #: The finished execution with the id of ``parent_id`` when this one started. Another execution with that
    #: id can still come and be the parent (pi reuses ids), so this one is the parent only at the end of the run.
    fallback_parent: _ToolExecution | None = None


@dataclass
class _PiEventCollector:
    """Accumulate pi JSON-mode events into an :class:`AgentRunResult`."""

    max_turns: int | None = None
    session_id: str | None = None
    turns: int = 0
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    #: Every tool execution, in start order. Pi can reuse a ``toolCallId``
    #: (even ""), so each occurrence is kept and the id is not a key.
    tool_executions: list[_ToolExecution] = field(default_factory=list)
    #: The execution of each id that started but did not end.
    _open_executions: dict[str, list[_ToolExecution]] = field(default_factory=dict, repr=False)
    #: The latest execution of each id: the parent of the calls that name it.
    _latest_executions: dict[str, _ToolExecution] = field(default_factory=dict, repr=False)
    #: Executions that named a parent that has not started yet, by parent id.
    _waiting_for_parent: dict[str, list[_ToolExecution]] = field(default_factory=dict, repr=False)
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    settled: bool = False
    aborted: bool = False
    retry_error: str | None = None
    failure: str | None = None
    invalid_lines: int = 0
    #: True when the run settled but pi did not exit, so the runner stopped it.
    #: The exit status is then not a run failure.
    stopped_after_settle: bool = False
    #: Receives every event with its arrival time, to make the spans.
    spans: PiSpanRecorder | None = None
    #: Number of JSON records read from stdout.
    records: int = 0
    #: Where pi gave a ``usage`` record that is missing or damaged (see
    #: :func:`is_valid_pi_usage`), one entry for each record. Their tokens are not counted.
    damaged_usage: list[str] = field(default_factory=list)
    #: Number of assistant messages whose ``content`` is not a list (or a string).
    calls_without_content: int = 0

    @property
    def stop_requested(self) -> bool:
        return self.failure is not None

    def fail(self, message: str) -> None:
        if self.failure is None:
            self.failure = message

    def feed_line(self, line: bytes, *, received_ns: int | None = None) -> str | None:
        """Process one JSONL record; return a short progress message, if any.

        ``received_ns`` is the time (epoch ns) when the record arrived.
        """
        try:
            # Strict: a replaced byte would change an answer or a tool result without a sign.
            text = line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            self.invalid_lines += 1
            logger.debug("Ignoring a pi stdout line that is not UTF-8 (%d bytes)", len(line))
            return None
        if not text.strip():
            return None
        try:
            # Like ``JSON.parse`` (what pi writes with): NaN and Infinity are errors, and an integer above the range
            # of a double is Infinity (so a damaged count is seen as damaged).
            event = js_json.loads(text)
        except ValueError:
            self.invalid_lines += 1
            # (Not the text: a record can hold an answer or a credential.)
            logger.debug("Ignoring a non-JSON pi stdout line (%d characters)", len(text))
            return None
        if not isinstance(event, dict):
            # Pi writes one JSON object per record; anything else is damage.
            self.invalid_lines += 1
            return None
        if _tool_event_problem(event) is not None or self._end_names_another_tool(event):
            # Not given to the spans either: they would coerce the same fields.
            self.invalid_lines += 1
            return None
        self.records += 1
        if self.spans is not None:
            self.spans.feed(event, received_ns=received_ns)
        return self.feed(event)

    def feed(self, event: dict[str, Any]) -> str | None:
        event_type = event.get("type")
        if event_type == "session":
            self.session_id = str(event.get("id") or "") or None
            return f"session {self.session_id or '<unknown>'}"
        if event_type == "turn_start":
            self.turns += 1
            if self.max_turns and self.turns > self.max_turns:
                self.fail(f"pi exceeded max_turns={self.max_turns}")
            return f"turn {self.turns}"
        if event_type == "message_end":
            message = event.get("message")
            if not isinstance(message, dict):
                # Pi always writes the message (agent-loop.js, 1.4.0 and 1.5.0). A record without it can be the
                # model call that holds the usage: the run must not succeed with a total that is too low.
                self.invalid_lines += 1
                return None
            role = message.get("role")
            if not isinstance(role, str) or not role:
                # Every message of pi has a role (also the custom ones). Without it the record can be a model
                # call, and the run must not succeed without its usage.
                self.invalid_lines += 1
                return None
            if message.get("role") == "assistant":
                self.assistant_messages.append(message)
                if not _valid_assistant_content(message.get("content")):
                    # Pi writes ``content`` of an assistant message as a list of blocks (an empty list for a
                    # response with no content; every provider of pi 1.4.0 and 1.5.0 does, also the test provider),
                    # and each block is a text, thinking or toolCall block. ``null``, no field, a string, or a
                    # block that is not one of these is damage, not an empty or a partial response.
                    self.calls_without_content += 1
                # A model call must have usage.
                self._add_usage(message.get("usage"), "model call")
            elif message.get("role") == "toolResult" and "usage" in message:
                # Nested model work done inside a tool (pi counts it in its
                # session totals too).
                self._add_usage(message["usage"], "tool result")
            return None
        if event_type in ("tool_execution_start", "tool_execution_end") and (
            _tool_event_problem(event) is not None or self._end_names_another_tool(event)
        ):
            self.invalid_lines += 1
            return None
        if event_type == "tool_execution_start":
            name = str(event.get("toolName") or "unknown")
            # A start record for an id that is still open begins a new
            # execution: the earlier one stays, unfinished.
            self._begin_execution(event, args=event.get("args"))
            return f"tool {name}"
        if event_type == "tool_execution_end":
            tool_id = str(event.get("toolCallId") or "")
            # The end record belongs to the open execution of this id. With no
            # start record, it makes an execution that is already over.
            execution = self._end_execution(tool_id) or self._begin_execution(event, track_open=False)
            execution.result = event.get("result")
            execution.is_error = bool(event.get("isError"))
            execution.duration_ms = event.get("durationMs")
            execution.finished = True
            return None
        if event_type == "compaction_end":
            # Compaction calls the model too; count that usage.
            result = event.get("result")
            # A compaction hook can leave usage out: that is valid. Usage that
            # is there must be usable.
            if isinstance(result, dict) and "usage" in result:
                self._add_usage(result["usage"], "compaction")
            return "compaction"
        if event_type == "entry_appended":
            # Model work outside the messages, for example prompt-cache warming
            # (``kind`` "cache_warm"). Pi counts it in its session totals.
            # ``pi.appendEntry()`` makes "custom" entries only, which are not
            # counted. (Repo code in pi can write any record to stdout: only
            # evaluate trusted repositories.)
            entry = event.get("entry")
            if isinstance(entry, dict) and entry.get("type") == "usage":
                # A usage entry with no usage (a cut record, a changed schema) is damaged too.
                self._add_usage(entry.get("usage"), "usage entry")
            return None
        if event_type == "auto_retry_end":
            if event.get("success") is False:
                self.retry_error = str(event.get("finalError") or "pi retries failed")
            return None
        if event_type == "auto_retry_start":
            return f"retry {event.get('attempt')}/{event.get('maxAttempts')}"
        if event_type == "agent_settled":
            aborted = event.get("aborted")
            if not isinstance(aborted, bool):
                # Pi always writes a boolean (agent-session.js, 1.4.0 and 1.5.0). Reading a missing or other value as
                # "not aborted" could turn a damaged record into a run that settled.
                self.invalid_lines += 1
                return None
            self.settled = True
            self.aborted = aborted
            return None
        return None

    def _end_names_another_tool(self, event: dict[str, Any]) -> bool:
        """Return True for an end record with the id of an open execution but the name of another tool.

        Pi writes the start and the end record of a call from the same tool call (``toolCall.id`` and
        ``toolCall.name`` in ``agent-loop.js``, ``id`` and ``name`` in ``nested-tool-calls.js``; pi 1.4.0 and 1.5.0), so
        the name is the same. A record that differs is damaged, and closing the execution by id alone would give the
        result to the wrong tool. The record is refused, and the run fails.
        """
        if event.get("type") != "tool_execution_end" or self.failure is not None:
            # (After a failure, the first one stays the cause: parallel calls with one id are such a failure, and their
            # records are those of pi, in any order.)
            return False
        open_executions = self._open_executions.get(str(event.get("toolCallId") or ""))
        if not open_executions:
            return False
        # (With several open calls of one id the run fails anyway, as the results cannot be matched: see ``_end_execution``.)
        name = str(event.get("toolName") or "unknown")
        return all(execution.tool_name != name for execution in open_executions)

    def _end_execution(self, tool_id: str) -> _ToolExecution | None:
        """Return the open execution that an end record closes (``None``: no start record).

        Pi starts all parallel tool calls of a turn before it waits for them, and ends them as they
        finish, in any order. Calls with the same id (even "") are then open together, and an end
        record does not tell which of them it is. Their results can not be matched to the calls, so
        the run fails (the first open execution gets the record, so that nothing else is lost).
        """
        open_executions = self._open_executions.get(tool_id)
        if not open_executions:
            return None
        if len(open_executions) > 1:
            self.fail(
                f"pi ran {len(open_executions)} tool calls with the id {tool_id!r} at the same time; "
                "their results cannot be matched to the calls"
            )
        execution = open_executions.pop(0)
        if not open_executions:
            del self._open_executions[tool_id]
        return execution

    def _add_usage(self, usage: Any, source: str) -> None:
        """Count a ``usage`` record; remember a record that is damaged."""
        if is_valid_pi_usage(usage):
            self.usage.add(UsageMetrics.from_pi(usage))
        else:
            self.damaged_usage.append(source)

    def _begin_execution(self, event: dict[str, Any], *, args: Any = None, track_open: bool = True) -> _ToolExecution:
        tool_id = str(event.get("toolCallId") or "")
        execution = _ToolExecution(
            tool_name=str(event.get("toolName") or "unknown"),
            tool_id=tool_id,
            args=args,
            parent_id=_parent_tool_id(event),
        )
        parent_id = execution.parent_id
        if parent_id is not None:
            latest = self._latest_executions.get(parent_id)
            if latest is not None and not latest.finished:
                execution.parent = latest
            else:
                # No parent, or one that has ended: the next execution with this id can be the parent.
                execution.fallback_parent = latest
                self._waiting_for_parent.setdefault(parent_id, []).append(execution)
        self.tool_executions.append(execution)
        if track_open:
            self._open_executions.setdefault(tool_id, []).append(execution)
        self._latest_executions[tool_id] = execution
        # Calls that came before their parent get it now.
        for waiting in self._waiting_for_parent.pop(tool_id, ()):
            if waiting is not execution:
                waiting.parent = execution
                waiting.fallback_parent = None
            else:  # A call that names its own id: wait for the next one.
                self._waiting_for_parent.setdefault(tool_id, []).append(waiting)
        return execution

    def result(self, *, model: ModelSpec, returncode: int | None, stderr_tail: str) -> AgentRunResult:
        for execution in self.tool_executions:
            # No later execution took the id: the finished one that was there when the call started is the parent.
            if execution.parent is None and execution.fallback_parent is not None:
                execution.parent = execution.fallback_parent
        segments, model_executions = self._segments()
        tool_calls = self._with_nested_calls(
            [segment for segment in segments if isinstance(segment, AgentToolCall)], model_executions
        )
        final = self.assistant_messages[-1] if self.assistant_messages else None
        # The text of the final message, also when it is empty (a valid response can have no text). The placeholder
        # is only for a run that has no final message: it is not model output.
        answer = _message_text(final) if final is not None else "pi did not produce a final response."
        error = self._error(model=model, final=final, returncode=returncode, stderr_tail=stderr_tail)
        return AgentRunResult(
            answer=answer,
            usage=self.usage,
            tool_calls=tool_calls,
            segments=segments,
            error=error,
        )

    def _error(
        self, *, model: ModelSpec, final: dict[str, Any] | None, returncode: int | None, stderr_tail: str
    ) -> RuntimeError | None:
        details = f"\npi stderr (tail):\n{stderr_tail}" if stderr_tail else ""
        if self.failure is not None:
            return RuntimeError(f"{self.failure} for {model.label}{details}")
        if returncode != 0 and not (self.settled and self.stopped_after_settle):
            return RuntimeError(f"pi exited with status {returncode} for {model.label}{details}")
        if final is None:
            return RuntimeError(f"pi produced no assistant message for {model.label}{details}")
        for message in self.assistant_messages:
            provider, model_id = message.get("provider"), message.get("model")
            if provider != model.provider or model_id != model.name:
                return RuntimeError(
                    f"pi ran {provider}/{model_id} instead of {model.label}; check that pi knows the model "
                    "(its catalog or [[providers.<provider>.pi_models]] in the gateway config)"
                )
        for message in self.assistant_messages:
            response_model = message.get("responseModel")
            if response_model and response_model != model.name:
                # The provider reports the concrete model (for example a dated
                # snapshot). Not an error, but make it visible.
                logger.warning("pi reports response model %r for %s", response_model, model.label)
        if final.get("stopReason") != PI_COMPLETE_STOP_REASON:
            reason = final.get("errorMessage") or self.retry_error or f"stopReason={final.get('stopReason')!r}"
            return RuntimeError(f"pi run did not complete for {model.label}: {reason}{details}")
        if self.aborted:
            return RuntimeError(f"pi run was aborted for {model.label}")
        if not self.settled:
            return RuntimeError(f"pi exited before the run settled for {model.label}{details}")
        if self.invalid_lines:
            return RuntimeError(
                f"pi emitted {self.invalid_lines} malformed JSONL record(s) for {model.label}; "
                "usage or tool data can be missing"
            )
        if self.calls_without_content:
            return RuntimeError(
                f"pi wrote {self.calls_without_content} assistant message(s) without a content list or with a damaged "
                f"block for {model.label}; the answer or the tool calls can be missing"
            )
        if self.damaged_usage:
            sources = ", ".join(f"{count} {source}" for source, count in Counter(self.damaged_usage).items())
            return RuntimeError(
                f"pi reported missing or damaged token usage ({sources}) for {model.label}; "
                "token and cost data are incomplete"
            )
        return None

    def _segments(self) -> tuple[list[AgentOutputSegment], list[_ToolExecution | None]]:
        """Return reasoning, intermediate text and tool calls in message order.

        The text of the last assistant message is the answer, so it is not a
        segment. The second value has the execution of each tool call, in the
        same order (``None``: pi did not run it). The n-th call of the model
        with an id gets the n-th top-level execution (no ``parentToolCallId``)
        with that id, so a reused id does not mix the results.
        """
        segments: list[AgentOutputSegment] = []
        executions: list[_ToolExecution | None] = []
        by_id: dict[str, list[_ToolExecution]] = {}
        for execution in self.tool_executions:
            # Only the calls of the model: a nested call can reuse the id of a
            # later call of the model, and must not take its place.
            if execution.parent_id is None:
                by_id.setdefault(execution.tool_id, []).append(execution)
        next_index: dict[str, int] = {}
        last_index = len(self.assistant_messages) - 1
        for index, message in enumerate(self.assistant_messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking":
                    text = str(block.get("thinking") or "").strip()
                    if text:
                        segments.append(AgentThought(text=text, kind="reasoning"))
                elif block_type == "text" and index != last_index:
                    text = str(block.get("text") or "").strip()
                    if text:
                        segments.append(AgentThought(text=text, kind="text"))
                elif block_type == "toolCall":
                    tool_id = str(block.get("id") or "")
                    execution = None
                    # Pi runs no tool for a response that failed or was aborted
                    # (it can be tried again, with the same ids): the call does
                    # not take an execution. (A response cut at its length can
                    # have executed tools.)
                    if message.get("stopReason") not in _PI_NO_TOOL_STOP_REASONS:
                        position = next_index.get(tool_id, 0)
                        next_index[tool_id] = position + 1
                        candidates = by_id.get(tool_id, [])
                        execution = candidates[position] if position < len(candidates) else None
                        if execution is not None and execution.tool_name != str(block.get("name") or "unknown"):
                            # Pi runs the tool calls of the message it wrote, and writes ``toolCall.name`` in
                            # both places (``agent-loop.js``, pi 1.4.0 and 1.5.0): another name is a damaged
                            # record. The result is not given to a call with another name.
                            self.fail(
                                "pi ran a tool call under a name that is not the one of the same call (same id) "
                                "in the assistant message; the results cannot be matched to the calls"
                            )
                            execution = None
                    executions.append(execution)
                    segments.append(self._tool_call(block, execution))
        return segments, executions

    def _with_nested_calls(
        self, model_calls: list[AgentToolCall], model_executions: list[_ToolExecution | None]
    ) -> list[AgentToolCall]:
        """Add the nested tool calls after the model call that made them.

        A tool such as ``codemode`` can call other tools. Pi reports these
        calls only as ``tool_execution_*`` events (with ``parentToolCallId``),
        not in the transcript, so they are not in ``segments`` (which shows
        what the model asked for). The nested calls of a model call come after
        it in start order (all depths together), each one after its own
        parent, with ``parent_tool_id`` in its metadata. A nested call
        whose parent is not known comes at the end.

        A parent is an execution, not an id: pi can reuse an id (even ""), and
        a call belongs to the latest execution with the id that it names when
        it starts.
        """
        children: dict[_ToolExecution, list[_ToolExecution]] = {}
        position = {execution: index for index, execution in enumerate(self.tool_executions)}
        for execution in self.tool_executions:
            if execution.parent is not None:
                children.setdefault(execution.parent, []).append(execution)
        if not any(execution.parent_id is not None for execution in self.tool_executions):
            return model_calls
        # A model call is never added again as a nested call.
        added: set[_ToolExecution] = {execution for execution in model_executions if execution is not None}
        result: list[AgentToolCall] = []

        def add_descendants(root: _ToolExecution) -> None:
            # In start order, but a call only after its parent: take the
            # earliest call whose parent is already in the result. (No
            # recursion: the nesting depth has no limit in pi.)
            ready = [(position[child], child) for child in children.get(root, ())]
            heapq.heapify(ready)
            while ready:
                _, child = heapq.heappop(ready)
                if child in added:  # damaged ids: never loop
                    continue
                added.add(child)
                result.append(self._nested_call(child))
                for grandchild in children.get(child, ()):
                    heapq.heappush(ready, (position[grandchild], grandchild))

        for call, execution in zip(model_calls, model_executions, strict=True):
            result.append(call)
            if execution is not None:
                add_descendants(execution)
        # Nested calls whose parent is not in the list: first the calls with a parent
        # that never started, then the rest (a parent that is a call of the model
        # without a transcript block, or ids that make a cycle). The call at the
        # top of a chain comes first, so that each call is after its own parent.
        unresolved = [e for e in self.tool_executions if e.parent_id is not None and e.parent is None]
        leftovers = [e for e in self.tool_executions if e.parent_id is not None and e.parent is not None]
        for execution in [*unresolved, *leftovers]:
            if execution in added:
                continue
            top, seen = execution, {execution}
            while top.parent is not None and top.parent.parent_id is not None and top.parent not in seen:
                top = top.parent
                seen.add(top)
            if top in added:
                continue
            added.add(top)
            result.append(self._nested_call(top))
            add_descendants(top)
        return result

    def _nested_call(self, execution: _ToolExecution) -> AgentToolCall:
        if execution.finished:
            result: Any = _tool_result_text(execution.result)
            error = (result or "tool error") if execution.is_error else None
        else:
            result, error = None, "tool did not finish"
        return AgentToolCall(
            name=execution.tool_name,
            arguments=execution.args if execution.args is not None else {},
            result=result,
            error=error,
            metadata={
                "source": SDK_NAME,
                "tool_id": execution.tool_id or None,
                "duration_ms": execution.duration_ms,
                "parent_tool_id": execution.parent_id,
            },
        )

    def _tool_call(self, block: dict[str, Any], execution: _ToolExecution | None) -> AgentToolCall:
        tool_id = str(block.get("id") or "")
        name = str(block.get("name") or "unknown")
        if execution is None or not execution.finished:
            error: str | None = "tool did not run"
            result: Any = None
        else:
            result = _tool_result_text(execution.result)
            error = (result or "tool error") if execution.is_error else None
        return AgentToolCall(
            name=name,
            arguments=block.get("arguments") if block.get("arguments") is not None else {},
            result=result,
            error=error,
            metadata={
                "source": SDK_NAME,
                "tool_id": tool_id or None,
                "duration_ms": execution.duration_ms if execution is not None else None,
            },
        )


def _valid_assistant_content(content: Any) -> bool:
    """Return True when ``content`` is a list of the blocks that pi writes in an assistant message.

    ``TextContent`` (``text`` is a string), ``ThinkingContent`` (``thinking`` is a string) and ``ToolCall`` (``id`` and
    ``name`` are strings, ``arguments`` is an object), as in ``AssistantMessage`` of pi-ai 1.4.0 and 1.5.0. An empty
    list is valid. Any other block, or a field of another type, would be dropped or turned into text when the answer
    is made, so it is damage.
    """
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        if block_type == "text":
            valid = isinstance(block.get("text"), str)
        elif block_type == "thinking":
            valid = isinstance(block.get("thinking"), str)
        elif block_type == "toolCall":
            valid = (
                isinstance(block.get("id"), str)
                and isinstance(block.get("name"), str)
                and isinstance(block.get("arguments"), dict)
            )
        else:
            valid = False
        if not valid:
            return False
    return True


#: ``stopReason`` of an assistant message whose tool calls pi does not run.
_PI_NO_TOOL_STOP_REASONS = frozenset({"error", "aborted"})


def _tool_event_problem(event: dict[str, Any]) -> str | None:
    """Return what is wrong with a ``tool_execution_start`` or ``tool_execution_end`` record, or ``None``.

    Pi writes ``toolCallId`` and ``toolName`` as strings (the id can be empty), ``parentToolCallId`` (only for a nested
    call) as a string, ``isError`` (required) as a boolean and ``durationMs`` (optional) as a number. Reading ``null``, a missing id or a
    number as the empty id (or as no parent, or as success) would close the execution of another call, or hide that
    a call failed.
    """
    event_type = event.get("type")
    if event_type not in ("tool_execution_start", "tool_execution_end"):
        return None
    for key in ("toolCallId", "toolName"):
        if not isinstance(event.get(key), str):
            return f"{key} is not a string"
    if "parentToolCallId" in event and not isinstance(event["parentToolCallId"], str):
        return "parentToolCallId is not a string"
    if event_type == "tool_execution_end":
        # Required: a missing flag would read as a success, and a failed tool would pass (the agent loop of pi 1.4.0 and
        # 1.5.0 writes a boolean in every end record).
        if not isinstance(event.get("isError"), bool):
            return "isError is not a boolean"
        duration = event.get("durationMs")
        if "durationMs" in event and (
            isinstance(duration, bool) or not isinstance(duration, int | float) or not 0 <= duration < float("inf")
        ):
            return "durationMs is not a number"
    return None


def _parent_tool_id(event: dict[str, Any]) -> str | None:
    """Return ``parentToolCallId`` (can be ""); ``None`` for a call of the model."""
    parent = event.get("parentToolCallId")
    return None if parent is None else str(parent)


_message_text = message_text
_tool_result_text = tool_result_text
