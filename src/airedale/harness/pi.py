# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Pi coding-agent runner with decorator-based LLMObs instrumentation.

A scenario with ``harness = "pi"`` runs its models through the `pi` coding
agent. Pi has no Python API and no ddtrace integration. This runner starts
``pi --mode json`` as a subprocess for each run, reads its JSONL event stream
(see pi ``docs/json.md``), and makes the same manual spans as the Codex runner
(``@agent`` + ``@llm`` + one ``tool`` span per tool call, see
:class:`DecoratorSpanAgentRunner`).

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
import json
import logging
import os
import shlex
import shutil
import signal
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from airedale.harness.base import (
    AgentOutputSegment,
    AgentRunResult,
    AgentThought,
    AgentToolCall,
    DecoratorSpanAgentRunner,
    ProgressCallback,
    _notify,
    json_dumps_compact,
)
from airedale.mcp import pi_literal
from airedale.skills import pi_skill_dirs
from airedale.types import ModelSpec, UsageMetrics

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from airedale.mcp import McpServerSpec

logger = logging.getLogger(__name__)

SDK_NAME = "pi"
PI_FRAMEWORK = "pi"
#: Environment variable that overrides the pi command (split with ``shlex``),
#: for example ``npx -y @earendil-works/pi-coding-agent``. Default: ``pi``.
PI_COMMAND_ENV = "AIREDALE_PI_COMMAND"
DEFAULT_PI_COMMAND = "pi"
#: Every built-in pi tool. Used when a scenario allows all built-in tools
#: (``allowed_builtin_tools`` omitted). ``powershell`` (Windows only) is left out.
PI_BUILTIN_TOOLS: tuple[str, ...] = ("read", "bash", "edit", "write", "grep", "find", "ls")
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
#: ``--tools`` pattern that matches every MCP tool.
PI_MCP_TOOLS_PATTERN = "mcp__*"
PI_THINKING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
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
        self.pi_models = [dict(model) for model in pi_models]
        self.pi_command = list(pi_command) if pi_command is not None else _default_pi_command()

    def _effective_builtin_tools(self) -> tuple[str, ...]:
        if self.allowed_builtin_tools is None:
            return PI_BUILTIN_TOOLS
        return tuple(dict.fromkeys(PI_TOOL_ALIASES.get(name, name) for name in self.allowed_builtin_tools))

    async def _run_agent(
        self,
        *,
        model: ModelSpec,
        system_prompt: str,
        user_prompt: str,
        trace_headers: Mapping[str, str],
        progress: ProgressCallback | None,
    ) -> AgentRunResult:
        self._check_project_mcp_collisions()
        skill_dirs = pi_skill_dirs(self.skills, self.cwd)
        agent_dir = Path(tempfile.mkdtemp(prefix="airedale-pi-agent-"))
        try:
            system_prompt_path = self._write_agent_dir(
                agent_dir, model=model, system_prompt=system_prompt, trace_headers=trace_headers
            )
            command = self._command(model=model, system_prompt_path=system_prompt_path, skill_dirs=skill_dirs)
            collector = _PiEventCollector(max_turns=self.max_turns)
            returncode, stderr_tail = await self._run_process(
                command,
                user_prompt=user_prompt,
                env=self._env(agent_dir),
                collector=collector,
                model=model,
                progress=progress,
            )
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
        system_prompt_path.write_text(system_prompt, encoding="utf-8")
        return system_prompt_path

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
            token = self.gateway_bearer_token or self.gateway_api_key
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

    def _check_project_mcp_collisions(self) -> None:
        """Raise when a repo ``.pi/mcp.json`` server would replace a scenario server.

        In pi a project ``mcp.json`` entry replaces the user-level entry with the
        same name (names that differ only in ``-``/``_`` are the same). The
        scenario server must win, as for the other harnesses, and pi cannot do
        that, so the run fails with a clear message.
        """
        if not self.mcp_servers:
            return
        project_file = Path(self.cwd) / ".pi" / "mcp.json"
        project_names = {_pi_server_key(name) for name in self._project_mcp_server_names()}
        collisions = sorted(server.name for server in self.mcp_servers if _pi_server_key(server.name) in project_names)
        if collisions:
            raise ValueError(
                f"Project MCP config {project_file} defines server(s) {', '.join(collisions)} with the same name as "
                "scenario MCP server(s); pi would use the project definition. Rename the scenario server(s)."
            )

    def _project_mcp_server_names(self) -> list[str]:
        """Return the server names of the repo ``<cwd>/.pi/mcp.json`` (if any)."""
        project_file = Path(self.cwd) / ".pi" / "mcp.json"
        if not project_file.is_file():
            return []
        try:
            data = json.loads(project_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Unable to parse project MCP config %s", project_file, exc_info=True)
            return []
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            return []
        return [str(name) for name in servers]

    def _has_env_auth(self, model: ModelSpec) -> bool:
        if self.gateway_base_url and (self.gateway_bearer_token or self.gateway_api_key):
            return True
        env_name = PI_PROVIDER_API_KEY_ENVS.get(model.provider)
        return bool(env_name and os.environ.get(env_name))

    def _env(self, agent_dir: Path) -> dict[str, str]:
        env = {name: value for name, value in os.environ.items() if name not in PI_INHERITED_ENV_TO_DROP}
        env["PI_CODING_AGENT_DIR"] = str(agent_dir)
        env["PI_OFFLINE"] = "1"
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env["PI_TELEMETRY"] = "0"
        return env

    # ------------------------------------------------------------------ #
    # Command line
    # ------------------------------------------------------------------ #
    def _command(self, *, model: ModelSpec, system_prompt_path: Path, skill_dirs: list[str]) -> list[str]:
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
            "--provider",
            model.provider,
            "--model",
            model.name,
            "--append-system-prompt",
            str(system_prompt_path),
        ]
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
        """Return the ``--tools`` allow-list: built-in tools, then ``mcp__*``.

        Plain names replace pi's default tool set. Pi declares a ``direct`` MCP
        tool only when a ``--tools`` entry names or matches it. ``mcp__*``
        matches every MCP tool whatever name pi gives it (pi shortens long
        names and adds a hash on collisions); the scenario ``tool_names``
        allow-list is enforced by ``toolExposure`` in ``mcp.json`` (other tools
        are ``hidden``). Tools of repo ``.pi/mcp.json`` servers keep the
        exposure the repo sets; for ``codemode`` exposure, the scenario must
        allow the ``codemode`` tool in ``allowed_builtin_tools``. It is never
        added implicitly, as it can call other models.
        """
        entries = list(self._effective_builtin_tools())
        if self.mcp_servers or self._project_mcp_server_names():
            entries.append(PI_MCP_TOOLS_PATTERN)
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
                # (stdio MCP servers, bash tool commands).
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"pi executable not found ({command[0]!r}); install pi or set {PI_COMMAND_ENV}") from exc

        stderr_tail = _TailBuffer(PI_STDERR_TAIL_BYTES)
        tasks = [
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
        return process.returncode, stderr_tail.text()


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
        message = collector.feed_line(line)
        if message:
            await _notify(progress, f"{SDK_NAME}/{model.label}: {message}")
        if collector.stop_requested or collector.settled:
            return


def _default_pi_command() -> list[str]:
    raw = os.environ.get(PI_COMMAND_ENV, "").strip()
    return shlex.split(raw) if raw else [DEFAULT_PI_COMMAND]


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

    def append(self, chunk: bytes) -> None:
        self._data.extend(chunk)
        if len(self._data) > self._limit:
            del self._data[: len(self._data) - self._limit]

    def text(self) -> str:
        return self._data.decode("utf-8", errors="replace").strip()


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
    """
    _signal_group(process, signal.SIGTERM)
    if process.returncode is None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_wait_for_exit(process), timeout=PI_TERMINATE_GRACE_SECONDS)
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
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
@dataclass
class _ToolExecution:
    tool_name: str
    args: Any = None
    result: Any = None
    is_error: bool = False
    duration_ms: Any = None
    finished: bool = False


@dataclass
class _PiEventCollector:
    """Accumulate pi JSON-mode events into an :class:`AgentRunResult`."""

    max_turns: int | None = None
    session_id: str | None = None
    turns: int = 0
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: dict[str, _ToolExecution] = field(default_factory=dict)
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    settled: bool = False
    aborted: bool = False
    retry_error: str | None = None
    failure: str | None = None
    invalid_lines: int = 0
    #: True when the run settled but pi did not exit, so the runner stopped it.
    #: The exit status is then not a run failure.
    stopped_after_settle: bool = False

    @property
    def stop_requested(self) -> bool:
        return self.failure is not None

    def fail(self, message: str) -> None:
        if self.failure is None:
            self.failure = message

    def feed_line(self, line: bytes) -> str | None:
        """Process one JSONL record; return a short progress message, if any."""
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text.strip():
            return None
        try:
            event = json.loads(text)
        except ValueError:
            self.invalid_lines += 1
            logger.debug("Ignoring non-JSON pi stdout line: %.200s", text)
            return None
        if not isinstance(event, dict):
            # Pi writes one JSON object per record; anything else is damage.
            self.invalid_lines += 1
            return None
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
                return None
            if message.get("role") == "assistant":
                self.assistant_messages.append(message)
                self.usage.add(UsageMetrics.from_pi(message.get("usage")))
            elif message.get("role") == "toolResult" and isinstance(message.get("usage"), dict):
                # Nested model work done inside a tool (pi counts it in its
                # session totals too).
                self.usage.add(UsageMetrics.from_pi(message["usage"]))
            return None
        if event_type == "tool_execution_start":
            tool_id = str(event.get("toolCallId") or "")
            name = str(event.get("toolName") or "unknown")
            self.tool_executions[tool_id] = _ToolExecution(tool_name=name, args=event.get("args"))
            return f"tool {name}"
        if event_type == "tool_execution_end":
            tool_id = str(event.get("toolCallId") or "")
            execution = self.tool_executions.setdefault(
                tool_id, _ToolExecution(tool_name=str(event.get("toolName") or "unknown"))
            )
            execution.result = event.get("result")
            execution.is_error = bool(event.get("isError"))
            execution.duration_ms = event.get("durationMs")
            execution.finished = True
            return None
        if event_type == "compaction_end":
            # Compaction calls the model too; count that usage.
            result = event.get("result")
            if isinstance(result, dict) and isinstance(result.get("usage"), dict):
                self.usage.add(UsageMetrics.from_pi(result["usage"]))
            return "compaction"
        if event_type == "auto_retry_end":
            if event.get("success") is False:
                self.retry_error = str(event.get("finalError") or "pi retries failed")
            return None
        if event_type == "auto_retry_start":
            return f"retry {event.get('attempt')}/{event.get('maxAttempts')}"
        if event_type == "agent_settled":
            self.settled = True
            self.aborted = bool(event.get("aborted"))
            return None
        return None

    def result(self, *, model: ModelSpec, returncode: int | None, stderr_tail: str) -> AgentRunResult:
        segments = self._segments()
        tool_calls = [segment for segment in segments if isinstance(segment, AgentToolCall)]
        final = self.assistant_messages[-1] if self.assistant_messages else None
        answer = _message_text(final) if final is not None else ""
        error = self._error(model=model, final=final, returncode=returncode, stderr_tail=stderr_tail)
        return AgentRunResult(
            answer=answer or "pi did not produce a final response.",
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
        return None

    def _segments(self) -> list[AgentOutputSegment]:
        """Return reasoning, intermediate text and tool calls in message order.

        The text of the last assistant message is the answer, so it is not a
        segment.
        """
        segments: list[AgentOutputSegment] = []
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
                    segments.append(self._tool_call(block))
        return segments

    def _tool_call(self, block: dict[str, Any]) -> AgentToolCall:
        tool_id = str(block.get("id") or "")
        name = str(block.get("name") or "unknown")
        execution = self.tool_executions.get(tool_id)
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


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _tool_result_text(result: Any) -> str:
    """Render a pi tool result (``{"content": [...], "details": ...}``) as text."""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts: list[str] = []
        for block in result["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(f"[{block.get('type') or 'content'}]")
        return "\n".join(parts)
    if isinstance(result, str):
        return result
    return json_dumps_compact(result)
