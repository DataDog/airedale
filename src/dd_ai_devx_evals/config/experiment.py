"""Experiment configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from dd_ai_devx_evals.config import ConfigError, read_toml_file
from dd_ai_devx_evals.types import ModelSpec


def _is_localhost_url(url: str) -> bool:
    """Return True if the URL's host is localhost or any loopback IP (v4/v6)."""
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for an MCP server in a scenario.

    Mirrors the ``.mcp.json`` field model: ``type``/``command``/``args``/``env``/
    ``url``/``headers``. ``type`` is optional and inferred when omitted.

    - ``stdio`` transport: ``command`` (+ optional ``args``/``env``), launched
      directly by the agent SDK. ``url`` must not be set.
    - ``http`` transport: ``url`` (+ optional ``headers``). ``command``
      (+ optional ``args``/``env``) may additionally name a command used to
      auto-start the server when it is unreachable; in that case ``url`` MUST
      point at localhost.
    """

    name: str
    type: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Resolve and validate the transport type from the provided fields."""
        object.__setattr__(self, "type", self._resolve_type())

    def _resolve_type(self) -> str:
        has_url = self.url is not None
        has_command = self.command is not None

        if self.type is not None and self.type not in ("stdio", "http"):
            raise ConfigError(
                f"MCP server '{self.name}' has unsupported type '{self.type}' (expected 'stdio' or 'http')"
            )

        if self.type is not None:
            resolved = self.type
        elif has_url:
            resolved = "http"
        elif has_command:
            resolved = "stdio"
        else:
            raise ConfigError(f"MCP server '{self.name}' must define 'url' (http) or 'command' (stdio)")

        if resolved == "stdio":
            if not has_command:
                raise ConfigError(f"MCP server '{self.name}' of type 'stdio' must define 'command'")
            if has_url:
                raise ConfigError(f"MCP server '{self.name}' of type 'stdio' must not define 'url'")
        else:  # http
            if not has_url:
                raise ConfigError(f"MCP server '{self.name}' of type 'http' must define 'url'")
            if has_command and not _is_localhost_url(self.url):
                raise ConfigError(
                    f"MCP server '{self.name}' defines a managed start 'command' but 'url' "
                    f"'{self.url}' is not localhost"
                )

        if (self.args or self.env) and not has_command:
            raise ConfigError(f"MCP server '{self.name}' sets 'args'/'env' without a 'command'")

        return resolved

    @property
    def is_managed(self) -> bool:
        """True for http servers that carry a command used to auto-start them."""
        return self.type == "http" and self.command is not None


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a scenario runtime.

    ``skills`` and ``mcp_servers`` hold values resolved from the top-level
    ``[skills]`` / ``[mcp_servers.<name>]`` registries (scenarios reference them
    by name). ``allowed_builtin_tools`` is ``None`` when the field is omitted,
    which means *all* built-in tools are allowed; an explicit empty tuple means
    no built-in tools are exposed.
    """

    name: str
    description: str | None = None
    system_prompt: str | None = None
    skills: tuple[str, ...] = ()
    allowed_builtin_tools: tuple[str, ...] | None = None
    max_turns: int | None = None
    effort: str | None = None
    mcp_servers: tuple[McpServerConfig, ...] = ()


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for an evaluation task."""

    id: str
    prompt: str
    criteria: tuple[str, ...]
    description: str | None = None
    context: str | None = None
    latency_threshold_ms: int | None = None

    def __post_init__(self) -> None:
        """Validate that criteria is non-empty."""
        if not self.criteria:
            raise ConfigError(f"Task '{self.id}' must have at least one criterion")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    project: str
    models: tuple[str, ...]
    scenarios: tuple[ScenarioConfig, ...]
    tasks: tuple[TaskConfig, ...]
    description: str | None = None
    judge_model: str = "anthropic/claude-sonnet-4-6"
    runs: int = 1
    dataset_name: str | None = None
    defaults: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate configuration integrity."""
        if not self.models:
            raise ConfigError("Experiment must have at least one model")
        if not self.scenarios:
            raise ConfigError("Experiment must have at least one scenario")
        if not self.tasks:
            raise ConfigError("Experiment must have at least one task")

        # Validate all model specs can be parsed
        for model in self.models:
            try:
                ModelSpec.parse(model)
            except ValueError as e:
                raise ConfigError(f"Invalid model '{model}': {e}") from e

        # Validate judge model
        try:
            ModelSpec.parse(self.judge_model)
        except ValueError as e:
            raise ConfigError(f"Invalid judge model '{self.judge_model}': {e}") from e


_SCENARIO_KEYS = {
    "description",
    "system_prompt",
    "skills",
    "allowed_builtin_tools",
    "max_turns",
    "effort",
    "mcp_servers",
}
_DEFAULTS_KEYS = {
    "max_turns",
    "effort",
    "system_prompt",
    "skills",
    "allowed_builtin_tools",
    "mcp_servers",
}


def _parse_mcp_server(name: str, config: dict[str, Any]) -> McpServerConfig:
    """Parse MCP server config from TOML data."""
    return McpServerConfig(
        name=name,
        type=config.get("type"),
        command=config.get("command"),
        args=tuple(config.get("args", [])),
        env=config.get("env") or {},
        url=config.get("url"),
        headers=config.get("headers") or {},
        tool_names=tuple(config.get("tool_names", [])),
    )


def _resolve_skill_names(names: Any, registry: dict[str, str], *, where: str) -> tuple[str, ...]:
    """Resolve a list of skill names against the top-level ``[skills]`` registry."""
    if isinstance(names, dict):
        raise ConfigError(
            f"{where} 'skills' must be a list of names referencing the top-level [skills] table, not a table"
        )
    if not isinstance(names, (list, tuple)):
        raise ConfigError(f"{where} 'skills' must be a list of skill names")
    resolved: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise ConfigError(f"{where} 'skills' entries must be skill names (strings)")
        if name not in registry:
            raise ConfigError(
                f"{where} references unknown skill '{name}'; skills must be defined once under the "
                f"top-level [skills] table and referenced by name"
            )
        resolved.append(registry[name])
    return tuple(resolved)


def _resolve_mcp_server_names(
    names: Any, registry: dict[str, McpServerConfig], *, where: str
) -> tuple[McpServerConfig, ...]:
    """Resolve a list of server names against the top-level ``[mcp_servers]`` registry."""
    if isinstance(names, dict):
        raise ConfigError(
            f"{where} 'mcp_servers' must be a list of names referencing top-level "
            f"[mcp_servers.<name>] blocks, not an inline table"
        )
    if not isinstance(names, (list, tuple)):
        raise ConfigError(f"{where} 'mcp_servers' must be a list of MCP server names")
    resolved: list[McpServerConfig] = []
    for name in names:
        if not isinstance(name, str):
            raise ConfigError(f"{where} 'mcp_servers' entries must be server names (strings)")
        if name not in registry:
            raise ConfigError(
                f"{where} references unknown MCP server '{name}'; MCP servers must be defined once under "
                f"top-level [mcp_servers.<name>] blocks and referenced by name"
            )
        resolved.append(registry[name])
    return tuple(resolved)


def _parse_scenario(
    name: str,
    config: dict[str, Any],
    defaults: dict[str, Any],
    skill_registry: dict[str, str],
    mcp_registry: dict[str, McpServerConfig],
) -> ScenarioConfig:
    """Parse scenario config from TOML data, applying defaults.

    A scenario that sets a list/table field (``skills``, ``allowed_builtin_tools``,
    ``mcp_servers``, ``system_prompt``) overrides the corresponding ``[defaults]``
    value entirely; values are never merged.
    """
    unknown = set(config.keys()) - _SCENARIO_KEYS
    if unknown:
        raise ConfigError(f"Scenario '{name}' has unknown keys: {', '.join(sorted(unknown))}")

    where = f"Scenario '{name}'"

    # Scenario value wins entirely when present; otherwise fall back to defaults.
    skill_names = config["skills"] if "skills" in config else defaults.get("skills", [])
    skills = _resolve_skill_names(skill_names, skill_registry, where=where)

    server_names = config["mcp_servers"] if "mcp_servers" in config else defaults.get("mcp_servers", [])
    mcp_servers = _resolve_mcp_server_names(server_names, mcp_registry, where=where)

    system_prompt = config["system_prompt"] if "system_prompt" in config else defaults.get("system_prompt")

    # ``None`` (omitted everywhere) means "all built-in tools allowed"; an explicit
    # list (including empty) is an exact allow-list.
    if "allowed_builtin_tools" in config:
        allowed = tuple(config["allowed_builtin_tools"])
    elif "allowed_builtin_tools" in defaults:
        allowed = tuple(defaults["allowed_builtin_tools"])
    else:
        allowed = None

    max_turns = config.get("max_turns") or defaults.get("max_turns")
    effort = config.get("effort") or defaults.get("effort")

    return ScenarioConfig(
        name=name,
        description=config.get("description"),
        system_prompt=system_prompt,
        skills=skills,
        allowed_builtin_tools=allowed,
        max_turns=max_turns,
        effort=effort,
        mcp_servers=mcp_servers,
    )


_TASK_KEYS = {"prompt", "criteria", "description", "context", "latency_threshold_ms"}


def _parse_task(task_id: str, config: dict[str, Any]) -> TaskConfig:
    """Parse one ``[tasks.<id>]`` block; the table key supplies the task id."""
    unknown = set(config.keys()) - _TASK_KEYS
    if unknown:
        raise ConfigError(f"Task '{task_id}' has unknown keys: {', '.join(sorted(unknown))}")
    if "prompt" not in config:
        raise ConfigError(f"Task '{task_id}' missing required 'prompt' field")
    if "criteria" not in config or not config["criteria"]:
        raise ConfigError(f"Task '{task_id}' must have at least one criterion")

    return TaskConfig(
        id=task_id,
        description=config.get("description"),
        prompt=config["prompt"],
        context=config.get("context"),
        criteria=tuple(config["criteria"]),
        latency_threshold_ms=config.get("latency_threshold_ms"),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file."""
    data = read_toml_file(path)

    # Check for unknown top-level keys
    known_keys = {
        "project",
        "description",
        "models",
        "judge_model",
        "runs",
        "dataset_name",
        "defaults",
        "skills",
        "mcp_servers",
        "scenarios",
        "tasks",
    }
    unknown_keys = set(data.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"Unknown top-level keys: {', '.join(sorted(unknown_keys))}")

    # Required fields
    if "project" not in data:
        raise ConfigError("Missing required 'project' field")
    if "models" not in data:
        raise ConfigError("Missing required 'models' field")
    if "scenarios" not in data:
        raise ConfigError("Missing required 'scenarios' field")
    if "tasks" not in data:
        raise ConfigError("Missing required 'tasks' field")

    defaults = data.get("defaults", {})
    unknown_defaults = set(defaults.keys()) - _DEFAULTS_KEYS
    if unknown_defaults:
        raise ConfigError(f"Unknown [defaults] keys: {', '.join(sorted(unknown_defaults))}")
    defaults.setdefault("max_turns", 64)
    defaults.setdefault("effort", "medium")

    # Parse the top-level shared registries referenced by name from scenarios.
    skill_registry: dict[str, str] = {}
    for skill_name, skill_path in data.get("skills", {}).items():
        if not isinstance(skill_path, str):
            raise ConfigError(f"Skill '{skill_name}' must map to a directory path (string)")
        skill_registry[skill_name] = skill_path

    mcp_registry: dict[str, McpServerConfig] = {}
    for server_name, server_config in data.get("mcp_servers", {}).items():
        mcp_registry[server_name] = _parse_mcp_server(server_name, server_config)

    # Parse scenarios
    scenarios = []
    for scenario_name, scenario_config in data["scenarios"].items():
        scenarios.append(_parse_scenario(scenario_name, scenario_config, defaults, skill_registry, mcp_registry))

    # Parse tasks. The `tasks` table is keyed by task id ([tasks.<id>]); TOML
    # forbids duplicate keys, so ids are unique by construction.
    if not isinstance(data["tasks"], dict):
        raise ConfigError("'tasks' must be a table keyed by id ([tasks.<id>]), not an array of tables")
    tasks = []
    for task_id, task_config in data["tasks"].items():
        tasks.append(_parse_task(task_id, task_config))

    dataset_name = data.get("dataset_name")
    if dataset_name is None:
        dataset_name = data["project"]

    return ExperimentConfig(
        project=data["project"],
        description=data.get("description"),
        models=tuple(data["models"]),
        judge_model=data.get("judge_model", "anthropic/claude-sonnet-4-6"),
        runs=data.get("runs", 1),
        dataset_name=dataset_name,
        defaults=defaults,
        scenarios=tuple(scenarios),
        tasks=tuple(tasks),
    )
