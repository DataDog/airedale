"""Experiment configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from dd_ai_devx_evals.config import ConfigError, read_toml_file
from dd_ai_devx_evals.types import ModelSpec


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for an MCP server in a scenario."""

    name: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    bearer_token_env_var: str | None = None
    tool_names: tuple[str, ...] = ()
    start_command: str | None = None
    start_env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one of url or command is set."""
        has_url = self.url is not None
        has_command = self.command is not None
        if has_url == has_command:  # Both set or both unset
            raise ConfigError(f"MCP server '{self.name}' must have exactly one of 'url' or 'command' set")


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a scenario runtime."""

    name: str
    description: str | None = None
    system_prompt: str | None = None
    skills: tuple[str, ...] = ()
    allowed_builtin_tools: tuple[str, ...] = ()
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


def _parse_mcp_server(name: str, config: dict[str, Any]) -> McpServerConfig:
    """Parse MCP server config from TOML data."""
    return McpServerConfig(
        name=name,
        url=config.get("url"),
        command=config.get("command"),
        args=tuple(config.get("args", [])),
        env=config.get("env") or {},
        headers=config.get("headers") or {},
        bearer_token_env_var=config.get("bearer_token_env_var"),
        tool_names=tuple(config.get("tool_names", [])),
        start_command=config.get("start_command"),
        start_env=config.get("start_env") or {},
    )


def _parse_scenario(name: str, config: dict[str, Any], defaults: dict[str, Any]) -> ScenarioConfig:
    """Parse scenario config from TOML data, applying defaults."""
    mcp_servers = []
    if "mcp_servers" in config:
        for server_name, server_config in config["mcp_servers"].items():
            mcp_servers.append(_parse_mcp_server(server_name, server_config))

    # Apply defaults for max_turns and effort if not specified
    max_turns = config.get("max_turns") or defaults.get("max_turns")
    effort = config.get("effort") or defaults.get("effort")

    return ScenarioConfig(
        name=name,
        description=config.get("description"),
        system_prompt=config.get("system_prompt"),
        skills=tuple(config.get("skills", [])),
        allowed_builtin_tools=tuple(config.get("allowed_builtin_tools", [])),
        max_turns=max_turns,
        effort=effort,
        mcp_servers=tuple(mcp_servers),
    )


def _parse_task(config: dict[str, Any]) -> TaskConfig:
    """Parse task config from TOML data."""
    if "id" not in config:
        raise ConfigError("Task missing required 'id' field")
    if "prompt" not in config:
        raise ConfigError(f"Task '{config['id']}' missing required 'prompt' field")
    if "criteria" not in config or not config["criteria"]:
        raise ConfigError(f"Task '{config['id']}' must have at least one criterion")

    return TaskConfig(
        id=config["id"],
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
    defaults.setdefault("max_turns", 64)
    defaults.setdefault("effort", "medium")

    # Parse scenarios
    scenarios = []
    for scenario_name, scenario_config in data["scenarios"].items():
        scenarios.append(_parse_scenario(scenario_name, scenario_config, defaults))

    # Parse tasks
    tasks = []
    for task_config in data["tasks"]:
        tasks.append(_parse_task(task_config))

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
