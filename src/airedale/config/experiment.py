"""Experiment configuration parsing and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airedale.config import ConfigError, read_toml_file
from airedale.types import ModelSpec


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
class RestoreStep:
    """`git restore --source=<from_ref> -- <paths...>` inside the workspace."""

    from_ref: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class RemoveStep:
    """Delete files/dirs matching glob ``paths`` from the workspace."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class WriteStep:
    """Create/overwrite a file in the workspace.

    Exactly one of ``content`` (inline) or ``source_path`` (a file copied in,
    already resolved to an absolute path at load time) is set.
    """

    path: str
    content: str | None = None
    source_path: str | None = None


WorkdirStep = RestoreStep | RemoveStep | WriteStep


@dataclass(frozen=True)
class WorkdirConfig:
    """Scenario working-directory configuration.

    ``repo`` is ``"self"`` (the git repo containing ``experiment.toml``), a git
    URL, or an absolute local path (local paths are resolved against the config
    dir at load time). ``None`` means start from an empty directory.
    """

    repo: str | None = None
    ref: str | None = None
    steps: tuple[WorkdirStep, ...] = ()


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
    workdir: WorkdirConfig | None = None


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
    config_path: Path | None = None

    @property
    def config_dir(self) -> Path:
        """Directory containing ``experiment.toml``; filesystem paths resolve here.

        Falls back to the current working directory when the config was not
        loaded from a file (e.g. synthesized in tests).
        """
        return self.config_path.parent if self.config_path is not None else Path.cwd()

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
    "workdir",
}
_DEFAULTS_KEYS = {
    "max_turns",
    "effort",
    "system_prompt",
    "skills",
    "allowed_builtin_tools",
    "mcp_servers",
    "workdir",
}
_WORKDIR_KEYS = {"repo", "ref", "steps"}
_RESTORE_STEP_KEYS = {"op", "from", "paths"}
_REMOVE_STEP_KEYS = {"op", "paths"}
_WRITE_STEP_KEYS = {"op", "path", "content", "source"}


def _looks_like_git_url(value: str) -> bool:
    """Return True for remote git sources (``scheme://...`` or ``user@host:path``)."""
    if "://" in value:
        return True
    return bool(re.match(r"^[\w.+-]+@[\w.-]+:", value))


def _git_toplevel(config_dir: Path) -> str | None:
    """Return the git toplevel containing ``config_dir``, or None if not a repo."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(config_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    toplevel = result.stdout.strip()
    return toplevel or None


def _require_str_paths(value: Any, *, where: str) -> tuple[str, ...]:
    """Validate a non-empty list of string pathspecs."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{where} 'paths' must be a non-empty list of strings")
    paths: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{where} 'paths' entries must be strings")
        paths.append(item)
    return tuple(paths)


def _validate_workspace_relative_path(path: str, *, where: str) -> None:
    """Reject absolute paths and ``..`` escapes for a workspace-relative path."""
    from pathlib import PurePosixPath

    if not path:
        raise ConfigError(f"{where} 'path' must be a non-empty string")
    pure = PurePosixPath(path)
    if pure.is_absolute() or Path(path).is_absolute():
        raise ConfigError(f"{where} 'path' must be relative to the workspace, got absolute '{path}'")
    if ".." in pure.parts:
        raise ConfigError(f"{where} 'path' must not escape the workspace with '..': '{path}'")


def _parse_workdir_step(name: str, index: int, raw: Any, *, config_dir: Path, has_repo: bool) -> WorkdirStep:
    """Parse and validate a single workdir step table."""
    where = f"Scenario '{name}' workdir step #{index + 1}"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table")
    op = raw.get("op")
    if not isinstance(op, str):
        raise ConfigError(f"{where} must define a string 'op'")

    if op == "restore":
        unknown = set(raw.keys()) - _RESTORE_STEP_KEYS
        if unknown:
            raise ConfigError(f"{where} (restore) has unknown keys: {', '.join(sorted(unknown))}")
        if not has_repo:
            raise ConfigError(f"{where} uses 'restore' but the workdir has no 'repo' to restore from")
        from_ref = raw.get("from")
        if not isinstance(from_ref, str) or not from_ref:
            raise ConfigError(f"{where} (restore) must define a non-empty 'from' ref")
        paths = _require_str_paths(raw.get("paths"), where=f"{where} (restore)")
        return RestoreStep(from_ref=from_ref, paths=paths)

    if op == "remove":
        unknown = set(raw.keys()) - _REMOVE_STEP_KEYS
        if unknown:
            raise ConfigError(f"{where} (remove) has unknown keys: {', '.join(sorted(unknown))}")
        paths = _require_str_paths(raw.get("paths"), where=f"{where} (remove)")
        return RemoveStep(paths=paths)

    if op == "write":
        unknown = set(raw.keys()) - _WRITE_STEP_KEYS
        if unknown:
            raise ConfigError(f"{where} (write) has unknown keys: {', '.join(sorted(unknown))}")
        path = raw.get("path")
        if not isinstance(path, str):
            raise ConfigError(f"{where} (write) must define a string 'path'")
        _validate_workspace_relative_path(path, where=f"{where} (write)")
        has_content = "content" in raw
        has_source = "source" in raw
        if has_content == has_source:
            raise ConfigError(f"{where} (write) must define exactly one of 'content' or 'source'")
        if has_content:
            content = raw.get("content")
            if not isinstance(content, str):
                raise ConfigError(f"{where} (write) 'content' must be a string")
            return WriteStep(path=path, content=content, source_path=None)
        source = raw.get("source")
        if not isinstance(source, str):
            raise ConfigError(f"{where} (write) 'source' must be a string")
        source_path = str((config_dir / source).resolve())
        return WriteStep(path=path, content=None, source_path=source_path)

    raise ConfigError(f"{where} has unknown op '{op}' (expected 'restore', 'remove', or 'write')")


def _parse_workdir(name: str, raw: Any, *, config_dir: Path) -> WorkdirConfig:
    """Parse and validate a scenario (or defaults) ``workdir`` block."""
    where = f"Scenario '{name}' workdir"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table")
    unknown = set(raw.keys()) - _WORKDIR_KEYS
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")

    repo_raw = raw.get("repo")
    repo: str | None = None
    if repo_raw is not None:
        if not isinstance(repo_raw, str) or not repo_raw:
            raise ConfigError(f"{where} 'repo' must be a non-empty string")
        if repo_raw == "self":
            # Validate eagerly so dry-run also catches a config outside any repo.
            if _git_toplevel(config_dir) is None:
                raise ConfigError(f'{where} sets repo="self" but the config directory is not inside a git repository')
            repo = "self"
        elif _looks_like_git_url(repo_raw):
            repo = repo_raw
        else:
            repo = str((config_dir / repo_raw).resolve())

    ref = raw.get("ref")
    if ref is not None and not isinstance(ref, str):
        raise ConfigError(f"{where} 'ref' must be a string")

    raw_steps = raw.get("steps", [])
    if not isinstance(raw_steps, (list, tuple)):
        raise ConfigError(f"{where} 'steps' must be an array of tables")
    steps = tuple(
        _parse_workdir_step(name, index, step, config_dir=config_dir, has_repo=repo is not None)
        for index, step in enumerate(raw_steps)
    )

    return WorkdirConfig(repo=repo, ref=ref, steps=steps)


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
    *,
    config_dir: Path,
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

    # Scenario workdir wins entirely over the [defaults] workdir (no merge).
    workdir_raw = config["workdir"] if "workdir" in config else defaults.get("workdir")
    workdir = _parse_workdir(name, workdir_raw, config_dir=config_dir) if workdir_raw is not None else None

    return ScenarioConfig(
        name=name,
        description=config.get("description"),
        system_prompt=system_prompt,
        skills=skills,
        allowed_builtin_tools=allowed,
        max_turns=max_turns,
        effort=effort,
        mcp_servers=mcp_servers,
        workdir=workdir,
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
    config_path = Path(path).resolve()
    config_dir = config_path.parent
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
    # Filesystem paths in the config (skill dirs) resolve relative to the config
    # file's directory, never the process CWD, so a relative path like
    # "./skills/apm" works regardless of where the tool is invoked from.
    skill_registry: dict[str, str] = {}
    for skill_name, skill_path in data.get("skills", {}).items():
        if not isinstance(skill_path, str):
            raise ConfigError(f"Skill '{skill_name}' must map to a directory path (string)")
        skill_registry[skill_name] = str((config_dir / skill_path).resolve())

    mcp_registry: dict[str, McpServerConfig] = {}
    for server_name, server_config in data.get("mcp_servers", {}).items():
        mcp_registry[server_name] = _parse_mcp_server(server_name, server_config)

    # Parse scenarios
    scenarios = []
    for scenario_name, scenario_config in data["scenarios"].items():
        scenarios.append(
            _parse_scenario(
                scenario_name, scenario_config, defaults, skill_registry, mcp_registry, config_dir=config_dir
            )
        )

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
        config_path=config_path,
    )
