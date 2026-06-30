# AGENTS.md — dd-ai-devx-evals

Context for AI agents and human contributors working on this codebase. Read this
fully before making changes. It is both a contributor guide and the architectural
contract that every module must honor.

---

## 1. What this project is

`dd-ai-devx-evals` is a **config-driven evaluation harness** for agentic LLM runs.
Given two TOML files it runs an evaluation matrix and reports every cell to
**Datadog LLM Observability Experiments**.

The evaluation matrix is:

```
model  ×  scenario  ×  task   (each cell repeated `runs` times)
```

- **model** — a provider-qualified model string `"<provider>/<model>"`
  (e.g. `anthropic/claude-sonnet-4-6`, `openai/gpt-5.5`). Providers supported:
  `anthropic`, `openai`.
- **scenario** — a named config block describing the *runtime* offered to the
  model: which MCP servers are visible (and how to start them if needed), which
  skills are exposed, and optional prompt/tool tweaks.
- **task** — a prompt (+ optional context) plus evaluation criteria used by an
  LLM-as-judge scorer.

Each `(model, scenario, task)` cell becomes **one LLMObs experiment** with a
single-record dataset view, run `runs` times.

### Execution model — provider-native agentic SDKs

The model provider selects the execution engine:

| provider    | engine                                  |
|-------------|-----------------------------------------|
| `anthropic` | `claude-agent-sdk` (Claude Code)        |
| `openai`    | `openai-codex` (Codex)                  |

MCP servers and skills are passed **natively** to these SDKs — we do NOT
reimplement a tool-calling loop. This is what enables:

- **Skills & subagents** support (both SDKs consume the open SKILL.md format).
- **Native LLMObs spans** for Claude (the ddtrace integration owns its spans);
  **decorator spans** (`@agent` / `@llm`) for Codex, which has no native
  integration.
- **Distributed tracing for complete token counting:** before each run we call
  `LLMObs.inject_distributed_headers(...)` and merge the resulting headers into
  the HTTP headers sent to MCP servers. When the MCP server is itself
  LLMObs-instrumented, its spans (orchestrator + sub-agents) link to the
  experiment span, so the experiment's `token_count` reflects the full cost.

---

## 2. Configuration contract

Two TOML files. Parsing lives in `config/experiment.py` and `config/gateway.py`.
Use the stdlib `tomllib` (Python ≥ 3.11). Unknown keys should raise a clear
`ConfigError` (defined in `config/__init__.py`).

### 2.1 `experiment.toml`

```toml
# Required: LLMObs ml_app / project name that experiments & datasets land in.
project = "my-evals"

# Optional human description applied to the dataset.
description = "Example eval suite"

# Required: provider-qualified models to evaluate.
models = ["anthropic/claude-sonnet-4-6", "openai/gpt-5.5"]

# Optional: judge model for the rubric scorer. Default below.
judge_model = "anthropic/claude-sonnet-4-6"

# Optional: how many times to run each (model, scenario, task) cell. Default 1.
runs = 1

# Optional: stable dataset name. Defaults to `project`.
dataset_name = "my-evals"

# --- Shared registries: defined once, referenced by name -----------------
# Skill registry. Table key = skill name; value = Agent-Skills SKILL.md dir.
[skills]
apm = "./skills/apm"

# MCP server registry. Table key = server name. Fields mirror the .mcp.json
# model: type/command/args/env/url/headers. `type` is optional and inferred
# ("http" when `url` is set, "stdio" when only `command` is set).
[mcp_servers.apm]
# HTTP transport (supports distributed-trace header injection):
url = "http://localhost:8000/mcp"
headers = { source = "evals" }        # optional static headers
tool_names = ["search_apm_libraries"] # tools allow-listed for this server
# Optional managed auto-start: on an http server, `command` (+ args/env) is the
# command used to launch the server if it is unreachable. Reachability is probed
# via the MCP protocol itself (a tools/list call). `url` MUST then be localhost
# (any loopback host, IPv4 or IPv6).
command = "python"
args = ["-m", "my_mcp_server"]
env = { MCP_MODE = "both" }

# stdio transport (alternative to url; launched directly by the agent SDK):
[mcp_servers.tools]
command = "python"
args = ["-m", "my_stdio_server"]
env = { FOO = "bar" }
tool_names = ["do_thing"]

# Optional defaults applied to every scenario unless the scenario sets the same
# field. A scenario that sets a field overrides the default ENTIRELY (no merge).
[defaults]
max_turns = 64          # outer agent loop cap
effort = "medium"       # reasoning effort hint passed to the SDKs
# May also default the list/table fields, all optional:
# system_prompt = "..."
# skills = ["apm"]
# allowed_builtin_tools = ["Read", "Grep", "Glob"]
# mcp_servers = ["apm"]
# A default workdir is also allowed; a scenario setting its own [scenarios.X.workdir]
# replaces it entirely (no merge).
# [defaults.workdir]
# repo = "self"

# --- Scenarios: named runtime configurations -----------------------------
# Table key = scenario name (used in experiment names / tags). Scenarios
# reference shared skills / MCP servers BY NAME (no inline definitions).
[scenarios.fat-mcp]
description = "Full MCP orchestrator tool"
system_prompt = "..."                 # optional; appended to the base system prompt
skills = ["apm"]                      # optional; names from the [skills] registry
# allowed_builtin_tools omitted -> ALL built-in tools allowed.
# allowed_builtin_tools = [] -> no built-in tools; = ["Read", ...] -> exactly those.
mcp_servers = ["apm"]                 # names from the [mcp_servers.<name>] registry
max_turns = 64                        # optional per-scenario override
effort = "medium"                     # optional per-scenario override

# --- Working directory: where the agent runs (optional) ------------------
# Without a `workdir` the cell runs in a fresh empty temp dir (hermetic
# sandbox). With one, each run/repetition gets a fresh git worktree of a
# cached clone (or an empty dir when `repo` is omitted), with ordered setup
# steps applied.
[scenarios.regression.workdir]
repo = "self"          # "self" (repo containing experiment.toml) | git URL | local path
ref  = "v2.3.0"        # optional; default = source repo's current HEAD
[[scenarios.regression.workdir.steps]]
op    = "restore"      # git restore --source=<from> -- <paths...> (needs a repo)
from  = "v2.2.0"
paths = ["src/api/**", "README.md"]
[[scenarios.regression.workdir.steps]]
op    = "remove"       # filesystem-delete matching globs from the worktree
paths = ["secrets/**"]
[[scenarios.regression.workdir.steps]]
op      = "write"      # create/overwrite a file (exactly one of content/source)
path    = "NOTES.md"
content = "Evaluate the migration."

# --- Tasks: prompts + evaluation criteria --------------------------------
# Table key = task id (unique by construction; TOML forbids duplicate keys).
[tasks.ssi_overview]
description = "[glossary] SSI definition"   # optional
prompt = "What is Single Step Instrumentation?"
context = "..."                              # optional extra context appended to the prompt
criteria = [                                 # rubric criteria, one judge call each
    "Defines SSI correctly",
    "Mentions supported languages",
]
latency_threshold_ms = 30000                 # optional, reported only
```

Notes:
- Skills and MCP servers are defined **once** in the top-level `skills` and
  `mcp_servers` registries and referenced **by name** from scenarios
  (and from `[defaults]`), so they can be reused across scenarios without
  copy-paste. A scenario naming a skill/server absent from the registry is a
  `ConfigError`. Inline `[scenarios.<name>.mcp_servers.<x>]` tables and raw skill
  paths inside a scenario are rejected.
- `[defaults]` may set `max_turns`, `effort`, `system_prompt`, `skills`,
  `allowed_builtin_tools`, and `mcp_servers`. When a scenario sets one of these,
  its value replaces the default **entirely** — lists/tables are never merged.
- `system_prompt` is optional on both scenarios and `[defaults]`; when present it
  is appended to the base system prompt.
- Tasks are a table keyed by id (`[tasks.<id>]`), not an array of tables. The id
  comes from the table key, so it is unique by construction (TOML rejects
  duplicate keys); an array-of-tables `[[tasks]]` form is a `ConfigError`, and an
  in-body `id` key is rejected as unknown.
- `allowed_builtin_tools` semantics: **omitted ⇒ all built-in tools allowed**; an
  explicit empty list `[]` ⇒ no built-in tools; a non-empty list ⇒ exactly those.
  In code this is the `None`-vs-`()`-vs-tuple distinction on
  `ScenarioConfig.allowed_builtin_tools`. For Claude (`dontAsk` permission mode)
  the "all" case pre-approves the full `CLAUDE_BUILTIN_TOOLS` set; for Codex the
  field is informational only (Codex does not gate built-ins on it).
- A scenario need not reference any MCP server — scenarios with no MCP servers
  are valid (model answers from skills/builtins only).
- `type` is optional; when omitted it is inferred ("http" if `url` is set,
  "stdio" if only `command` is set). Only `stdio` and `http` are supported.
- A `stdio` server defines `command` (+ optional `args`/`env`) and MUST NOT set
  `url`. An `http` server defines `url` (+ optional `headers`); it MAY also set
  `command` (+ `args`/`env`) as a managed auto-start command, in which case `url`
  MUST be localhost (any loopback host, IPv4 or IPv6).
- `tool_names` is the allow-list of MCP tools exposed; empty/omitted means "all
  tools the server advertises".
- `workdir` is a defaultable scenario block (override-entirely, like `skills`):
  - `repo`: `"self"` (the git toplevel containing `experiment.toml`, resolved &
    validated eagerly at load), a git URL, or a local path (resolved relative to
    the config dir). Omitted ⇒ start from an **empty** directory.
  - `ref`: branch/tag/commit checked out as the workspace base; default = the
    source repo's current HEAD.
  - `steps`: ordered ops, each discriminated by `op`:
    - `restore` — `from` (ref) + `paths` (non-empty pathspecs); requires a `repo`.
    - `remove` — `paths` (non-empty globs); filesystem-deletes from the worktree.
    - `write` — `path` (workspace-relative, no `..`/absolute) + exactly one of
      `content` (inline) or `source` (file copied in, resolved against the config
      dir).
  - Each run/repetition gets a **fresh** workspace: a repo source is cloned once
    (lazily, `--no-hardlinks` for local sources so `repo="self"` never touches
    your checkout) into a cache, and every workspace is a `git worktree` of it.
  - **Project files are always honored.** A cloned repo's own `.claude/`,
    `.codex/`, `AGENTS.md`/`CLAUDE.md`, project subagents/skills, and `.mcp.json`
    are discovered automatically; a bare temp dir has none, so it stays hermetic.
  - **MCP discovery is harness-specific.** Claude: `strict_mcp_config` stays
    `True` and we parse `<cwd>/.mcp.json` ourselves into specs (repo HTTP servers
    thus also get distributed-trace headers); Codex does **not** read `.mcp.json`
    (its servers come from `$CODEX_HOME/config.toml`). Scenario-configured servers
    **win** on a name collision; discovered servers are not auto-started.
  - **Codex hermeticity.** Codex reads MCP servers (and all other global config)
    from `$CODEX_HOME/config.toml` (global, default `~/.codex`), so an operator's
    ambient servers would leak into runs (the Codex analog of Claude's
    `strict_mcp_config`). `CodexRunner._codex_env()` therefore **always** isolates
    `CODEX_HOME` to a fresh, empty per-run dir under `cwd`, so no global config
    ever loads. Auth is preserved on both paths: env auth (gateway token /
    `OPENAI_API_KEY`) is used directly; otherwise only `auth.json` is **copied**
    from the operator's real `CODEX_HOME` into the isolated dir, so `codex login`
    keeps working without inheriting the global `config.toml`. A repo-scoped
    `<cwd>/.codex/config.toml` is **not** read by Codex for MCP servers
    (verified), so there is no Codex project MCP discovery.
  - Filesystem paths in the config (`[skills]` entries, workdir local `repo`
    paths, `write.source`) resolve relative to `experiment.toml`'s directory.
    Command-style values (MCP `command`/`args`, gateway `credentials_helper`) are
    left untouched.

### 2.2 `gateway.toml` (optional, separate file)

Maps a provider name to gateway settings. Passed via `--gateway-config PATH`.
When omitted, the standard provider APIs and env-var API keys are used
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

```toml
[providers.anthropic]
base_url = "https://ai-gateway.example.com"
# A shell command whose stdout yields a bearer token (JWT or raw). Run on demand.
credentials_helper = "mytool auth token --datacenter us1"
# OR a static key from the environment instead of a helper:
# api_key_env = "ANTHROPIC_API_KEY"
headers = { source = "evals", "org-id" = "2", provider = "anthropic" }

[providers.openai]
base_url = "https://ai-gateway.example.com/v1"
credentials_helper = "mytool auth token --datacenter us1"
headers = { source = "evals" }
```

Resolution rules (`gateway.py`):
- If a provider has a `credentials_helper`, run it once (cached for the process),
  trim output, and use it as the bearer token.
- Else if `api_key_env` is set, read that env var.
- Else fall back to the provider SDK default (env var).
- `headers` are static extra headers merged into every request to that provider.
- For Claude, the gateway is wired through Claude Code's env + `apiKeyHelper`
  settings (the SDK runs the helper itself with a TTL). For Codex, through config
  overrides (base URL + headers + bearer). For the judge (direct SDK clients), we
  run the helper ourselves and set `base_url` / `api_key` / `default_headers`.

---

## 3. Module layout & responsibilities

```
src/dd_ai_devx_evals/
  __init__.py            __version__
  __main__.py            python -m dd_ai_devx_evals -> cli.main
  cli.py                 argparse CLI; parses flags, loads config, runs matrix

  types.py               ModelSpec, UsageMetrics, HarnessResult, slugify
  config/
    __init__.py          ConfigError; shared TOML helpers
    experiment.py        ExperimentConfig, ScenarioConfig, McpServerConfig, TaskConfig,
                         WorkdirConfig + RestoreStep/RemoveStep/WriteStep (+ load_experiment)
    gateway.py           GatewayConfig, ProviderGatewayConfig (+ load_gateway)

  gateway.py             credential-helper exec + per-provider header/base-url/key resolution (runtime)
  tracing.py             current_trace_headers() via LLMObs.inject_distributed_headers
  observability.py       enable_llmobs(project, *, agentless, integrations_enabled)

  mcp.py                 McpServerSpec (from McpServerConfig); HTTP/stdio rendering for both SDKs;
                         merged_headers (static + trace); managed-server start/health;
                         tools/list metadata catalog; discover_claude_project_mcp_servers (.mcp.json)
  skills.py              cross-provider skill staging (Claude + Codex) + collision guard;
                         discover_claude_skill_names (project skills allow-list)
  workdir.py             WorkspaceManager: clone-cache + git-worktree per-run workspaces;
                         restore/remove/write step application; cache-confined teardown

  harness/
    __init__.py          create_runner(model, *, scenario, gateway, cwd) -> AgentRunner;
                         merges scenario + project-discovered MCP servers (scenario wins)
    base.py              AgentRunner protocol; AgentRunResult; AgentToolCall
    claude.py            claude-agent-sdk runner (native LLMObs; skills; MCP; gateway)
    codex.py             openai-codex runner (decorator spans; skills; MCP; gateway)

  scoring/
    __init__.py
    rubric.py            RubricEvaluator(BaseAsyncEvaluator): per-criterion LLM-as-judge

  dataset.py             build/sync an LLMObs Dataset from tasks; single-record views
  experiment.py          build & run the matrix; parallelism; ties harness+scoring+dataset together
  progress.py            console/rich progress display
  summary.py             post-run token/latency summary table
```

### Source of truth for the proven implementation

This project is an extraction/generalization of an internal harness. When in
doubt about distributed-tracing, SDK wiring, or usage accounting, consult the
original (do NOT copy hardcoded project names / gateway URLs / credential tools):

- `../apm-libraries-devx-ai/app/evals/llm.py` — `AgentSdkRunner`, MCP specs,
  distributed-trace header injection, native-vs-decorator span strategy, Claude
  & Codex wiring, usage accounting. **This is the canonical reference.**
- `../apm-libraries-devx-ai/app/evals/types.py` — `ModelSpec`, `UsageMetrics`.
- `../apm-libraries-devx-ai/app/evals/harnesses.py` — harness abstraction & system prompt.
- `../apm-libraries-devx-ai/app/evals/judge.py` — `QEvalRubricEvaluator`.
- `../apm-libraries-devx-ai/app/evals/runner.py` — experiment loop, dataset sync,
  managed MCP server, `enable_llmobs`.
- Top-level `evals/` package (in git history of that repo) — `progress.py`,
  `summary.py` patterns worth reusing.

---

## 4. Key type contracts

```python
# types.py
ModelProvider = Literal["anthropic", "openai"]

@dataclass(frozen=True)
class ModelSpec:
    provider: ModelProvider
    name: str       # provider-native model id, e.g. "claude-sonnet-4-6"
    label: str      # original "<provider>/<name>"
    @classmethod
    def parse(cls, value: str) -> "ModelSpec": ...  # rejects non-provider-qualified / unknown providers

@dataclass
class UsageMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    def add(self, other) -> None: ...
    def to_llmobs_metrics(self) -> dict[str, int | float]: ...
    @classmethod
    def from_anthropic(cls, usage): ...
    @classmethod
    def from_openai(cls, usage): ...
    @classmethod
    def from_claude_sdk(cls, usage, *, total_cost_usd=None): ...
    @classmethod
    def from_codex(cls, usage): ...

@dataclass
class HarnessResult:
    answer: str
    usage: UsageMetrics
    tool_calls: list[dict]
    harness: str       # scenario name
    def to_output_data(self) -> dict: ...   # {"answer", "usage", "tool_calls", "harness"}

def slugify(value: str) -> str: ...
```

---

## 5. LLMObs reporting contract

- Enable once via `observability.enable_llmobs(project, agentless=..., integrations_enabled=True)`.
  Use `DD_API_KEY` / `DD_APP_KEY` / `DD_SITE` from env. `agentless` defaults to True.
- One `LLMObs.async_experiment(...)` per `(model, scenario, task)` cell, with:
  - `name` = stable, slugified `"<scenario>|<model.label>|<task.id>"` (≤180 chars)
  - `dataset` = a single-record view of the synced dataset
  - `evaluators` = `[RubricEvaluator(judge_model=..., gateway=...)]`
  - `project_name` = `project`
  - `config` carrying `model_name`, `scenario`, `task` id, `judge_model`,
    `mcp_servers` (redacted via `to_safe_dict()`), `gateway_enabled`
  - `runs` = configured runs
- **Prefer native integrations** (`integrations_enabled=True`); only add manual
  spans where a provider has no native integration. Concretely: Claude runs emit
  no synthetic agent/LLM/tool spans (the SDK integration owns them); Codex runs
  are wrapped in `@agent` + `@llm` decorator spans and annotated with usage,
  tool definitions, and I/O.
- Always `LLMObs.annotate(metrics=usage.to_llmobs_metrics(), ...)` on the active
  experiment span so token totals are complete even when the harness sees usage
  the integration didn't.
- Headers from `tracing.current_trace_headers()` are merged into MCP HTTP headers
  by `McpServerSpec.merged_headers(...)`. Never log bearer tokens; redact headers
  in any LLMObs metadata via `to_safe_dict()`.

---

## 6. Skills staging contract (`skills.py`)

`ScenarioConfig.skills` is a list of skill *directories* (each an Agent-Skills
SKILL.md package), already resolved from the top-level `skills` registry at
config-load time (scenarios reference skills by name). The harness stages them
into each engine's discoverable location:

- **Claude (`claude-agent-sdk`):** make dirs discoverable in the run `cwd`
  (e.g. `<cwd>/.claude/skills/<name>`), set `setting_sources` accordingly (or use
  `add_dirs`/`plugins`), and pass `ClaudeAgentOptions(skills=[names])` to
  allow-list them. The SDK injects the `Skill` tool.
- **Codex (`openai-codex`):** Codex has no per-thread skills argument. Stage dirs
  into a repo-scope location under the run `cwd` (e.g. `<cwd>/.codex/skills/<name>`)
  so Codex discovers them by scope, and enable/approve via the thread `config`
  (`skill_approval`) so the non-interactive run may invoke them.

Both engines consume the same SKILL.md package format, so a single configured
directory is staged for whichever engine the model uses. Keep staging idempotent
and confined to the per-run temp `cwd`.

---

## 7. CLI contract (`cli.py`)

```
dd-ai-devx-evals CONFIG [options]

  CONFIG                   experiment TOML (required positional argument)
  --gateway-config PATH    gateway TOML; omit to use provider default APIs
  --model M                run only these models (repeatable / comma-separated)
  --scenario S             run only these scenarios (repeatable / comma-separated)
  --task T                 run only these task ids (repeatable / comma-separated)
  --runs N                 override runs per cell
  --judge-model M          override judge model
  --jobs N                 total cells run concurrently across the matrix (default 1)
  --dry-run                print the matrix and exit
  --no-progress            disable the live progress display
  --agentless / --no-agentless   LLMObs submission mode (default agentless)
  --fail-fast              stop on the first task/evaluator error
```

`--jobs` is the single concurrency knob: a global semaphore bounds how many
cells run at once, and each cell runs its `runs` repetitions sequentially, so
`--jobs` equals the maximum number of in-flight agent runs. Exit non-zero on
config errors or when `--fail-fast` and a cell fails.

---

## 8. Development workflow

- **Python:** ≥ 3.11, < 3.14. Use `uv` for envs (`uv sync`, `uv run ...`).
- **Lint/format:** `uv run ruff check .` and `uv run ruff format .` (line length 120).
- **Tests:** `uv run pytest`. Unit tests MUST run fully offline — no network, no
  real provider/LLMObs calls. Mock SDK clients and `LLMObs`. Cover: config TOML
  parsing & validation, gateway resolution, MCP spec rendering (Claude config &
  Codex overrides), `ModelSpec`/`UsageMetrics`, judge JSON parsing, skill staging.
- **Type hints:** everywhere; `from __future__ import annotations` at the top of
  every module. Prefer dataclasses for value objects.
- **Secrets:** never commit tokens; never log bearer tokens; redact headers in
  LLMObs metadata.

### Version control — jj (Jujutsu), never git

This repo uses **jj**. Never run `git` mutating commands. Commit each substantial
unit separately with a **conventional-commit** subject (`feat(scope): ...`,
`fix(...)`, `docs:`, `test:`, `chore:`), a blank line, then a body explaining the
*what* and *why* wrapped at 72 columns. Subject ≤ 72 chars, imperative mood, not
capitalized, no trailing period, code/paths in backticks. Use `jj commit -m` /
`jj describe -m` + `jj new`.

---

## 9. Conventions & invariants

- `<provider>/<model>` is the only accepted model format; reject others early.
- A scenario name, task id, and model label uniquely identify an experiment cell.
- Keep Datadog-specific defaults OUT of code: project name, gateway URLs, and
  credential commands all come from config. There are no hardcoded internal URLs.
- Distributed tracing only applies to HTTP MCP transports; stdio servers can't
  receive per-request trace headers (document this where relevant).
- Fail loudly when an SDK completes without reporting token usage (a silent zero
  means broken accounting).
