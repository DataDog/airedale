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

# Optional defaults applied to every scenario unless overridden.
[defaults]
max_turns = 64          # outer agent loop cap
effort = "medium"       # reasoning effort hint passed to the SDKs

# --- Scenarios: named runtime configurations -----------------------------
# Table key = scenario name (used in experiment names / tags).
[scenarios.fat-mcp]
description = "Full MCP orchestrator tool"
system_prompt = "..."                 # optional; appended to the base system prompt
skills = ["./skills/apm"]             # optional list of skill directories
allowed_builtin_tools = ["Read", "Grep", "Glob"]  # optional; Claude built-ins to allow
max_turns = 64                        # optional per-scenario override
effort = "medium"                     # optional per-scenario override

# MCP servers visible in this scenario. Table key = server name.
[scenarios.fat-mcp.mcp_servers.apm]
# HTTP transport (supports distributed-trace header injection):
url = "http://localhost:8000/mcp"
headers = { source = "evals" }        # optional static headers
bearer_token_env_var = "APM_TOKEN"    # optional; Authorization: Bearer <env value>
tool_names = ["search_apm_libraries"] # tools allow-listed for this scenario
# Optional managed auto-start (started only if health check fails):
start_command = "python -m my_mcp_server"
start_env = { MCP_MODE = "both" }
health_url = "http://localhost:8000/health"  # default: <scheme>://<host>/health

# stdio transport (alternative to url; launched directly by the agent SDK):
[scenarios.local.mcp_servers.tools]
command = "python"
args = ["-m", "my_stdio_server"]
env = { FOO = "bar" }
tool_names = ["do_thing"]

# --- Tasks: prompts + evaluation criteria --------------------------------
[[tasks]]
id = "ssi_overview"
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
- A scenario MUST define at least one MCP server is NOT required — scenarios with
  no MCP servers are valid (model answers from skills/builtins only).
- Each MCP server MUST define either `url` or `command` (never both).
- `tool_names` is the allow-list of MCP tools exposed; empty/omitted means "all
  tools the server advertises".

### 2.2 `gateway.toml` (optional, separate file)

Maps a provider name to gateway settings. Passed via `--gateway-config PATH`.
When omitted (or `--no-gateway`), the standard provider APIs and env-var API keys
are used (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).

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
    experiment.py        ExperimentConfig, ScenarioConfig, McpServerConfig, TaskConfig (+ load_experiment)
    gateway.py           GatewayConfig, ProviderGatewayConfig (+ load_gateway)

  gateway.py             credential-helper exec + per-provider header/base-url/key resolution (runtime)
  tracing.py             current_trace_headers() via LLMObs.inject_distributed_headers
  observability.py       enable_llmobs(project, *, agentless, integrations_enabled)

  mcp.py                 McpServerSpec (from McpServerConfig); HTTP/stdio rendering for both SDKs;
                         merged_headers (static + bearer + trace); managed-server start/health;
                         tools/list metadata catalog
  skills.py              cross-provider skill staging (Claude + Codex)

  harness/
    __init__.py          create_runner(model, *, scenario, gateway) -> AgentRunner
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

`scenario.skills` is a list of skill *directories* (each an Agent-Skills SKILL.md
package). The harness stages them into each engine's discoverable location:

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
dd-ai-devx-evals --config experiment.toml [options]

  --config PATH            experiment TOML (required)
  --gateway-config PATH    gateway TOML (optional)
  --no-gateway             ignore gateway config; use provider default APIs
  --model M                run only these models (repeatable / comma-separated)
  --scenario S             run only these scenarios (repeatable / comma-separated)
  --task T                 run only these task ids (repeatable / comma-separated)
  --runs N                 override runs per cell
  --judge-model M          override judge model
  --jobs N                 concurrent tasks within one experiment (default 1)
  --parallel-experiments N concurrent experiments (default 1 = sequential)
  --dry-run                print the matrix and exit
  --no-progress            disable the live progress display
  --agentless / --no-agentless   LLMObs submission mode (default agentless)
```

Exit non-zero on config errors or when `--raise-errors` and a cell fails.

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
