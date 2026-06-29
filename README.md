# dd-ai-devx-evals

> Config-driven evaluation harness for agentic LLM runs, reporting to Datadog
> LLM Observability Experiments.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

`dd-ai-devx-evals` runs an evaluation matrix of **model × scenario × task**
through provider-native agentic SDKs, exposes MCP servers and Agent Skills to
the model, and reports every run to
[Datadog LLM Observability Experiments](https://docs.datadoghq.com/llm_observability/).

---

## What it does

Given two TOML files the harness:

1. **Builds the matrix** — every combination of `model × scenario × task`,
   optionally filtered by `--model`, `--scenario`, and `--task` flags.
2. **Runs each cell** with a provider-native agentic SDK: Anthropic models use
   [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/) (Claude Code);
   OpenAI models use [`openai-codex`](https://pypi.org/project/openai-codex/).
   MCP servers and skills are passed natively — no custom tool-calling loop.
3. **Scores each run** with a per-criterion LLM-as-judge
   (`RubricEvaluator`). Each criterion is an independent judge call; the final
   score is the mean across criteria.
4. **Reports to LLMObs** — one `async_experiment` per cell, with the run's
   token usage, tool calls, judge scores, and gateway metadata.

### Provider-native execution

| Provider    | Engine                      | LLMObs spans                        |
|-------------|-----------------------------|-------------------------------------|
| `anthropic` | `claude-agent-sdk`          | Native integration (ddtrace owns spans) |
| `openai`    | `openai-codex`              | Decorator spans (`@agent` / `@llm`) |

### Distributed tracing for complete token accounting

Trace-context headers are injected into every HTTP MCP request, so an
LLMObs-instrumented server's spans link back to the experiment and the tokens it
consumes — including those used by sub-agents inside the server — are rolled into
the experiment's `token_count`. See
[How distributed tracing works](#how-distributed-tracing-works) for details.

---

## Installation

```bash
# From source (development)
uv pip install -e .

# From PyPI (once published)
# uv pip install dd-ai-devx-evals
```

### External SDK requirements

The provider execution engines are not bundled as transitive dependencies on all
platforms. Install them explicitly:

```bash
uv pip install "claude-agent-sdk==0.2.82"   # Anthropic / Claude Code
uv pip install "openai-codex>=0.1.0b2"       # OpenAI Codex
```

### Required environment variables

| Variable        | Purpose                                        | Required when          |
|-----------------|------------------------------------------------|------------------------|
| `DD_API_KEY`    | Datadog API key for LLMObs                     | Always                 |
| `DD_APP_KEY`    | Datadog App key for LLMObs                     | Always                 |
| `DD_SITE`       | Datadog site (e.g. `datadoghq.com`)            | Always                 |
| `ANTHROPIC_API_KEY` | Anthropic API key                          | Without a gateway      |
| `OPENAI_API_KEY`    | OpenAI API key                             | Without a gateway      |

---

## Quickstart

### 1. Write your experiment config

```toml
# experiment.toml
project = "my-evals"
models   = ["anthropic/claude-sonnet-4-6", "openai/gpt-4.1"]
judge_model = "anthropic/claude-sonnet-4-6"
runs = 1

[defaults]
max_turns = 64
effort    = "medium"

[scenarios.fat-mcp]
description = "Full MCP orchestrator"
skills      = ["./skills/apm"]

[scenarios.fat-mcp.mcp_servers.apm]
url          = "http://localhost:8000/mcp"
headers      = { source = "evals" }
tool_names   = ["search_apm_libraries"]

[[tasks]]
id       = "ssi_overview"
prompt   = "What is Single Step Instrumentation?"
criteria = [
  "Correctly defines SSI",
  "Mentions supported languages",
]
```

### 2. Optionally write a gateway config

```toml
# gateway.toml
[providers.anthropic]
base_url           = "https://ai-gateway.example.com"
credentials_helper = "mytool auth token --datacenter us1"
headers            = { source = "evals", provider = "anthropic" }

[providers.openai]
base_url           = "https://ai-gateway.example.com/v1"
credentials_helper = "mytool auth token --datacenter us1"
```

### 3. Run

```bash
dd-ai-devx-evals \
  --config experiment.toml \
  --gateway-config gateway.toml
```

Preview the matrix without running:

```bash
dd-ai-devx-evals --config experiment.toml --dry-run
```

See `examples/` for fully-worked configs and a sample skill directory.

---

## Configuration reference

### `experiment.toml`

See [`AGENTS.md §2.1`](./AGENTS.md) for the full contract.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `project` | string | yes | LLMObs `ml_app` / project name |
| `models` | list of strings | yes | Provider-qualified models (`"<provider>/<model>"`) |
| `scenarios` | table | yes | Named scenario blocks (see below) |
| `tasks` | array of tables | yes | Prompt + criteria definitions (see below) |
| `description` | string | no | Human description applied to the dataset |
| `judge_model` | string | no | Model for rubric scoring (default `anthropic/claude-sonnet-4-6`) |
| `runs` | int | no | Runs per cell (default `1`) |
| `dataset_name` | string | no | Stable LLMObs dataset name (default: `project`) |
| `defaults.max_turns` | int | no | Default outer agent loop cap (default `64`) |
| `defaults.effort` | string | no | Reasoning effort hint: `"low"` / `"medium"` / `"high"` (default `"medium"`) |

#### Scenario fields (`[scenarios.<name>]`)

| Key | Type | Description |
|-----|------|-------------|
| `description` | string | Human description |
| `system_prompt` | string | Appended to the base system prompt |
| `skills` | list of strings | Paths to Agent Skill directories |
| `allowed_builtin_tools` | list of strings | Built-in tools to allow (e.g. `"Read"`, `"Grep"`) |
| `max_turns` | int | Per-scenario override for `defaults.max_turns` |
| `effort` | string | Per-scenario override for `defaults.effort` |
| `mcp_servers.<name>` | table | MCP server config (see below) |

#### MCP server fields (`[scenarios.<name>.mcp_servers.<name>]`)

Server blocks use the same field model as a standard `.mcp.json` file. Two
transports are supported:

- **stdio** — the agent SDK launches a local process and speaks MCP over its
  standard streams.
- **http** — the agent connects to an MCP endpoint over HTTP. This is the only
  transport that participates in distributed tracing: when the server is
  LLMObs-instrumented, the trace headers injected into its requests let its
  agent and sub-agent spans link back to the experiment (see
  [How distributed tracing works](#how-distributed-tracing-works)).

An http server may additionally carry `command`/`args`/`env`, which the harness
uses to auto-start the server when it is unreachable; `url` must then resolve to
localhost. Reachability is probed through the MCP protocol itself with a
`tools/list` call.

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | `"stdio"` or `"http"`; inferred from `url`/`command` when omitted |
| `command` | string | Stdio executable, or (http) command to auto-start the server |
| `args` | list of strings | Arguments for `command` |
| `env` | table | Extra environment variables for `command` |
| `url` | string | HTTP(S) MCP endpoint (enables trace-header injection) |
| `headers` | table | Static HTTP headers (http transport only) |
| `tool_names` | list of strings | Allow-list of MCP tools; empty = all tools |

#### Task fields (`[[tasks]]`)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | string | yes | Stable task identifier (used in experiment names and datasets) |
| `prompt` | string | yes | User prompt sent to the model |
| `criteria` | list of strings | yes | Rubric criteria (one judge call each; must be non-empty) |
| `description` | string | no | Human-readable task description |
| `context` | string | no | Extra context appended to the prompt |
| `latency_threshold_ms` | int | no | Latency threshold (reported in metadata only) |

---

### `gateway.toml`

See [`AGENTS.md §2.2`](./AGENTS.md) for the full contract.
Pass via `--gateway-config PATH`; omit or use `--no-gateway` for direct provider APIs.

| Key | Type | Description |
|-----|------|-------------|
| `providers.<name>.base_url` | string | Gateway base URL for this provider |
| `providers.<name>.credentials_helper` | string | Shell command yielding a bearer token |
| `providers.<name>.api_key_env` | string | Env var holding a static API key (alternative to helper) |
| `providers.<name>.headers` | table | Static extra headers for every request to this provider |

Credential resolution priority (per provider):

1. `credentials_helper` → runs the command once, caches for 30 minutes, prefers JWT-shaped output.
2. `api_key_env` → reads the named environment variable.
3. Falls back to the provider SDK default (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

---

## Skills

Scenarios can expose **Agent Skills** (SKILL.md packages) to both execution
engines. List skill directories under `scenario.skills` — the harness stages
each one into the engine's discoverable location within the per-run temporary
working directory before the agent starts.

For **Claude** (`claude-agent-sdk`), skills are copied to
`<cwd>/.claude/skills/<name>` and allow-listed in `ClaudeAgentOptions`.
For **Codex** (`openai-codex`), skills are copied to `<cwd>/.codex/skills/<name>`
and approved via the thread config. Both engines share the same SKILL.md package
format, so a single configured directory works for whichever engine the model
uses.

See `examples/skills/example-skill/SKILL.md` for a minimal skill template.

---

## How distributed tracing works

Before each run the harness calls `LLMObs.inject_distributed_headers()` and
merges the resulting trace-context headers into the static HTTP headers sent to
every HTTP-transport MCP server (`McpServerSpec.merged_headers`). When the MCP
server is itself LLMObs-instrumented, its spans (including any sub-agent or LLM
spans inside the server) link to the experiment span, making the experiment's
`token_count` a true full-stack count. Stdio-transport servers cannot receive
per-request headers; for those servers, only the tokens reported directly by the
provider SDK are counted.

---

## CLI reference

```
dd-ai-devx-evals --config experiment.toml [options]

  --config PATH              experiment TOML file (required)
  --gateway-config PATH      gateway TOML file (optional)
  --no-gateway               ignore gateway config; use provider default APIs
  --model M                  run only these models (repeatable / comma-separated)
  --scenario S               run only these scenarios (repeatable / comma-separated)
  --task T                   run only these task ids (repeatable / comma-separated)
  --runs N                   override runs per cell
  --judge-model M            override judge model
  --jobs N                   concurrent tasks within one experiment (default 1)
  --parallel-experiments N   concurrent experiments (default 1 = sequential)
  --dry-run                  print the matrix and exit without running
  --no-progress              disable the live progress display
  --agentless / --no-agentless  LLMObs submission mode (default: agentless)
  --raise-errors             stop on the first task/evaluator error
```

Exit codes: `0` success, `1` runtime error, `2` configuration error.

---

## Development

```bash
# Install dependencies (including dev extras)
uv sync

# Run tests (fully offline — no network, no real provider calls)
uv run pytest

# Lint and format
uv run ruff check . --fix
uv run ruff format .
```

This project uses **[jj (Jujutsu)](https://github.com/jj-vcs/jj)** for version
control. Never run vanilla `git` mutating commands.

For the full architectural contract, module responsibilities, and invariants,
read [`AGENTS.md`](./AGENTS.md).

---

## Contributing

Pull requests welcome. Please:

- Keep all tests fully offline (mock SDK clients and LLMObs).
- Follow the conventional-commit subject format (`feat(scope):`, `fix:`,
  `test:`, `docs:`, `chore:`).
- Run `uv run ruff check . --fix && uv run ruff format .` before submitting.

---

## License

Licensed under the [Apache License 2.0](./LICENSE).
