# airedale

> Config-driven evaluation harness for agentic LLM runs, reporting to Datadog
> LLM Observability Experiments.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

`airedale` runs an evaluation matrix of **model × scenario × task**
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

When an MCP server is itself LLMObs-instrumented, its spans link back to the
experiment so the tokens it consumes are rolled into the experiment's total. See
[How distributed tracing works](#how-distributed-tracing-works) for details.

---

## Installation

```bash
# From source (development)
uv pip install -e .

# From PyPI (once published)
# uv pip install airedale
```

The provider execution engines (`claude-agent-sdk`, `openai-codex`) are regular
dependencies and are installed automatically.

### Environment variables

| Variable        | Purpose                                        | Required when          |
|-----------------|------------------------------------------------|------------------------|
| `DD_API_KEY`    | Datadog API key for LLMObs                      | Always                 |
| `DD_APP_KEY`    | Datadog App key (datasets & experiments API)   | Always                 |
| `DD_SITE`       | Datadog site (default `datadoghq.com`)          | Only for a non-default site |
| `ANTHROPIC_API_KEY` | Anthropic API key                          | Anthropic models, without a gateway |
| `OPENAI_API_KEY`    | OpenAI API key                             | OpenAI models, without a gateway    |

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

# Shared skills and MCP servers are defined once and referenced by name.
[skills]
apm = "./skills/apm"

[mcp_servers.apm]
url          = "http://localhost:8000/mcp"
headers      = { source = "evals" }
tool_names   = ["search_apm_libraries"]

[scenarios.fat-mcp]
description = "Full MCP orchestrator"
skills      = ["apm"]   # reference by name
mcp_servers = ["apm"]   # reference by name

[tasks.ssi_overview]
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
airedale experiment.toml \
  --gateway-config gateway.toml
```

Preview the matrix without running:

```bash
airedale experiment.toml --dry-run
```

See `examples/` for fully-worked configs and a sample skill directory.

---

## Configuration reference

### `experiment.toml`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `project` | string | yes | LLMObs `ml_app` / project name |
| `models` | list of strings | yes | Provider-qualified models (`"<provider>/<model>"`) |
| `scenarios` | table | yes | Named scenario blocks (see below) |
| `tasks` | table | yes | Prompt + criteria definitions, keyed by task id (see below) |
| `description` | string | no | Human description applied to the dataset |
| `judge_model` | string | no | Model for rubric scoring (default `anthropic/claude-sonnet-4-6`) |
| `runs` | int | no | Runs per cell (default `1`) |
| `dataset_name` | string | no | Stable LLMObs dataset name (default: `project`) |
| `skills` | table | no | Skill registry: each `<name>` maps to an Agent Skill directory path (referenced by name from scenarios) |
| `mcp_servers` | table | no | MCP server registry: each `<name>` maps to a server block (referenced by name from scenarios; see below) |
| `defaults` | table | no | Defaults applied to every scenario (see below) |

#### Shared registries (`skills` / `mcp_servers`)

Skills and MCP servers are defined **once** at the top level and referenced
**by name** from scenarios (and from `[defaults]`), so they can be reused across
scenarios without copy-paste. Scenarios reference registry entries by name only:
they cannot define an MCP server inline or give a raw skill path, and naming an
entry absent from the registry is a configuration error.

#### Defaults (`[defaults]`)

Defaults are applied to every scenario that does not set the same field. When a
scenario sets a field it overrides the default **entirely** — values are never
merged (e.g. a scenario `skills` list replaces, rather than extends, the default
list).

| Key | Type | Description |
|-----|------|-------------|
| `max_turns` | int | Default outer agent loop cap (default `64`) |
| `effort` | string | Reasoning effort hint: `"low"` / `"medium"` / `"high"` (default `"medium"`) |
| `system_prompt` | string | Default appended system prompt |
| `skills` | list of strings | Default skill names (from the `skills` registry) |
| `allowed_builtin_tools` | list of strings | Default built-in tool allow-list |
| `mcp_servers` | list of strings | Default MCP server names (from the `mcp_servers` registry) |

#### Scenario fields (`[scenarios.<name>]`)

| Key | Type | Description |
|-----|------|-------------|
| `description` | string | Human description |
| `system_prompt` | string | Optional; appended to the base system prompt |
| `skills` | list of strings | Skill names referencing the `skills` registry |
| `allowed_builtin_tools` | list of strings | Built-in tool allow-list. **Omitted = all built-in tools allowed**; an explicit empty list `[]` = no built-in tools; a list = exactly those (e.g. `["Read", "Grep"]`) |
| `max_turns` | int | Per-scenario override for `defaults.max_turns` |
| `effort` | string | Per-scenario override for `defaults.effort` |
| `mcp_servers` | list of strings | MCP server names referencing the `mcp_servers` registry |
| `workdir` | table | Optional working-directory config (see [Working directories](#working-directories)) |

#### MCP server fields (`[mcp_servers.<name>]`)

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

#### Task fields (`[tasks.<id>]`)

Each task is its own table keyed by a stable task **id** (`[tasks.ssi_overview]`).
The id is used in experiment names and datasets, and must be unique — TOML
forbids duplicate keys, so a repeated id is a parse error.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `prompt` | string | yes | User prompt sent to the model |
| `criteria` | list of strings | yes | Rubric criteria (one judge call each; must be non-empty) |
| `description` | string | no | Human-readable task description |
| `context` | string | no | Extra context appended to the prompt |
| `latency_threshold_ms` | int | no | Latency threshold (reported in metadata only) |

---

### Working directories

By default every `(model, scenario, task)` repetition runs in a **fresh empty
temporary directory** — a hermetic sandbox. A scenario can instead configure a
`workdir` so the agent runs inside a checkout of a git repository (optionally at
a ref), with files staged in. Each run/repetition gets its **own fresh
workspace**, so file mutations never leak between repetitions.

```toml
[scenarios.regression.workdir]
repo = "self"          # "self" (repo containing experiment.toml) | a git URL | a local path
ref  = "v2.3.0"        # optional; default = the source repo's current HEAD

# Ordered setup steps applied inside each fresh worktree (all optional):
[[scenarios.regression.workdir.steps]]
op    = "restore"      # git restore --source=<from> -- <paths...> (requires a repo)
from  = "v2.2.0"
paths = ["src/api/**", "README.md"]

[[scenarios.regression.workdir.steps]]
op    = "remove"       # delete matching globs from the worktree
paths = ["secrets/**"]

[[scenarios.regression.workdir.steps]]
op      = "write"      # create/overwrite a file; exactly one of content/source
path    = "NOTES.md"
content = "Evaluate the migration."

[[scenarios.regression.workdir.steps]]
op     = "write"
path   = "fixtures/input.json"
source = "./fixtures/input.json"   # resolved relative to experiment.toml's dir
```

| Key | Type | Description |
|-----|------|-------------|
| `repo` | string | `"self"`, a git URL, or a local path. Omitted = start from an empty directory |
| `ref` | string | Branch/tag/commit to check out; default = source repo's current HEAD |
| `steps` | array of tables | Ordered setup steps, each discriminated by `op` (`restore` / `remove` / `write`) |

Notes:

- **`repo = "self"`** refers to the git repository that contains
  `experiment.toml` (validated at load — an error if the config is not inside a
  git repo). It is cloned (`--no-hardlinks`) into an isolated cache, so the eval
  **never mutates your real checkout**.
- A repo source is **cloned once**, lazily; each workspace is a `git worktree` of
  that clone. The whole cache is deleted at the end of the run.
- `restore` requires a `repo`; `remove`/`write` work with or without one. A
  `write` `path` must stay inside the workspace (no `..`/absolute paths).
- **The repo's own project config is honored.** If a cloned repo ships
  `.claude/`, `.codex/`, `AGENTS.md`/`CLAUDE.md`, project subagents/skills, or a
  `.mcp.json`, those are discovered automatically. Claude's `.mcp.json` servers
  are merged in (and get distributed-trace headers); a scenario-configured MCP
  server **wins** on a name collision. Codex does not read `.mcp.json`.
- **Codex hermeticity.** Codex reads MCP servers (and other global config) from
  `$CODEX_HOME/config.toml` (global, default `~/.codex`). The harness **always**
  isolates `CODEX_HOME` to a fresh per-run directory so the operator's global
  Codex config never leaks into a run. Authentication still works: env auth (a
  gateway token or `OPENAI_API_KEY`) is used directly, and when relying on
  `codex login` the harness copies only `auth.json` into the isolated home — so
  login auth keeps working without pulling in the global `config.toml`. A
  repo-committed `.codex/config.toml` is not read by Codex for MCP servers.

---

### `gateway.toml`

Pass via `--gateway-config PATH`; omit it to use the standard provider APIs and
env-var keys (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`).

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
engines. Define each skill directory once in the top-level `skills` registry
(`<name> = "<path>"`) and reference it by name from `scenario.skills` — the
harness stages each referenced skill into the engine's discoverable location
within the per-run temporary working directory before the agent starts.

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
airedale CONFIG [options]

  CONFIG                     experiment TOML file (required positional argument)
  --gateway-config PATH      gateway TOML file; omit to use provider default APIs
  --model M                  run only these models (repeatable / comma-separated)
  --scenario S               run only these scenarios (repeatable / comma-separated)
  --task T                   run only these task ids (repeatable / comma-separated)
  --runs N                   override runs per cell
  --judge-model M            override judge model
  --jobs N                   total cells run concurrently across the matrix (default 1 = sequential)
  --dry-run                  print the matrix and exit without running
  --no-progress              disable the live progress display
  --[no-]agentless           LLMObs submission mode (default: agentless)
  --fail-fast                stop on the first task/evaluator error
```

Each cell runs its `runs` repetitions sequentially, so `--jobs` is the total
number of in-flight agent runs at any time.

Exit codes: `0` success, `1` runtime error, `2` configuration error.

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the
development environment, dependency policy, testing requirements, coding style,
and pull-request conventions, and [`AGENTS.md`](./AGENTS.md) for the
architectural contract.

---

## License

Licensed under the [Apache License 2.0](./LICENSE).
