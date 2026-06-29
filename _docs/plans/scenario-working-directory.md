# Plan: configurable scenario working directories

Status: **proposed** — awaiting review before implementation.

## 1. Goal

Today every `(model, scenario, task)` cell runs in a throwaway empty temp dir.
`experiment.py._run_cell` creates a `tempfile.TemporaryDirectory` and passes its
path as `cwd` into `create_runner`, so exactly one temp dir is created per cell.
`harness/base.py` *also* has a `cwd=None` fallback that creates its own temp dir
and cleans it up in `__del__`; it never fires in the production path (the caller
always passes a `cwd`) and this plan removes it (decision #5, §6.3) so the caller
is the single owner of the working directory. We want scenarios to be able to
**configure the working directory** the agent runs in:

1. **Clone a git repository** (optionally at a ref). The repo is cloned **once**
   into a cache the first time it is needed; each workspace is then a fresh
   `git worktree` from that clone. There is a first-class way to refer to *the
   git repository that contains `experiment.toml`* (the dominant use case).
2. **Restore** one or more files/patterns from another ref.
3. **Remove** files/patterns.
4. **Create/update** files with specified content.

Because a cloned repo can carry its own agent configuration (`.claude/`,
`.codex/`, `AGENTS.md`, `CLAUDE.md`, project subagents, project skills,
`.mcp.json`), the harness must account for those project-level files, which in
turn affects how scenario-configured `skills` and `mcp_servers` combine with the
repo's own configuration.

## 2. Decisions locked (from review Q&A)

1. **Project files: always honor whatever is in `cwd` (no `project_mode`
   flag).** Project-scoped discovery is always enabled and self-adjusts to the
   working directory's actual contents (Claude `setting_sources` is always
   `["project"]`; repo MCP servers come from parsing `<cwd>/.mcp.json` while
   `strict_mcp_config` stays `True` — see decision #2; skills allow-list =
   whatever is staged/present under `<cwd>/.claude/skills`; Codex always discovers
   repo `.codex/` + `AGENTS.md`). A bare temp dir has no
   project files, so it stays hermetic — today's sandbox behavior is preserved
   without gating on "is a workdir configured." (Refines the original
   workdir-gated "project mode" framing; see §6.)
2. **MCP servers: merge repo + scenario, harness-specifically.** Project-level
   MCP config differs per engine, so discovery is per-harness. **Claude:** keep
   `strict_mcp_config=True` (no user-global/enterprise leakage) and parse
   `<cwd>/.mcp.json` ourselves into `McpServerSpec`s; repo HTTP servers then also
   get distributed-trace headers (no longer a blind spot). **Codex:** MCP comes
   from `$CODEX_HOME/config.toml` (global), not `.mcp.json`, so we do **not**
   cross-feed `.mcp.json` to Codex; Codex-side repo discovery is empty pending
   verification of any repo-committed Codex MCP file (§6.0/§12). In both cases
   discovered servers are **not** auto-started/health-managed by us (discovered
   too late for the matrix-start pass), and name collisions resolve in favor of
   the scenario-configured server.
3. **Workspace mechanism: cached clone + `git worktree`.** Each distinct repo
   source is cloned **once, lazily** (first use) into a cache. Each workspace is a
   `git worktree add --detach`. Clone, worktree-add, and worktree-remove are
   guarded by a **per-repo lock** so concurrent `--jobs` cells are safe.
4. **Workspace granularity: per run/repetition.** Each of a cell's `runs`
   repetitions gets its own fresh workspace and re-applies the setup steps. Full
   isolation: agent file mutations from one repetition never leak into the next.
   (This also unifies the empty-temp-dir sandbox case to per-run, a minor change
   from today's per-cell temp dir.)
5. **`cwd` is required on runners.** The `cwd=None` fallback in
   `harness/base.py` (which creates an internal `tempfile.TemporaryDirectory` and
   cleans it up in `__del__`) is removed. Callers — in production the
   `WorkspaceManager`, in tests a `tmp_path` — own the working directory and its
   lifecycle. This makes cwd ownership single-source and cleanup deterministic
   (no GC-timed `__del__`), and avoids two competing temp-dir owners once
   `WorkspaceManager` (§7) is the source of truth.

## 3. Configuration schema

`workdir` is a scenario block. It is **defaultable** via `[defaults]` with the
established *override-entirely* semantics (a scenario that sets `workdir`
replaces the default `workdir` wholesale — no merge).

```toml
[defaults]
max_turns = 64
effort = "medium"
# Optional shared default; scenarios that set their own [scenarios.X.workdir]
# replace this entirely.
[defaults.workdir]
repo = "self"          # the git repo containing this experiment.toml

# --- A scenario that uses the repo as-is at its current HEAD ---------------
[scenarios.repo-baseline.workdir]
repo = "self"          # "self" | a git URL | a local path
ref  = "main"          # optional; default = source repo's current HEAD

# --- A scenario that stages a specific file layout -------------------------
[scenarios.regression.workdir]
repo = "self"
ref  = "v2.3.0"

# Ordered list of setup steps applied inside each fresh workspace.
# Each step is discriminated by an explicit `op` key.
[[scenarios.regression.workdir.steps]]
op    = "restore"                 # git restore --source=<from> -- <paths...>
from  = "v2.2.0"                  # required: source ref
paths = ["src/api/**", "README.md"]

[[scenarios.regression.workdir.steps]]
op    = "remove"                  # delete matching paths from the worktree
paths = ["secrets/**", "config/local.yaml"]

[[scenarios.regression.workdir.steps]]
op      = "write"                 # create/overwrite a file
path    = "NOTES.md"
content = "Evaluate the migration in this branch."

[[scenarios.regression.workdir.steps]]
op     = "write"                  # alternatively, copy from an external file
path   = "fixtures/input.json"
source = "./fixtures/input.json"  # resolved relative to experiment.toml's dir

# --- A scenario with no repo: empty dir + a couple of files ----------------
[scenarios.scratch.workdir]
# no repo -> start from an empty dir (today's behavior, but now you can seed it)
[[scenarios.scratch.workdir.steps]]
op = "write"
path = "main.py"
content = "print('hello')\n"
```

### Schema rules

- `repo` (optional string): the git source.
  - `"self"` → the git toplevel that contains `experiment.toml` (resolved via
    `git -C <config_dir> rev-parse --show-toplevel`). Error if the config file is
    not inside a git repo.
  - any other value → a git URL or a local filesystem path. Local paths are
    resolved relative to the `experiment.toml` directory.
  - omitted → no clone; the workspace starts as an **empty** directory. `restore`
    steps are then a `ConfigError` (nothing to restore from); `remove`/`write`
    are still valid.
- `ref` (optional string): branch / tag / commit to check out as the workspace
  base. Default: the source repo's current HEAD (for `"self"`, the working
  repo's current commit at clone time; for URLs, the remote default branch).
- `steps` (optional array of tables): ordered operations, each with `op`:
  - `op = "restore"`: `from` (required ref string) + `paths` (required non-empty
    list of pathspecs). Implemented as `git restore --source=<from> --
    <paths...>`. Requires a `repo` (else `ConfigError`).
  - `op = "remove"`: `paths` (required non-empty list of glob patterns). Deletes
    matching files/dirs from the worktree (filesystem delete; works with or
    without a repo).
  - `op = "write"`: `path` (required, relative to the workspace root) + exactly
    one of `content` (inline string) or `source` (file path resolved relative to
    the config dir, copied in). Parent dirs are created. `path` must stay inside
    the workspace (reject `..` escapes / absolute paths).
  - any other `op` value → `ConfigError`.

The clone is a **full** clone (no `--depth`) so that `restore` from arbitrary
refs has the objects available.

## 4. Config parsing changes (`config/experiment.py`)

New value objects (frozen dataclasses):

```python
@dataclass(frozen=True)
class RestoreStep:
    from_ref: str
    paths: tuple[str, ...]

@dataclass(frozen=True)
class RemoveStep:
    paths: tuple[str, ...]

@dataclass(frozen=True)
class WriteStep:
    path: str
    content: str | None        # exactly one of content / source_path
    source_path: str | None    # already resolved to an absolute path at load time

WorkdirStep = RestoreStep | RemoveStep | WriteStep

@dataclass(frozen=True)
class WorkdirConfig:
    repo: str | None = None          # "self" | URL | local path (raw, unresolved)
    ref: str | None = None
    steps: tuple[WorkdirStep, ...] = ()
```

- Add `workdir: WorkdirConfig | None = None` to `ScenarioConfig`.
- Add `workdir` to `_SCENARIO_KEYS` and `_DEFAULTS_KEYS`.
- `_parse_scenario`: resolve `workdir` with the same "scenario value wins
  entirely, else defaults" rule used for `skills`/`mcp_servers`.
- New `_parse_workdir(name, raw, *, config_dir)` validates `repo`/`ref`/`steps`,
  rejects unknown keys, resolves `write.source` and local `repo` paths relative
  to `config_dir`, and enforces the per-op rules above.
- `ExperimentConfig` gains `config_path: Path` (and a `config_dir` property).
  `load_experiment(path)` records the resolved path so `repo="self"`, local repo
  paths, and `write.source` can be resolved. `config_dir` is threaded into
  `_parse_scenario` → `_parse_workdir`.
- **General path-resolution rule:** any *filesystem path* in the config
  (`[skills]` registry entries, workdir local `repo` paths, `write.source`) is
  resolved relative to the directory containing `experiment.toml` at load time,
  producing absolute paths downstream. Command-style values (MCP `command`/
  `args`, gateway `credentials_helper`) are **not** paths and are left untouched.

### 4.1 Bug fix — skill paths resolved relative to `experiment.toml`

Today the `[skills]` registry stores the raw string (`config/experiment.py`
lines ~369-372) and `skills.py` later calls `Path(skill_dir).resolve()`, which
resolves against the **process CWD**. A relative skill path such as
`apm = "./skills/apm"` therefore only works when the tool is run from the
config's own directory — a latent bug. As part of this work (and landing first,
since it introduces the `config_path`/`config_dir` plumbing the workdir feature
reuses):

- `load_experiment` resolves each `[skills]` registry path against `config_dir`
  to an absolute path before storing it in the registry.
- `skills.py` then receives absolute paths; its `Path(...).resolve()` becomes a
  no-op for these (still fine for any absolute path passed directly).
- Tests: cover that a relative `[skills]` path resolves against the config dir
  regardless of process CWD (run the loader from a different CWD).

## 5. Runtime: workspace manager (`workdir.py`, new module)

A `WorkspaceManager` owns the clone cache and per-repo locks for the whole
matrix run. It is itself an **async context manager**: entering it allocates the
cache root, exiting it tears everything down (worktrees first, then clones). It
is created in `run_experiments` and wraps the whole matrix.

```python
class WorkspaceManager:
    def __init__(self, *, config_dir: Path) -> None: ...

    async def __aenter__(self) -> "WorkspaceManager":
        # allocate self._cache_root = TemporaryDirectory(prefix="dd-ai-devx-clones-")
        ...
    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        # Remove every worktree we created, then delete the entire cache root
        # (all clones). Idempotent. See "Cache lifecycle & safety" below: this
        # ONLY ever deletes paths under self._cache_root, never any source repo.
        ...

    # Lazily clone the repo for a workdir config (once per resolved source),
    # then create a fresh worktree, apply the steps, and yield the path.
    @contextlib.asynccontextmanager
    async def workspace(self, workdir: WorkdirConfig | None) -> AsyncIterator[Path]:
        # workdir is None or has no repo -> yield a fresh empty TemporaryDirectory,
        #   still applying any remove/write steps.
        # else:
        #   key = resolved source path/URL
        #   async with self._lock_for(key):
        #       clone_path = self._ensure_clone(key, source)   # git clone once,
        #                                                       # always under cache_root
        #       worktree = git worktree add --detach <cache_root>/wt-<uuid> <ref or HEAD>
        #   try:
        #       self._apply_steps(worktree, workdir.steps)
        #       yield worktree
        #   finally:
        #       async with self._lock_for(key):
        #           git worktree remove --force <worktree>
```

Details:

- **Clone cache**: `dict[str, ClonedRepo]` keyed by resolved source
  (abs local path or URL). `ClonedRepo` holds the clone path and an
  `asyncio.Lock`. First access clones, subsequent accesses reuse. The clone path
  is **always a freshly created directory under `self._cache_root`** (e.g.
  `<cache_root>/clone-<slug>`) — the source path is **never** stored as the clone
  path, even for `repo="self"`. For local sources (including `self`) the clone
  uses `git clone --no-hardlinks <source> <cache_root>/clone-<slug>` so the cache
  shares no object storage with the source and is safe to delete wholesale.

- **Cache lifecycle & safety (`self` must never be disastrous)**:
  - `aclose()` deletes worktrees, then the entire `cache_root`, then clears the
    cache dict. Worktree removal uses `git worktree remove --force`; clone
    removal is a plain recursive delete of `cache_root`.
  - **Invariant:** every path the manager deletes is asserted to live **inside
    `cache_root`** (resolved, `Path.is_relative_to`). A delete targeting
    anything outside `cache_root` is a bug and raises instead of removing.
  - Because `repo="self"` is materialised as an independent `--no-hardlinks`
    clone under `cache_root`, tearing the cache down can never touch the user's
    real checkout. The source repo is only ever read from (during `git clone`)
    and is never a worktree parent nor a deletion target.
- **Per-repo lock**: guards clone creation + every `git worktree add` /
  `git worktree remove` for that clone (worktree metadata mutations race under
  concurrency). The actual agent run and step application happen **outside** the
  lock so cells still run in parallel.
- **`ref` resolution**: for `"self"` with no `ref`, snapshot the source repo's
  current HEAD SHA at clone time and check that out (`worktree add --detach
  <sha>`); otherwise use the configured `ref`, defaulting to the clone's default
  branch HEAD.
- **Step application** (`_apply_steps`):
  - `restore`: `git -C <worktree> restore --source=<from> -- <paths...>`.
  - `remove`: glob-expand each pattern within the worktree and delete (files and
    dirs); patterns matching nothing are a no-op (logged).
  - `write`: write `content` or copy `source_path`; create parent dirs; assert
    the resolved target stays within the worktree.
- All git invocations use `subprocess`/`asyncio.create_subprocess_exec`, never
  the user's `git` config surprises beyond defaults; capture stderr for clear
  errors. Per AGENTS.md/global rules this module uses plain `git` only as an
  *internal mechanism for creating eval workspaces* — it never touches the user's
  own checkout (it clones into an isolated cache and operates on worktrees of
  that clone). The project's "no git, use jj" rule governs how *we* version this
  repo, not the runtime behavior of the harness.

## 6. Harness changes — project-scoped discovery (always on)

There is **no `project_mode` flag**. Project-scoped discovery is always enabled;
each individual behavior keys off the *actual contents of `cwd`*, so a bare temp
dir stays hermetic while a cloned repo (or a `write` step that seeds project
files) is honored automatically. This removes the wart where today's code already
flips `setting_sources` based on whether skills happen to be configured.

### 6.0 Project-level MCP discovery is **harness-specific**

Each engine has its *own* project-level MCP convention, and honoring the file
that engine actually reads is what makes the eval reflect real behavior. So
discovery is **per-harness**, not a single shared `.mcp.json` parser fed to both.
`create_runner` runs a provider-specific discovery, then merges with the scenario
specs (scenario-configured server **wins** on name collision):

```python
mcp_servers = scenario_specs + [s for s in discovered if s.name not in scenario_names]
```

The workspace is fully staged before `create_runner` runs (steps execute inside
`workspace()` which then yields `cwd`), so any project files are present.

**Claude** — project MCP convention is `<cwd>/.mcp.json`
(`{"mcpServers": {<name>: {...}}}`, the field model we already parse for config).
Because we keep `strict_mcp_config=True` (hermetic; no user-global/enterprise
leakage), Claude will not read `.mcp.json` itself, so we parse it and inject the
servers as `McpServerSpec`s. Helper:
`discover_claude_project_mcp_servers(cwd) -> list[McpServerSpec]` (missing/empty
file → `[]`). Everything downstream falls out of existing logic because the
servers become ordinary `self.mcp_servers` entries:
  - `_claude_available_tools()` emits their `mcp__<server>` allow-list entries
    automatically (no separate `.mcp.json` allow-list parsing).
  - HTTP servers get **distributed-trace headers** via
    `merged_headers(trace_headers)` — no trace-accounting blind spot.
  - They appear in the LLMObs `mcp_servers` config/metadata (redacted).

**Codex** — Codex does **not** read `.mcp.json`. Its MCP servers come from
`$CODEX_HOME/config.toml` (`[mcp_servers.<name>]`, global), launched via
`codex app-server --config <kv>` with `cwd`; instructions come from `AGENTS.md`
in the cwd hierarchy. Therefore:
  - We do **not** feed `.mcp.json` into Codex — it is not Codex's convention and
    would misrepresent how Codex behaves in the repo.
  - The Codex-side discovery is `[]` **unless verification (below) finds a
    repo-committed file Codex treats as project-scoped MCP config.** Scenario
    servers continue to be rendered via `config_overrides` exactly as today.
  - **Hermeticity gap (pre-existing):** Codex reads the operator's global
    `~/.codex/config.toml`, so ambient MCP servers already leak into runs today
    (the Codex analog of the leak `strict_mcp_config` prevents for Claude).
    Closing it means setting `CODEX_HOME` to an isolated dir in `_codex_env()` —
    but `CODEX_HOME` also holds `auth.json`, so isolating it requires auth via
    `OPENAI_API_KEY`/gateway env (breaks reliance on `codex login`). Flagged as
    an open decision (§12), not mandated by this change.
  - **To verify during implementation:** whether the installed Codex honors any
    repo-committed config (e.g. a project `.codex/config.toml`) for MCP. If yes
    and we want to honor it, plug a `discover_codex_project_mcp_servers(cwd)` into
    the same merge seam (injecting via `config_overrides` with trace headers).

- **Not auto-started / not health-managed by us** (both engines): the
  matrix-start managed-server pass runs over config-level scenario servers before
  any workspace exists, so it never sees discovered servers. stdio servers are
  launched by the SDK; HTTP servers must already be reachable and **participate
  in Claude's readiness gate** (`_wait_for_claude_mcp_servers`) — an unreachable
  one fails the run loudly (consistent with the "fail loudly" ethos).

### 6.1 Claude (`harness/claude.py`)

- `setting_sources`: **always `["project"]`** (today: `["project"] if skills else
  []`). Project-scoped only — never `"user"` — so user-global `settings.json`
  never leaks. In a bare temp dir there are no project files, so this is a no-op;
  when the repo ships them it loads `.claude/settings.json`, project subagents,
  and `CLAUDE.md`.
- `strict_mcp_config`: **stays `True`, always** (unchanged from today). Repo
  `.mcp.json` servers are honored via the §6.0 Claude-specific discover-&-merge,
  not by relaxing strict mode, so no user-global/enterprise MCP config ever
  leaks. Repo MCP tool allow-listing is automatic because discovered servers are
  now in `self.mcp_servers` (see §6.0).
- **Skills merge**: `stage_skills_for_claude` copies scenario skills into
  `<cwd>/.claude/skills/<name>`; a cloned repo may already have `.claude/skills`.
  Staging must **not** clobber a repo skill of the same name: on collision, raise
  a clear error (names must be unique). The `ClaudeAgentOptions(skills=...)`
  allow-list is **always the set of skill names discovered in
  `<cwd>/.claude/skills` after staging** (scenario ∪ repo). In a bare temp dir
  that set equals today's scenario-only list.
- **Keep staged scenario skills out of the repo's tracked view**: append staged
  paths to `<cwd>/.git/info/exclude` when the workspace is a git worktree, so the
  agent doesn't perceive scenario skills as untracked repo changes. (Nice-to-have;
  low risk.)

> Net effect for existing sandbox scenarios (no workdir): `setting_sources` goes
> `[] → ["project"]` (no-op in an empty dir), the skills allow-list is derived
> from the staged dir instead of the passed list (same names), and MCP discovery
> finds no `.mcp.json` so the server set is unchanged. `strict_mcp_config` stays
> `True`. So behavior is unchanged in practice, but the code path is now uniform.

### 6.2 Codex (`harness/codex.py`)

- Codex runs with `cwd=<workspace>` and reads project **instructions** from
  `AGENTS.md` in the cwd hierarchy. Its **MCP servers**, however, are *not*
  project-scoped: they come from `$CODEX_HOME/config.toml` (global) plus the
  `--config` overrides we pass. Codex does **not** read `.mcp.json`. So
  honoring "the repo's MCP setup" for Codex is handled by the harness-specific
  rules in §6.0, **not** by cross-feeding `.mcp.json`.
- Scenario-configured servers are rendered into `config_overrides`
  (`to_codex_config_overrides`) exactly as today, with trace headers.
- **Hermeticity (open item, §12):** because Codex inherits the operator's global
  `~/.codex/config.toml`, ambient MCP servers leak into runs today. Closing this
  means isolating `CODEX_HOME` in `_codex_env()`, which trades against
  `codex login` auth (needs `OPENAI_API_KEY`/gateway). Not mandated here.
- Skills: `stage_skills_for_codex` stages into `<cwd>/.codex/skills`. Same
  no-clobber rule on name collisions with repo skills.
- Sandbox stays `Sandbox.read_only` (the agent reads the repo; eval shouldn't
  let it mutate the host outside the worktree). The pre-applied setup steps are
  what shape the tree, not the agent.

### 6.3 `harness/base.py` + `harness/__init__.py`

- **`AgentRunner.__init__`**: make `cwd` a required `str | Path` (drop the
  `cwd: ... | None = None` default). Remove the `self._temp_cwd` field, the
  `if cwd is None:` branch, the `__del__` cleanup, and the now-unused `tempfile`
  import. `__init__` simplifies to `self.cwd = str(cwd)`. Both `ClaudeRunner` and
  `CodexRunner` `__init__` signatures change `cwd` to required to match.
- **`create_runner`**: `cwd` becomes required; drop the "when `cwd` is `None`
  the runner creates a temp dir" line from the docstring. **No `project_mode`
  parameter** — project-scoped discovery is always on and self-adjusts to `cwd`
  contents (§6.1). `cwd` is supplied per-run by the caller (see §7).
- **Tests**: the 5 direct `ClaudeRunner(...)` constructions in
  `tests/test_harness_claude.py` gain a `cwd=tmp_path` argument (pytest fixture).

## 7. `experiment.py` restructure — per-run workspaces

Currently `_run_cell` creates one temp dir and one runner for the cell, then
`experiment.run(jobs=1)` invokes `experiment_task` `runs` times against that
single cwd. To get a **fresh workspace per repetition**:

- Move workspace creation **and runner construction** inside `experiment_task`,
  which LLMObs calls once per (record × run):

```python
async def experiment_task(input_data, cfg=None):
    async with workspace_manager.workspace(scenario.workdir) as cwd:
        runner = create_runner(
            model, scenario=scenario, gateway=gateway, cwd=str(cwd),
        )
        result = await runner.run(...)
        _annotate_experiment_usage(...)
        return result.to_output_data()
```

- The cell-level `tempfile.TemporaryDirectory` in `_run_cell` is removed; the
  workspace manager now owns cwd lifecycle (it yields an empty temp dir when no
  workdir is configured).
- `run_experiments` wraps the matrix in `async with WorkspaceManager(
  config_dir=config.config_dir) as workspace_manager:` and passes it down to
  `_run_cell`. The manager's `__aexit__`/`aclose()` removes all worktrees and
  deletes the clone cache; this runs alongside `progress.stop()`.
- Construction cost per run is negligible (gateway credential helper is already
  process-cached); worktree add/remove dominate and are bounded by the per-repo
  lock.

## 8. Concurrency & cleanup

- Cross-cell concurrency is unchanged (`--jobs` semaphore). Within the workspace
  manager, only clone creation and `git worktree add`/`remove` are serialized
  per repo via the per-repo `asyncio.Lock`; step application and the agent run
  happen outside the lock.
- Cleanup order at run end (`WorkspaceManager.aclose()`): remove every worktree,
  then recursively delete the clone `cache_root`, then `progress.stop()`. All
  deletions are asserted to live inside `cache_root` (see §5 "Cache lifecycle &
  safety"); `repo="self"` is a `--no-hardlinks` clone under `cache_root`, so
  teardown never touches the user's checkout.
- Failure handling: if a worktree/step setup fails, the cell errors like any
  other task failure; `--fail-fast` still applies. The per-workspace `finally`
  removes that partial worktree; `aclose()` then sweeps anything left over.

## 9. Edge cases & validation

- `repo="self"` but `experiment.toml` not in a git repo → `ConfigError` at load
  (resolved eagerly so dry-run also catches it).
- `restore` step with no `repo` → `ConfigError`.
- `write.path` escaping the workspace (absolute or `..`) → `ConfigError`.
- `write` with both/neither of `content`/`source` → `ConfigError`.
- Unknown `op` or unknown keys within a step → `ConfigError`.
- Skill name collision between scenario skills and repo `.claude/skills` /
  `.codex/skills` → runtime error with both sources named.
- Repo `.mcp.json` server name equal to a scenario MCP server name → scenario
  wins; log a warning.
- `--dry-run` prints the matrix (unchanged) and now also validates workdir config
  (since parsing happens at load).

## 10. Testing (`tests/`, offline)

- `test_config_experiment.py`: parse all step types; defaults inheritance &
  override-entirely for `workdir`; every validation error above; `repo="self"`
  resolution against a fake config dir; `write.source` path resolution.
- New `test_workdir.py`: against a small **local** throwaway git repo created in
  the test (allowed — it's the harness's own mechanism, fully offline):
  - clone-once caching (second workspace reuses the clone),
  - worktree creation at a ref,
  - `restore`/`remove`/`write` step effects on the worktree,
  - empty-dir path (no repo) still applies remove/write,
  - `aclose()` removes worktrees and deletes the clone cache,
  - **`self` safety:** after a full run + `aclose()` against a `repo="self"`
    workdir, the source repo dir still exists with its working tree and
    `.git` intact (assert the source is untouched and was cloned
    `--no-hardlinks`, i.e. modifying/deleting the clone left the source's
    objects intact),
  - the manager refuses to delete a path outside `cache_root` (the safety
    assertion raises),
  - per-repo lock serializes concurrent worktree adds (smoke test).
- `test_mcp.py`: `discover_claude_project_mcp_servers(cwd)` parses a
  `<cwd>/.mcp.json` (stdio + http entries) into `McpServerSpec`s; missing/empty
  file → `[]`. The merge keeps scenario servers on name collision.
- `test_harness_claude.py`: `setting_sources` is always `["project"]`;
  `strict_mcp_config` stays `True`; discovered `.mcp.json` servers are merged into
  `self.mcp_servers` with scenario servers winning on name collision; their
  `mcp__<server>` allow-list entries appear; an http discovered server gets trace
  headers via `merged_headers`; skills allow-list reflects names discovered under
  `<cwd>/.claude/skills` (scenario ∪ any repo skills).
- `test_harness_codex` (or `test_mcp.py`): Codex-side project MCP discovery does
  **not** read `<cwd>/.mcp.json` (a `.mcp.json` present in cwd does not add Codex
  servers); scenario servers still render into `config_overrides`.
- `test_skills.py`: name-collision error when a repo already has a skill of the
  same name.
- `test_experiment.py`: a fresh workspace is created per run (assert the workspace
  manager is entered `runs` times); runner constructed per run.

## 11. Docs to update

- `AGENTS.md` §2 (config contract): add the `workdir` block, `repo="self"`,
  step ops, project-mode semantics, and the merge/trace-header limitation.
- `AGENTS.md` §3 (module layout): add `workdir.py`.
- `README.md`: a "Working directories" section with the example config.
- `examples/experiment.toml`: add a commented `workdir` example scenario.

## 12. Open questions / future work (not in this change)

- Optional explicit `use_project_settings` / `mcp_from_project` overrides if some
  scenario wants a cloned repo *without* honoring its project files. The current
  design always honors `cwd` contents and has no opt-out; if hermetic evaluation
  of a repo's *code* (ignoring its `.claude`/`.codex` config) is ever needed,
  these toggles would be the way to add it.
- **Codex MCP hermeticity:** decide whether to isolate `CODEX_HOME` so the
  operator's global `~/.codex/config.toml` MCP servers don't leak into runs
  (Codex analog of Claude `strict_mcp_config`). Tradeoff: `CODEX_HOME` also holds
  `auth.json`, so isolation requires `OPENAI_API_KEY`/gateway auth.
- **Codex repo-scoped MCP:** verify whether the installed Codex honors any
  repo-committed file (e.g. project `.codex/config.toml`) for MCP servers; if so,
  add `discover_codex_project_mcp_servers(cwd)` into the §6.0 merge seam.
- Capturing the source repo's **uncommitted** changes (clone only sees committed
  state). Out of scope; documented.
- A top-level `[repos]` registry (analogous to `[skills]`/`[mcp_servers]`) if repo
  reuse across many scenarios becomes verbose.

## 13. Implementation order (commit breakdown)

1. `fix(config): resolve config paths relative to experiment.toml` — add
   `config_path`/`config_dir` to `ExperimentConfig`, resolve `[skills]` registry
   paths against the config dir (fixes the CWD bug), drop the CWD-relative
   assumption in `skills.py` + tests. (Standalone bug fix; foundation for the
   workdir path resolution.)
2. `feat(config): parse scenario workdir block` — `WorkdirConfig`/step
   dataclasses, `_parse_workdir`, `repo`/`write.source` path resolution (reusing
   `config_dir`), validation + tests.
3. `feat(workdir): clone-cache + worktree workspace manager` — `workdir.py`,
   step application, per-repo locks, cleanup + tests.
4. `refactor(harness): require cwd on runners` — make `cwd` required in
   `AgentRunner`/`ClaudeRunner`/`CodexRunner`/`create_runner`, delete the
   `_temp_cwd`/`__del__` fallback, update the direct-construction tests to pass
   `cwd=tmp_path`. (Standalone prerequisite; no behavior change in production.)
5. `feat(harness): always-on project-scoped discovery` — harness-specific MCP
   discovery in `mcp.py` (`discover_claude_project_mcp_servers(cwd)` reads
   `<cwd>/.mcp.json`; Codex side empty pending verification) + `create_runner`
   merge (scenario wins), Claude `setting_sources` always `["project"]`
   (strict_mcp_config unchanged), skills allow-list derived from the staged dir +
   skill-collision guard. No `project_mode` param, no `.mcp.json` cross-feed to
   Codex. + tests.
6. `refactor(experiment): per-run workspaces` — move workspace + runner creation
   into `experiment_task`, wire `WorkspaceManager` through `run_experiments` +
   tests.
7. `docs: document workdir feature` — AGENTS.md, README, example.
