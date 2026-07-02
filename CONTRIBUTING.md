# Contributing

Thanks for your interest in contributing! This is an open source project, so we
appreciate community contributions.

Pull requests for bug fixes are welcome, but before submitting new features or
changes to current functionality [open an issue][new-issue] and discuss your
ideas or propose the changes you wish to make. After a resolution is reached a PR
can be submitted for review. PRs created before a decision has been reached may
be closed.

## License

`airedale` is licensed under the [Apache License 2.0](./LICENSE). By submitting a
PR to this repository, you are making the contribution under the terms of the
[`Apache-2.0` license](./LICENSE), and you affirm that you are authorized to do
so.

Every source file should begin with the Apache-2.0 license line:

```
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
```

Files authored or heavily modified by a Datadog employee should additionally
carry the Datadog copyright line:

```
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright <year>-present Datadog, Inc.
```

where `<year>` is the 4-digit year the file was introduced. Files created or
heavily modified by third-party (non-Datadog) authors need not include the
Datadog copyright line.

Code copied from another repository should live in separate files containing only
code from that same origin, and must **retain** the original repository's
copyright/license header and a reference to its source. Such code must be
governed by a license compatible with [`Apache-2.0`](./LICENSE).

## Development environment

`airedale` targets **Python ≥ 3.11, < 3.15** and uses [`uv`](https://docs.astral.sh/uv/)
for environment and dependency management.

```bash
# Install dependencies (including the dev group)
uv sync

# Run the CLI from your checkout
uv run airedale --help
```

The two provider execution engines (`claude-agent-sdk`, `openai-codex`) are
declared as dependencies, but may need to be installed explicitly on some
platforms — see the [README](./README.md#external-sdk-requirements).

## Dependency management

`airedale` deliberately takes **minimal dependencies**. Every dependency is a
long-term maintenance and security liability, so a new one should only be
introduced when it either:

- **meaningfully reduces the maintenance burden** of the project (i.e. it
  replaces a non-trivial amount of code we would otherwise have to write, test,
  and maintain ourselves), or
- is **essential** — there is no reasonable way to deliver the functionality
  without it.

When in doubt, prefer the standard library or a small amount of in-repo code
over a new dependency.

Non-dev dependencies (the `[project].dependencies` list in
[`pyproject.toml`](./pyproject.toml)) MUST be pinned with an exact `==` version
so the dependency closure — and [`LICENSE-3rdparty.csv`](./LICENSE-3rdparty.csv)
— stays reproducible. If a dependency needs a more open range (`>=`, `<`, etc.),
add an inline comment explaining **why**. For example:

```toml
"openai-codex-cli-bin>=0.137.0a4", # 0.136.0 is missing a manylinux wheel
```

Dev dependencies (the `[dependency-groups].dev` list) are not subject to the
`==` pinning rule.

## Pull requests

`airedale` uses the [conventional commits][conventional-commits] specification
for commit messages and PR titles, with the following structure:

```text
<type>(scope): <description>
```

Where:

- **type** — one of `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
- **scope** — the module or area affected (e.g. `config`, `harness`, `mcp`,
  `workdir`, `scoring`). Optional but encouraged.
- **description** — a concise, imperative summary. For bug fixes, describe the
  bug being fixed, not how it is fixed:
  - :x: `fix(codex): copy auth.json into the isolated CODEX_HOME`
  - :white_check_mark: `fix(codex): global config.toml leaks into isolated runs`

Examples:

- `feat(mcp): infer transport type from url/command`
- `fix(workdir): confine clone-cache teardown to the cache dir`
- `docs: document distributed-trace header injection`
- `test(harness): cover Codex span shape`

The PR title and body should explain:

- **Why** the change is being made — for bug fixes, the root cause; for
  enhancements, the use case or added value.
- **What** changed, in plain English, so reviewers can make sense of the diff
  without reading every line.

Link relevant issues (including in other repositories) using header-style footers
in the PR body:

- `Fixes: #123` — fixes issue `123` in this repository.
- `Depends-On: DataDog/dd-trace-go#456` — depends on PR/issue `456` in another
  repository.

## Continuous integration

All automated checks must pass before a PR is eligible for merging. PRs with
failing checks will not be reviewed as a priority.

The [CI workflow](./.github/workflows/ci.yml) runs two jobs:

1. **checks (format, lint):**
   - `uv run ruff format --check .` — formatting.
   - `uv run ruff check --no-fix .` — lint.
2. **test (py3.11 – py3.14):** `uv run pytest` on every supported Python version.

A separate [LICENSE-3rdparty workflow](./.github/workflows/license-3rdparty.yml)
verifies that [`LICENSE-3rdparty.csv`](./LICENSE-3rdparty.csv) matches the current
dependency closure. If a transitive dependency publishes a new release the file
can drift; regenerate it (on **Python 3.12**, to match how it was created) with
the command printed on failure:

```bash
python -m pip install "dd-license-attribution @ git+https://github.com/DataDog/dd-license-attribution.git@v0.5.0"
dd-license-attribution generate-sbom \
  --no-github-sbom-strategy --no-scancode-strategy --no-npm-strategy --no-gopkg-strategy \
  https://github.com/DataDog/airedale > LICENSE-3rdparty.csv
```

### Running the checks locally

Run the same checks before submitting to save a review round-trip:

```bash
# Format and lint (with autofix) while iterating
uv run ruff format .
uv run ruff check . --fix

# Reproduce CI exactly (no autofix)
uv run ruff format --check .
uv run ruff check --no-fix .

# Tests
uv run pytest
```

## Testing

We expect PRs to add tests for any new or significantly changed functionality,
unless existing tests already cover the surface. Reviewers may request additional
tests before approving.

**Unit tests MUST run fully offline** — no network, no real provider calls, no
real LLMObs submissions. Mock the SDK clients and `LLMObs`. This is a hard
requirement: a test that reaches the network will fail in CI and is not
acceptable. The existing suite covers config TOML parsing/validation, gateway
resolution, MCP spec rendering (Claude config and Codex overrides),
`ModelSpec`/`UsageMetrics`, judge JSON parsing, skill staging, workspace
management, and span shapes — mirror those patterns.

```bash
uv run pytest                      # whole suite
uv run pytest tests/test_mcp.py    # one file
uv run pytest -k rubric            # by keyword
```

## Code style

- **Formatting & lint:** [`ruff`](https://docs.astral.sh/ruff/), configured in
  [`pyproject.toml`](./pyproject.toml) (line length 120). Run `ruff format` and
  `ruff check` before submitting.
- **Type hints everywhere**, and `from __future__ import annotations` at the top
  of every module. Prefer dataclasses for value objects.
- **Comments** explain non-obvious intent, trade-offs, or constraints the code
  can't carry. Don't narrate what the diff already shows.
- **Secrets:** never commit tokens; never log bearer tokens; redact headers in
  any LLMObs metadata.

For the architectural contract every module must honor (configuration semantics,
LLMObs reporting rules, distributed-tracing invariants, module responsibilities),
read [`AGENTS.md`](./AGENTS.md) before making non-trivial changes.

## Getting a PR reviewed

If your PR passes all checks and has been waiting for a review, feel free to
comment on it to bubble it up.

<!-- Links -->
[new-issue]: https://github.com/DataDog/airedale/issues/new
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/
