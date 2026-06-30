"""Tests for the Codex runner's environment wiring.

`CodexRunner._codex_env()` **always** isolates `CODEX_HOME` to a fresh per-run
dir so the operator's global `~/.codex/config.toml` (MCP servers etc.) never
leaks into a run. Auth is preserved either via env (gateway token /
`OPENAI_API_KEY`) or by seeding only `auth.json` from the operator's real
`CODEX_HOME` into the isolated dir (so `codex login` keeps working without the
global config).
"""

from __future__ import annotations

from dd_ai_devx_evals.harness.codex import ISOLATED_CODEX_HOME_DIRNAME, CodexRunner


def _operator_home_with_auth(tmp_path, monkeypatch, *, with_auth: bool):
    """Point CODEX_HOME at a throwaway operator home, optionally with auth.json."""
    operator = tmp_path / "operator-codex-home"
    operator.mkdir()
    if with_auth:
        (operator / "auth.json").write_text('{"tokens": "operator"}')
    # Also drop a config.toml the run must NOT pick up.
    (operator / "config.toml").write_text("[mcp_servers.leaky]\nurl = 'http://x'\n")
    monkeypatch.setenv("CODEX_HOME", str(operator))
    return operator


class TestCodexEnv:
    def test_always_isolates_codex_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        isolated = cwd / ISOLATED_CODEX_HOME_DIRNAME
        assert env["CODEX_HOME"] == str(isolated)
        assert isolated.is_dir()
        # The isolated home does not contain the operator's global config.
        assert not (isolated / "config.toml").exists()

    def test_codex_login_auth_seeded_without_env_auth(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        seeded = cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json"
        # auth.json is copied in (codex login preserved) but config.toml is not.
        assert seeded.read_text() == '{"tokens": "operator"}'
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "config.toml").exists()
        assert "OPENAI_BASE_URL" not in env

    def test_env_api_key_does_not_seed_auth_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        # With env auth we rely on OPENAI_API_KEY, not the operator's auth.json.
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)

    def test_gateway_sets_base_url_token_and_isolates(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=True)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(
            cwd=cwd,
            gateway_base_url="https://gw.example.com/v1",
            gateway_token="bearer-xyz",
        )
        env = runner._codex_env()
        assert env["OPENAI_BASE_URL"] == "https://gw.example.com/v1"
        assert env["OPENAI_API_KEY"] == "bearer-xyz"
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)
        # Gateway token is env auth -> no auth.json seeding.
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()

    def test_missing_operator_auth_is_tolerated(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _operator_home_with_auth(tmp_path, monkeypatch, with_auth=False)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = CodexRunner(cwd=cwd)
        env = runner._codex_env()
        # No env auth and no operator auth.json: still isolates, nothing seeded.
        assert env["CODEX_HOME"] == str(cwd / ISOLATED_CODEX_HOME_DIRNAME)
        assert not (cwd / ISOLATED_CODEX_HOME_DIRNAME / "auth.json").exists()
