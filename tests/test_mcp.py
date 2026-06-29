"""Tests for dd_ai_devx_evals.mcp — McpServerSpec and rendering helpers."""

from __future__ import annotations

import pytest

from dd_ai_devx_evals.config.experiment import McpServerConfig
from dd_ai_devx_evals.mcp import (
    McpServerSpec,
    _toml_key,
    _toml_key_path,
    _toml_string,
    _toml_value,
    provider_mcp_tool_name,
)

# ---------------------------------------------------------------------------
# McpServerSpec.from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_http_server(self):
        cfg = McpServerConfig(
            name="apm",
            url="http://localhost:8000/mcp",
            headers={"source": "evals"},
            bearer_token_env_var="APM_TOKEN",
            tool_names=("search_apm",),
            start_command="python -m server",
        )
        spec = McpServerSpec.from_config(cfg)
        assert spec.name == "apm"
        assert spec.url == "http://localhost:8000/mcp"
        assert spec.command is None
        assert dict(spec.headers) == {"source": "evals"}
        assert spec.bearer_token_env_var == "APM_TOKEN"
        assert spec.tool_names == ("search_apm",)
        assert spec.start_command == "python -m server"

    def test_stdio_server(self):
        cfg = McpServerConfig(
            name="tools",
            command="python",
            args=("-m", "my_server"),
            env={"FOO": "bar"},
        )
        spec = McpServerSpec.from_config(cfg)
        assert spec.name == "tools"
        assert spec.url is None
        assert spec.command == "python"
        assert spec.args == ("-m", "my_server")
        assert dict(spec.env) == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# merged_headers
# ---------------------------------------------------------------------------


class TestMergedHeaders:
    def test_static_headers_only(self):
        spec = McpServerSpec(name="s", url="http://localhost/mcp", headers={"source": "evals"})
        result = spec.merged_headers({})
        assert result == {"source": "evals"}

    def test_bearer_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret-token")
        spec = McpServerSpec(
            name="s",
            url="http://localhost/mcp",
            headers={"source": "evals"},
            bearer_token_env_var="MY_TOKEN",
        )
        result = spec.merged_headers({})
        assert result["Authorization"] == "Bearer secret-token"
        assert result["source"] == "evals"

    def test_bearer_env_unset_skipped(self, monkeypatch):
        monkeypatch.delenv("MISSING_TOKEN", raising=False)
        spec = McpServerSpec(name="s", url="http://localhost/mcp", bearer_token_env_var="MISSING_TOKEN")
        result = spec.merged_headers({})
        assert "Authorization" not in result

    def test_trace_headers_override_static(self):
        spec = McpServerSpec(name="s", url="http://localhost/mcp", headers={"x-foo": "static"})
        result = spec.merged_headers({"x-foo": "from-trace", "x-dd-trace-id": "123"})
        assert result["x-foo"] == "from-trace"
        assert result["x-dd-trace-id"] == "123"

    def test_include_bearer_false_skips_auth(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret")
        spec = McpServerSpec(name="s", url="http://localhost/mcp", bearer_token_env_var="MY_TOKEN")
        result = spec.merged_headers({}, include_bearer=False)
        assert "Authorization" not in result


# ---------------------------------------------------------------------------
# to_claude_config — HTTP and stdio
# ---------------------------------------------------------------------------


class TestToClaudeConfig:
    def test_http_config_type_and_fields(self, monkeypatch):
        monkeypatch.setenv("APM_TOKEN", "mytoken")
        spec = McpServerSpec(
            name="apm",
            url="http://localhost:8000/mcp",
            headers={"source": "evals"},
            bearer_token_env_var="APM_TOKEN",
        )
        result = spec.to_claude_config({})
        # McpHttpServerConfig returns a dict
        assert result["type"] == "http"
        assert result["url"] == "http://localhost:8000/mcp"
        assert result["headers"]["source"] == "evals"
        assert result["headers"]["Authorization"] == "Bearer mytoken"

    def test_stdio_config_type_and_fields(self):
        spec = McpServerSpec(
            name="tools",
            command="python",
            args=("-m", "my_server"),
            env={"FOO": "bar"},
        )
        result = spec.to_claude_config({})
        assert result["type"] == "stdio"
        assert result["command"] == "python"
        assert result["args"] == ["-m", "my_server"]
        assert result["env"] == {"FOO": "bar"}

    def test_no_url_no_command_raises(self):
        # Directly construct a spec in an invalid state to test the guard
        spec = McpServerSpec.__new__(McpServerSpec)
        object.__setattr__(spec, "name", "bad")
        object.__setattr__(spec, "url", None)
        object.__setattr__(spec, "command", None)
        object.__setattr__(spec, "args", ())
        object.__setattr__(spec, "env", {})
        object.__setattr__(spec, "headers", {})
        object.__setattr__(spec, "tool_names", ())
        object.__setattr__(spec, "bearer_token_env_var", None)
        object.__setattr__(spec, "start_command", None)
        object.__setattr__(spec, "start_env", {})
        with pytest.raises(ValueError, match="url or command"):
            spec.to_claude_config({})


# ---------------------------------------------------------------------------
# to_codex_config_overrides
# ---------------------------------------------------------------------------


class TestToCodexConfigOverrides:
    def test_http_basic_overrides(self):
        spec = McpServerSpec(name="apm", url="http://localhost:8000/mcp")
        overrides = spec.to_codex_config_overrides({})
        url_override = next(o for o in overrides if ".url=" in o)
        assert '"http://localhost:8000/mcp"' in url_override

    def test_http_headers_in_overrides(self):
        spec = McpServerSpec(name="apm", url="http://localhost/mcp", headers={"source": "evals"})
        overrides = spec.to_codex_config_overrides({"x-dd-trace-id": "123"})
        headers_override = next(o for o in overrides if "http_headers" in o)
        assert "source" in headers_override
        assert "x-dd-trace-id" in headers_override

    def test_http_bearer_env_var_override(self):
        spec = McpServerSpec(name="apm", url="http://localhost/mcp", bearer_token_env_var="MY_TOKEN")
        overrides = spec.to_codex_config_overrides({})
        bearer_override = next(o for o in overrides if "bearer_token_env_var" in o)
        assert "MY_TOKEN" in bearer_override

    def test_stdio_command_override(self):
        spec = McpServerSpec(name="tools", command="python", args=("-m", "server"), env={"FOO": "bar"})
        overrides = spec.to_codex_config_overrides({})
        cmd_override = next(o for o in overrides if ".command=" in o)
        assert '"python"' in cmd_override
        args_override = next(o for o in overrides if ".args=" in o)
        assert '"-m"' in args_override
        env_override = next(o for o in overrides if ".env=" in o)
        assert "FOO" in env_override

    def test_key_prefix_uses_server_name(self):
        spec = McpServerSpec(name="my-server", url="http://localhost/mcp")
        overrides = spec.to_codex_config_overrides({})
        assert all(o.startswith("mcp_servers.my-server.") for o in overrides)


# ---------------------------------------------------------------------------
# to_safe_dict
# ---------------------------------------------------------------------------


class TestToSafeDict:
    def test_header_values_redacted(self):
        spec = McpServerSpec(
            name="apm",
            url="http://localhost/mcp",
            headers={"Authorization": "Bearer secret", "source": "evals"},
        )
        safe = spec.to_safe_dict()
        assert safe["headers"]["Authorization"] == "<redacted>"
        assert safe["headers"]["source"] == "<redacted>"
        # Keys are still present, just values replaced
        assert set(safe["headers"].keys()) == {"Authorization", "source"}

    def test_url_and_name_preserved(self):
        spec = McpServerSpec(name="apm", url="http://localhost/mcp")
        safe = spec.to_safe_dict()
        assert safe["name"] == "apm"
        assert safe["url"] == "http://localhost/mcp"

    def test_bearer_token_env_var_preserved(self):
        spec = McpServerSpec(name="apm", url="http://localhost/mcp", bearer_token_env_var="TOKEN_ENV")
        safe = spec.to_safe_dict()
        assert safe["bearer_token_env_var"] == "TOKEN_ENV"


# ---------------------------------------------------------------------------
# provider_mcp_tool_name
# ---------------------------------------------------------------------------


class TestProviderMcpToolName:
    def test_claude_sdk_format(self):
        name = provider_mcp_tool_name("apm", "search_libraries", sdk_name="claude-agent-sdk")
        assert name == "mcp__apm__search_libraries"

    def test_codex_format(self):
        name = provider_mcp_tool_name("apm", "search_libraries", sdk_name="openai-codex")
        assert name == "mcp.apm.search_libraries"

    def test_other_sdk_uses_dot_format(self):
        name = provider_mcp_tool_name("srv", "do_thing", sdk_name="other-sdk")
        assert name == "mcp.srv.do_thing"


# ---------------------------------------------------------------------------
# TOML helpers
# ---------------------------------------------------------------------------


class TestTomlValue:
    def test_string(self):
        assert _toml_value("hello") == '"hello"'

    def test_string_with_quotes(self):
        # json.dumps escapes internal quotes
        assert _toml_value('say "hi"') == '"say \\"hi\\""'

    def test_bool_true(self):
        assert _toml_value(True) == "true"

    def test_bool_false(self):
        assert _toml_value(False) == "false"

    def test_int(self):
        assert _toml_value(42) == "42"

    def test_dict(self):
        result = _toml_value({"key": "val"})
        assert result == '{key = "val"}'

    def test_list(self):
        result = _toml_value(["a", "b"])
        assert result == '["a", "b"]'

    def test_none(self):
        assert _toml_value(None) == '""'


class TestTomlKey:
    def test_simple_key_unquoted(self):
        assert _toml_key("simple") == "simple"
        assert _toml_key("my-key") == "my-key"
        assert _toml_key("my_key123") == "my_key123"

    def test_key_with_space_quoted(self):
        result = _toml_key("with space")
        assert result == '"with space"'

    def test_key_with_dot_quoted(self):
        # dot is NOT in `[A-Za-z0-9_-]` so it needs quoting
        result = _toml_key("my.key")
        assert result == '"my.key"'


class TestTomlKeyPath:
    def test_two_simple_parts(self):
        assert _toml_key_path("mcp_servers", "my-server") == "mcp_servers.my-server"

    def test_part_with_space_quoted(self):
        result = _toml_key_path("mcp_servers", "my server")
        assert result == 'mcp_servers."my server"'


class TestTomlString:
    def test_simple(self):
        assert _toml_string("hello") == '"hello"'

    def test_escapes_newline(self):
        assert "\\n" in _toml_string("line\nnewline")

    def test_escapes_backslash(self):
        assert "\\\\" in _toml_string("back\\slash")
