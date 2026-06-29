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
            tool_names=("search_apm",),
        )
        spec = McpServerSpec.from_config(cfg)
        assert spec.name == "apm"
        assert spec.type == "http"
        assert spec.url == "http://localhost:8000/mcp"
        assert spec.command is None
        assert dict(spec.headers) == {"source": "evals"}
        assert spec.tool_names == ("search_apm",)
        assert not spec.is_managed

    def test_managed_http_server(self):
        cfg = McpServerConfig(
            name="apm",
            url="http://localhost:8000/mcp",
            command="my-server",
            args=("--port", "8000"),
            env={"MODE": "both"},
        )
        spec = McpServerSpec.from_config(cfg)
        assert spec.type == "http"
        assert spec.is_managed
        assert spec.command == "my-server"
        assert spec.args == ("--port", "8000")
        assert dict(spec.env) == {"MODE": "both"}

    def test_stdio_server(self):
        cfg = McpServerConfig(
            name="tools",
            command="python",
            args=("-m", "my_server"),
            env={"FOO": "bar"},
        )
        spec = McpServerSpec.from_config(cfg)
        assert spec.name == "tools"
        assert spec.type == "stdio"
        assert spec.url is None
        assert spec.command == "python"
        assert spec.args == ("-m", "my_server")
        assert dict(spec.env) == {"FOO": "bar"}
        assert not spec.is_managed


# ---------------------------------------------------------------------------
# merged_headers
# ---------------------------------------------------------------------------


class TestMergedHeaders:
    def test_static_headers_only(self):
        spec = McpServerSpec(name="s", url="http://localhost/mcp", headers={"source": "evals"})
        result = spec.merged_headers({})
        assert result == {"source": "evals"}

    def test_trace_headers_override_static(self):
        spec = McpServerSpec(name="s", url="http://localhost/mcp", headers={"x-foo": "static"})
        result = spec.merged_headers({"x-foo": "from-trace", "x-dd-trace-id": "123"})
        assert result["x-foo"] == "from-trace"
        assert result["x-dd-trace-id"] == "123"


# ---------------------------------------------------------------------------
# to_claude_config — HTTP and stdio
# ---------------------------------------------------------------------------


class TestToClaudeConfig:
    def test_http_config_type_and_fields(self):
        spec = McpServerSpec(
            name="apm",
            url="http://localhost:8000/mcp",
            headers={"source": "evals"},
        )
        result = spec.to_claude_config({})
        # McpHttpServerConfig returns a dict
        assert result["type"] == "http"
        assert result["url"] == "http://localhost:8000/mcp"
        assert result["headers"]["source"] == "evals"

    def test_managed_http_renders_as_http(self):
        spec = McpServerSpec(name="apm", url="http://localhost:8000/mcp", command="my-server", args=("--port", "8000"))
        result = spec.to_claude_config({})
        # The launch command is internal; Claude only sees the http transport.
        assert result["type"] == "http"
        assert result["url"] == "http://localhost:8000/mcp"

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

    def test_unsupported_type_raises(self):
        # Force an unsupported transport type to exercise the guard.
        spec = McpServerSpec(name="bad", url="http://localhost/mcp")
        object.__setattr__(spec, "type", "sse")
        with pytest.raises(ValueError, match="unsupported type"):
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

    def test_url_name_and_type_preserved(self):
        spec = McpServerSpec(name="apm", url="http://localhost/mcp")
        safe = spec.to_safe_dict()
        assert safe["name"] == "apm"
        assert safe["url"] == "http://localhost/mcp"
        assert safe["type"] == "http"


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
