# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the pi harness (``airedale.harness.pi``).

Everything runs offline. The subprocess tests use a small Python script as a
fake ``pi`` executable: it records its argv, environment and agent-dir files,
then prints a scripted JSONL event stream.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from airedale.config.experiment import HARNESS_PI, ScenarioConfig
from airedale.config.gateway import GatewayConfig, ProviderGatewayConfig
from airedale.harness import create_runner
from airedale.harness.base import AgentThought, AgentToolCall
from airedale.harness.pi import PI_BUILTIN_TOOLS, PiRunner, _PiEventCollector
from airedale.mcp import McpServerSpec, pi_literal, pi_mcp_tool_name, provider_mcp_tool_name
from airedale.skills import pi_skill_dirs
from airedale.types import ModelSpec, UsageMetrics

MODEL = ModelSpec.parse("anthropic/claude-sonnet-4-6")


def _usage(input_tokens: int, output: int, cache_read: int = 0, cache_write: int = 0, cost: float = 0.0) -> dict:
    return {
        "input": input_tokens,
        "output": output,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": input_tokens + output + cache_read + cache_write,
        "cost": {"total": cost},
    }


def _assistant(content: list[dict], *, usage: dict, stop: str = "stop", model: ModelSpec = MODEL, **extra) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "provider": model.provider,
        "model": model.name,
        "usage": usage,
        "stopReason": stop,
        **extra,
    }


def _events_tool_run(model: ModelSpec = MODEL) -> list[dict[str, Any]]:
    """A two-turn run: thinking + text + tool call, then the final answer."""
    first = _assistant(
        [
            {"type": "thinking", "thinking": "I should list files."},
            {"type": "text", "text": "Let me look."},
            {"type": "toolCall", "id": "call_1", "name": "ls", "arguments": {"path": "."}},
        ],
        usage=_usage(3, 8, cache_write=1447, cost=0.005),
        stop="toolUse",
        model=model,
    )
    final = _assistant(
        [{"type": "text", "text": "The directory is empty."}], usage=_usage(1, 12, 1447, 90, 0.001), model=model
    )
    return [
        {"type": "session", "id": "sess-1", "version": 3},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": "hi"}},
        {"type": "message_end", "message": first},
        {"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "ls", "args": {"path": "."}},
        {
            "type": "tool_execution_end",
            "toolCallId": "call_1",
            "toolName": "ls",
            "result": {"content": [{"type": "text", "text": "(empty directory)"}], "details": {}},
            "isError": False,
            "durationMs": 2,
        },
        {"type": "turn_end"},
        {"type": "turn_start"},
        {"type": "message_end", "message": final},
        {"type": "turn_end"},
        {"type": "agent_end", "messages": []},
        {"type": "agent_settled", "aborted": False},
    ]


def _collect(events: list[dict[str, Any]], **kwargs: Any) -> _PiEventCollector:
    collector = _PiEventCollector(**kwargs)
    for event in events:
        collector.feed_line((json.dumps(event) + "\n").encode())
    return collector


# --------------------------------------------------------------------------- #
# Usage and literal helpers
# --------------------------------------------------------------------------- #
class TestUsageFromPi:
    def test_effective_input_includes_cache(self):
        usage = UsageMetrics.from_pi(
            {
                "input": 1,
                "output": 12,
                "cacheRead": 1447,
                "cacheWrite": 90,
                "totalTokens": 1550,
                "reasoning": 4,
                "cost": {"total": 0.0009},
            }
        )
        assert usage.input_tokens == 1538
        assert usage.output_tokens == 12
        assert usage.total_tokens == 1550
        assert usage.cache_read_input_tokens == 1447
        assert usage.cache_write_input_tokens == 90
        assert usage.reasoning_output_tokens == 4
        assert usage.estimated_cost_usd == pytest.approx(0.0009)

    def test_missing_or_bad_values_are_zero(self):
        usage = UsageMetrics.from_pi({"input": "x", "cost": "bad"})
        assert usage.total_tokens == 0
        assert usage.estimated_cost_usd == 0.0
        assert UsageMetrics.from_pi(None).total_tokens == 0


class TestPiLiteral:
    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [("plain", "plain"), ("$HOME", "$$HOME"), ("!rm -rf /", "$!rm -rf /"), ("a$b!c", "a$$b!c"), ("!$X", "$!$$X")],
    )
    def test_escapes_interpolation_and_commands(self, raw, escaped):
        assert pi_literal(raw) == escaped

    def test_tool_name_sanitized(self):
        assert pi_mcp_tool_name("my-server", "get.docs") == "mcp__my_server__get_docs"
        assert provider_mcp_tool_name("apm", "search", sdk_name="pi") == "mcp__apm__search"


# --------------------------------------------------------------------------- #
# MCP rendering
# --------------------------------------------------------------------------- #
class TestToPiConfig:
    def test_http_merges_trace_headers_and_escapes(self):
        spec = McpServerSpec(name="apm", url="http://localhost:8000/mcp", headers={"x-token": "$SECRET"})
        entry = spec.to_pi_config({"traceparent": "00-abc"})
        assert entry == {
            "type": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {"x-token": "$$SECRET", "traceparent": "00-abc"},
            "exposure": "direct",
        }

    def test_stdio_with_tool_allow_list(self):
        spec = McpServerSpec(name="tools", command="python", args=("-m", "srv"), env={"K": "!cmd"}, tool_names=("a",))
        entry = spec.to_pi_config({"traceparent": "ignored-for-stdio"})
        assert entry == {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "srv"],
            "env": {"K": "$!cmd"},
            "exposure": "hidden",
            "toolExposure": {"a": "direct"},
        }

    def test_wildcard_tool_name_rejected(self):
        with pytest.raises(ValueError, match="exact tool names"):
            McpServerSpec(name="s", command="x", tool_names=("*",)).to_pi_config({})

    def test_invalid_server_name_rejected(self):
        with pytest.raises(ValueError, match="not valid for pi"):
            McpServerSpec(name="bad.name", command="x").to_pi_config({})


# --------------------------------------------------------------------------- #
# Event collection
# --------------------------------------------------------------------------- #
class TestEventCollector:
    def test_tool_run_result(self):
        collector = _collect(_events_tool_run())
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert result.answer == "The directory is empty."
        assert result.usage.input_tokens == 3 + 1447 + 1 + 1447 + 90
        assert result.usage.output_tokens == 20
        assert result.usage.estimated_cost_usd == pytest.approx(0.006)
        kinds = [type(segment).__name__ for segment in result.segments]
        assert kinds == ["AgentThought", "AgentThought", "AgentToolCall"]
        thought, text, call = result.segments
        assert isinstance(thought, AgentThought) and thought.kind == "reasoning"
        assert isinstance(text, AgentThought) and text.kind == "text"
        assert isinstance(call, AgentToolCall)
        assert call.name == "ls"
        assert call.arguments == {"path": "."}
        assert call.result == "(empty directory)"
        assert call.error is None
        assert call.metadata == {"source": "pi", "tool_id": "call_1", "duration_ms": 2}
        assert result.tool_calls == [call]
        assert collector.settled

    def test_tool_error_and_missing_execution(self):
        events = _events_tool_run()
        events[6]["isError"] = True
        events[6]["result"] = {"content": [{"type": "text", "text": "boom"}]}
        # A second tool call that never ran.
        events[4]["message"]["content"].append({"type": "toolCall", "id": "call_2", "name": "bash", "arguments": {}})
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        ls_call, bash_call = result.tool_calls
        assert ls_call.error == "boom"
        assert bash_call.error == "tool did not run"

    def test_model_mismatch_is_an_error(self):
        other = ModelSpec.parse("anthropic/claude-sonnet-4-5")
        result = _collect(_events_tool_run(model=other)).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        assert "instead of anthropic/claude-sonnet-4-6" in str(result.error)

    def test_failed_final_message_is_an_error(self):
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "error"
        events[9]["message"]["errorMessage"] = "529 overloaded"
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert "529 overloaded" in str(result.error)

    def test_non_zero_exit_includes_stderr(self):
        result = _collect(_events_tool_run()).result(model=MODEL, returncode=2, stderr_tail="bad flag")
        assert "status 2" in str(result.error)
        assert "bad flag" in str(result.error)

    def test_no_assistant_message(self):
        result = _collect([{"type": "agent_settled"}]).result(model=MODEL, returncode=0, stderr_tail="")
        assert "no assistant message" in str(result.error)
        assert result.answer == "pi did not produce a final response."

    def test_max_turns_requests_stop(self):
        collector = _collect(_events_tool_run(), max_turns=1)
        assert collector.stop_requested
        assert "max_turns=1" in str(collector.result(model=MODEL, returncode=-15, stderr_tail="").error)

    def test_compaction_usage_is_counted(self):
        events = [*_events_tool_run()[:-1], {"type": "compaction_end", "result": {"usage": _usage(100, 10)}}]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.usage.output_tokens == 30

    def test_non_json_lines_are_counted(self):
        collector = _PiEventCollector()
        assert collector.feed_line(b"not json\n") is None
        assert collector.feed_line(b"[1, 2]\n") is None
        assert collector.feed_line(b"null\n") is None
        assert collector.invalid_lines == 3


# --------------------------------------------------------------------------- #
# Runner configuration
# --------------------------------------------------------------------------- #
class TestRunnerConfig:
    def test_models_config_bearer_token(self, tmp_path):
        runner = PiRunner(
            cwd=tmp_path,
            gateway_base_url="https://gw.example.com",
            gateway_headers={"source": "evals", "weird": "$X"},
            gateway_bearer_token="tok$en",
        )
        assert runner._models_config(MODEL) == {
            "providers": {
                "anthropic": {
                    "baseUrl": "https://gw.example.com",
                    "headers": {"source": "evals", "weird": "$$X"},
                    "apiKey": "tok$$en",
                    "authHeader": True,
                }
            }
        }

    def test_models_config_static_key_uses_native_auth(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, gateway_base_url="https://gw/v1", gateway_api_key="k")
        entry = runner._models_config(ModelSpec.parse("openai/gpt-5.5"))["providers"]["openai"]
        assert entry == {"baseUrl": "https://gw/v1", "apiKey": "k"}

    def test_models_config_without_gateway(self, tmp_path):
        assert PiRunner(cwd=tmp_path)._models_config(MODEL) == {}

    def test_models_config_pi_models_passthrough(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, pi_models=[{"id": "claude-new", "contextWindow": 1000}])
        assert runner._models_config(MODEL) == {
            "providers": {"anthropic": {"models": [{"id": "claude-new", "contextWindow": 1000}]}}
        }

    def test_command_lists_builtin_and_mcp_tools(self, tmp_path):
        runner = PiRunner(
            cwd=tmp_path,
            pi_command=["pi"],
            allowed_builtin_tools=["read"],
            mcp_servers=[
                McpServerSpec(name="apm", url="http://localhost/mcp", tool_names=("search",)),
                McpServerSpec(name="all-tools", command="srv"),
            ],
            effort="high",
        )
        command = runner._command(model=MODEL, system_prompt_path=Path("/p.md"), skill_dirs=["/s/a"])
        assert command[:3] == ["pi", "--mode", "json"]
        for flag in ("--no-session", "--approve", "--no-skills"):
            assert flag in command
        assert command[command.index("--provider") + 1] == "anthropic"
        assert command[command.index("--model") + 1] == "claude-sonnet-4-6"
        assert command[command.index("--thinking") + 1] == "high"
        assert command[command.index("--skill") + 1] == "/s/a"
        assert command[command.index("--append-system-prompt") + 1] == "/p.md"
        # `mcp__*` matches pi's real MCP tool names (hashed when long or
        # colliding); `toolExposure` in mcp.json enforces `tool_names`.
        assert command[command.index("--tools") + 1] == "read,mcp__*"
        # The prompt is never on the command line (pi reads `@path` args as files).
        assert "--" not in command

    def test_claude_tool_names_are_mapped(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, allowed_builtin_tools=["Read", "Grep", "Glob", "Edit", "MultiEdit", "custom"])
        assert runner._effective_builtin_tools() == ("read", "grep", "find", "edit", "custom")

    def test_command_all_builtins_when_omitted(self, tmp_path):
        command = PiRunner(cwd=tmp_path, pi_command=["pi"])._command(
            model=MODEL, system_prompt_path=Path("/p"), skill_dirs=[]
        )
        assert command[command.index("--tools") + 1] == ",".join(PI_BUILTIN_TOOLS)
        assert "--thinking" not in command

    def test_command_no_tools(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, pi_command=["pi"], allowed_builtin_tools=[], effort="bogus")
        command = runner._command(model=MODEL, system_prompt_path=Path("/p"), skill_dirs=[])
        assert "--no-tools" in command
        assert "--tools" not in command
        assert "--thinking" not in command

    def test_env_isolates_agent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_SESSION_ID", "parent-session")
        monkeypatch.setenv("PI_CODING_AGENT_DIR", "/operator/agent")
        monkeypatch.setenv("KEEP_ME", "1")
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent")
        assert env["PI_CODING_AGENT_DIR"] == str(tmp_path / "agent")
        assert "PI_SESSION_ID" not in env
        assert env["PI_OFFLINE"] == "1"
        assert env["KEEP_ME"] == "1"

    @pytest.mark.parametrize(
        "name",
        [
            "DD_API_KEY",
            "DD_APP_KEY",
            "DD_APPLICATION_KEY",
            "DD_CLIENT_TOKEN",
            "DD_API_KEY_SECRET_ARN",
            "DD_CLIENT_SECRET",
            "DD_DB_PASSWORD",
            "DD_AUTH",
            "DATADOG_MCP_AUTHORIZATION",
            "DD_CREDENTIALS",
            "DD_MCP_BEARER",
            "DATADOG_API_KEY",
            "DATADOG_APP_KEY",
            "_DD_API_KEY",
            "dd_api_key",
        ],
    )
    def test_env_drops_datadog_credentials(self, tmp_path, monkeypatch, name):
        secret = f"operator-secret-{name}"
        monkeypatch.setenv(name, secret)
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent")
        assert name not in env
        assert secret not in env.values()

    def test_env_keeps_other_variables(self, tmp_path, monkeypatch):
        # Datadog settings that are not credentials, and provider API keys
        # (pi reads ANTHROPIC_API_KEY / OPENAI_API_KEY itself), stay.
        keep = {
            "DD_SITE": "datadoghq.com",
            "DD_SERVICE": "svc",
            "DD_ENV": "dev",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "OPENAI_API_KEY": "openai-key",
            "MY_DD_API_KEY_NOTE": "not a datadog variable",
        }
        for name, value in keep.items():
            monkeypatch.setenv(name, value)
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent")
        for name, value in keep.items():
            assert env[name] == value

    def test_auth_json_copied_only_without_env_auth(self, tmp_path, monkeypatch):
        operator = tmp_path / "operator"
        operator.mkdir()
        (operator / "auth.json").write_text('{"anthropic": "x"}')
        (operator / "settings.json").write_text('{"packages": ["leak"]}')
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(operator))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        agent = tmp_path / "agent1"
        agent.mkdir()
        PiRunner(cwd=tmp_path)._write_agent_dir(agent, model=MODEL, system_prompt="S", trace_headers={})
        assert (agent / "auth.json").read_text() == '{"anthropic": "x"}'
        assert json.loads((agent / "settings.json").read_text()) == {}
        assert (agent / "airedale-system-prompt.md").read_text() == "S"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        agent2 = tmp_path / "agent2"
        agent2.mkdir()
        PiRunner(cwd=tmp_path)._write_agent_dir(agent2, model=MODEL, system_prompt="S", trace_headers={})
        assert not (agent2 / "auth.json").exists()

    def test_project_mcp_collision_raises(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"my_server": {"command": "x"}}}))
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="my-server", command="y")])
        with pytest.raises(ValueError, match="my-server"):
            runner._check_project_mcp_collisions()
        PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="other", command="y")])._check_project_mcp_collisions()


class TestPiSkillDirs:
    def _skill(self, base: Path, name: str) -> Path:
        path = base / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n")
        return path

    def test_scenario_and_project_skills(self, tmp_path):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        project = self._skill(cwd / ".pi" / "skills", "beta")
        agents = self._skill(cwd / ".agents" / "skills", "gamma")
        assert pi_skill_dirs([str(scenario)], str(cwd)) == [str(scenario.resolve()), str(project), str(agents)]

    def test_scenario_project_collision_raises(self, tmp_path):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        self._skill(cwd / ".agents" / "skills", "alpha")
        with pytest.raises(ValueError, match="collides"):
            pi_skill_dirs([str(scenario)], str(cwd))

    def test_project_duplicates_keep_first(self, tmp_path):
        cwd = tmp_path / "cwd"
        first = self._skill(cwd / ".pi" / "skills", "alpha")
        self._skill(cwd / ".agents" / "skills", "alpha")
        assert pi_skill_dirs([], str(cwd)) == [str(first)]

    def test_missing_scenario_skill_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            pi_skill_dirs([str(tmp_path / "nope")], str(tmp_path))


# --------------------------------------------------------------------------- #
# create_runner dispatch
# --------------------------------------------------------------------------- #
def test_create_runner_selects_pi(tmp_path):
    gateway = GatewayConfig(
        providers={
            "openai": ProviderGatewayConfig(
                provider="openai", base_url="https://gw/v1", api_key_env="GW_KEY", pi_models=({"id": "m"},)
            )
        }
    )
    scenario = ScenarioConfig(name="s", harness=HARNESS_PI, max_turns=5, effort="low")
    runner = create_runner(ModelSpec.parse("openai/gpt-5.5"), scenario=scenario, gateway=gateway, cwd=str(tmp_path))
    assert isinstance(runner, PiRunner)
    assert runner.gateway_base_url == "https://gw/v1"
    assert runner.pi_models == [{"id": "m"}]
    assert runner.max_turns == 5


# --------------------------------------------------------------------------- #
# Subprocess runs against a fake pi executable
# --------------------------------------------------------------------------- #
FAKE_PI = textwrap.dedent(
    """
    import json, os, sys, time
    record = os.environ["FAKE_PI_RECORD"]
    agent_dir = os.environ["PI_CODING_AGENT_DIR"]
    files = {}
    for name in sorted(os.listdir(agent_dir)):
        with open(os.path.join(agent_dir, name)) as handle:
            files[name] = handle.read()
    stdin = sys.stdin.read()
    with open(record, "w") as handle:
        json.dump({"argv": sys.argv[1:], "agent_dir": agent_dir, "files": files, "stdin": stdin,
                   "session": os.environ.get("PI_SESSION_ID"), "cwd": os.getcwd(),
                   "dd_env": {k: v for k, v in os.environ.items() if k.upper().startswith(("DD_", "DATADOG_", "_DD_"))}},
                  handle)
    events = json.loads(os.environ["FAKE_PI_EVENTS"])
    if os.environ.get("FAKE_PI_GARBAGE"):
        print("not json", flush=True)
    for event in events:
        print(json.dumps(event), flush=True)
    if os.environ.get("FAKE_PI_CHILD_HOLDS_STDOUT"):
        # A child keeps stdout open after pi exits; it records its pid.
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        with open(record + ".child", "w") as handle:
            handle.write(str(child.pid))
        sys.exit(0)
    if os.environ.get("FAKE_PI_HANG"):
        time.sleep(600)
    sys.stderr.write(os.environ.get("FAKE_PI_STDERR", ""))
    sys.exit(int(os.environ.get("FAKE_PI_EXIT", "0")))
    """
)


@pytest.fixture
def fake_pi(tmp_path, monkeypatch):
    script = tmp_path / "fake_pi.py"
    script.write_text(FAKE_PI)
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_PI_RECORD", str(record))
    monkeypatch.setenv("PI_SESSION_ID", "parent")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _set(events: list[dict[str, Any]], **env: str) -> tuple[list[str], Path]:
        monkeypatch.setenv("FAKE_PI_EVENTS", json.dumps(events))
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return [sys.executable, str(script)], record

    return _set


#: Hard limit for one fake-pi run, so a shutdown regression fails the test
#: instead of hanging the suite (the fake can sleep for 600 seconds).
RUN_TIMEOUT_SECONDS = 30


def _run(runner: PiRunner, model: ModelSpec = MODEL):
    return _bounded(
        runner._run_agent(
            model=model, system_prompt="SYS", user_prompt="Q", trace_headers={"traceparent": "00-x"}, progress=None
        )
    )


def _bounded(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=RUN_TIMEOUT_SECONDS))


class TestSubprocess:
    def test_successful_run(self, tmp_path, fake_pi):
        command, record = fake_pi(_events_tool_run())
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        runner = PiRunner(
            cwd=cwd,
            pi_command=command,
            mcp_servers=[McpServerSpec(name="apm", url="http://localhost/mcp")],
            gateway_base_url="https://gw",
            gateway_bearer_token="secret-token",
        )
        result = _run(runner)
        assert result.error is None
        assert result.answer == "The directory is empty."

        seen = json.loads(record.read_text())
        assert seen["session"] is None
        assert Path(seen["cwd"]).resolve() == cwd.resolve()
        assert seen["stdin"] == "Q"
        assert "Q" not in seen["argv"]
        models = json.loads(seen["files"]["models.json"])
        assert models["providers"]["anthropic"]["apiKey"] == "secret-token"
        mcp = json.loads(seen["files"]["mcp.json"])
        assert mcp["autoEnableCodemode"] is False
        assert mcp["mcpServers"]["apm"]["headers"] == {"traceparent": "00-x"}
        assert seen["files"]["airedale-system-prompt.md"] == "SYS"
        # The private agent dir lives outside the workspace and is removed.
        assert not Path(seen["agent_dir"]).exists()
        assert not Path(seen["agent_dir"]).resolve().is_relative_to(cwd.resolve())
        assert list(cwd.iterdir()) == []

    def test_datadog_credentials_do_not_reach_pi(self, tmp_path, fake_pi):
        command, record = fake_pi(
            _events_tool_run(),
            DD_API_KEY="dd-api",
            DD_APP_KEY="dd-app",
            DATADOG_API_KEY="dd-api-2",
            _DD_API_KEY="dd-api-3",
            DD_SITE="site",
        )
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is None
        seen = json.loads(record.read_text())
        for name in ("DD_API_KEY", "DD_APP_KEY", "DATADOG_API_KEY", "_DD_API_KEY"):
            assert name not in seen["dd_env"]
        assert not {"dd-api", "dd-app", "dd-api-2", "dd-api-3"} & set(seen["dd_env"].values())
        assert seen["dd_env"]["DD_SITE"] == "site"

    def test_non_zero_exit(self, tmp_path, fake_pi):
        command, _ = fake_pi([], FAKE_PI_EXIT="3", FAKE_PI_STDERR="unknown model")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "status 3" in str(result.error)
        assert "unknown model" in str(result.error)

    def test_max_turns_stops_hanging_process(self, tmp_path, fake_pi):
        events = [{"type": "turn_start"}, {"type": "turn_start"}, {"type": "turn_start"}]
        command, _ = fake_pi(events, FAKE_PI_HANG="1")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command, max_turns=2))
        assert "max_turns=2" in str(result.error)

    def test_settled_but_not_exiting_is_stopped(self, tmp_path, fake_pi, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 0.5)
        command, _ = fake_pi(_events_tool_run(), FAKE_PI_HANG="1")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        # Pi was stopped by a signal after the run settled: that is not a failure.
        assert result.answer == "The directory is empty."
        assert result.error is None

    def test_missing_executable(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, pi_command=[str(tmp_path / "no-such-pi")])
        with pytest.raises(RuntimeError, match="pi executable not found"):
            _run(runner)

    def test_full_run_returns_harness_result(self, tmp_path, fake_pi):
        command, _ = fake_pi(_events_tool_run())
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        result = _bounded(
            runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="pi-scenario")
        )
        assert result.answer == "The directory is empty."
        assert result.harness == "pi-scenario"
        assert result.tool_calls[0]["name"] == "ls"
        assert result.usage.total_tokens > 0


# --------------------------------------------------------------------------- #
# Review follow-ups: stricter completion, usage, names, process lifecycle
# --------------------------------------------------------------------------- #
class TestCompletionRules:
    def test_tool_result_usage_is_counted(self):
        events = _events_tool_run()
        nested = {"role": "toolResult", "toolCallId": "call_1", "usage": _usage(50, 5)}
        events.insert(7, {"type": "message_end", "message": nested})
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert result.usage.output_tokens == 25

    @pytest.mark.parametrize("stop", ["length", "toolUse", "deferred"])
    def test_incomplete_final_stop_reason_is_an_error(self, stop):
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = stop
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert f"stopReason={stop!r}" in str(result.error)

    def test_not_settled_is_an_error(self):
        events = [event for event in _events_tool_run() if event["type"] != "agent_settled"]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert "before the run settled" in str(result.error)

    def test_aborted_is_an_error(self):
        events = _events_tool_run()
        events[-1]["aborted"] = True
        assert "aborted" in str(_collect(events).result(model=MODEL, returncode=0, stderr_tail="").error)

    def test_malformed_record_is_an_error(self):
        collector = _collect(_events_tool_run())
        collector.feed_line(b"{broken\n")
        assert "malformed JSONL" in str(collector.result(model=MODEL, returncode=0, stderr_tail="").error)

    def test_response_model_is_only_a_warning(self, caplog):
        events = _events_tool_run()
        events[9]["message"]["responseModel"] = "claude-sonnet-4-6-20991231"
        with caplog.at_level("WARNING"):
            result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert "claude-sonnet-4-6-20991231" in caplog.text


class TestPiToolNames:
    def test_long_name_gets_pi_hash_suffix(self):
        import hashlib

        server, tool = "server", "t" * 80
        digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:8]
        name = pi_mcp_tool_name(server, tool)
        assert len(name) == 64
        assert name == f"mcp__server__{'t' * (64 - len('mcp__server__') - 9)}_{digest}"

    def test_project_mcp_servers_are_kept_in_tool_selection(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"repo-docs": {"command": "x"}}}))
        runner = PiRunner(cwd=tmp_path, allowed_builtin_tools=[], mcp_servers=[McpServerSpec(name="apm", command="y")])
        assert runner._tool_entries() == ["mcp__*"]
        # Without any scenario server, project servers still keep MCP selected.
        assert PiRunner(cwd=tmp_path, allowed_builtin_tools=["read"])._tool_entries() == ["read", "mcp__*"]

    def test_codemode_only_when_allowed_explicitly(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="apm", command="y")])
        assert "codemode" not in runner._tool_entries()
        runner = PiRunner(
            cwd=tmp_path, allowed_builtin_tools=["codemode"], mcp_servers=[McpServerSpec(name="a", command="y")]
        )
        assert runner._tool_entries() == ["codemode", "mcp__*"]


def _pid_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestProcessLifecycle:
    def test_prompt_with_at_sign_goes_to_stdin(self, tmp_path, fake_pi):
        command, record = fake_pi(_events_tool_run())
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        _bounded(
            runner._run_agent(
                model=MODEL, system_prompt="S", user_prompt="@/etc/hosts", trace_headers={}, progress=None
            )
        )
        seen = json.loads(record.read_text())
        assert seen["stdin"] == "@/etc/hosts"
        assert all("/etc/hosts" not in arg for arg in seen["argv"])

    def test_exit_with_child_holding_stdout_does_not_hang(self, tmp_path, fake_pi, monkeypatch):
        import time

        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 0.5)
        events = [event for event in _events_tool_run() if event["type"] != "agent_settled"]
        command, record = fake_pi(events, FAKE_PI_CHILD_HOLDS_STDOUT="1")
        started = time.monotonic()
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert time.monotonic() - started < 10
        assert "before the run settled" in str(result.error)
        child_pid = int(Path(str(record) + ".child").read_text())
        for _ in range(50):
            if not _pid_alive(child_pid):
                break
            time.sleep(0.05)
        assert not _pid_alive(child_pid)

    def test_garbage_stdout_fails_the_run(self, tmp_path, fake_pi):
        command, _ = fake_pi(_events_tool_run(), FAKE_PI_GARBAGE="1")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "malformed JSONL" in str(result.error)

    def test_cancellation_stops_pi_and_removes_agent_dir(self, tmp_path, fake_pi, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 0.5)
        command, record = fake_pi([{"type": "turn_start"}], FAKE_PI_HANG="1")
        runner = PiRunner(cwd=tmp_path, pi_command=command)

        async def scenario():
            task = asyncio.create_task(
                runner._run_agent(model=MODEL, system_prompt="S", user_prompt="Q", trace_headers={}, progress=None)
            )
            for _ in range(100):
                await asyncio.sleep(0.05)
                if record.exists():
                    break
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        seen = json.loads(record.read_text())
        assert not Path(seen["agent_dir"]).exists()


class TestPiSpans:
    """The pi path makes the same agent -> llm + tool spans as Codex."""

    def test_spans_and_annotations(self, tmp_path, fake_pi, monkeypatch):
        import contextlib

        import airedale.harness.base as base

        class _FakeLLMObs:
            enabled = True

            def __init__(self) -> None:
                self.annotations: list[dict[str, Any]] = []
                self.tool_spans: list[str] = []

            def annotate(self, **kwargs: Any) -> None:
                self.annotations.append(kwargs)

            @contextlib.contextmanager
            def tool(self, *, name: str):
                self.tool_spans.append(name)
                self.tool_parents.append(active[-1] if active else None)
                yield object()

        _FakeLLMObs.tool_parents = []
        fake = _FakeLLMObs()
        spans: list[tuple[str, dict[str, Any]]] = []
        active: list[str] = []
        parents: dict[str, str | None] = {}

        def _decorator(kind):
            def factory(**kwargs):
                def wrap(func):
                    async def inner(*args, **kw):
                        spans.append((kind, kwargs))
                        parents[kind] = active[-1] if active else None
                        active.append(kind)
                        try:
                            return await func(*args, **kw)
                        finally:
                            active.pop()

                    return inner

                return wrap

            return factory

        monkeypatch.setattr(base, "LLMObs", fake)
        monkeypatch.setattr(base, "agent", _decorator("agent"))
        monkeypatch.setattr(base, "llm", _decorator("llm"))

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        monkeypatch.setattr(base, "current_trace_headers", lambda: {"traceparent": "00-t"})

        command, record = fake_pi(_events_tool_run())
        runner = PiRunner(
            cwd=tmp_path,
            pi_command=command,
            mcp_servers=[McpServerSpec(name="apm", url="http://x/mcp", tool_names=("s",))],
        )
        result = _bounded(
            runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s")
        )
        assert result.answer == "The directory is empty."
        assert [kind for kind, _ in spans] == ["agent", "llm"]
        assert spans[1][1]["model_provider"] == "anthropic"
        assert spans[1][1]["model_name"] == "claude-sonnet-4-6"
        assert fake.tool_spans == ["ls"]
        # Topology: agent -> llm, and tool spans are children of the agent span
        # (siblings of the llm span), like the native Claude integration.
        assert parents == {"agent": None, "llm": "agent"}
        assert fake.tool_parents == ["agent"]

        llm_annotation = next(a for a in fake.annotations if "tool_definitions" in a)
        assert llm_annotation["metadata"]["sdk"] == "pi"
        assert llm_annotation["metrics"]["output_tokens"] == 20
        names = [definition["name"] for definition in llm_annotation["tool_definitions"]]
        assert names == [*PI_BUILTIN_TOOLS, "mcp__apm__s"]
        assert llm_annotation["input_data"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
        assert llm_annotation["output_data"][-1] == {"role": "assistant", "content": "The directory is empty."}

        agent_annotation = next(a for a in fake.annotations if "_dd" in a.get("metadata", {}))
        manifest = agent_annotation["metadata"]["_dd"]["agent_manifest"]
        assert manifest["framework"] == "pi"
        assert {"name": "mcp__apm__s"} in manifest["tools"]
        # Trace headers reach the MCP server config pi receives.
        seen = json.loads(record.read_text())
        assert json.loads(seen["files"]["mcp.json"])["mcpServers"]["apm"]["headers"] == {"traceparent": "00-t"}


def test_finish_cleanup_completes_and_keeps_cancellation():
    from airedale.harness.pi import _finish_cleanup

    done: list[bool] = []

    async def cleanup():
        await asyncio.sleep(0.05)
        done.append(True)

    async def scenario():
        caller = asyncio.create_task(_finish_cleanup(cleanup()))
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.sleep(0)
        caller.cancel()  # A second cancellation during cleanup.
        with pytest.raises(asyncio.CancelledError):
            await caller

    _bounded(scenario())
    assert done == [True]
