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
import math
import os
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
from airedale.types import ModelSpec, UsageMetrics, is_valid_pi_usage

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

    @pytest.mark.parametrize("digits", [400, 5000])
    def test_collector_does_not_raise_for_a_cost_above_the_float_range(self, digits):
        """A record through ``feed_line`` (the path of the real run): damaged usage, no exception, no cost of 0."""
        events = _events_tool_run()
        line_of = lambda event: json.dumps(event).encode() + b"\n"  # noqa: E731
        collector = _PiEventCollector()
        for event in events:
            line = line_of(event)
            if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                line = line.replace(b'"total": 0.005', b'"total": 1' + b"0" * digits)
            collector.feed_line(line)  # Must not raise.
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        # (400 digits: a record that is damaged usage; 5000 digits: Python cannot even parse it, a malformed record.)
        assert "damaged token usage" in str(result.error) or "malformed JSONL" in str(result.error)
        # The record is not counted: the cost is not replaced by 0 for a call that holds a cost.
        assert collector.usage.estimated_cost_usd == pytest.approx(0.001)

    @pytest.mark.parametrize(
        ("usage", "valid"),
        [
            ({"input": 1, "output": 10, "reasoning": 10}, True),  # All the output is thinking: a subset, equal.
            ({"input": 1, "output": 10, "reasoning": 4}, True),
            ({"input": 1, "output": 10, "reasoning": 0}, True),
            ({"input": 1, "output": 10}, True),  # No reasoning field (Bedrock and others).
            ({"input": 1, "output": 0, "reasoning": 0}, True),
            ({"input": 1, "output": 10, "reasoning": 11}, False),
            ({"input": 1, "output": 1, "reasoning": 100}, False),
        ],
    )
    def test_reasoning_tokens_are_a_subset_of_output_tokens(self, usage, valid):
        # Anthropic (thinking_tokens), OpenAI completions and responses (reasoning_tokens inside completion_tokens /
        # output_tokens) and Google (thoughts added to output) all give reasoning <= output.
        usage = {**usage, "cost": {"total": 0.0}}
        assert is_valid_pi_usage(usage) is valid

    def test_google_shape_of_pi_has_reasoning_inside_output(self):
        usage = {"input": 10, "output": 8, "reasoning": 3, "totalTokens": 18, "cost": {"total": 0.0}}
        assert is_valid_pi_usage(usage)

    def test_cost_above_the_float_range_is_damaged_and_does_not_raise(self):
        usage = {"input": 1, "output": 1, "cost": {"total": 10**400}}
        assert not is_valid_pi_usage(usage)
        assert UsageMetrics.from_pi(usage).estimated_cost_usd == 0.0  # No exception for a record that is not checked.

    def test_huge_integers_do_not_raise(self):
        # An integer above the range of a float: it is finite, and it must not abort the run with OverflowError.
        usage = {"input": 10**400, "output": 1, "cost": {"total": 0}}
        assert is_valid_pi_usage(usage)
        assert not is_valid_pi_usage({**usage, "cost": {"total": -(10**400)}})
        assert not is_valid_pi_usage({"input": -(10**400), "output": 1, "cost": {"total": 0}})
        assert UsageMetrics.from_pi(usage).output_tokens == 1

    @pytest.mark.parametrize(
        ("label", "usage", "total"),
        [
            # The example of the review: the total says 1, the parts are 115.
            ("inconsistent", {"input": 100, "output": 10, "cacheRead": 5, "totalTokens": 1}, 115),
            # Anthropic: pi writes input + output + cacheRead + cacheWrite.
            ("anthropic", {"input": 10, "output": 5, "cacheRead": 100, "cacheWrite": 20, "totalTokens": 135}, 135),
            # OpenAI Responses: pi writes the total of the provider (input and output, cache inside the input).
            ("openai responses", {"input": 4, "output": 5, "cacheRead": 6, "totalTokens": 15}, 15),
            # Bedrock: pi writes the total of the provider, or input + output (no cache).
            ("bedrock", {"input": 10, "output": 5, "cacheRead": 100, "totalTokens": 15}, 115),
            # Google, the shape pi writes: output is candidates (5) + thinking (3), reasoning is the 3 thinking tokens
            # (a subset of output), and the total of the provider is the sum.
            ("google", {"input": 10, "output": 8, "reasoning": 3, "totalTokens": 18}, 18),
            # Google, a provider total that is more than the parts (for example tool-use prompt tokens): synthetic.
            (
                "google, provider total above the parts",
                {"input": 10, "output": 8, "reasoning": 3, "totalTokens": 25},
                18,
            ),
            ("no total", {"input": 10, "output": 5}, 15),
            ("total zero", {"input": 10, "output": 5, "totalTokens": 0}, 15),
            ("total too high", {"input": 1, "output": 1, "totalTokens": 1_000_000}, 2),
        ],
    )
    def test_total_is_the_sum_of_the_parts(self, label, usage, total):
        usage = {**usage, "cost": {"total": 0.0}}
        assert UsageMetrics.from_pi(usage).total_tokens == total, label
        assert is_valid_pi_usage(usage)  # A different total is not an error.

    def test_run_with_a_different_total_succeeds_with_the_sum(self):
        events = _events_tool_run()
        events[4]["message"]["usage"]["totalTokens"] = 1
        collector = _collect(events)
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert result.usage.total_tokens == 3 + 8 + 1447 + 1 + 12 + 1447 + 90  # the parts of both calls

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
        assert "another model than anthropic/claude-sonnet-4-6" in str(result.error)
        assert "claude-sonnet-4-5" not in str(result.error)  # What pi ran is a private note, not part of the error.

    def test_failed_final_message_is_an_error(self):
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "error"
        events[9]["message"]["errorMessage"] = "529 overloaded"
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert "provider error (stopReason=error)" in str(result.error)
        assert "529 overloaded" not in str(result.error)

    def test_non_zero_exit_does_not_include_stderr(self):
        result = _collect(_events_tool_run()).result(model=MODEL, returncode=2, stderr_tail="bad flag")
        assert "status 2" in str(result.error)
        assert "bad flag" not in str(result.error)

    def test_no_assistant_message(self):
        result = _collect([{"type": "agent_settled"}]).result(model=MODEL, returncode=0, stderr_tail="")
        assert "no assistant message" in str(result.error)
        assert result.answer == "pi did not produce a final response."

    @pytest.mark.parametrize(
        "content",
        [
            [],
            [{"type": "thinking", "thinking": "hmm"}],
            [{"type": "text", "text": ""}],
            [{"type": "text", "text": "  "}],
        ],
    )
    def test_valid_empty_final_response_is_kept_empty(self, content):
        # A response with no text is valid: the answer is empty, it is not the diagnostic text.
        events = _events_tool_run()
        events[-4]["message"]["content"] = content
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert result.answer.strip() == ""
        assert "did not produce" not in result.answer

    def test_max_turns_requests_stop(self):
        collector = _collect(_events_tool_run(), max_turns=1)
        assert collector.stop_requested
        assert "max_turns=1" in str(collector.result(model=MODEL, returncode=-15, stderr_tail="").error)

    def test_compaction_usage_is_counted(self):
        events = [*_events_tool_run()[:-1], {"type": "compaction_end", "result": {"usage": _usage(100, 10)}}]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.usage.output_tokens == 30

    def test_usage_entries_are_counted(self):
        # Prompt-cache warming: model work outside the messages.
        warm = {"type": "usage", "id": "u1", "kind": "cache_warm", "usage": _usage(0, 0, cache_read=500, cost=0.002)}
        events = [
            *_events_tool_run()[:-1],
            {"type": "entry_appended", "entry": warm},
            # Other entries (for example extension "custom" entries) are not usage.
            {"type": "entry_appended", "entry": {"type": "custom", "usage": _usage(1000, 1000)}},
            _events_tool_run()[-1],
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert result.usage.cache_read_input_tokens == 1447 + 500
        assert result.usage.output_tokens == 20
        assert result.usage.estimated_cost_usd == pytest.approx(0.008)

    @pytest.mark.parametrize("entry", [{"type": "usage"}, {"type": "usage", "kind": "cache_warm"}])
    def test_usage_entry_without_usage_fails_the_run(self, entry):
        # A cut record, or a changed schema: the accounting is incomplete.
        events = [*_events_tool_run()[:-1], {"type": "entry_appended", "entry": entry}, _events_tool_run()[-1]]
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "1 usage entry" in str(error)

    @pytest.mark.parametrize("content", ["missing", None, 5, {}, True, "text", ""])
    def test_assistant_message_without_a_content_list_fails_the_run(self, content):
        # content [] is a valid empty response; null, no field or another type is damage.
        events = _events_tool_run()
        if content == "missing":
            del events[-4]["message"]["content"]
        else:
            events[-4]["message"]["content"] = content
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "without a content list" in str(error)

    @pytest.mark.parametrize(
        "block",
        [
            "answer",
            None,
            5,
            ["text"],
            {},
            {"type": "text"},
            {"type": "text", "text": None},
            {"type": "text", "text": {"a": 1}},
            {"type": "text", "text": 5},
            {"type": "text", "text": ["x"]},
            {"type": "thinking"},
            {"type": "thinking", "thinking": {"a": 1}},
            {"type": "toolCall", "name": "ls", "arguments": {}},
            {"type": "toolCall", "id": 5, "name": "ls", "arguments": {}},
            {"type": "toolCall", "id": "x", "arguments": {}},
            {"type": "toolCall", "id": "x", "name": ["ls"], "arguments": {}},
            {"type": "toolCall", "id": "x", "name": "ls"},
            {"type": "toolCall", "id": "x", "name": "ls", "arguments": "{}"},
            {"type": "toolCall", "id": "x", "name": "ls", "arguments": [1]},
            {"type": "image", "data": "x"},
            {"type": "unknown", "text": "x"},
            {"type": 5, "text": "x"},
        ],
        ids=repr,
    )
    def test_damaged_block_in_the_final_response_fails_the_run(self, block):
        # The answer would be made without the block (or from a stringified object) and scored.
        events = _events_tool_run()
        events[-4]["message"]["content"] = [{"type": "text", "text": "The directory is empty."}, block]
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "damaged block" in str(error)

    def test_valid_blocks_of_every_kind_are_accepted(self):
        events = _events_tool_run()
        events[-4]["message"]["content"] = [
            {"type": "thinking", "thinking": "hmm", "thinkingSignature": "sig"},
            {"type": "text", "text": "The directory is empty.", "textSignature": "sig"},
            {"type": "toolCall", "id": "", "name": "ls", "arguments": {}, "thoughtSignature": "sig"},
            {"type": "text", "text": ""},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None

    def test_blocks_of_the_recorded_events_of_pi_agent_loop_are_valid(self, pi_reference):
        for action in ("retry", "parallel"):
            for version, events in pi_reference({"action": action}).items():
                collector = _collect(events)
                assert collector.calls_without_content == 0, (action, version)
                assert collector.assistant_messages, (action, version)

    def test_content_of_a_tool_call_message_must_be_a_list_too(self):
        events = _events_tool_run()
        events[4]["message"]["content"] = None
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "without a content list" in str(error)

    @pytest.mark.parametrize("case", ["project replaces a server", "tool names collide"])
    def test_refused_run_does_not_probe_the_mcp_servers(self, tmp_path, monkeypatch, fake_span_llmobs, case):
        """With LLMObs on, the metadata probe starts stdio servers: the local checks come first."""
        import airedale.harness.base as base

        probed: list[object] = []

        async def _catalog(servers, headers):
            probed.append(servers)
            return {}

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _catalog)
        if case == "project replaces a server":
            (tmp_path / ".pi").mkdir()
            (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"same": {"command": "project"}}}))
            servers = [McpServerSpec(name="same", command="scenario")]
        else:
            servers = [
                McpServerSpec(name="a", command="x", tool_names=("b__c",)),
                McpServerSpec(name="a__b", command="y", tool_names=("c",)),
            ]
        runner = PiRunner(cwd=tmp_path, mcp_servers=servers)
        with pytest.raises(ValueError, match=r"same|Possible"):
            _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert probed == []

    @pytest.mark.parametrize(
        ("server", "message"),
        [
            (McpServerSpec(name="a.b", command="x"), "not valid for pi"),
            (McpServerSpec(name="ok", command="x", tool_names=("get_*",)), "must be exact tool names"),
            (McpServerSpec(name="ok", url="ftp://example.test/mcp"), "not valid for pi"),
            (McpServerSpec(name="ok", type="sse", url="https://example.test/mcp"), "unsupported type|not valid for pi"),
            (McpServerSpec(name="ok", type="stdio"), "no 'command'|not valid for pi"),
            # Pi rejects these hosts; the URL reader cannot decide them exactly, so the strict check refuses them.
            (McpServerSpec(name="ok", url="http://[not-an-ip]/mcp"), "cannot be checked exactly"),
            (McpServerSpec(name="ok", url="http://[1::2::3]/mcp"), "cannot be checked exactly"),
            (McpServerSpec(name="ok", url="https://\u00ad127.1/mcp"), "cannot be checked exactly"),
            (McpServerSpec(name="ok", url="http://exämple.test/mcp"), "cannot be checked exactly"),
        ],
        ids=[
            "name",
            "wildcard",
            "url scheme",
            "sse",
            "no command",
            "not an ip",
            "double colon",
            "soft hyphen host",
            "idna host",
        ],
    )
    def test_scenario_server_that_pi_rejects_is_refused_before_the_probe(
        self, tmp_path, monkeypatch, fake_span_llmobs, server, message
    ):
        """An invalid scenario server turned into an empty config and the probe started or called it (LLMObs on)."""
        import airedale.harness.base as base

        probed: list[object] = []

        async def _catalog(servers, headers):
            probed.append(servers)
            return {}

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _catalog)
        runner = PiRunner(cwd=tmp_path, mcp_servers=[server])
        with pytest.raises(ValueError, match=message):
            _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert probed == []

    @pytest.mark.parametrize(
        ("servers", "skills", "message"),
        [
            ([McpServerSpec(name="bad.name", command="DO_NOT_START")], False, "not valid for pi"),
            ([McpServerSpec(name="docs", command="DO_NOT_START", tool_names=("*",))], False, "exact tool names"),
            ([McpServerSpec(name="docs", command="DO_NOT_START")], True, "collides"),
        ],
        ids=["invalid server name", "wildcard tool name", "skill name collision"],
    )
    def test_every_local_refusal_comes_before_the_probe(
        self, tmp_path, monkeypatch, fake_span_llmobs, servers, skills, message
    ):
        """The three refusals of the review, LLMObs on: the probe never starts a server."""
        import airedale.harness.base as base

        probed: list[object] = []

        async def _catalog(servers, headers):
            probed.append(servers)
            return {}

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _catalog)
        scenario_skills = []
        if skills:
            scenario = tmp_path / "scenario"
            scenario.mkdir()
            (scenario / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n")
            project = tmp_path / ".pi" / "skills" / "project"
            project.mkdir(parents=True)
            (project / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\n")
            scenario_skills = [str(scenario)]
        runner = PiRunner(
            cwd=tmp_path,
            skills=scenario_skills,
            mcp_servers=servers,
            pi_command=[sys.executable, "-c", "raise AssertionError('Unexpected subprocess')"],
        )
        with pytest.raises(ValueError, match=message):
            _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert probed == []

    def test_accepted_run_probes_the_mcp_servers(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        probed: list[object] = []

        async def _catalog(servers, headers):
            probed.append(servers)
            return {}

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _catalog)
        command, _ = fake_pi(_events_tool_run())
        runner = PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=[McpServerSpec(name="docs", command="x")])
        _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert len(probed) == 1

    def test_unused_auth_json_credential_with_an_unpaired_surrogate_does_not_crash_a_good_run(
        self, tmp_path, fake_pi, monkeypatch
    ):
        """json.loads gives a str with a lone surrogate for "\\ud800": .encode() raised and crashed a run that succeeded."""
        operator = tmp_path / "operator"
        operator.mkdir()
        (operator / "auth.json").write_text('{"openai": {"type": "api_key", "key": "sk-lone-\\ud800-surrogate-0123"}}')
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(operator))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        command, _ = fake_pi(_events_tool_run())
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        result = _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert result.answer == "The directory is empty."

    def test_empty_final_response_through_a_run(self, tmp_path, fake_pi):
        events = _events_tool_run()
        events[-4]["message"]["content"] = []
        command, _ = fake_pi(events)
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        result = _bounded(
            runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s")
        )
        assert result.answer == ""

    def test_all_zero_usage_is_not_a_missing_usage_error(self, tmp_path, fake_pi):
        # Every record is valid and zero (no tokens, or a price of zero): that is data, not missing usage.
        events = _events_tool_run()
        for event in events:
            if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                event["message"]["usage"] = _usage(0, 0)
        command, _ = fake_pi(events)
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        result = _bounded(
            runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s")
        )
        assert result.answer == "The directory is empty."
        assert result.usage.total_tokens == 0

    def test_nested_tool_calls_follow_their_parent(self):
        events = _events_tool_run()
        nested = [
            {
                "type": "tool_execution_start",
                "toolCallId": "call_1/1",
                "toolName": "mcp__apm__search",
                "args": {"q": "x"},
                "parentToolCallId": "call_1",
            },
            {
                "type": "tool_execution_start",
                "toolCallId": "call_1/1/1",
                "toolName": "read",
                "args": {"path": "a"},
                "parentToolCallId": "call_1/1",
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "call_1/1/1",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": "A"}]},
                "isError": False,
                "parentToolCallId": "call_1/1",
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "call_1/1",
                "toolName": "mcp__apm__search",
                "result": {"content": [{"type": "text", "text": "boom"}]},
                "isError": True,
                "durationMs": 7,
                "parentToolCallId": "call_1",
            },
            # Started but never ended; and a call whose parent is not known.
            {"type": "tool_execution_start", "toolCallId": "call_1/2", "toolName": "ls", "parentToolCallId": "call_1"},
            {"type": "tool_execution_start", "toolCallId": "zz/1", "toolName": "grep", "parentToolCallId": "zz"},
        ]
        events[6:6] = nested
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.name for call in result.tool_calls] == ["ls", "mcp__apm__search", "read", "ls", "grep"]
        model_call, search, read, unfinished, orphan = result.tool_calls
        assert model_call.metadata == {"source": "pi", "tool_id": "call_1", "duration_ms": 2}
        assert search.arguments == {"q": "x"}
        assert search.error == "boom"
        assert search.metadata == {
            "source": "pi",
            "tool_id": "call_1/1",
            "duration_ms": 7,
            "parent_tool_id": "call_1",
        }
        assert read.result == "A"
        assert read.error is None
        assert read.metadata["parent_tool_id"] == "call_1/1"
        assert unfinished.error == "tool did not finish"
        assert unfinished.arguments == {}
        assert orphan.metadata["parent_tool_id"] == "zz"
        # The segments show what the model asked for: the nested calls are not in them.
        assert [s for s in result.segments if isinstance(s, AgentToolCall)] == [model_call]

    def test_parallel_nested_calls_keep_start_order(self):
        # B starts before A1 (a child of A): the flat list is A, B, A1.
        events = _events_tool_run()

        def start(tool_id: str, parent: str) -> dict[str, Any]:
            return {
                "type": "tool_execution_start",
                "toolCallId": tool_id,
                "toolName": tool_id,
                "parentToolCallId": parent,
            }

        events[6:6] = [start("A", "call_1"), start("B", "call_1"), start("A1", "A"), start("B1", "B")]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "A", "B", "A1", "B1"]

    def test_nested_call_that_started_before_its_parent_comes_after_it(self):
        # The start record of the child came first (damaged order): the parent
        # still comes before it.
        events = _events_tool_run()
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "kid", "toolName": "kid", "parentToolCallId": "mid"},
            {"type": "tool_execution_start", "toolCallId": "mid", "toolName": "mid", "parentToolCallId": "call_1"},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "mid", "kid"]

    def test_deep_nested_tool_calls(self):
        # Pi has no nesting limit; a chain deeper than the recursion limit works.
        depth = sys.getrecursionlimit() + 500
        events = _events_tool_run()
        chain = []
        parent = "call_1"
        for level in range(depth):
            tool_id = f"{parent}/1"
            chain.append(
                {
                    "type": "tool_execution_start",
                    "toolCallId": tool_id,
                    "toolName": f"t{level}",
                    "parentToolCallId": parent,
                }
            )
            parent = tool_id
        # An orphan chain too.
        chain.append({"type": "tool_execution_start", "toolCallId": "o/1", "toolName": "o0", "parentToolCallId": "o"})
        chain.append(
            {"type": "tool_execution_start", "toolCallId": "o/1/1", "toolName": "o1", "parentToolCallId": "o/1"}
        )
        events[6:6] = chain
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        names = [call.name for call in result.tool_calls]
        assert names == ["ls", *(f"t{level}" for level in range(depth)), "o0", "o1"]

    def test_nested_calls_of_a_model_call_with_an_empty_id(self):
        events = _events_tool_run()
        events[4]["message"]["content"][-1]["id"] = ""
        events[5]["toolCallId"] = ""
        events[6]["toolCallId"] = ""
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "read", "parentToolCallId": ""},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "/1",
                "toolName": "read",
                "parentToolCallId": "",
                "result": {},
            },
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "read"]
        assert result.tool_calls[1].metadata["parent_tool_id"] == ""
        assert result.tool_calls[1].error is None

    def test_reused_empty_id_keeps_each_execution(self):
        # Two sequential calls use the id "": each call gets its own result.
        events = _events_tool_run()
        events[4]["message"]["content"].append({"type": "toolCall", "id": "", "name": "read", "arguments": {"p": 2}})
        events[4]["message"]["content"][2]["id"] = ""

        def end(name: str, text: str, ms: int, **extra: Any) -> dict[str, Any]:
            return {
                "type": "tool_execution_end",
                "toolCallId": "",
                "toolName": name,
                "result": {"content": [{"type": "text", "text": text}]},
                "isError": False,
                "durationMs": ms,
                **extra,
            }

        events[5:7] = [
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "ls", "args": {"path": "."}},
            end("ls", "first", 1),
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "read", "args": {"p": 2}},
            # A nested call of the second "" call, and one of the first.
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "grep", "parentToolCallId": ""},
            end("grep", "nested", 3, toolCallId="/1", parentToolCallId=""),
            end("read", "second", 2),
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.name for call in result.tool_calls] == ["ls", "read", "grep"]
        first, second, nested = result.tool_calls
        assert (first.result, first.metadata["duration_ms"]) == ("first", 1)
        assert (second.result, second.metadata["duration_ms"]) == ("second", 2)
        assert (nested.result, nested.metadata["parent_tool_id"]) == ("nested", "")

    def test_reused_empty_id_keeps_the_nested_calls_of_the_first_execution(self):
        # A nested call that came while the first "" call ran stays with it,
        # even when a second "" call starts later.
        events = _events_tool_run()
        events[4]["message"]["content"].append({"type": "toolCall", "id": "", "name": "second", "arguments": {}})
        events[4]["message"]["content"][2].update({"id": "", "name": "first"})
        events[5:7] = [
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "first"},
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "child1", "parentToolCallId": ""},
            {"type": "tool_execution_end", "isError": False, "toolCallId": "", "toolName": "first", "result": {}},
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "second"},
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "child2", "parentToolCallId": ""},
            {"type": "tool_execution_end", "isError": False, "toolCallId": "", "toolName": "second", "result": {}},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        # Each nested call follows the execution that it started under.
        assert [call.name for call in result.tool_calls] == ["first", "child1", "second", "child2"]

    def _parallel_calls(self, ids, ends):
        """A turn with parallel calls: all starts, then the ends in the order of ``ends``."""
        events = _events_tool_run()
        message = events[4]["message"]
        message["content"] = [
            {"type": "toolCall", "id": tool_id, "name": f"t{index}", "arguments": {}}
            for index, tool_id in enumerate(ids)
        ]
        starts = [
            {"type": "tool_execution_start", "toolCallId": tool_id, "toolName": f"t{index}", "args": {}}
            for index, tool_id in enumerate(ids)
        ]
        finishes = [
            {
                "type": "tool_execution_end",
                "toolCallId": ids[index],
                "toolName": f"t{index}",
                "result": {"content": [{"type": "text", "text": f"result {index}"}]},
                "isError": False,
            }
            for index in ends
        ]
        events[5:7] = [*starts, *finishes]
        return events

    @pytest.mark.parametrize("ids", [["", ""], ["", "", ""], ["a", "a"]])
    def test_parallel_calls_with_the_same_id_fail_the_run(self, ids):
        # Pi ends parallel calls as they finish, so the end records of calls with one id can not be matched.
        events = self._parallel_calls(ids, ends=list(reversed(range(len(ids)))))
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "cannot be matched" in str(error)
        assert f"{len(ids)} tool calls" in str(error)

    def test_parallel_calls_with_distinct_ids_finish_in_any_order(self):
        events = self._parallel_calls(["a", "b", "c"], ends=[2, 0, 1])
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.result for call in result.tool_calls] == ["result 0", "result 1", "result 2"]

    def test_calls_with_the_same_id_one_after_the_other_keep_their_results(self):
        events = _events_tool_run()
        events[4]["message"]["content"] = [
            {"type": "toolCall", "id": "", "name": "t0", "arguments": {}},
            {"type": "toolCall", "id": "", "name": "t1", "arguments": {}},
        ]
        events[5:7] = [
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "t0"},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "",
                "toolName": "t0",
                "result": {"content": [{"type": "text", "text": "r0"}]},
            },
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "t1"},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "",
                "toolName": "t1",
                "result": {"content": [{"type": "text", "text": "r1"}]},
            },
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.result for call in result.tool_calls] == ["r0", "r1"]

    def test_parallel_events_of_pi_agent_loop(self, pi_reference):
        """Pi's real agent loop: parallel calls with the id "" end in the order of completion."""
        for version, events in pi_reference({"action": "parallel"}).items():
            error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
            assert error is not None and "cannot be matched" in str(error), version

    def test_nested_execution_is_not_the_execution_of_a_model_call(self):
        # A damaged stream: call_2 is a model call, and the only execution with
        # this id says that call_1 made it. It is a nested call, so the model
        # call did not run, and the nested call is not lost.
        events = _events_tool_run()
        events[4]["message"]["content"].append({"type": "toolCall", "id": "call_2", "name": "bash", "arguments": {}})
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "call_2", "toolName": "bash", "parentToolCallId": "call_1"},
            {
                "type": "tool_execution_start",
                "toolCallId": "call_2/1",
                "toolName": "read",
                "parentToolCallId": "call_2",
            },
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "bash", "read", "bash"]
        model_call, nested, child, unrun = result.tool_calls
        assert model_call.error is None
        assert nested.metadata["parent_tool_id"] == "call_1"
        assert child.metadata["parent_tool_id"] == "call_2"
        assert "parent_tool_id" not in unrun.metadata
        assert unrun.error == "tool did not run"

    def test_nested_execution_before_a_model_call_with_the_same_id(self):
        # A nested call with id "" comes before the model call with id "": the
        # model call gets its own execution, not the nested one.
        events = _events_tool_run()
        events[4]["message"]["content"][2]["id"] = ""
        events[5:7] = [
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "inner", "parentToolCallId": "outer"},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "",
                "toolName": "inner",
                "parentToolCallId": "outer",
                "result": {"content": [{"type": "text", "text": "nested"}]},
                "durationMs": 9,
            },
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "ls", "args": {"path": "."}},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "",
                "toolName": "ls",
                "result": {"content": [{"type": "text", "text": "top"}]},
                "durationMs": 2,
            },
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.name for call in result.tool_calls] == ["ls", "inner"]
        model_call, nested = result.tool_calls
        assert (model_call.result, model_call.metadata["duration_ms"]) == ("top", 2)
        assert (nested.result, nested.metadata["parent_tool_id"]) == ("nested", "outer")

    def test_orphan_chain_starts_with_its_top_call(self):
        # ``kid`` started before ``mid``, but its parent is ``mid`` (which has a
        # parent that never started): ``mid`` comes first.
        events = _events_tool_run()
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "kid", "toolName": "kid", "parentToolCallId": "mid"},
            {"type": "tool_execution_start", "toolCallId": "mid", "toolName": "mid", "parentToolCallId": "gone"},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "mid", "kid"]

    def test_chain_under_a_call_without_transcript_block_starts_with_its_top_call(self):
        # ``top`` is a call that is not in the transcript, so it is not listed.
        # Its descendants are, and each one comes after its own parent.
        events = _events_tool_run()
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "b", "toolName": "b", "parentToolCallId": "a"},
            {"type": "tool_execution_start", "toolCallId": "top", "toolName": "top"},
            {"type": "tool_execution_start", "toolCallId": "a", "toolName": "a", "parentToolCallId": "top"},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert [call.name for call in result.tool_calls] == ["ls", "a", "b"]

    @pytest.mark.parametrize("stop", ["error", "aborted"])
    def test_failed_response_does_not_take_the_execution_of_a_retry(self, stop):
        # A response that failed has a tool call with id "" and no execution. The
        # retry uses the id "" again: its execution belongs to the retry.
        failed = _assistant(
            [{"type": "toolCall", "id": "", "name": "failed", "arguments": {}}],
            usage=_usage(0, 0),
            stop=stop,
            errorMessage="overloaded",
        )
        retry = _assistant(
            [{"type": "toolCall", "id": "", "name": "worked", "arguments": {}}], usage=_usage(1, 1), stop="toolUse"
        )
        events = _events_tool_run()
        events[4:7] = [
            {"type": "message_end", "message": failed},
            {"type": "message_end", "message": retry},
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "worked", "args": {}},
            {
                "type": "tool_execution_end",
                "toolCallId": "",
                "toolName": "worked",
                "result": {"content": [{"type": "text", "text": "worked result"}]},
                "isError": False,
            },
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        failed_call, worked = result.tool_calls
        assert (failed_call.name, failed_call.result, failed_call.error) == ("failed", None, "tool did not run")
        assert (worked.name, worked.result, worked.error) == ("worked", "worked result", None)
        # The usage of the failed response is counted (it is data), the run is fine.
        assert result.error is None

    def test_retry_events_of_pi_agent_loop(self, pi_reference):
        """The events of pi's real agent loop: the failed call has no execution."""
        for version, events in pi_reference({"action": "retry"}).items():
            result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
            assert [(call.name, call.result, call.error) for call in result.tool_calls] == [
                ("failed", None, "tool did not run"),
                ("worked", "worked result", None),
            ], version

    def test_response_cut_at_its_length_keeps_its_executions(self):
        # Pi can run tools for a response with stopReason "length".
        events = _events_tool_run()
        events[4]["message"]["stopReason"] = "length"
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.tool_calls[0].result == "(empty directory)"

    def test_nested_tool_call_id_cycle_does_not_loop(self):
        events = _events_tool_run()
        events[6:6] = [
            {"type": "tool_execution_start", "toolCallId": "a", "toolName": "x", "parentToolCallId": "b"},
            {"type": "tool_execution_start", "toolCallId": "b", "toolName": "y", "parentToolCallId": "a"},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert sorted(call.name for call in result.tool_calls) == ["ls", "x", "y"]

    @pytest.mark.parametrize(
        "usage",
        [
            "missing",
            "n/a",
            True,
            [],
            {},
            {"output": 1},  # no input
            {"input": 1},  # no output
            {"input": None, "output": 1},
            {"input": "n/a", "output": 1},
            {"input": False, "output": 1},
            {"input": True, "output": 1},
            {"input": -1, "output": 1},
            {"input": 1.5, "output": 1},
            {"input": {}, "output": 1},
            {"input": [], "output": 1},
            {"input": 1, "output": 1, "cacheRead": None},
            {"input": 1, "output": 1, "cacheWrite": "x"},
            {"input": 1, "output": 1, "totalTokens": None},
            {"input": 1, "output": 1, "reasoning": []},
            {"input": 1, "output": 1, "reasoning": 100},  # More thinking tokens than output tokens.
            {"input": 1, "output": 0, "reasoning": 1},
            {"input": 1, "output": 1},  # no cost
            {"input": 1, "output": 1, "cost": {}},  # no cost.total
            {"input": 1, "output": 1, "cost": {"input": 0.1, "output": 0.2}},
            {"input": 1, "output": 1, "cost": 5},
            {"input": 1, "output": 1, "cost": {"total": "Infinity"}},
            {"input": 1, "output": 1, "cost": {"total": float("inf")}},
            {"input": 1, "output": 1, "cost": {"total": 10**400}},  # An integer that a float cannot hold.
            {"input": 1, "output": 1, "cost": {"total": -1}},
            {"input": 1, "output": 1, "cost": {"total": None}},
            {"input": float("nan"), "output": 1},
        ],
        ids=repr,
    )
    def test_run_fails_when_a_model_call_has_unusable_usage(self, usage):
        # The first call has damaged usage, the final call has usage: the
        # total is not zero, but the accounting is incomplete.
        events = _events_tool_run()
        if usage == "missing":
            del events[4]["message"]["usage"]
        else:
            events[4]["message"]["usage"] = usage
        collector = _PiEventCollector()
        for event in events:
            collector.feed(event)
        assert collector.usage.output_tokens > 0
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "damaged token usage (1 model call)" in str(error)

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "message_end", "message": {"role": "toolResult", "usage": {"input": "bad", "output": 1}}},
            {"type": "message_end", "message": {"role": "toolResult", "usage": None}},
            {"type": "compaction_end", "result": {"usage": {"input": 1, "output": None}}},
            {"type": "compaction_end", "result": {"usage": "bad"}},
            {"type": "entry_appended", "entry": {"type": "usage", "usage": {"input": 1, "output": 1, "cacheRead": []}}},
            {"type": "entry_appended", "entry": {"type": "usage", "usage": None}},
            {"type": "message_end", "message": {"role": "toolResult", "usage": {"input": 1, "output": 1}}},
            {"type": "compaction_end", "result": {"usage": {"input": 1, "output": 1, "cost": {}}}},
            {"type": "entry_appended", "entry": {"type": "usage", "usage": {"input": 1, "output": 1}}},
        ],
        ids=lambda event: json.dumps(event)[:60],
    )
    def test_run_fails_when_other_usage_is_damaged(self, event):
        events = _events_tool_run()
        events.insert(-1, event)
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "damaged token usage" in str(error)

    def test_compaction_without_usage_is_valid(self):
        events = _events_tool_run()
        events.insert(-1, {"type": "compaction_end", "result": {"summary": "s"}})
        assert _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_damaged_usage_is_not_counted(self):
        events = _events_tool_run()
        events.insert(-1, {"type": "compaction_end", "result": {"usage": _usage(1000, 1000) | {"cacheRead": "x"}}})
        collector = _collect(events)
        assert collector.usage.output_tokens == 20

    @pytest.mark.parametrize("stop", ["error", "aborted"])
    def test_complete_zero_usage_of_a_failed_call_is_valid(self, stop):
        events = _events_tool_run()
        failed = _assistant([{"type": "text", "text": ""}], usage=_usage(0, 0), stop=stop, errorMessage="x")
        events.insert(4, {"type": "message_end", "message": failed})
        assert _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_infinite_cost_does_not_reach_the_metrics(self):
        usage = UsageMetrics.from_pi({"input": 1, "output": 1, "cost": {"total": "Infinity"}})
        assert usage.estimated_cost_usd == 0.0
        json.dumps(usage.to_llmobs_metrics(), allow_nan=False)

    def test_zero_usage_record_is_not_missing_usage(self):
        # Pi writes zeros for a call that produced no tokens: that is data.
        events = _events_tool_run()
        events[4]["message"]["usage"] = _usage(0, 0)
        assert _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_invalid_utf8_in_a_record_fails_the_run(self):
        # A byte that is not UTF-8 inside a JSON string: it must not be replaced by U+FFFD and accepted.
        events = _events_tool_run()
        collector = _PiEventCollector()
        for event in events:
            line = (json.dumps(event, ensure_ascii=False) + "\n").encode()
            if event.get("type") == "message_end" and event["message"].get("role") == "user":
                line = line.replace(b'"hi"', b'"h\xffi"')
            collector.feed_line(line)
        assert collector.invalid_lines == 1
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "malformed JSONL" in str(error)

    @pytest.mark.parametrize("message", ["missing", None, "text", [], 5])
    def test_message_end_without_a_message_object_fails_the_run(self, message):
        # The record of a model call has no message: its usage is lost, a later valid call must not hide that.
        events = _events_tool_run()
        if message == "missing":
            del events[4]["message"]
        else:
            events[4]["message"] = message
        collector = _collect(events)
        assert collector.invalid_lines == 1
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "malformed JSONL" in str(error)

    @pytest.mark.parametrize("role", ["missing", None, 7, "", [], {}])
    def test_message_end_without_a_role_fails_the_run(self, role):
        # The message of a model call has no usable role: its usage would be lost (1,550 tokens instead of 3,008).
        events = _events_tool_run()
        if role == "missing":
            del events[4]["message"]["role"]
        else:
            events[4]["message"]["role"] = role
        collector = _collect(events)
        assert collector.invalid_lines == 1
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "malformed JSONL" in str(error)

    def test_message_end_with_an_empty_object_fails_the_run(self):
        events = _events_tool_run()
        events[4]["message"] = {}
        collector = _collect(events)
        assert collector.invalid_lines == 1
        assert collector.result(model=MODEL, returncode=0, stderr_tail="").error is not None

    @pytest.mark.parametrize("role", ["custom", "bashExecution", "branchSummary", "compactionSummary", "system"])
    def test_message_end_with_another_role_name_is_accepted(self, role):
        events = _events_tool_run()
        events.insert(5, {"type": "message_end", "message": {"role": role, "content": "x"}})
        collector = _collect(events)
        assert collector.invalid_lines == 0
        assert collector.result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_message_end_records_of_pi_agent_loop_all_have_a_message(self, pi_reference):
        """Pi's real loop (recorded): every message_end has a message object."""
        for version, events in pi_reference({"action": "parallel"}).items():
            ends = [event for event in events if event["type"] == "message_end"]
            assert ends and all(isinstance(event.get("message"), dict) for event in ends), version
            assert all(isinstance(event["message"].get("role"), str) and event["message"]["role"] for event in ends), (
                version
            )
            assert _collect(events).invalid_lines == 0, version

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_javascript_invalid_constants_in_a_record_fail_the_run(self, constant):
        # JSON.parse rejects these; Python accepted them and scored "nan" as the answer.
        events = _events_tool_run()
        collector = _PiEventCollector()
        for event in events:
            line = json.dumps(event).encode() + b"\n"
            if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                line = line.replace(b'"The directory is empty."', constant.encode()) if b"empty" in line else line
            collector.feed_line(line)
        assert collector.invalid_lines == 1
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None

    def test_text_block_with_a_constant_is_not_the_answer(self):
        line = (
            b'{"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": NaN}]}}\n'
        )
        collector = _PiEventCollector()
        collector.feed_line(line)
        assert collector.invalid_lines == 1
        assert collector.assistant_messages == []

    @pytest.mark.parametrize("field", ["input", "output", "cacheRead", "cacheWrite"])
    def test_count_of_400_digits_is_infinity_and_damaged(self, field):
        # 400 digits are legal JSON for Python, and Infinity for JSON.parse: the count is not a finite number.
        events = _events_tool_run()
        collector = _PiEventCollector()
        for event in events:
            if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                event["message"]["usage"][field] = None  # replaced in the text below
            line = json.dumps(event).encode() + b"\n"
            if event.get("type") == "message_end" and event["message"].get("role") == "assistant":
                line = line.replace(f'"{field}": null'.encode(), f'"{field}": 1'.encode() + b"0" * 400, 1)
            collector.feed_line(line)
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "damaged token usage" in str(error)

    @pytest.mark.parametrize("kind", ["tool_execution_start", "tool_execution_end"])
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("toolCallId", "missing"),
            ("toolCallId", None),
            ("toolCallId", 5),
            ("toolCallId", ["x"]),
            ("toolName", "missing"),
            ("toolName", None),
            ("toolName", 5),
            ("parentToolCallId", None),
            ("parentToolCallId", 5),
            ("parentToolCallId", {}),
        ],
    )
    def test_tool_event_with_a_field_of_the_wrong_type_is_malformed(self, kind, field, value):
        # null or a missing id used to become the empty id, and could close the execution of another call.
        events = _events_tool_run()
        index = 5 if kind == "tool_execution_start" else 6
        if value == "missing":
            del events[index][field]
        else:
            events[index][field] = value
        collector = _collect(events)
        assert collector.invalid_lines == 1
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "malformed JSONL" in str(error)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("isError", None),
            ("isError", "false"),
            ("isError", 0),
            ("durationMs", None),
            ("durationMs", "5"),
            ("durationMs", True),
            ("durationMs", -1),
            ("durationMs", float("inf")),
        ],
    )
    def test_tool_end_with_is_error_or_duration_of_the_wrong_type_is_malformed(self, field, value):
        events = _events_tool_run()
        events[6][field] = value
        collector = _collect(events)
        assert collector.invalid_lines == 1

    def test_tool_end_without_is_error_is_malformed_not_a_success(self):
        # A failed tool whose end record lost isError used to become a call with no error.
        events = _events_tool_run()
        del events[6]["isError"]
        events[6]["result"] = {"content": [{"type": "text", "text": "boom"}]}
        collector = _collect(events)
        assert collector.invalid_lines == 1
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        assert "malformed JSONL" in str(result.error)
        assert result.tool_calls[0].error == "tool did not run"  # Not a success.

    def test_failed_tool_is_reported_with_its_error(self):
        events = _events_tool_run()
        events[6]["isError"] = True
        events[6]["result"] = {"content": [{"type": "text", "text": "boom"}]}
        collector = _collect(events)
        assert collector.invalid_lines == 0
        assert collector.result(model=MODEL, returncode=0, stderr_tail="").tool_calls[0].error == "boom"

    def test_duration_is_optional_on_a_tool_end(self):
        events = _events_tool_run()
        events[6].pop("durationMs", None)
        assert _collect(events).invalid_lines == 0

    def test_every_tool_end_of_pi_agent_loop_has_a_boolean_is_error(self, pi_reference):
        for action in ("retry", "parallel"):
            for version, events in pi_reference({"action": action}).items():
                ends = [event for event in events if event["type"] == "tool_execution_end"]
                assert ends and all(isinstance(event.get("isError"), bool) for event in ends), (action, version)

    def test_end_with_a_null_id_does_not_close_an_empty_id_execution(self):
        # The execution with the empty id stays open: the bad record is ignored and counted.
        events = _events_tool_run()
        events[4]["message"]["content"][-1]["id"] = ""
        events[5]["toolCallId"] = ""
        events[6]["toolCallId"] = None
        collector = _collect(events)
        assert collector.invalid_lines == 1
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.tool_calls[0].error == "tool did not run"
        assert result.tool_calls[0].result is None

    def test_tool_events_of_pi_agent_loop_are_valid(self, pi_reference):
        for action in ("retry", "parallel"):
            for version, events in pi_reference({"action": action}).items():
                assert _collect(events).invalid_lines == 0, (action, version)

    def test_valid_tool_events_are_accepted(self):
        events = _events_tool_run()
        events[5]["parentToolCallId"] = "parent"
        events[6].update({"parentToolCallId": "parent", "isError": False, "durationMs": 0})
        assert _collect(events).invalid_lines == 0

    @pytest.mark.parametrize("aborted", ["missing", None, "false", "true", 0, 1, [], {}])
    def test_agent_settled_without_a_boolean_aborted_is_malformed(self, aborted):
        events = _events_tool_run()
        settled = events[-1]
        assert settled["type"] == "agent_settled"
        if aborted == "missing":
            del settled["aborted"]
        else:
            settled["aborted"] = aborted
        collector = _collect(events)
        assert collector.invalid_lines == 1
        assert not collector.settled  # The damaged record does not settle the run.
        error = collector.result(model=MODEL, returncode=0, stderr_tail="").error
        assert error is not None
        assert "malformed JSONL" in str(error) or "before the run settled" in str(error)

    @pytest.mark.parametrize("aborted", [True, False])
    def test_agent_settled_with_a_boolean_aborted_is_read(self, aborted):
        events = _events_tool_run()
        events[-1]["aborted"] = aborted
        collector = _collect(events)
        assert (collector.invalid_lines, collector.settled, collector.aborted) == (0, True, aborted)

    def test_valid_utf8_and_escaped_surrogates_are_accepted(self):
        collector = _PiEventCollector()
        line = b'{"type": "session", "id": "caf\\u00e9 \\ud83d\\ude00 \xc3\xa9 \\ud800"}\n'
        collector.feed_line(line)
        assert collector.invalid_lines == 0
        assert collector.session_id is not None

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

    BASE_URLS = [
        "https://gw.test/v1/$TENANT",
        "https://gw.test/${TENANT}/x",
        "https://gw.test/$$x",
        "https://gw.test/$!y",
        "https://gw.test/!cmd",
        "https://gw.test/v1/$",
    ]

    def test_base_url_with_a_dollar_is_written_as_it_is_and_pi_does_not_expand_it(self, pi_reference, tmp_path):
        """Pi 1.4.0 and 1.5.0 expand ``$NAME`` in the API key and the headers of ``models.json``, not in ``baseUrl``.

        An escape (``$$``) would change the URL that the model calls, so the URL is written raw.
        """
        for url in self.BASE_URLS:
            runner = PiRunner(cwd=tmp_path, gateway_base_url=url)
            assert runner._models_config(MODEL)["providers"]["anthropic"]["baseUrl"] == url
        request = {"action": "base_url", "provider": "anthropic", "env": {"TENANT": "RESOLVED"}, "urls": self.BASE_URLS}
        for version, answers in pi_reference(request).items():
            assert [answer["model"] for answer in answers] == self.BASE_URLS, version
            assert all(answer["error"] is None for answer in answers), version

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
        for flag in ("--no-session", "--approve", "--no-skills", "--no-context-files"):
            assert flag in command
        assert command[command.index("--provider") + 1] == "anthropic"
        assert command[command.index("--model") + 1] == "claude-sonnet-4-6"
        assert command[command.index("--thinking") + 1] == "high"
        assert command[command.index("--skill") + 1] == "/s/a"
        assert command[command.index("--append-system-prompt") + 1] == "/p.md"
        # `mcp__*` matches pi's real MCP tool names (hashed when long or
        # colliding); `toolExposure` in mcp.json enforces `tool_names`.
        # The MCP resource tools do not match `mcp__*`: they are named.
        assert (
            command[command.index("--tools") + 1]
            == "read,mcp__*,list_mcp_resources,list_mcp_resource_templates,read_mcp_resource"
        )
        # The prompt is never on the command line (pi reads `@path` args as files).
        assert "--" not in command

    def test_claude_tool_names_are_mapped(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, allowed_builtin_tools=["Read", "Grep", "Glob", "Edit", "MultiEdit", "custom"])
        assert runner._effective_builtin_tools() == ("read", "grep", "find", "edit", "custom")

    @pytest.mark.parametrize(
        ("windows", "expected"),
        [(False, ",".join(PI_BUILTIN_TOOLS)), (True, ",".join([*PI_BUILTIN_TOOLS, "powershell"]))],
    )
    def test_command_all_builtins_when_omitted(self, tmp_path, monkeypatch, windows, expected):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "_is_windows", lambda: windows)
        command = PiRunner(cwd=tmp_path, pi_command=["pi"])._command(
            model=MODEL, system_prompt_path=Path("/p"), skill_dirs=[]
        )
        assert command[command.index("--tools") + 1] == expected
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
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent", model=MODEL)
        assert env["PI_CODING_AGENT_DIR"] == str(tmp_path / "agent")
        assert "PI_SESSION_ID" not in env
        assert env["PI_OFFLINE"] == "1"
        assert env["KEEP_ME"] == "1"

    @pytest.mark.parametrize(
        "name",
        [
            # Credentials.
            "DD_API_KEY",
            "DD_APP_KEY",
            "DATADOG_API_KEY",
            "_DD_API_KEY",
            "dd_api_key",
            # Credentials inside the value, whatever the name.
            "DD_ADDITIONAL_ENDPOINTS",
            "DD_PROXY_HTTPS",
            "DD_ML_JOB_ENV",
            "DD_TRACE_AGENT_URL",
            # Settings: not for the pi process either.
            "DD_SITE",
            "DD_ENV",
            "DD_SERVICE",
            "DD_LLMOBS_ML_APP",
            "DD_GIT_COMMIT_AUTHOR_NAME",
            "DD_SECRET_BACKEND_COMMAND",
            "DD_APPSEC_OBFUSCATION_PARAMETER_KEY_REGEXP",
            "DATADOG_ANYTHING",
        ],
    )
    def test_env_drops_datadog_variables(self, tmp_path, monkeypatch, name):
        secret = f"operator-value-{name}"
        monkeypatch.setenv(name, secret)
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent", model=MODEL)
        assert name not in env
        assert secret not in env.values()

    def test_env_datadog_allowlist(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "PI_DATADOG_ENV_ALLOWLIST", frozenset({"DD_SITE"}))
        monkeypatch.setenv("DD_SITE", "datadoghq.com")
        monkeypatch.setenv("dd_site_other", "x")
        monkeypatch.setenv("DD_API_KEY", "operator-secret")
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent", model=MODEL)
        assert env["DD_SITE"] == "datadoghq.com"
        assert "DD_API_KEY" not in env
        assert "dd_site_other" not in env

    def test_env_allowlist_is_empty(self):
        # Pi needs no Datadog variable. Adding one is a decision: see the constant.
        from airedale.harness.pi import PI_DATADOG_ENV_ALLOWLIST

        assert frozenset() == PI_DATADOG_ENV_ALLOWLIST

    def test_env_keeps_other_variables(self, tmp_path, monkeypatch):
        # The key of the evaluated provider (pi reads ANTHROPIC_API_KEY itself
        # for MODEL) and variables that only contain "DD_" stay.
        keep = {
            "ANTHROPIC_API_KEY": "anthropic-key",
            "MY_DD_API_KEY_NOTE": "not a datadog variable",
            "ADD_ONE": "1",
            "HTTPS_PROXY": "http://user:pass@proxy:3128",
        }
        for name, value in keep.items():
            monkeypatch.setenv(name, value)
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent", model=MODEL)
        for name, value in keep.items():
            assert env[name] == value

    @pytest.mark.parametrize(
        ("model", "kept", "dropped"),
        [
            (MODEL, "ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
            (ModelSpec.parse("openai/gpt-5.5"), "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        ],
    )
    def test_env_keeps_only_the_key_of_the_evaluated_provider(self, tmp_path, monkeypatch, model, kept, dropped):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        env = PiRunner(cwd=tmp_path)._env(tmp_path / "agent", model=model)
        assert kept in env
        assert dropped not in env

    def test_env_drops_every_provider_key_when_models_json_has_the_token(self, tmp_path, monkeypatch):
        # Pi authenticates with the token in models.json: it needs no key variable.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        runner = PiRunner(cwd=tmp_path, gateway_base_url="https://gw", gateway_bearer_token="helper-token")
        env = runner._env(tmp_path / "agent", model=MODEL)
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env

    @pytest.mark.parametrize(("provider", "native"), [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
    def test_empty_gateway_key_keeps_the_native_key(self, tmp_path, monkeypatch, provider, native):
        # The api_key_env variable is empty: models.json has no token, so pi
        # needs the native key of the evaluated provider.
        monkeypatch.setenv("MY_GATEWAY_TOKEN", "")
        monkeypatch.setenv(native, "native-key")
        gateway = GatewayConfig(
            providers={
                provider: ProviderGatewayConfig(
                    provider=provider, base_url="https://gw", api_key_env="MY_GATEWAY_TOKEN"
                )
            }
        )
        model = MODEL if provider == "anthropic" else ModelSpec.parse("openai/gpt-5.5")
        runner = create_runner(
            model, scenario=ScenarioConfig(name="s", harness=HARNESS_PI), gateway=gateway, cwd=str(tmp_path)
        )
        assert isinstance(runner, PiRunner)
        assert runner._models_json_token() is None
        assert "apiKey" not in runner._models_config(model).get("providers", {}).get(provider, {})
        assert runner._env(tmp_path / "agent", model=model)[native] == "native-key"

    def test_unrelated_provider_key_does_not_reach_pi(self, tmp_path, fake_pi, monkeypatch):
        command, record = fake_pi(_events_tool_run())
        monkeypatch.setenv("OPENAI_API_KEY", "gateway-secret-openai")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is None
        seen = json.loads(record.read_text())
        assert "OPENAI_API_KEY" not in seen["env_names"]
        assert seen["secret_in_env"] is False

    @pytest.mark.parametrize("token_kind", ["api_key", "bearer"])
    def test_env_drops_gateway_api_key_env(self, tmp_path, monkeypatch, token_kind):
        # The gateway key goes to pi in models.json only, not in its environment.
        monkeypatch.setenv("MY_GATEWAY_TOKEN", "gateway-secret")
        monkeypatch.setenv("KEEP_ME", "1")
        runner = PiRunner(
            cwd=tmp_path,
            gateway_base_url="https://gw",
            gateway_api_key="gateway-secret" if token_kind == "api_key" else None,
            gateway_bearer_token="helper-token" if token_kind == "bearer" else None,
            gateway_api_key_env="MY_GATEWAY_TOKEN",
        )
        env = runner._env(tmp_path / "agent", model=MODEL)
        assert "MY_GATEWAY_TOKEN" not in env
        assert "gateway-secret" not in env.values()
        assert env["KEEP_ME"] == "1"

    def test_env_drops_gateway_native_key_env_when_in_models_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "gateway-secret")
        runner = PiRunner(
            cwd=tmp_path,
            gateway_base_url="https://gw",
            gateway_api_key="gateway-secret",
            gateway_api_key_env="ANTHROPIC_API_KEY",
        )
        assert "ANTHROPIC_API_KEY" not in runner._env(tmp_path / "agent", model=MODEL)

    def test_env_drops_custom_key_env_without_gateway_base_url(self, tmp_path, monkeypatch):
        # Pi does not read a custom variable, so it is removed also without a base URL.
        monkeypatch.setenv("MY_GATEWAY_TOKEN", "gateway-secret")
        runner = PiRunner(cwd=tmp_path, gateway_api_key="gateway-secret", gateway_api_key_env="MY_GATEWAY_TOKEN")
        assert "MY_GATEWAY_TOKEN" not in runner._env(tmp_path / "agent", model=MODEL)

    def test_env_drops_other_gateway_provider_key_envs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTHER_GATEWAY_TOKEN", "other-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-gateway-secret")
        runner = PiRunner(
            cwd=tmp_path,
            gateway_base_url="https://gw",
            gateway_api_key="k",
            other_gateway_api_key_envs=["OTHER_GATEWAY_TOKEN", "OPENAI_API_KEY"],
        )
        env = runner._env(tmp_path / "agent", model=MODEL)
        assert "OTHER_GATEWAY_TOKEN" not in env
        assert "OPENAI_API_KEY" not in env

    def test_env_without_models_json_token_keeps_only_the_evaluated_native_key(self, tmp_path, monkeypatch):
        # No base URL: pi can need ANTHROPIC_API_KEY (MODEL is an anthropic
        # model) to authenticate. The OpenAI gateway key is not needed.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-gateway-key")
        monkeypatch.setenv("OTHER_GATEWAY_TOKEN", "other-secret")
        runner = PiRunner(
            cwd=tmp_path, other_gateway_api_key_envs=["OPENAI_API_KEY", "OTHER_GATEWAY_TOKEN", "ANTHROPIC_API_KEY"]
        )
        env = runner._env(tmp_path / "agent", model=MODEL)
        assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
        assert "OPENAI_API_KEY" not in env
        assert "OTHER_GATEWAY_TOKEN" not in env

    def test_native_key_exception_ignores_case_on_windows(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        runner = PiRunner(cwd=tmp_path, gateway_api_key="k", gateway_api_key_env="anthropic_api_key")
        with monkeypatch.context() as patch:
            patch.setattr(pi_module.os, "name", "nt")
            dropped_on_windows = runner._dropped_gateway_envs("anthropic")
        with monkeypatch.context() as patch:
            patch.setattr(pi_module.os, "name", "posix")
            dropped_on_posix = runner._dropped_gateway_envs("anthropic")
        assert dropped_on_windows == []
        # On POSIX, names are case-sensitive: "anthropic_api_key" is another variable.
        assert dropped_on_posix == ["anthropic_api_key"]

    def test_env_keeps_native_key_env_that_pi_needs(self, tmp_path, monkeypatch):
        # No base URL: models.json has no key, and pi reads ANTHROPIC_API_KEY itself.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-key")
        runner = PiRunner(cwd=tmp_path, gateway_api_key="provider-key", gateway_api_key_env="ANTHROPIC_API_KEY")
        assert not runner._models_config(MODEL)
        assert runner._env(tmp_path / "agent", model=MODEL)["ANTHROPIC_API_KEY"] == "provider-key"

    def test_private_agent_dir_skips_parents_inside_workspace(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        workspace = tmp_path / "ws"
        (workspace / "tmp").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(
            pi_module,
            "_agent_dir_parent_candidates",
            lambda: [workspace / "tmp", workspace, outside / "missing", outside],
        )
        agent_dir = pi_module._make_private_agent_dir(workspace)
        try:
            assert agent_dir.parent.resolve() == outside.resolve()
            assert agent_dir.stat().st_mode & 0o777 == 0o700
        finally:
            agent_dir.rmdir()
        assert list(workspace.rglob("*")) == [workspace / "tmp"]

    def test_private_agent_dir_detects_workspace_with_other_letter_case(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        workspace = tmp_path / "Workspace"
        (workspace / "tmp").mkdir(parents=True)
        other_case = tmp_path / "wORKSPACE" / "tmp"
        if not other_case.exists():
            pytest.skip("the file system is case-sensitive")
        assert pi_module._is_inside(other_case, workspace)
        outside = tmp_path / "outside"
        outside.mkdir()
        monkeypatch.setattr(pi_module, "_agent_dir_parent_candidates", lambda: [other_case, outside])
        agent_dir = pi_module._make_private_agent_dir(workspace)
        try:
            assert agent_dir.parent.resolve() == outside.resolve()
        finally:
            agent_dir.rmdir()
        assert list((workspace / "tmp").iterdir()) == []

    def test_is_inside_uses_file_identity(self, tmp_path):
        from airedale.harness.pi import _is_inside

        workspace = tmp_path / "ws"
        (workspace / "a" / "b").mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        link = tmp_path / "link"
        link.symlink_to(workspace / "a", target_is_directory=True)
        assert _is_inside(workspace, workspace)
        assert _is_inside(workspace / "a" / "b", workspace)
        assert _is_inside(link / "b", workspace)
        assert not _is_inside(tmp_path / "outside", workspace)
        assert not _is_inside(tmp_path, workspace)

    def test_is_inside_detects_an_alias_by_file_identity(self, tmp_path, monkeypatch):
        # The alias has another name than the workspace but the same device and
        # inode (a case variant, a hard link, ...). Works on every file system.
        from airedale.harness.pi import _is_inside

        workspace = tmp_path / "ws"
        workspace.mkdir()
        alias = tmp_path / "alias"
        (alias / "tmp").mkdir(parents=True)
        real_alias = Path(os.path.realpath(alias))
        workspace_stat = workspace.stat()
        real_stat = Path.stat

        def fake_stat(self, *args, **kwargs):
            if self == real_alias:
                return workspace_stat
            return real_stat(self, *args, **kwargs)

        assert not _is_inside(alias / "tmp", workspace)
        monkeypatch.setattr(Path, "stat", fake_stat)
        assert _is_inside(alias / "tmp", workspace)

    def test_is_inside_is_true_when_a_parent_cannot_be_examined(self, tmp_path, monkeypatch):
        from airedale.harness.pi import _is_inside

        workspace = tmp_path / "ws"
        workspace.mkdir()
        candidate = tmp_path / "outside"
        candidate.mkdir()
        real_candidate = Path(os.path.realpath(candidate))
        real_stat = Path.stat

        def failing_stat(self, *args, **kwargs):
            if self == real_candidate:
                raise PermissionError("denied")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", failing_stat)
        assert _is_inside(candidate, workspace)

    def test_private_agent_dir_raises_when_every_parent_is_inside_workspace(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        workspace = tmp_path / "ws"
        (workspace / "tmp").mkdir(parents=True)
        monkeypatch.setattr(pi_module, "_agent_dir_parent_candidates", lambda: [workspace / "tmp", workspace])
        with pytest.raises(RuntimeError, match="outside the workspace"):
            pi_module._make_private_agent_dir(workspace)
        assert list(workspace.rglob("*")) == [workspace / "tmp"]

    def test_private_agent_dir_rejects_a_dir_that_ends_up_inside_workspace(self, tmp_path, monkeypatch):
        # The parent check passes, but the created directory is in the workspace
        # (the parent changed between the check and mkdtemp): it is removed.
        import airedale.harness.pi as pi_module

        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        real_mkdtemp = pi_module.tempfile.mkdtemp
        monkeypatch.setattr(pi_module, "_agent_dir_parent_candidates", lambda: [outside])
        monkeypatch.setattr(
            pi_module.tempfile, "mkdtemp", lambda **kw: real_mkdtemp(prefix=kw["prefix"], dir=workspace)
        )
        with pytest.raises(RuntimeError):
            pi_module._make_private_agent_dir(workspace)
        assert list(workspace.iterdir()) == []

    def test_workspace_context_file_order(self, tmp_path):
        runner = PiRunner(cwd=tmp_path)
        assert runner._workspace_context_section() == ""
        (tmp_path / "CLAUDE.md").write_text("claude")
        assert "claude" in runner._workspace_context_section()
        (tmp_path / "AGENTS.md").write_text("agents")
        assert "agents" in runner._workspace_context_section()
        assert "claude" not in runner._workspace_context_section()
        # As in pi, AGENTS.override.md replaces the other files of the directory.
        (tmp_path / "AGENTS.override.md").write_text("override")
        section = runner._workspace_context_section()
        assert "override" in section
        assert "agents" not in section
        # As in pi, invalid UTF-8 is replaced and a BOM is removed.
        (tmp_path / "AGENTS.override.md").write_bytes(b"\xef\xbb\xbfDon\x92t")
        section = runner._workspace_context_section()
        assert "\nDon\ufffdt\n" in section
        # A directory with a context file name is not a context file.
        (tmp_path / "AGENTS.override.md").unlink()
        (tmp_path / "AGENTS.override.md").mkdir()
        assert "agents" in runner._workspace_context_section()

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

    @pytest.mark.parametrize("names", [("docs-api", "docs_api"), ("a_b-c", "a-b_c")])
    def test_scenario_mcp_servers_with_the_same_pi_name_raise(self, tmp_path, names):
        servers = [McpServerSpec(name=name, url="http://localhost/mcp") for name in names]
        servers.append(McpServerSpec(name="other", url="http://localhost/mcp"))
        runner = PiRunner(cwd=tmp_path, mcp_servers=servers)
        with pytest.raises(ValueError, match="same name for pi") as error:
            runner._check_project_mcp_collisions()
        assert ", ".join(names) in str(error.value)
        assert "other" not in str(error.value)

    def test_scenario_mcp_name_collision_fails_before_pi_starts(self, tmp_path, fake_pi):
        command, record = fake_pi(_events_tool_run())
        servers = [McpServerSpec(name=name, url="http://localhost/mcp") for name in ("docs-api", "docs_api")]
        with pytest.raises(ValueError, match="same name for pi"):
            _run(PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=servers))
        assert not record.exists()

    @pytest.mark.parametrize(
        "entry",
        [
            "read,bash",
            "*",
            "mcp__*",
            "re*d",
            "+bash",
            "-read",
            " read",
            "read ",
            "",
            "  ",
            "read,",
            ",",
            "a,b,c",
            "\ufeff+bash",
            "\ufeff+ls",
            "\ufeff\ufeffread\ufeff",
            "\ufeffread",
            "read\ufeff",
            "\ufeff",
            "\u00a0read",
            "read\u2028",
            "\u2003read",
            "re ad",
            "read\n",
            "\tread",
            "r\u0435ad",
            "read;bash",
            "read|bash",
            "'read'",
            "a=b",
            ".hidden",
            "x+y",
            "é",
        ],
    )
    def test_builtin_tool_entry_with_pi_cli_syntax_is_refused_before_pi_starts(self, tmp_path, fake_pi, entry):
        # Pi splits ``--tools`` at "," and trims, reads "*" as a pattern and a leading "+"/"-" as a change to its
        # defaults: the entries must be exact names.
        command, record = fake_pi(_events_tool_run())
        with pytest.raises(ValueError, match="not an exact tool name"):
            _run(PiRunner(cwd=tmp_path, pi_command=command, allowed_builtin_tools=["read", entry]))
        assert not record.exists()

    def test_builtin_tool_entry_with_pi_cli_syntax_is_refused_for_the_scenario_too(self, tmp_path):
        with pytest.raises(ValueError, match="'read,bash'"):
            PiRunner(cwd=tmp_path, allowed_builtin_tools=["read,bash"]).check_scenario()

    @pytest.mark.parametrize(
        "entry", ["read", "Bash", "my_extension_tool", "ext.tool", "a-b", "_x", "9lives", "mcp__docs__q"]
    )
    def test_exact_builtin_tool_names_are_accepted(self, tmp_path, entry):
        runner = PiRunner(cwd=tmp_path, allowed_builtin_tools=[entry])
        runner.check_scenario()
        runner._check_before_probe({})

    def test_no_builtin_tools_and_the_default_are_accepted(self, tmp_path):
        PiRunner(cwd=tmp_path, allowed_builtin_tools=[]).check_scenario()
        PiRunner(cwd=tmp_path, allowed_builtin_tools=None).check_scenario()

    def test_extension_tool_names_go_to_pi_unchanged(self, tmp_path):
        # Repo extension tools are declared only when the scenario names them.
        runner = PiRunner(cwd=tmp_path, allowed_builtin_tools=["Read", "my_extension_tool"])
        command = runner._command(model=MODEL, system_prompt_path=Path("/p"), skill_dirs=[])
        assert command[command.index("--tools") + 1] == "read,my_extension_tool"

    def test_project_mcp_collision_raises(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"my-server": {"command": "x"}}}))
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="my-server", command="y")])
        with pytest.raises(ValueError, match="my-server"):
            runner._check_project_mcp_collisions()
        PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="other", command="y")])._check_project_mcp_collisions()


class TestPiSkillDirs:
    def _skill(self, base: Path, name: str, *, declared: str | None = None) -> Path:
        path = base / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(f"---\nname: {declared or name}\ndescription: d\n---\n")
        return path

    def test_scenario_and_project_skills(self, tmp_path):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        self._skill(cwd / ".pi" / "skills", "beta")
        self._skill(cwd / ".agents" / "skills", "gamma")
        # Project skill roots are passed whole: pi discovers their skills.
        # .pi/skills goes whole (pi's directory loader has the same rules as its
        # discovery); .agents/skills goes as the files pi's discovery finds.
        assert pi_skill_dirs([str(scenario)], str(cwd)) == [
            str(scenario.resolve()),
            str(cwd / ".pi" / "skills"),
            str(cwd / ".agents" / "skills" / "gamma" / "SKILL.md"),
        ]

    def test_nested_project_skill_is_checked_for_collisions(self, tmp_path):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        self._skill(cwd / ".pi" / "skills" / "team", "alpha")
        with pytest.raises(ValueError, match="collides"):
            pi_skill_dirs([str(scenario)], str(cwd))

    def test_collisions_use_the_frontmatter_name(self, tmp_path):
        cwd = tmp_path / "cwd"
        # Same directory name, different declared names: no collision.
        scenario = self._skill(tmp_path / "src", "tool", declared="scenario-tool")
        self._skill(cwd / ".pi" / "skills", "tool", declared="repo-tool")
        assert pi_skill_dirs([str(scenario)], str(cwd))[0] == str(scenario.resolve())
        # Different directory names, same declared name: collision.
        other = self._skill(tmp_path / "src2", "other", declared="repo-tool")
        with pytest.raises(ValueError, match="'repo-tool'"):
            pi_skill_dirs([str(other)], str(cwd))

    def test_scenario_project_collision_raises(self, tmp_path):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        self._skill(cwd / ".agents" / "skills", "alpha")
        with pytest.raises(ValueError, match="collides"):
            pi_skill_dirs([str(scenario)], str(cwd))

    def test_project_duplicates_are_left_to_pi(self, tmp_path, caplog):
        cwd = tmp_path / "cwd"
        self._skill(cwd / ".pi" / "skills", "alpha")
        self._skill(cwd / ".agents" / "skills", "alpha")
        # Pi keeps the first skill of a name (.pi/skills comes first).
        assert pi_skill_dirs([], str(cwd)) == [
            str(cwd / ".pi" / "skills"),
            str(cwd / ".agents" / "skills" / "alpha" / "SKILL.md"),
        ]
        assert "shadowed" in caplog.text

    def test_settings_skill_paths(self, tmp_path):
        cwd = tmp_path / "cwd"
        extra = self._skill(cwd / "tools", "delta")
        absolute = self._skill(tmp_path / "shared", "epsilon")
        (cwd / ".pi").mkdir(parents=True)
        (cwd / ".pi" / "settings.json").write_text(
            json.dumps({"skills": ["../tools/delta", str(absolute), "!skills/old", "+x", "-y", "../missing", 3]})
        )
        assert pi_skill_dirs([], str(cwd)) == [str(cwd / ".pi" / "../tools/delta"), str(absolute)]
        assert extra.exists()

    def test_settings_skills_come_before_discovered_skills(self, tmp_path, caplog):
        # Pi gives a settings entry precedence over automatic discovery.
        cwd = tmp_path / "cwd"
        discovered = self._skill(cwd / ".pi" / "skills", "alpha")
        configured = self._skill(cwd / "tools", "alpha-v2", declared="alpha")
        (cwd / ".pi" / "settings.json").write_text(json.dumps({"skills": [str(configured)]}))
        assert pi_skill_dirs([], str(cwd)) == [str(configured), str(discovered.parent)]
        assert "shadowed" in caplog.text

    def test_standalone_markdown_rules_per_root(self, tmp_path):
        cwd = tmp_path / "cwd"
        pi_root = cwd / ".pi" / "skills"
        agents_root = cwd / ".agents" / "skills"
        (pi_root / "team").mkdir(parents=True)
        (agents_root / "team").mkdir(parents=True)
        skill = "---\nname: {}\ndescription: d\n---\n"
        (pi_root / "root.md").write_text(skill.format("pi-root"))
        (pi_root / "team" / "nested.md").write_text(skill.format("pi-nested"))
        (agents_root / "root.md").write_text(skill.format("agents-root"))
        (agents_root / "team" / "nested.md").write_text(skill.format("agents-nested"))
        # As pi: root .md files in .pi/skills, nested .md files in .agents/skills.
        assert pi_skill_dirs([], str(cwd)) == [str(pi_root), str(agents_root / "team" / "nested.md")]
        from airedale.skills import _pi_skills

        assert [name for name, _ in _pi_skills(pi_root)] == ["pi-root"]

    @pytest.mark.parametrize(
        "line", ["name: alpha # project copy", "name: 'alpha' # quoted", 'name: "alpha"', "name:   alpha   "]
    )
    def test_yaml_name_forms_are_compared(self, tmp_path, line):
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        other = cwd / ".pi" / "skills" / "other"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text(f"---\n{line}\ndescription: d\n---\n")
        with pytest.raises(ValueError, match="'alpha'"):
            pi_skill_dirs([str(scenario)], str(cwd))

    @pytest.mark.parametrize(
        "lines",
        [
            ['"name": alpha', "description: d"],
            ["'name': alpha", "description: d"],
            ['"name" : "alpha"', "'description': d"],
            ['"na\\u006de": alpha', '"description": d'],  # an escape in the key
        ],
    )
    def test_quoted_yaml_keys_are_compared(self, tmp_path, lines):
        # YAML reads a quoted key as its text, so pi sees the name "alpha".
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        other = cwd / ".pi" / "skills" / "other"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text("---\n" + "\n".join(lines) + "\n---\n")
        with pytest.raises(ValueError, match="'alpha'"):
            pi_skill_dirs([str(scenario)], str(cwd))

    @pytest.mark.parametrize(
        ("header", "expected"), [(">-", "alpha"), ("|-", "alpha"), ("|", "alpha\n"), (">", "alpha\n")]
    )
    def test_block_scalar_name(self, tmp_path, header, expected):
        from airedale.skills import _pi_skills

        skill = tmp_path / "dir-name"
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"---\nname: {header}\n  alpha\ndescription: d\n---\n")
        assert [name for name, _ in _pi_skills(skill)] == [expected]

    @pytest.mark.parametrize("description", ["", "description:", "description: ''", "description:    "])
    def test_skill_without_description_is_not_loaded(self, tmp_path, description):
        # As pi: no non-blank description, no skill (also for SKILL.md), so no collision.
        scenario = self._skill(tmp_path / "src", "alpha")
        cwd = tmp_path / "cwd"
        other = cwd / ".agents" / "skills" / "other"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text(f"---\nname: alpha\n{description}\n---\n")
        assert pi_skill_dirs([str(scenario)], str(cwd))[0] == str(scenario.resolve())

    def test_non_string_name_uses_the_directory_name(self, tmp_path):
        from airedale.skills import _pi_skills

        skill = tmp_path / "dir-name"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: 123\ndescription: |\n  text\n---\n")
        assert [name for name, _ in _pi_skills(skill)] == ["dir-name"]

    def test_agents_skills_above_the_workspace_are_not_loaded(self, tmp_path):
        self._skill(tmp_path / ".agents" / "skills", "ambient")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        assert pi_skill_dirs([], str(cwd)) == []

    def test_standalone_markdown_skill_names(self, tmp_path):
        from airedale.skills import _pi_skills

        root = tmp_path / "skills"
        root.mkdir()
        (root / "quick.md").write_text("---\nname: quick\ndescription: 'd'\n---\n")
        (root / "notes.md").write_text("# not a skill\n")
        (root / ".hidden").mkdir()
        (root / ".hidden" / "SKILL.md").write_text("---\nname: hidden\n---\n")
        (root / "node_modules" / "dep").mkdir(parents=True)
        (root / "node_modules" / "dep" / "SKILL.md").write_text("---\nname: dep\n---\n")
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\ndescription: d\n---\n")
        assert [name for name, _ in _pi_skills(root)] == ["b", "quick"]

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
    assert runner.gateway_api_key_env == "GW_KEY"
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
                   "env_names": sorted(os.environ),
                   "secret_in_env": any("gateway-secret" in v for v in os.environ.values()),
                   "dd_env": {k: v for k, v in os.environ.items() if k.upper().startswith(("DD_", "DATADOG_", "_DD_"))}},
                  handle)
    with open(record + ".runs", "a") as handle:
        handle.write(json.dumps(sys.argv[1:]) + chr(10))
    if os.environ.get("FAKE_PI_REJECT_EXTENSION") and "--extension" in sys.argv:
        path = sys.argv[sys.argv.index("--extension") + 1]
        sys.stderr.write(f'Error: Failed to load extension "{path}": unsupported' + chr(10))
        sys.exit(1)
    events = json.loads(os.environ["FAKE_PI_EVENTS"])
    if os.environ.get("FAKE_PI_GARBAGE"):
        print("not json", flush=True)
    for event in events:
        if "__sleep" in event:  # a pause between two records
            time.sleep(event["__sleep"])
            continue
        if "__now_ms" in event:  # set the timestamp of a message to now
            event["message"]["timestamp"] = time.time() * 1000 + event.pop("__now_ms")
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

    def test_gateway_api_key_env_does_not_reach_pi(self, tmp_path, fake_pi, monkeypatch):
        command, record = fake_pi(_events_tool_run())
        monkeypatch.setenv("MY_GATEWAY_TOKEN", "gateway-secret")
        # The key of the other gateway provider does not reach pi either.
        monkeypatch.setenv("OTHER_GATEWAY_TOKEN", "other-gateway-secret")
        gateway = GatewayConfig(
            providers={
                "anthropic": ProviderGatewayConfig(
                    provider="anthropic", base_url="https://gw", api_key_env="MY_GATEWAY_TOKEN"
                ),
                "openai": ProviderGatewayConfig(
                    provider="openai", base_url="https://gw2", api_key_env="OTHER_GATEWAY_TOKEN"
                ),
            }
        )
        scenario = ScenarioConfig(name="s", harness=HARNESS_PI)
        runner = create_runner(MODEL, scenario=scenario, gateway=gateway, cwd=str(tmp_path))
        assert isinstance(runner, PiRunner)
        runner.pi_command = command
        result = _run(runner)
        assert result.error is None

        seen = json.loads(record.read_text())
        assert "MY_GATEWAY_TOKEN" not in seen["env_names"]
        assert "OTHER_GATEWAY_TOKEN" not in seen["env_names"]
        assert seen["secret_in_env"] is False
        # Pi still gets the key, in its private models.json.
        assert json.loads(seen["files"]["models.json"])["providers"]["anthropic"]["apiKey"] == "gateway-secret"

    @pytest.mark.parametrize("variable", ["TMPDIR", "TEMP", "TMP"])
    @pytest.mark.parametrize("through_symlink", [False, True])
    def test_agent_dir_outside_workspace_when_tmpdir_is_inside(
        self, tmp_path, fake_pi, monkeypatch, variable, through_symlink
    ):
        import tempfile

        command, record = fake_pi(_events_tool_run())
        cwd = tmp_path / "cwd"
        (cwd / "tmp").mkdir(parents=True)
        temp_value = cwd / "tmp"
        if through_symlink:
            # A link outside the workspace that points into it.
            temp_value = tmp_path / "link-to-workspace-tmp"
            temp_value.symlink_to(cwd / "tmp", target_is_directory=True)
        for name in ("TMPDIR", "TEMP", "TMP"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(variable, str(temp_value))
        # tempfile keeps the directory it found first: make it read the variables again.
        monkeypatch.setattr(tempfile, "tempdir", None)
        assert Path(tempfile.gettempdir()).resolve() == (cwd / "tmp").resolve()

        result = _run(PiRunner(cwd=cwd, pi_command=command, gateway_base_url="https://gw", gateway_api_key="k"))
        assert result.error is None
        seen = json.loads(record.read_text())
        assert not Path(seen["agent_dir"]).resolve().is_relative_to(cwd.resolve())
        assert "models.json" in seen["files"]
        assert not Path(seen["agent_dir"]).exists()
        assert list((cwd / "tmp").iterdir()) == []

    def test_only_the_workspace_context_file_reaches_pi(self, tmp_path, fake_pi):
        command, record = fake_pi(_events_tool_run())
        # A context file above the workspace (for example /tmp/AGENTS.md) is
        # not part of the evaluated repo.
        (tmp_path / "AGENTS.md").write_text("PARENT INSTRUCTIONS")
        cwd = tmp_path / "ws"
        cwd.mkdir()
        (cwd / "CLAUDE.md").write_text("REPO INSTRUCTIONS")
        result = _run(PiRunner(cwd=cwd, pi_command=command))
        assert result.error is None
        seen = json.loads(record.read_text())
        assert "--no-context-files" in seen["argv"]
        prompt = seen["files"]["airedale-system-prompt.md"]
        assert prompt.startswith("SYS\n\n<project_context>")
        assert (
            f'<project_instructions path="{cwd / "CLAUDE.md"}">\nREPO INSTRUCTIONS\n</project_instructions>' in prompt
        )
        assert "PARENT INSTRUCTIONS" not in prompt

    def test_datadog_variables_do_not_reach_pi(self, tmp_path, fake_pi):
        command, record = fake_pi(
            _events_tool_run(),
            DD_API_KEY="dd-api",
            DD_APP_KEY="dd-app",
            DATADOG_API_KEY="dd-api-2",
            _DD_API_KEY="dd-api-3",
            DD_PROXY_HTTPS="http://user:dd-proxy-pass@proxy:3128",
            DD_SITE="site",
        )
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is None
        seen = json.loads(record.read_text())
        assert seen["dd_env"] == {}

    def test_non_zero_exit(self, tmp_path, fake_pi):
        command, _ = fake_pi([], FAKE_PI_EXIT="3", FAKE_PI_STDERR="unknown model")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "status 3" in str(result.error)
        assert "unknown model" not in str(result.error)
        # The raw stderr is in the private file that the error names.
        report = Path(str(result.error).rsplit(" ", 1)[1])
        assert "unknown model" in report.read_text()

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
        assert "claude-sonnet-4-6-20991231" not in caplog.text  # The value is not logged.
        assert "response model that is not the configured one" in caplog.text


class TestPiToolNames:
    def test_tools_whose_names_collide_all_get_the_hash_suffix(self):
        import hashlib

        from airedale.mcp import pi_mcp_tool_names

        def digest(tool: str) -> str:
            return hashlib.sha256(f"docs\0{tool}".encode()).hexdigest()[:8]

        expected = {
            "get-doc": f"mcp__docs__get_doc_{digest('get-doc')}",
            "get_doc": f"mcp__docs__get_doc_{digest('get_doc')}",
            "search": "mcp__docs__search",
        }
        assert pi_mcp_tool_names("docs", ["get-doc", "search", "get_doc"]) == expected
        # The order of the list does not change the names (as in pi).
        assert pi_mcp_tool_names("docs", ["get_doc", "get-doc", "search", "search"]) == expected

    @pytest.mark.parametrize(
        ("sdk", "expected"),
        [("pi", ["mcp__docs__search", "mcp__docs__read"]), ("openai-codex", ["mcp.docs.search", "mcp.docs.read"])],
    )
    def test_tool_definitions_without_allow_list_declare_every_probed_tool(self, sdk, expected):
        from airedale.harness.base import _llmobs_tool_definitions
        from airedale.mcp import McpToolMetadata

        server = McpServerSpec(name="docs", url="http://localhost/mcp")
        catalog = {
            ("docs", "search"): McpToolMetadata(description="S", input_schema={"type": "object"}),
            ("docs", "read"): McpToolMetadata(),
            ("other", "x"): McpToolMetadata(),
        }
        definitions = _llmobs_tool_definitions(
            [server], sdk_name=sdk, allowed_builtin_tools=[], mcp_tool_metadata=catalog
        )
        assert [definition["name"] for definition in definitions] == expected
        assert definitions[0]["description"] == "S"
        # No catalog (LLMObs off or the probe failed): nothing is known.
        assert _llmobs_tool_definitions([server], sdk_name=sdk, allowed_builtin_tools=[]) == []

    def test_tool_definitions_use_the_complete_tool_list_of_the_server(self):
        import hashlib

        from airedale.harness.base import _llmobs_tool_definitions
        from airedale.mcp import McpToolMetadata

        server = McpServerSpec(name="docs", url="http://localhost/mcp", tool_names=("get-doc",))
        # `get_doc` is not in tool_names, but the server has it: pi hashes both.
        catalog = {("docs", "get-doc"): McpToolMetadata(description="D"), ("docs", "get_doc"): McpToolMetadata()}
        definitions = _llmobs_tool_definitions(
            [server], sdk_name="pi", allowed_builtin_tools=[], mcp_tool_metadata=catalog
        )
        digest = hashlib.sha256(b"docs\0get-doc").hexdigest()[:8]
        assert definitions == [{"name": f"mcp__docs__get_doc_{digest}", "description": "D"}]
        # Without the catalog, only the configured names are known.
        plain = _llmobs_tool_definitions([server], sdk_name="pi", allowed_builtin_tools=[])
        assert plain == [{"name": "mcp__docs__get_doc"}]
        # Other SDKs keep their own names.
        claude = _llmobs_tool_definitions([server], sdk_name="claude-agent-sdk", mcp_tool_metadata=catalog)
        assert claude == [{"name": "mcp__docs__get-doc", "description": "D"}]

    def test_reported_mcp_servers_include_project_servers(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "repo-docs": {"command": "node", "args": ["docs.js"]},
                        "repo-http": {"url": "https://example.test/mcp", "headers": {"Authorization": "x"}},
                        "off": {"command": "x", "enabled": False},
                    }
                }
            )
        )
        scenario = McpServerSpec(name="apm", url="http://localhost/mcp")
        runner = PiRunner(cwd=tmp_path, mcp_servers=[scenario])
        reported = runner._reported_mcp_servers()
        assert [server.name for server in reported] == ["apm", "repo-docs", "repo-http"]
        assert reported[1].type == "stdio"
        assert reported[2].to_safe_dict()["headers"] == {"Authorization": "<redacted>"}
        # Pi reads the project file itself: the generated mcp.json has only the scenario server.
        assert list(runner._mcp_config({})["mcpServers"]) == ["apm"]

    def test_reported_project_servers_have_no_secrets(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "repo-cli": {
                            "command": "node",
                            "args": ["--token", "arg-secret"],
                            "env": {"TOKEN": "env-secret"},
                        },
                        "repo-http": {
                            "url": "https://user:pw-secret@example.test:8443/s/path-secret/mcp?key=query-secret#frag"
                        },
                        # Wrong types: pi skips these entries, so does the report (no error).
                        "bad-env": {"command": "x", "env": 1},
                        "bad-args": {"command": "x", "args": [1]},
                        "bad-url": {"url": 5},
                    }
                }
            )
        )
        runner = PiRunner(cwd=tmp_path)
        reported = [server.to_safe_dict() for server in runner._reported_mcp_servers()]
        assert [server["name"] for server in reported] == ["repo-cli", "repo-http"]
        text = json.dumps(reported)
        for secret in ("arg-secret", "env-secret", "pw-secret", "path-secret", "query-secret", "frag"):
            assert secret not in text
        assert reported[0]["env"] == {"TOKEN": "<set>"}
        assert reported[0]["args"] == ["<redacted>", "<redacted>"]
        assert reported[1]["url"] == "https://example.test:8443"

    @pytest.mark.parametrize(
        ("url", "reported"),
        [
            ("http://host\\s\\TOKEN/mcp", "http://host"),  # backslash is a path separator for pi
            ("http://host:8080\\s\\TOKEN/mcp", "http://host:8080"),
            ("https://user:pw@host/s/TOKEN/mcp?k=TOKEN#TOKEN", "https://host"),
            ("https://user:pw@host\\TOKEN@x/", "https://host"),
            ("ht\ttps://ho\nst/TOKEN", "https://host"),  # tab and newline are removed
            ("https:\\\\host\\TOKEN", "https://host"),
            ("https://[::1]:9/TOKEN", "https://[::1]:9"),
            ("https://example.test:443/TOKEN", "https://example.test"),
            ("not a url TOKEN", "<redacted>"),
            ("https:///host/TOKEN", "https://host"),  # slashes are skipped, as for pi
            ("https://", "<redacted>"),
        ],
    )
    def test_reported_url_has_only_scheme_host_and_port(self, url, reported):
        from airedale.mcp import url_without_secrets

        assert url_without_secrets(url) == reported
        assert "TOKEN" not in (url_without_secrets(url) or "")

    @pytest.mark.parametrize(
        "url",
        [
            "http://host\uff0fSECRET/mcp",  # full-width solidus: a path separator after IDNA mapping
            "http://host\uff20SECRET/mcp",  # full-width commercial at
            "http://host\uff1aSECRET/mcp",  # full-width colon
            "http://[SECRET]/mcp",
            "http://host%EF%BC%8FSECRET/mcp",  # percent-encoded full-width solidus
            "http://user\uff20SECRET.example/mcp",
            "https://exämple.test/SECRET",
        ],
    )
    def test_project_server_url_with_an_uncertain_host_is_not_reported(self, tmp_path, url):
        # The reader cannot tell what pi reads as the host: nothing of the URL is reported (to_safe_dict goes to
        # LLMObs metadata and the agent manifest).
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"repo": {"url": url}}}))
        runner = PiRunner(cwd=tmp_path)
        reported = runner._reported_mcp_servers()
        assert [server.name for server in reported] == ["repo"]
        assert "SECRET" not in json.dumps([server.to_safe_dict() for server in reported], ensure_ascii=False)
        assert reported[0].url == "<redacted>"

    @pytest.mark.parametrize(
        ("port", "reported"),
        [
            ("0" * 5000 + "80", "http://host"),  # A valid port (80, the default) with thousands of leading zeros.
            ("0" * 5000 + "8080", "http://host:8080"),
            ("8080", "http://host:8080"),
        ],
    )
    def test_project_server_with_a_port_of_leading_zeros_is_reported(self, tmp_path, port, reported):
        (tmp_path / ".pi").mkdir()
        url = f"http://host:{port}/SECRET/mcp"
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"repo": {"url": url}}}))
        servers = PiRunner(cwd=tmp_path)._reported_mcp_servers()
        assert [(server.name, server.url) for server in servers] == [("repo", reported)]

    @pytest.mark.parametrize("port", ["9" * 5000, "9" * 5, "65536"])
    def test_project_server_with_a_port_that_pi_rejects_is_not_loaded_and_does_not_raise(self, tmp_path, port):
        (tmp_path / ".pi").mkdir()
        url = f"http://host:{port}/SECRET/mcp"
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"repo": {"url": url}}}))
        assert PiRunner(cwd=tmp_path)._reported_mcp_servers() == []

    def test_redaction_never_raises(self, monkeypatch):
        import airedale.whatwg_url as whatwg
        from airedale.mcp import url_without_secrets

        def boom(value):
            raise RuntimeError("bug in the reader")

        monkeypatch.setattr(whatwg, "parse_url", boom)
        assert url_without_secrets("https://host/SECRET") == "<redacted>"

    def test_long_tool_names_collide_after_shortening(self):
        from airedale.mcp import pi_mcp_tool_name, pi_mcp_tool_names

        long_tool = "t" * 80
        shortened = pi_mcp_tool_name("docs", long_tool)
        other = shortened.removeprefix("mcp__docs__")
        names = pi_mcp_tool_names("docs", [long_tool, other])
        # Pi compares the shortened names: both tools get their own hash suffix.
        assert len(set(names.values())) == 2
        assert names[long_tool] == shortened
        assert names[other] != shortened

    def test_project_servers_are_those_of_the_start_of_the_run(self, tmp_path, monkeypatch):
        """A task that deletes or edits ``.pi/mcp.json`` does not change what is reported about the run."""
        import airedale.harness.base as base
        from airedale.harness.base import AgentRunResult
        from airedale.types import UsageMetrics

        mcp_file = tmp_path / ".pi" / "mcp.json"
        mcp_file.parent.mkdir()
        mcp_file.write_text(json.dumps({"mcpServers": {"repo-docs": {"command": "node"}}}))
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="apm", url="http://localhost/mcp")])

        async def _agent(self, **kwargs):
            # The task runs: pi has read the file, the task deletes it (and adds another server).
            mcp_file.write_text(json.dumps({"mcpServers": {"late": {"command": "node"}}}))
            return AgentRunResult(
                answer="done", usage=UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2), tool_calls=[]
            )

        reported: list[list[str]] = []
        monkeypatch.setattr(PiRunner, "_run_agent", _agent)
        monkeypatch.setattr(
            base, "_annotate_agent_loop_span", lambda **kwargs: reported.append([s.name for s in kwargs["mcp_servers"]])
        )
        _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert reported == [["apm", "repo-docs"]]
        # The next run reads the file again.
        mcp_file.unlink()
        _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        assert reported[-1] == ["apm"]

    def test_reported_mcp_servers_without_project_file(self, tmp_path):
        scenario = McpServerSpec(name="apm", url="http://localhost/mcp")
        assert PiRunner(cwd=tmp_path, mcp_servers=[scenario])._reported_mcp_servers() == [scenario]

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
        assert runner._tool_entries() == [
            "mcp__*",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
        ]
        # Without any scenario server, project servers still keep MCP selected.
        assert PiRunner(cwd=tmp_path, allowed_builtin_tools=["read"])._tool_entries() == [
            "read",
            "mcp__*",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
        ]

    def test_codemode_only_when_allowed_explicitly(self, tmp_path):
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="apm", command="y")])
        assert "codemode" not in runner._tool_entries()
        runner = PiRunner(
            cwd=tmp_path, allowed_builtin_tools=["codemode"], mcp_servers=[McpServerSpec(name="a", command="y")]
        )
        assert runner._tool_entries() == [
            "codemode",
            "mcp__*",
            "list_mcp_resources",
            "list_mcp_resource_templates",
            "read_mcp_resource",
        ]

    def test_no_mcp_resource_tools_without_mcp_servers(self, tmp_path):
        entries = PiRunner(cwd=tmp_path, allowed_builtin_tools=["read"])._tool_entries()
        assert entries == ["read"]

    def test_all_builtins_include_powershell_on_windows(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "_is_windows", lambda: True)
        assert PiRunner(cwd=tmp_path)._effective_builtin_tools() == (*PI_BUILTIN_TOOLS, "powershell")
        # An explicit list stays as it is.
        assert PiRunner(cwd=tmp_path, allowed_builtin_tools=["Bash"])._effective_builtin_tools() == ("bash",)
        monkeypatch.setattr(pi_module, "_is_windows", lambda: False)
        assert PiRunner(cwd=tmp_path)._effective_builtin_tools() == PI_BUILTIN_TOOLS


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
    """The pi path makes agent -> (llm per model call + tool per execution) spans."""

    def test_spans_and_annotations(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        fake = fake_span_llmobs
        agent_spans: list[dict[str, Any]] = []

        def _agent(**kwargs):
            def wrap(func):
                async def inner(*args, **kw):
                    # The agent span is the active parent of the manual spans.
                    span = fake._start("agent", kwargs.get("name"))
                    agent_spans.append(kwargs)
                    try:
                        return await func(*args, **kw)
                    finally:
                        span.finish()

                return inner

            return wrap

        def _no_llm_decorator(**kwargs):
            raise AssertionError("the pi runner must not make the single llm span")

        monkeypatch.setattr(base, "LLMObs", fake)
        monkeypatch.setattr(base, "agent", _agent)
        monkeypatch.setattr(base, "llm", _no_llm_decorator)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        monkeypatch.setattr(base, "current_trace_headers", lambda: {"traceparent": "00-t"})

        command, record = fake_pi(_events_tool_run())
        # A repo server that pi reads itself: it is reported, not configured.
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(json.dumps({"mcpServers": {"repo-docs": {"command": "node"}}}))
        runner = PiRunner(
            cwd=tmp_path,
            pi_command=command,
            mcp_servers=[McpServerSpec(name="apm", url="http://x/mcp", tool_names=("s",))],
        )
        result = _bounded(
            runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s")
        )
        assert result.answer == "The directory is empty."
        assert [(span.kind, span.name) for span in fake.spans] == [
            ("agent", "eval.anthropic_agent_loop"),
            ("llm", "eval.anthropic_agent"),
            ("tool", "ls"),
            ("llm", "eval.anthropic_agent"),
        ]
        agent_span = fake.spans[0]
        first_llm, tool, final_llm = fake.spans[1:]
        # Topology: every manual span is a child of the agent span.
        assert all(span.parent is agent_span for span in fake.spans[1:])
        assert first_llm.kwargs == {"model_name": "claude-sonnet-4-6", "model_provider": "anthropic"}
        # Real times: each span starts after the previous one started.
        assert first_llm.start_ns <= tool.start_ns <= final_llm.start_ns
        assert all(span.end_ns >= span.start_ns for span in fake.spans[1:])

        first = first_llm.annotations
        assert first["metadata"]["sdk"] == "pi"
        assert first["metadata"]["turn"] == 1
        assert first["metadata"]["stop_reason"] == "toolUse"
        assert first["metrics"]["output_tokens"] == 8
        assert final_llm.annotations["metrics"]["output_tokens"] == 12
        names = [definition["name"] for definition in first["tool_definitions"]]
        builtins = [*PI_BUILTIN_TOOLS, "powershell"] if os.name == "nt" else list(PI_BUILTIN_TOOLS)
        assert names == [*builtins, "mcp__apm__s"]
        # The input of a call is the transcript before it.
        assert first["input_data"] == [{"role": "user", "content": "hi"}]
        assert first["output_data"][-1]["role"] == "assistant"
        final_input = final_llm.annotations["input_data"]
        # The first reply: its thinking block, then its text with the tool call.
        assert [message["role"] for message in final_input] == ["user", "assistant", "assistant"]
        assert final_input[1]["content"] == "I should list files."
        assert final_input[-1]["tool_calls"][0]["tool_id"] == "call_1"
        assert final_llm.annotations["output_data"][-1] == {"role": "assistant", "content": "The directory is empty."}
        assert tool.annotations["input_data"] == {"path": "."}
        assert tool.annotations["output_data"] == "(empty directory)"
        assert tool.annotations["metadata"]["duration_ms"] == 2

        agent_annotation = next(a for a in fake.annotations if "_dd" in a.get("metadata", {}))
        manifest = agent_annotation["metadata"]["_dd"]["agent_manifest"]
        assert manifest["framework"] == "pi"
        assert {"name": "mcp__apm__s"} in manifest["tools"]
        assert [server["name"] for server in manifest["dependencies"]["mcp_servers"]] == ["apm", "repo-docs"]
        assert [server["name"] for server in agent_annotation["metadata"]["mcp_servers"]] == ["apm", "repo-docs"]
        assert agent_annotation["metrics"]["output_tokens"] == 20
        # Trace headers reach the MCP server config pi receives, and pi loads
        # the collector extension.
        seen = json.loads(record.read_text())
        assert json.loads(seen["files"]["mcp.json"])["mcpServers"]["apm"]["headers"] == {"traceparent": "00-t"}
        assert "--extension" in seen["argv"]

    def test_manifest_names_tools_with_the_full_catalog(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        # `get-doc` is in tool_names, but the server also has `get_doc`: pi adds a
        # hash suffix to both. The manifest must use the same name as the
        # per-call tool definitions.
        import hashlib

        import airedale.harness.base as base
        from airedale.mcp import McpToolMetadata

        fake = fake_span_llmobs

        def _agent(**kwargs):
            def wrap(func):
                async def inner(*args, **kw):
                    span = fake._start("agent", kwargs.get("name"))
                    try:
                        return await func(*args, **kw)
                    finally:
                        span.finish()

                return inner

            return wrap

        async def _catalog(servers, headers):
            return {("docs", "get-doc"): McpToolMetadata(), ("docs", "get_doc"): McpToolMetadata()}

        monkeypatch.setattr(base, "LLMObs", fake)
        monkeypatch.setattr(base, "agent", _agent)
        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _catalog)
        command, _ = fake_pi(_events_tool_run())
        runner = PiRunner(
            cwd=tmp_path,
            pi_command=command,
            mcp_servers=[McpServerSpec(name="docs", url="http://x/mcp", tool_names=("get-doc",))],
        )
        _bounded(runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s"))
        digest = hashlib.sha256(b"docs\0get-doc").hexdigest()[:8]
        hashed = f"mcp__docs__get_doc_{digest}"
        first_llm = next(span for span in fake.spans if span.kind == "llm")
        assert hashed in [definition["name"] for definition in first_llm.annotations["tool_definitions"]]
        agent_annotation = next(a for a in fake.annotations if "_dd" in a.get("metadata", {}))
        manifest_names = [tool["name"] for tool in agent_annotation["metadata"]["_dd"]["agent_manifest"]["tools"]]
        assert hashed in manifest_names
        assert "mcp__docs__get_doc" not in manifest_names

    def test_no_spans_when_llmobs_is_disabled(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        fake_span_llmobs.enabled = False
        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        command, record = fake_pi(_events_tool_run())
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        result = _bounded(
            runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="pi-s")
        )
        assert result.answer == "The directory is empty."
        assert fake_span_llmobs.spans == []
        # No spans: pi does not load the collector extension.
        assert "--extension" not in json.loads(record.read_text())["argv"]

    def test_real_times_through_the_subprocess(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        messages = [event["message"] for event in _events_tool_run() if event["type"] == "message_end"]
        first, final = messages[1], messages[2]
        events = [
            {"type": "turn_start"},
            {"type": "message_end", "message": {"role": "user", "content": "Q"}},
            # The request started 300 ms before the message ends.
            {"type": "message_end", "message": first, "__now_ms": -300},
            {"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "ls", "args": {}},
            {"__sleep": 0.4},
            {
                "type": "tool_execution_end",
                "isError": False,
                "toolCallId": "call_1",
                "toolName": "ls",
                "result": "x",
                "durationMs": 250,
            },
            {"type": "turn_start"},
            {"type": "compaction_start", "reason": "threshold"},
            {"__sleep": 0.2},
            {"type": "compaction_end", "reason": "threshold", "result": {"summary": "S"}},
            {"type": "message_end", "message": final, "__now_ms": -100},
            {"type": "agent_settled", "aborted": False},
        ]
        command, _ = fake_pi(events)
        result = _bounded(
            PiRunner(cwd=tmp_path, pi_command=command).run(
                model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
            )
        )
        assert result.answer == "The directory is empty."
        llm_1, tool, compaction, llm_2 = fake_span_llmobs.spans
        assert [span.kind for span in fake_span_llmobs.spans] == ["llm", "tool", "task", "llm"]
        ms = 1_000_000
        # Durations come from the recorded times, not from the time of the
        # emission (all spans are sent at the same time, after the run). The
        # bounds allow a slow reader: a record can only arrive late.
        # - llm: from the message timestamp (300 ms before it was written).
        assert 300 * ms <= llm_1.end_ns - llm_1.start_ns <= 30_000 * ms
        assert 100 * ms <= llm_2.end_ns - llm_2.start_ns <= 30_000 * ms
        # - tool: the pi duration, but never past the arrival of its end record.
        assert 0 <= tool.end_ns - tool.start_ns <= 250 * ms + 1000
        assert tool.annotations["metadata"]["duration_ms"] == 250
        assert compaction.end_ns >= compaction.start_ns
        assert llm_1.start_ns < tool.start_ns <= compaction.start_ns < llm_2.end_ns

    def test_cancelled_run_still_sends_spans(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 0.5)
        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        events = [
            {"type": "turn_start"},
            {"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "bash", "args": {}},
        ]
        command, _ = fake_pi(events, FAKE_PI_HANG="1")
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        tool_started = asyncio.Event()

        class _SignalingRecorder(pi_module.PiSpanRecorder):
            def feed(self, event, *, received_ns=None):
                super().feed(event, received_ns=received_ns)
                if event.get("type") == "tool_execution_start":
                    tool_started.set()

        monkeypatch.setattr(pi_module, "PiSpanRecorder", _SignalingRecorder)

        async def scenario():
            task = asyncio.create_task(
                runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
            )
            # Cancel only after the recorder has the tool start.
            await asyncio.wait_for(tool_started.wait(), timeout=10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(asyncio.wait_for(scenario(), timeout=20))
        [tool] = fake_span_llmobs.spans
        assert (tool.kind, tool.name, tool.error) == ("tool", "bash", 1)
        assert tool.annotations["metadata"]["error"] == "tool did not finish"
        assert fake_span_llmobs.active == []

    def test_run_without_the_collector_when_pi_cannot_load_it(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        command, record = fake_pi(_events_tool_run(), FAKE_PI_REJECT_EXTENSION="1")
        result = _bounded(
            PiRunner(cwd=tmp_path, pi_command=command).run(
                model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
            )
        )
        assert result.answer == "The directory is empty."
        runs = [json.loads(line) for line in Path(str(record) + ".runs").read_text().splitlines()]
        assert ["--extension" in argv for argv in runs] == [True, False]
        # The spans use the stream data only.
        assert [span.kind for span in fake_span_llmobs.spans] == ["llm", "tool", "llm"]

    def test_load_failure_is_found_in_the_raw_stderr_when_the_redaction_hides_the_path(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog
    ):
        import airedale.harness.base as base
        from airedale.harness.pi import PI_LLMOBS_EXTENSION_PATH

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        command, record = fake_pi(_events_tool_run(), FAKE_PI_REJECT_EXTENSION="1")
        # A credential that is part of the path of the extension: the redacted text has no path in it.
        secret = PI_LLMOBS_EXTENSION_PATH.name
        assert len(secret) >= 8
        with caplog.at_level("WARNING"):
            result = _bounded(
                PiRunner(
                    cwd=tmp_path, pi_command=command, gateway_base_url="https://gw.example", gateway_api_key=secret
                ).run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
            )
        assert result.answer == "The directory is empty."
        runs = [json.loads(line) for line in Path(str(record) + ".runs").read_text().splitlines()]
        assert ["--extension" in argv for argv in runs] == [True, False]
        # What is shown has no credential.
        assert secret not in caplog.text

    def test_other_startup_failures_are_not_run_again(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        command, record = fake_pi([], FAKE_PI_EXIT="1", FAKE_PI_STDERR="Error: Failed to load extension other.ts")
        with pytest.raises(RuntimeError, match="exited with status 1"):
            _bounded(
                PiRunner(cwd=tmp_path, pi_command=command).run(
                    model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
                )
            )
        assert len(Path(str(record) + ".runs").read_text().splitlines()) == 1

    def test_spans_are_sent_when_the_run_fails(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        events = _events_tool_run()[:6]  # pi stops while the tool runs.
        command, _ = fake_pi(events, FAKE_PI_EXIT="1")
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        with pytest.raises(RuntimeError, match="exited with status 1"):
            _bounded(runner.run(model=MODEL, system_prompt="SYS", user_prompt="Q", prompt_version="v", harness="h"))
        kinds = [(span.kind, span.name) for span in fake_span_llmobs.spans]
        assert kinds == [("llm", "eval.anthropic_agent"), ("tool", "ls")]
        tool = fake_span_llmobs.spans[1]
        assert tool.error == 1
        assert tool.annotations["metadata"]["error"] == "tool did not finish"


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


# --------------------------------------------------------------------------- #
# Process stop on Windows (no process groups, no os.killpg)
# --------------------------------------------------------------------------- #
class _FakeWindowsProcess:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1


class TestWindowsStop:
    @pytest.fixture
    def windows(self, monkeypatch):
        import airedale.harness.pi as pi_module

        def no_killpg(*args):
            raise AssertionError("os.killpg must not be used on Windows")

        monkeypatch.setattr(pi_module, "_is_windows", lambda: True)
        monkeypatch.setattr(pi_module.os, "killpg", no_killpg, raising=False)
        monkeypatch.setattr(pi_module, "PI_TERMINATE_GRACE_SECONDS", 1.0)
        calls: list[tuple[str, ...]] = []
        return pi_module, calls

    def _fake_exec(
        self,
        calls,
        process,
        *,
        stops_pi: bool = True,
        error: type[Exception] | None = None,
        hangs: bool = False,
        helpers: list | None = None,
    ):
        async def fake_exec(*argv, **kwargs):
            calls.append(argv)
            if error is not None:
                raise error("taskkill")

            class _Taskkill:
                returncode: int | None = None
                killed = False

                async def wait(self) -> int:
                    if hangs:
                        await asyncio.sleep(600)
                    if stops_pi:
                        process.returncode = 1
                    self.returncode = 0
                    return 0

                def kill(self) -> None:
                    self.killed = True
                    self.returncode = 1

            helper = _Taskkill()
            if helpers is not None:
                helpers.append(helper)
            return helper

        return fake_exec

    def test_taskkill_that_does_not_end_is_stopped(self, windows, monkeypatch):
        pi_module, calls = windows
        process = _FakeWindowsProcess()
        helpers: list = []
        monkeypatch.setattr(
            pi_module.asyncio,
            "create_subprocess_exec",
            self._fake_exec(calls, process, hangs=True, helpers=helpers),
        )
        _bounded(pi_module._stop_process_group(process))
        (helper,) = helpers
        assert helper.killed
        assert helper.returncode is not None
        assert process.killed

    def test_taskkill_stops_the_process_tree(self, windows, monkeypatch):
        pi_module, calls = windows
        process = _FakeWindowsProcess()
        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", self._fake_exec(calls, process))
        _bounded(pi_module._stop_process_group(process))
        assert calls == [("taskkill", "/PID", "4242", "/T", "/F")]
        assert process.returncode == 1
        assert not process.killed

    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
    def test_pi_is_killed_when_taskkill_cannot_run(self, windows, monkeypatch, error):
        pi_module, calls = windows
        process = _FakeWindowsProcess()
        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", self._fake_exec(calls, process, error=error))
        _bounded(pi_module._stop_process_group(process))
        assert len(calls) == 1
        assert process.killed

    def test_pi_is_killed_when_taskkill_does_not_stop_it(self, windows, monkeypatch):
        pi_module, calls = windows
        process = _FakeWindowsProcess()
        monkeypatch.setattr(
            pi_module.asyncio, "create_subprocess_exec", self._fake_exec(calls, process, stops_pi=False)
        )
        _bounded(pi_module._stop_process_group(process))
        assert process.killed

    def test_nothing_to_do_when_pi_already_exited(self, windows, monkeypatch):
        pi_module, calls = windows
        process = _FakeWindowsProcess(returncode=0)
        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", self._fake_exec(calls, process))
        _bounded(pi_module._stop_process_group(process))
        assert calls == []
        assert not process.killed

    def test_pi_starts_without_a_new_session_on_windows(self, windows, monkeypatch, tmp_path):
        pi_module, _ = windows
        seen: dict[str, Any] = {}

        async def fake_exec(*argv, **kwargs):
            seen.update(kwargs)
            raise FileNotFoundError(argv[0])

        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", fake_exec)
        # The Windows lookup needs a program to find: a fake ``pi.cmd`` on a test ``PATH`` (no real pi is needed).
        (tmp_path / "pi.cmd").write_text("@echo off\n")
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("PATHEXT", ".cmd")
        with pytest.raises(RuntimeError, match="pi executable not found"):
            _run(PiRunner(cwd=tmp_path, pi_command=["pi"]))
        assert seen["start_new_session"] is False


class TestResolvePiProgram:
    """POSIX: the program is passed as it is. Windows: own lookup with the ``PATH`` / ``PATHEXT`` / cwd of the child."""

    @staticmethod
    def _touch(path, text="x"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)
        return path

    @staticmethod
    def _capture(monkeypatch):
        import airedale.harness.pi as pi_module

        seen: list[Any] = []

        async def fake_exec(*argv, **kwargs):
            seen.append(argv[0])
            raise FileNotFoundError(argv[0])

        monkeypatch.setattr(pi_module.asyncio, "create_subprocess_exec", fake_exec)
        return pi_module, seen

    @pytest.mark.parametrize("through_env", [False, True], ids=["default-pi", "AIREDALE_PI_COMMAND"])
    def test_posix_relative_path_entry_is_searched_from_the_child_directory(
        self, tmp_path, fake_pi, monkeypatch, through_env
    ):
        command, _ = fake_pi(_events_tool_run())
        airedale_dir, workspace, fallback = tmp_path / "airedale", tmp_path / "ws", tmp_path / "fallback"
        workspace.mkdir()
        # Only in the directory of Airedale: a broken ``pi`` that must not run.
        self._touch(airedale_dir / "relative-bin" / "pi", "#!/bin/sh\nexit 99\n")
        self._touch(fallback / "pi", f'#!/bin/sh\nexec "{command[0]}" "{command[1]}" "$@"\n')
        monkeypatch.chdir(airedale_dir)
        monkeypatch.setenv("PATH", f"relative-bin{os.pathsep}{fallback}")
        if through_env:
            monkeypatch.setenv("AIREDALE_PI_COMMAND", "pi")
            runner = PiRunner(cwd=workspace)
        else:
            runner = PiRunner(cwd=workspace, pi_command=["pi"])
        assert _run(runner).error is None

    def test_posix_command_is_not_resolved(self, tmp_path, monkeypatch):
        pi_module, seen = self._capture(monkeypatch)
        monkeypatch.setattr(pi_module, "_is_windows", lambda: False)
        monkeypatch.setenv("PATH", str(tmp_path))
        self._touch(tmp_path / "pi")
        with pytest.raises(RuntimeError, match="pi executable not found"):
            _run(PiRunner(cwd=tmp_path, pi_command=["pi"]))
        assert seen == ["pi"]

    def test_posix_path_wins_over_path_with_another_case(self, tmp_path, fake_pi, monkeypatch):
        command, _ = fake_pi(_events_tool_run())
        good = tmp_path / "good"
        self._touch(good / "pi", f'#!/bin/sh\nexec "{command[0]}" "{command[1]}" "$@"\n')
        bad = tmp_path / "bad"
        self._touch(bad / "pi", "#!/bin/sh\nexit 99\n")
        monkeypatch.setenv("PATH", str(good))
        monkeypatch.setenv("Path", str(bad))
        assert _run(PiRunner(cwd=tmp_path, pi_command=["pi"])).error is None

    # -- simulated Windows ------------------------------------------------------------------------------------
    @pytest.fixture
    def windows(self, monkeypatch):
        pi_module, seen = self._capture(monkeypatch)
        monkeypatch.setattr(pi_module, "_is_windows", lambda: True)
        monkeypatch.setattr(pi_module, "WindowsJob", type("J", (), {"attach": staticmethod(lambda pid: None)}))
        return pi_module, seen

    def test_windows_cmd_shim_in_path_is_found(self, windows, tmp_path):
        pi_module, seen = windows
        shim = self._touch(tmp_path / "bin" / "pi.cmd")
        resolved = pi_module._resolve_windows_program(
            "pi", {"Path": str(tmp_path / "bin"), "PATHEXT": ".exe" + os.pathsep + ".cmd"}, tmp_path
        )
        assert resolved == str(shim)

    def test_windows_current_directory_of_airedale_is_not_a_candidate(self, windows, tmp_path, monkeypatch):
        pi_module, seen = windows
        airedale_dir, workspace = tmp_path / "airedale", tmp_path / "ws"
        workspace.mkdir()
        self._touch(airedale_dir / "pi.cmd")  # Planted next to Airedale.
        trusted = self._touch(tmp_path / "trusted" / "pi.cmd")
        monkeypatch.chdir(airedale_dir)
        env = {"PATH": str(tmp_path / "trusted"), "PATHEXT": ".cmd"}
        assert pi_module._resolve_windows_program("pi", env, workspace) == str(trusted)
        # And nothing is found when only the current directory has it.
        assert pi_module._resolve_windows_program("pi", {"PATH": "", "PATHEXT": ".cmd"}, workspace) is None
        assert pi_module._resolve_windows_program("pi", {"PATHEXT": ".cmd"}, airedale_dir) is None

    def test_windows_relative_path_entry_and_relative_command_use_the_child_directory(
        self, windows, tmp_path, monkeypatch
    ):
        pi_module, seen = windows
        airedale_dir, workspace = tmp_path / "airedale", tmp_path / "ws"
        self._touch(airedale_dir / "rel" / "pi.cmd")  # Only in the directory of Airedale: not used.
        in_workspace = self._touch(workspace / "rel" / "pi.cmd")
        tool = self._touch(workspace / "tools" / "mypi.cmd")
        monkeypatch.chdir(airedale_dir)
        env = {"PATH": "rel", "PATHEXT": ".cmd"}
        assert pi_module._resolve_windows_program("pi", env, workspace) == str(in_workspace)
        assert pi_module._resolve_windows_program("pi", env, tmp_path / "other") is None
        assert pi_module._resolve_windows_program("tools\\mypi", env, workspace) == str(tool)
        assert pi_module._resolve_windows_program("tools/mypi.cmd", env, workspace) == str(tool)
        assert pi_module._resolve_windows_program(str(tool), env, tmp_path / "other") == str(tool)

    def test_windows_child_pathext_is_used_not_the_one_of_airedale(self, windows, tmp_path, monkeypatch):
        pi_module, seen = windows
        shim = self._touch(tmp_path / "bin" / "pi.cmd")
        monkeypatch.setenv("PATHEXT", ".exe")
        resolved = pi_module._resolve_windows_program(
            "pi", {"pAtH": str(tmp_path / "bin"), "PaThExT": ".cmd"}, tmp_path
        )
        assert resolved == str(shim)

    def test_windows_extensionless_shell_script_is_not_chosen_over_the_cmd_shim(self, windows, tmp_path):
        pi_module, seen = windows
        self._touch(tmp_path / "bin" / "pi")
        shim = self._touch(tmp_path / "bin" / "pi.cmd")
        env = {"PATH": str(tmp_path / "bin"), "PATHEXT": ".exe" + os.pathsep + ".cmd"}
        assert pi_module._resolve_windows_program("pi", env, tmp_path) == str(shim)

    def test_windows_spawn_receives_the_absolute_path(self, windows, tmp_path, monkeypatch):
        pi_module, seen = windows
        shim = self._touch(tmp_path / "bin" / "pi.cmd")
        monkeypatch.setenv("PATH", str(tmp_path / "bin"))
        monkeypatch.setenv("PATHEXT", ".cmd")
        with pytest.raises(RuntimeError, match="pi executable not found"):
            _run(PiRunner(cwd=tmp_path, pi_command=["pi", "--flag"]))
        assert seen == [str(shim)]

    def test_windows_not_found_raises_before_the_spawn(self, windows, tmp_path, monkeypatch):
        pi_module, seen = windows
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(RuntimeError, match=r"pi executable not found \('pi'\)"):
            _run(PiRunner(cwd=tmp_path, pi_command=["pi"]))
        assert seen == []


# --------------------------------------------------------------------------- #
# MCP metadata probe (LLMObs tool definitions) environment
# --------------------------------------------------------------------------- #
PROBE_SERVER = textwrap.dedent(
    """
    import json, os, sys
    with open(sys.argv[1], "w") as handle:
        json.dump(dict(os.environ), handle)
    """
)


@pytest.mark.parametrize("server_env", [{}, {"SERVER_SETTING": "1"}])
def test_mcp_metadata_probe_gets_no_operator_credentials(tmp_path, monkeypatch, server_env):
    # The probe starts a stdio server before pi runs. The MCP SDK gives it only
    # its default variables (HOME, PATH, ...) and the server's own ``env``, not
    # the airedale environment. This test fails if that changes.
    from airedale.mcp import _mcp_server_tool_metadata

    secrets = {
        "DD_API_KEY": "dd-secret",
        "DD_APP_KEY": "dd-app-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "OPENAI_API_KEY": "openai-secret",
        "MY_GATEWAY_TOKEN": "gateway-secret",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    script = tmp_path / "probe_server.py"
    script.write_text(PROBE_SERVER)
    record = tmp_path / "probe_env.json"
    server = McpServerSpec(name="probe", command=sys.executable, args=(str(script), str(record)), env=server_env)
    catalog = _bounded(_mcp_server_tool_metadata(server, {}))
    assert catalog == {}  # the fake server does not speak MCP
    seen = json.loads(record.read_text())
    for name, value in server_env.items():
        assert seen[name] == value
    for name, value in secrets.items():
        assert name not in seen
        assert value not in seen.values()


def test_tool_list_follows_every_page():
    from types import SimpleNamespace

    from airedale.mcp import _list_all_tools

    pages = {
        None: SimpleNamespace(tools=["a"], nextCursor="p2"),
        "p2": SimpleNamespace(tools=["b"], nextCursor="p3"),
        "p3": SimpleNamespace(tools=["c"], nextCursor=None),
    }
    calls: list[str | None] = []

    class _Session:
        async def list_tools(self, cursor: str | None = None):
            calls.append(cursor)
            return pages[cursor]

    assert _bounded(_list_all_tools(_Session())) == ["a", "b", "c"]  # type: ignore[arg-type]
    assert calls == [None, "p2", "p3"]


def test_tool_list_stops_on_a_repeated_cursor():
    from types import SimpleNamespace

    from airedale.mcp import _list_all_tools

    class _Session:
        calls = 0

        async def list_tools(self, cursor: str | None = None):
            self.calls += 1
            return SimpleNamespace(tools=[self.calls], nextCursor="same")

    session = _Session()
    assert _bounded(_list_all_tools(session)) == [1, 2]  # type: ignore[arg-type]


def test_tool_list_with_more_pages_than_the_cap_fails_and_gives_no_partial_catalog(monkeypatch):
    from types import SimpleNamespace

    import airedale.mcp as mcp_module
    from airedale.mcp import MCP_TOOL_LIST_MAX_PAGES, McpToolListIncompleteError, _list_all_tools

    def session(pages):
        class _Session:
            calls = 0

            async def list_tools(self, cursor: str | None = None):
                self.calls += 1
                more = self.calls < pages
                return SimpleNamespace(
                    tools=[SimpleNamespace(name=f"t{self.calls}")], nextCursor=f"c{self.calls}" if more else None
                )

        return _Session()

    # Exactly the cap: complete. One more page: failed.
    assert len(_bounded(_list_all_tools(session(MCP_TOOL_LIST_MAX_PAGES)))) == MCP_TOOL_LIST_MAX_PAGES  # type: ignore[arg-type]
    with pytest.raises(McpToolListIncompleteError, match="incomplete"):
        _bounded(_list_all_tools(session(MCP_TOOL_LIST_MAX_PAGES + 1)))  # type: ignore[arg-type]

    # The probe of the metadata catalog treats it as a failure: nothing of that server is in the catalog.
    async def fake_list(server, headers, *, all_pages=False):
        return await _list_all_tools(session(MCP_TOOL_LIST_MAX_PAGES + 1))  # type: ignore[arg-type]

    monkeypatch.setattr(mcp_module, "_list_mcp_tools", fake_list)
    server = McpServerSpec(name="s", url="http://localhost/mcp")
    assert _bounded(mcp_module._mcp_tool_metadata_catalog([server], {})) == {}


def test_health_probe_reads_only_the_first_tools_page(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    import airedale.mcp as mcp_module

    calls: list[str | None] = []

    class _Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, cursor: str | None = None):
            calls.append(cursor)
            return SimpleNamespace(tools=[cursor or "first"], nextCursor=None if cursor else "p2")

    @asynccontextmanager
    async def _client(*args, **kwargs):
        yield (None, None, None)

    monkeypatch.setattr(mcp_module, "ClientSession", _Session)
    monkeypatch.setattr(mcp_module, "streamablehttp_client", _client)
    server = McpServerSpec(name="s", url="http://localhost/mcp")
    assert _bounded(mcp_module._list_mcp_tools(server, {})) == ["first"]
    assert calls == [None]
    assert _bounded(mcp_module._list_mcp_tools(server, {}, all_pages=True)) == ["first", "p2"]


# --------------------------------------------------------------------------- #
# AIREDALE_PI_COMMAND parsing
# --------------------------------------------------------------------------- #
class TestPiCommandEnv:
    @pytest.fixture
    def pi_module(self, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.delenv("AIREDALE_PI_COMMAND", raising=False)
        return pi_module

    def test_default(self, pi_module):
        assert pi_module._default_pi_command() == ["pi"]

    def test_posix_rules(self, pi_module, monkeypatch):
        monkeypatch.setattr(pi_module, "_is_windows", lambda: False)
        monkeypatch.setenv("AIREDALE_PI_COMMAND", "npx -y '@earendil-works/pi-coding-agent'")
        assert pi_module._default_pi_command() == ["npx", "-y", "@earendil-works/pi-coding-agent"]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (r"C:\tools\pi.exe", [r"C:\tools\pi.exe"]),
            (r'"C:\Program Files\pi\pi.exe" --flag', [r"C:\Program Files\pi\pi.exe", "--flag"]),
            (r'node C:\pi\cli.js "a b" c', ["node", r"C:\pi\cli.js", "a b", "c"]),
            (r'x \\"quoted\\" y', ["x", "\\quoted\\", "y"]),
            (r"x \"literal\" y", ["x", '"literal"', "y"]),
            (r'x "" y', ["x", "", "y"]),
            ("  x\t y  ", ["x", "y"]),
            # The program name: up to the next quote, backslashes literal.
            (r'"C:\Program Files\pi\pi.exe"--flag', [r"C:\Program Files\pi\pi.exe", "--flag"]),
            # In a quoted part, "" is a literal quote.
            ('x "a"""', ["x", 'a"']),
            ('node -e "console.log("""hello""")"', ["node", "-e", 'console.log("hello")']),
        ],
    )
    def test_windows_rules(self, pi_module, monkeypatch, raw, expected):
        monkeypatch.setattr(pi_module, "_is_windows", lambda: True)
        monkeypatch.setenv("AIREDALE_PI_COMMAND", raw)
        assert pi_module._default_pi_command() == expected

    @pytest.mark.parametrize("windows", [False, True])
    def test_json_array(self, pi_module, monkeypatch, windows):
        monkeypatch.setattr(pi_module, "_is_windows", lambda: windows)
        monkeypatch.setenv("AIREDALE_PI_COMMAND", json.dumps([r"C:\tools\pi.exe", "--x", "a b"]))
        assert pi_module._default_pi_command() == [r"C:\tools\pi.exe", "--x", "a b"]

    @pytest.mark.parametrize("raw", ["[1]", "[]", '[""]', '["", "x"]'])
    def test_bad_json_array(self, pi_module, monkeypatch, raw):
        monkeypatch.setenv("AIREDALE_PI_COMMAND", raw)
        with pytest.raises(ValueError, match="AIREDALE_PI_COMMAND"):
            pi_module._default_pi_command()

    def test_empty_arguments_after_the_program_are_kept(self, pi_module, monkeypatch):
        monkeypatch.setenv("AIREDALE_PI_COMMAND", json.dumps(["sh", "-c", 'exec pi "$@"', ""]))
        assert pi_module._default_pi_command() == ["sh", "-c", 'exec pi "$@"', ""]

    def test_value_starting_with_a_bracket_that_is_not_json(self, pi_module, monkeypatch):
        # A path such as "[tools]/pi" stays a command line, as before.
        monkeypatch.setattr(pi_module, "_is_windows", lambda: False)
        monkeypatch.setenv("AIREDALE_PI_COMMAND", "[tools]/pi --x")
        assert pi_module._default_pi_command() == ["[tools]/pi", "--x"]


class TestStrictScenarioValidation:
    """The strict check wants an exact accept; the guard for project entries keeps its fail-closed meaning."""

    @pytest.mark.parametrize("url", ["https://example.test/mcp", "http://127.0.0.1:8080/mcp", "http://[::1]:9/mcp"])
    def test_urls_that_are_decided_exactly_are_accepted(self, tmp_path, url):
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="docs", url=url)])
        assert list(runner._scenario_pi_configs({}, strict=True)) == ["docs"]

    def test_the_punycode_form_of_an_idna_host_is_accepted(self, tmp_path):
        # The documented way around the limit: the ASCII form of the host.
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="docs", url="http://xn--exmple-cua.test/mcp")])
        assert list(runner._scenario_pi_configs({}, strict=True)) == ["docs"]

    def test_undecided_url_is_refused_in_strict_mode_only(self, tmp_path):
        from airedale.pi_mcp import validate_server_config

        config = {"url": "http://[not-an-ip]/mcp"}
        assert isinstance(validate_server_config("x", config), dict)  # The guard: possibly accepted.
        assert "cannot be checked exactly" in str(validate_server_config("x", config, exact=True))

    def test_project_entry_with_an_undecided_url_still_replaces_the_scenario_server(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "mcp.json").write_text(
            json.dumps({"mcpServers": {"docs": {"url": "http://[not-an-ip]/mcp"}}})
        )
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="docs", command="scenario")])
        with pytest.raises(ValueError, match="docs"):
            runner._check_project_mcp_collisions({})

    def test_oauth_urls_of_a_scenario_server_are_checked_exactly_too(self):
        from airedale.pi_mcp import validate_server_config

        config = {"url": "https://example.test", "oauth": {"authServerMetadataUrl": "http://[x]/meta"}}
        assert "oauth.authServerMetadataUrl" in str(validate_server_config("x", config, exact=True))


class TestNoRawValueInDiagnostics:
    """A URL, a header, an environment value or a stdout line must not be copied into an error or a log."""

    SECRET = "SECRET-api-key-0123"

    @pytest.mark.parametrize(
        "url",
        [
            f"http://[not-an-ip]/mcp?api_key={SECRET}",
            f"http://user:{SECRET}@[1::2::3]/mcp",
            f"https://\u00ad127.1/mcp#{SECRET}",
            f"http://exämple.test/{SECRET}/mcp",
        ],
    )
    def test_refused_server_url_is_not_in_the_error(self, tmp_path, url):
        runner = PiRunner(cwd=tmp_path, mcp_servers=[McpServerSpec(name="docs", url=url)])
        with pytest.raises(ValueError, match="cannot be checked exactly") as error:
            runner._scenario_pi_configs({}, strict=True)
        assert self.SECRET not in str(error.value)
        assert "docs" in str(error.value)  # The server name and the field stay.
        assert "url" in str(error.value)

    @pytest.mark.parametrize("field", ["callbackUrl", "authServerMetadataUrl"])
    def test_refused_oauth_url_is_not_in_the_error(self, field):
        from airedale.pi_mcp import validate_server_config

        config = {"url": "https://example.test", "oauth": {field: f"http://[x]/cb?token={self.SECRET}"}}
        problem = str(validate_server_config("docs", config, exact=True))
        assert field in problem
        assert self.SECRET not in problem

    def test_every_error_of_the_scenario_validation_is_free_of_the_values(self, tmp_path):
        # Headers and environment values with a credential: the messages name the field, not the value.
        from airedale.pi_mcp import validate_server_config

        for config in (
            {"url": "https://e.test", "headers": {"X-Api-Key": 5}},
            {"command": "x", "env": {"TOKEN": 5}},
            {"command": "x", "args": [self.SECRET, 5]},
            {"url": "ftp://e.test/?k=" + self.SECRET},
            {"command": "x", "cwd": 5, "env": {"TOKEN": self.SECRET}},
        ):
            assert self.SECRET not in str(validate_server_config("docs", config, exact=True))

    def test_stdout_lines_are_not_logged_with_their_text(self, caplog):
        collector = _PiEventCollector()
        with caplog.at_level("DEBUG"):
            collector.feed_line(f"not json {self.SECRET}\n".encode())
            collector.feed_line(b"\xff " + self.SECRET.encode() + b"\n")
        assert collector.invalid_lines == 2
        assert self.SECRET not in caplog.text
        assert "Ignoring" in caplog.text


class TestEndRecordNamesTheSameTool:
    """Pi repeats the name of a tool call in its end record (pi 1.4.0 and 1.5.0): another name is damage."""

    def _events(self, end_name):
        events = _events_tool_run()
        end = next(event for event in events if event["type"] == "tool_execution_end")
        start = next(event for event in events if event["type"] == "tool_execution_start")
        end["toolName"] = end_name
        return events, start, end

    def test_end_with_another_tool_name_fails_the_run(self):
        events, start, end = self._events("write")
        assert start["toolName"] != "write"
        error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
        assert "malformed JSONL" in str(error)

    def test_the_execution_is_not_closed_by_that_record(self):
        events, _, _ = self._events("write")
        collector = _collect(events)
        assert [call.tool_name for call in collector.tool_executions] == [events[5]["toolName"]]
        assert not collector.tool_executions[0].finished
        assert collector.tool_executions[0].result is None

    def test_end_with_the_same_name_is_fine(self):
        events, start, _ = self._events("write")
        events[events.index(next(e for e in events if e["type"] == "tool_execution_end"))]["toolName"] = start[
            "toolName"
        ]
        assert _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_end_without_a_start_keeps_its_own_name(self):
        events, _, end = self._events("write")
        events = [event for event in events if event["type"] != "tool_execution_start"]
        events[
            next(
                i
                for i, e in enumerate(events)
                if e.get("type") == "message_end" and e["message"].get("stopReason") == "toolUse"
            )
        ]["message"]["content"][2]["name"] = "write"
        collector = _collect(events)
        assert collector.result(model=MODEL, returncode=0, stderr_tail="").error is None
        assert [call.tool_name for call in collector.tool_executions] == ["write"]

    def test_the_span_recorder_does_not_give_the_result_to_another_tool(self):
        from airedale.harness.pi_spans import PiSpanRecorder

        recorder = PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v")
        recorder.feed({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read", "args": {}})
        recorder.feed(
            {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "result": "SECRET", "isError": False}
        )
        assert recorder.records == []  # The record is ignored: the span of "read" is still open.
        recorder.close_tool("c1", 1)
        assert [record.name for record in recorder.records] == ["read"]
        assert not recorder.records[0].output_data


class TestTranscriptCallNamesTheSameTool:
    """The tool call of the assistant message and its execution (same id) have the same name in pi 1.4.0 and 1.5.0."""

    def test_a_transcript_call_with_another_name_than_its_execution_fails_the_run(self):
        events = _events_tool_run()
        events[4]["message"]["content"][2]["name"] = "read"  # The execution is "ls".
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        assert "not the one of the same call" in str(result.error)

    def test_the_result_of_the_execution_is_not_given_to_the_call_with_another_name(self):
        events = _events_tool_run()
        events[4]["message"]["content"][2]["name"] = "read"
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        [call] = [call for call in result.tool_calls if call.name == "read"]
        assert call.result is None
        assert call.error == "tool did not run"
        assert "(empty directory)" not in str(call.result)

    def test_same_names_are_matched(self):
        result = _collect(_events_tool_run()).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [(call.name, call.result) for call in result.tool_calls] == [("ls", "(empty directory)")]

    def test_a_call_without_a_name_matches_an_execution_without_a_name(self):
        events = _events_tool_run()
        events[4]["message"]["content"][2]["name"] = ""
        for event in events:
            if event["type"].startswith("tool_execution"):
                event["toolName"] = ""
        assert _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error is None

    def test_parallel_calls_with_one_id_keep_their_own_message(self, pi_reference):
        for version, events in pi_reference({"action": "parallel"}).items():
            error = _collect(events).result(model=MODEL, returncode=0, stderr_tail="").error
            assert error is not None and "at the same time" in str(error), version


class TestEmptyNameIsNotTheToolCalledUnknown:
    """An empty name is shown as "unknown", but it is not the tool that is literally called "unknown"."""

    @staticmethod
    def _events(start, end, block):
        events = _events_tool_run()
        events[4]["message"]["content"][2]["name"] = block
        for event in events:
            if event["type"] == "tool_execution_start":
                event["toolName"] = start
            elif event["type"] == "tool_execution_end":
                event["toolName"] = end
        return events

    @pytest.mark.parametrize(("start", "end"), [("unknown", ""), ("", "unknown")])
    def test_end_record_with_the_other_spelling_is_malformed(self, start, end):
        collector = _collect(self._events(start, end, start))
        assert collector.invalid_lines == 1
        assert not collector.tool_executions[0].finished
        assert "malformed JSONL" in str(collector.result(model=MODEL, returncode=0, stderr_tail="").error)

    @pytest.mark.parametrize(("execution", "block"), [("unknown", ""), ("", "unknown")])
    def test_transcript_call_with_the_other_spelling_does_not_take_the_result(self, execution, block):
        result = _collect(self._events(execution, execution, block)).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None and "not the one of the same call" in str(result.error)
        assert [call.result for call in result.tool_calls] == [None]

    @pytest.mark.parametrize("name", ["", "unknown", "ls"])
    def test_the_same_name_in_the_three_places_is_matched(self, name):
        result = _collect(self._events(name, name, name)).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.result for call in result.tool_calls] == ["(empty directory)"]

    @pytest.mark.parametrize(("start", "end"), [("unknown", ""), ("", "unknown")])
    def test_span_recorder_does_not_give_the_result_to_the_other_spelling(self, start, end):
        from airedale.harness.pi_spans import PiSpanRecorder

        recorder = PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v")
        recorder.feed({"type": "tool_execution_start", "toolCallId": "c1", "toolName": start, "args": {}})
        recorder.feed(
            {"type": "tool_execution_end", "toolCallId": "c1", "toolName": end, "result": "SECRET", "isError": False}
        )
        recorder.close_tool("c1", 1)
        assert [record.name for record in recorder.records] == ["unknown"]
        assert not recorder.records[0].output_data

    @pytest.mark.parametrize("name", ["", "unknown"])
    def test_span_recorder_matches_the_same_spelling(self, name):
        from airedale.harness.pi_spans import PiSpanRecorder

        recorder = PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v")
        recorder.feed({"type": "tool_execution_start", "toolCallId": "c1", "toolName": name, "args": {}})
        recorder.feed(
            {"type": "tool_execution_end", "toolCallId": "c1", "toolName": name, "result": "out", "isError": False}
        )
        assert [record.output_data for record in recorder.records] == ["out"]


class TestAllEightNameMutations:
    """Every change of a name between ``""`` and ``"unknown"`` (end record and transcript call, both ways), in a stream
    with the collector and the span recorder together: malformed, no result attached, no span closed with the result."""

    @staticmethod
    def _stream(name):
        events = _events_tool_run()
        events[4]["message"]["content"][2]["name"] = name
        for event in events:
            if event["type"] in ("tool_execution_start", "tool_execution_end"):
                event["toolName"] = name
        return events

    @pytest.mark.parametrize("name", ["", "unknown", "ls"])
    def test_unchanged_names_are_matched_everywhere(self, name):
        from airedale.harness.pi_spans import PiSpanRecorder

        recorder = PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v")
        collector = _collect(self._stream(name), spans=recorder)
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert [call.result for call in result.tool_calls] == ["(empty directory)"]
        assert [record.output_data for record in recorder.records if record.kind == "tool"] == ["(empty directory)"]

    @pytest.mark.parametrize(
        ("name", "other"), [("", "unknown"), ("unknown", ""), ("ls", "write"), ("", "ls")], ids=lambda v: repr(v)
    )
    @pytest.mark.parametrize("where", ["end", "transcript"])
    def test_changed_name_is_refused_everywhere(self, name, other, where):
        from airedale.harness.pi_spans import PiSpanRecorder

        events = self._stream(name)
        if where == "end":
            for event in events:
                if event["type"] == "tool_execution_end":
                    event["toolName"] = other
        else:
            events[4]["message"]["content"][2]["name"] = other
        recorder = PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v")
        collector = _collect(events, spans=recorder)
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        assert all(call.result is None for call in result.tool_calls)
        if where == "end":
            # (A changed transcript name leaves the execution and its span right: the span has no transcript.)
            assert all(not record.output_data for record in recorder.records if record.kind == "tool")


class TestSumOfCostsMustBeFinite:
    """Each cost can be finite and the sum of several infinite: the total cost of the run is damaged usage."""

    @staticmethod
    def _with_costs(first, second):
        events = _events_tool_run()
        for event in events:
            message = event.get("message") if event.get("type") == "message_end" else None
            if message and message.get("role") == "assistant":
                message["usage"]["cost"] = {"total": first if message["stopReason"] == "toolUse" else second}
        return events

    def test_two_costs_that_add_up_to_infinity_fail_the_run(self):
        import sys

        events = self._with_costs(sys.float_info.max, sys.float_info.max)
        collector = _collect(events)
        result = collector.result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None and "damaged token usage" in str(result.error)
        assert "total cost is not finite" in str(result.error)
        assert math.isfinite(result.usage.estimated_cost_usd)
        assert "inf" not in json.dumps(result.usage.to_llmobs_metrics(), default=str).lower()

    def test_costs_that_fit_are_added(self):
        import sys

        events = self._with_costs(sys.float_info.max / 4, sys.float_info.max / 4)
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is None
        assert math.isfinite(result.usage.estimated_cost_usd)
        assert result.usage.estimated_cost_usd == sys.float_info.max / 2

    def test_three_records_cannot_sneak_in_after_the_overflow(self):
        import sys

        events = self._with_costs(sys.float_info.max, sys.float_info.max)
        events.insert(
            -1, {"type": "entry_appended", "entry": {"type": "usage", "usage": _usage(1, 1, cost=sys.float_info.max)}}
        )
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        assert result.error is not None
        assert math.isfinite(result.usage.estimated_cost_usd)


# --------------------------------------------------------------------------- #
# Option S: no raw value of pi, a provider or the model in an error, a log or a span
# --------------------------------------------------------------------------- #
SECRETS = {
    "gateway key": "gw-key-0123456789abcdef",
    "short token": "a1b2c",
    "bearer": "Bearer tok-ABCDEFGHIJKLMNOP",
    "url key": "https://user:URL-CRED-0123456789@host.example/v1?api_key=URL-QUERY-0123456789",
    "escaped": "esc-\\u00e9-secret-0123456789",
    "cut fragment": "FRAGMENT-0123456789",
}


def _all_text(result, caplog, fake_span_llmobs=None) -> str:
    """Everything that a reader of the run can see: the error, the logs, the spans and the result."""
    parts = [str(result.error), caplog.text, result.answer, repr(result.tool_calls), repr(result.segments)]
    if fake_span_llmobs is not None:
        for span in fake_span_llmobs.spans:
            parts.append(
                json.dumps({"tags": span.tags, "annotations": span.annotations, "error": span.error}, default=str)
            )
        parts.append(json.dumps(fake_span_llmobs.annotations, default=str))
    return "\n".join(parts)


def _report_files(directory):
    return sorted(Path(directory).glob("pi-run-*.log"))


class TestNoRawValues:
    @pytest.mark.parametrize("secret", SECRETS.values(), ids=SECRETS.keys())
    def test_stderr_of_a_failed_run_is_only_in_the_private_file(
        self, tmp_path, fake_pi, caplog, private_report_dir, secret
    ):
        command, _ = fake_pi([], FAKE_PI_EXIT="3", FAKE_PI_STDERR=f"first line {secret} last line\n")
        with caplog.at_level("DEBUG"):
            result = _run(
                PiRunner(cwd=tmp_path, pi_command=command, gateway_base_url="https://gw", gateway_api_key="x")
            )
        assert "status 3" in str(result.error)
        assert secret not in _all_text(result, caplog)
        [report] = _report_files(private_report_dir)
        assert str(report) in str(result.error)
        assert secret.encode("utf-8", "surrogatepass") in report.read_bytes() or secret in report.read_text(
            errors="replace"
        )
        assert report.stat().st_mode & 0o777 == 0o600

    def test_stderr_written_just_before_exit_is_not_lost_when_the_reader_is_late(
        self, tmp_path, fake_pi, monkeypatch, private_report_dir
    ):
        """The reader of stderr may run after pi and its stdout are gone: the cleanup must let it finish."""
        import airedale.harness.pi as pi_module

        real_drain = pi_module._drain

        async def late_drain(stream, buffer):
            await asyncio.sleep(0.5)  # (Pi has exited and stdout has ended by now.)
            await real_drain(stream, buffer)

        monkeypatch.setattr(pi_module, "_drain", late_drain)
        command, _ = fake_pi([], FAKE_PI_EXIT="3", FAKE_PI_STDERR="LATE-STDERR-0123456789\n")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "status 3" in str(result.error)
        [report] = _report_files(private_report_dir)
        assert "LATE-STDERR-0123456789" in report.read_text()

    def test_no_file_when_the_run_did_not_fail(self, tmp_path, fake_pi, private_report_dir):
        command, _ = fake_pi(_events_tool_run(), FAKE_PI_STDERR="warning: something printed SECRET-NOISE\n")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is None
        assert _report_files(private_report_dir) == []

    def test_no_file_when_a_failed_run_has_nothing_raw_to_say(self, tmp_path, fake_pi, private_report_dir):
        command, _ = fake_pi([], FAKE_PI_EXIT="3")  # No stderr, no provider message.
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "status 3" in str(result.error)
        assert "private file" not in str(result.error)
        assert _report_files(private_report_dir) == []

    def test_provider_error_message_of_the_final_message_is_only_in_the_private_file(
        self, tmp_path, fake_pi, caplog, private_report_dir
    ):
        secret = SECRETS["url key"]
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "error"
        events[9]["message"]["errorMessage"] = f"401 for {secret}"
        command, _ = fake_pi(events)
        with caplog.at_level("DEBUG"):
            result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "provider error (stopReason=error)" in str(result.error)
        assert "URL-CRED" not in _all_text(result, caplog) and "URL-QUERY" not in _all_text(result, caplog)
        [report] = _report_files(private_report_dir)
        assert secret in report.read_text()

    def test_final_error_of_the_retries_is_only_in_the_private_file(
        self, tmp_path, fake_pi, caplog, private_report_dir
    ):
        secret = "FINAL-ERROR-CRED-0123456789"
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "length"
        events.insert(-1, {"type": "auto_retry_end", "success": False, "finalError": f"gateway said {secret}"})
        command, _ = fake_pi(events)
        with caplog.at_level("DEBUG"):
            result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "pi retries failed" in str(result.error)
        assert secret not in _all_text(result, caplog)
        [report] = _report_files(private_report_dir)
        assert secret in report.read_text()

    def test_unknown_stop_reason_is_not_shown(self, tmp_path, fake_pi, caplog):
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "SECRET-STOP-REASON-0123"
        command, _ = fake_pi(events)
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "SECRET-STOP-REASON" not in str(result.error)
        assert "stopReason='<other>'" in str(result.error)

    def test_model_pi_ran_is_not_in_the_error(self, tmp_path, fake_pi, private_report_dir):
        other = ModelSpec.parse("anthropic/SECRET-MODEL-0123456789")
        command, _ = fake_pi(_events_tool_run(model=other))
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "SECRET-MODEL" not in str(result.error)
        [report] = _report_files(private_report_dir)
        assert "SECRET-MODEL-0123456789" in report.read_text()

    def test_attempt_that_could_not_load_the_collector_goes_to_the_file_if_the_run_fails(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir
    ):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "error"
        events[9]["message"]["errorMessage"] = "SECOND-ATTEMPT-ERROR-0123"
        command, record = fake_pi(events, FAKE_PI_REJECT_EXTENSION="1")
        with caplog.at_level("DEBUG"), pytest.raises(RuntimeError) as error:
            _bounded(
                PiRunner(cwd=tmp_path, pi_command=command).run(
                    model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
                )
            )
        runs = [json.loads(line) for line in Path(str(record) + ".runs").read_text().splitlines()]
        assert ["--extension" in argv for argv in runs] == [True, False]
        assert "pi cannot load the airedale collector extension" in caplog.text
        everything = (
            caplog.text
            + str(error.value)
            + json.dumps([(span.tags, span.annotations) for span in fake_span_llmobs.spans], default=str)
        )
        assert "unsupported" not in everything and "SECOND-ATTEMPT" not in everything
        [report] = _report_files(private_report_dir)
        text = report.read_text()
        assert "unsupported" in text and "SECOND-ATTEMPT-ERROR-0123" in text

    @staticmethod
    def _events_with_a_duplicate_tool_id(tool_id):
        events = _events_tool_run()
        start = next(e for e in events if e["type"] == "tool_execution_start")
        first_call = events[4]["message"]["content"][2]
        second_call = {**first_call, "arguments": {"path": "other"}}
        events[4]["message"]["content"].append(second_call)
        for block in (first_call, second_call):
            block["id"] = tool_id
        for event in events:
            if event["type"].startswith("tool_execution"):
                event["toolCallId"] = tool_id
        events.insert(events.index(start) + 1, {**start, "args": {"path": "other"}})
        return events

    @pytest.mark.parametrize("telemetry", [False, True], ids=["telemetry-off", "telemetry-on"])
    def test_a_duplicate_tool_call_id_is_only_in_the_private_file(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir, telemetry
    ):
        secret = "call_SECRET-ID-0123456789"
        if telemetry:
            import airedale.harness.base as base

            monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
            monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

            async def _no_catalog(servers, headers):
                return {}

            monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        command, _ = fake_pi(self._events_with_a_duplicate_tool_id(secret))
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        with caplog.at_level("DEBUG"):
            if telemetry:
                with pytest.raises(RuntimeError) as raised:
                    _bounded(
                        runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
                    )
                error = raised.value
            else:
                error = _run(runner).error
        # The error, the logs and the error data of the spans. (The tool call id is ordinary data of the tool call
        # itself, like its name and arguments; it is not in an error.)
        everything = caplog.text + str(error)
        if telemetry:
            everything += json.dumps([(span.error, span.tags) for span in fake_span_llmobs.spans], default=str)
        assert "pi ran 2 tool calls with the same id" in str(error)
        assert secret not in everything
        [report] = _report_files(private_report_dir)
        assert secret in report.read_text()

    @pytest.mark.parametrize("telemetry", [False, True], ids=["telemetry-off", "telemetry-on"])
    def test_calls_with_one_id_and_other_names_keep_the_id_out_of_errors_and_logs(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir, telemetry
    ):
        secret = "call_MIXED-ID-0123456789"
        if telemetry:
            import airedale.harness.base as base

            monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
            monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

            async def _no_catalog(servers, headers):
                return {}

            monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        events = _events_tool_run()
        start = next(e for e in events if e["type"] == "tool_execution_start")
        end = next(e for e in events if e["type"] == "tool_execution_end")
        events.insert(events.index(start) + 1, {**start, "toolName": "bash", "args": {"command": "ls"}})
        for event in (start, end):
            event["toolName"] = "read"
        for event in events:
            if event["type"].startswith("tool_execution"):
                event["toolCallId"] = secret
        for block in events[4]["message"]["content"]:
            if block.get("type") == "toolCall":
                block["id"], block["name"] = secret, "read"
        command, _ = fake_pi(events)
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        with caplog.at_level("DEBUG"):
            if telemetry:
                with pytest.raises(RuntimeError) as raised:
                    _bounded(
                        runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
                    )
                error = raised.value
            else:
                error = _run(runner).error
        assert error is not None
        shown = caplog.text + str(error)
        if telemetry:
            shown += json.dumps([(span.error, span.tags) for span in fake_span_llmobs.spans], default=str)
            assert "Ignoring an end record of the tool" in caplog.text  # The log line of the span recorder ran.
        assert secret not in shown
        [report] = _report_files(private_report_dir)
        assert secret in report.read_text()

    @staticmethod
    def _failed_tool_events(output):
        events = _events_tool_run()
        end = next(e for e in events if e["type"] == "tool_execution_end")
        end["isError"] = True
        end["result"] = {"content": [{"type": "text", "text": output}], "details": {}}
        return events

    @pytest.mark.parametrize("telemetry", [False, True], ids=["telemetry-off", "telemetry-on"])
    def test_output_of_a_failed_tool_is_once_in_the_report_of_a_failed_run(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir, telemetry
    ):
        secret = "TOOL-OUTPUT-SECRET-0123456789"
        if telemetry:
            import airedale.harness.base as base

            monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
            monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

            async def _no_catalog(servers, headers):
                return {}

            monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        command, _ = fake_pi(self._failed_tool_events(f"permission denied for {secret}"), FAKE_PI_EXIT="3")
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        with caplog.at_level("DEBUG"):
            if telemetry:
                with pytest.raises(RuntimeError) as raised:
                    _bounded(
                        runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
                    )
                error = raised.value
            else:
                error = _run(runner).error
        assert "status 3" in str(error)
        shown = caplog.text + str(error)
        if telemetry:
            tool_spans = [s for s in fake_span_llmobs.spans if s.error]
            assert any("tool call failed" in json.dumps(s.tags, default=str) for s in tool_spans)
            shown += json.dumps([(span.error, span.tags) for span in fake_span_llmobs.spans], default=str)
        assert secret not in shown
        [report] = _report_files(private_report_dir)
        assert report.read_text().count(secret) == 1

    @pytest.mark.parametrize("char", ["a", "é", "😀"], ids=["ascii", "2-byte", "4-byte"])
    @pytest.mark.parametrize("extra", [-1, 0, 1, 3])
    def test_output_of_a_failed_tool_is_cut_at_8192_bytes_without_a_partial_character(
        self, tmp_path, fake_pi, private_report_dir, char, extra
    ):
        size = len(char.encode())
        count = (8192 + extra) // size + (1 if extra > 0 else 0)
        output = char * count
        command, _ = fake_pi(self._failed_tool_events(output), FAKE_PI_EXIT="3")
        _run(PiRunner(cwd=tmp_path, pi_command=command))
        [report] = _report_files(private_report_dir)
        line = next(
            line for line in report.read_text(encoding="utf-8").splitlines() if "output of the failed tool call" in line
        )
        kept = line.split(": ", 1)[1].removesuffix("...[cut]")
        assert len(kept.encode("utf-8")) <= 8192
        assert set(kept) <= {char}  # No partial character (no replacement character).
        if len(output.encode()) <= 8192:
            assert kept == output and "...[cut]" not in line
        else:
            assert line.endswith("...[cut]")
            assert len(kept.encode()) > 8192 - size  # As much as fits.

    @pytest.mark.parametrize("telemetry", [False, True], ids=["telemetry-off", "telemetry-on"])
    @pytest.mark.parametrize("count", [1365, 1366, 8192], ids=["fits", "one-over", "8192-surrogates"])
    def test_lone_surrogates_in_a_failed_tool_output_are_measured_as_written(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, private_report_dir, telemetry, count
    ):
        if telemetry:
            import airedale.harness.base as base

            monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
            monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

            async def _no_catalog(servers, headers):
                return {}

            monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        # (Pi's stdout is JSON: a lone surrogate arrives as the escape ``\ud800``.)
        command, _ = fake_pi(self._failed_tool_events("\ud800" * count), FAKE_PI_EXIT="3")
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        if telemetry:
            with pytest.raises(RuntimeError):
                _bounded(runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"))
        else:
            _run(runner)
        [report] = _report_files(private_report_dir)
        line = next(
            line for line in report.read_text(encoding="utf-8").splitlines() if "output of the failed tool call" in line
        )
        kept = line.split(": ", 1)[1].removesuffix("...[cut]")
        assert len(kept.encode("utf-8")) <= 8192
        assert set(kept) <= set("\\ud800")  # Whole six-byte escapes only.
        assert len(kept) % 6 == 0
        if count * 6 <= 8192:
            assert "...[cut]" not in line and len(kept) == count * 6
        else:
            assert line.endswith("...[cut]")

    def test_cut_utf8_does_not_allocate_in_proportion_to_a_large_input(self):
        import tracemalloc

        from airedale.harness.pi import _cut_utf8

        text = "😀" * (4 * 1024 * 1024 // 4) + "\ud800" * 1000  # 4 MiB of UTF-8, then lone surrogates.
        tracemalloc.start()
        try:
            result = _cut_utf8(text, 8192)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result == "😀" * 2048 + "...[cut]"  # The same result as before: 8192 bytes of 4-byte characters.
        assert peak < 256 * 1024, peak  # (The old list of forms took about 75 MB.)
        assert _cut_utf8("\ud800" * 1365, 8192) == "\\ud800" * 1365  # Lone surrogates: six bytes each.
        assert _cut_utf8("\ud800" * 1366, 8192) == "\\ud800" * 1365 + "...[cut]"

    def test_cut_utf8_returns_the_written_form_also_when_it_fits(self):
        from airedale.harness.pi import _cut_utf8

        assert _cut_utf8("a\ud800b", 100) == "a\\ud800b"
        assert _cut_utf8("é😀", 100) == "é😀"

    def test_progress_has_no_session_id_and_no_tool_name_that_is_not_of_the_run(self, tmp_path, fake_pi):
        session_secret, tool_secret = "sess-SECRET-0123456789", "sk-TOOL-SECRET-0123456789"
        events = _events_tool_run()
        events[0]["id"] = session_secret
        for event in events:
            if event["type"].startswith("tool_execution"):
                event["toolName"] = tool_secret
        events[4]["message"]["content"][2]["name"] = tool_secret
        command, _ = fake_pi(events)
        progress: list[str] = []

        async def receive(message):
            progress.append(message)

        _bounded(
            PiRunner(cwd=tmp_path, pi_command=command)._run_agent(
                model=MODEL, system_prompt="S", user_prompt="Q", trace_headers={}, progress=receive
            )
        )
        text = "\n".join(progress)
        assert "session started" in text and "tool <other>" in text
        assert "SECRET" not in text

    def test_progress_shows_a_tool_name_of_the_run(self):
        collector = _PiEventCollector(tool_names=("read", "bash", "mcp__*"))
        start = {"type": "tool_execution_start", "toolCallId": "1", "args": {}}
        shown = {name: collector.feed({**start, "toolName": name}) for name in ("read", "mcp__srv_tool", "other")}
        assert shown == {"read": "tool read", "mcp__srv_tool": "tool mcp__srv_tool", "other": "tool <other>"}
        assert collector.feed({"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3}) == "retry 1/3"
        assert collector.feed({"type": "auto_retry_start", "attempt": "SECRET", "maxAttempts": 3}) == "retry"

    @pytest.mark.parametrize("telemetry", [False, True], ids=["telemetry-off", "telemetry-on"])
    def test_compaction_reason_and_counts_from_pi_are_only_in_the_private_file(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir, telemetry
    ):
        reason_secret, count_secret = "sk-REASON-SECRET-0123456789", "sk-COUNT-SECRET-0123456789"
        if telemetry:
            import airedale.harness.base as base

            monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
            monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

            async def _no_catalog(servers, headers):
                return {}

            monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        events = _events_tool_run()
        position = next(i for i, e in enumerate(events) if e["type"] == "agent_end")
        events[position:position] = [
            {"type": "compaction_start", "reason": reason_secret},
            {
                "type": "compaction_end",
                "result": {"summary": "S", "tokensBefore": count_secret, "estimatedTokensAfter": 7},
            },
        ]
        command, _ = fake_pi(events, FAKE_PI_EXIT="3")
        runner = PiRunner(cwd=tmp_path, pi_command=command)
        with caplog.at_level("DEBUG"):
            if telemetry:
                with pytest.raises(RuntimeError) as raised:
                    _bounded(
                        runner.run(model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h")
                    )
                error = raised.value
            else:
                error = _run(runner).error
        shown = caplog.text + str(error)
        if telemetry:
            compaction = [s for s in fake_span_llmobs.spans if s.name == "pi.compaction"]
            assert compaction, [s.name for s in fake_span_llmobs.spans]
            metadata = json.dumps([s.annotations for s in compaction], default=str)
            assert "<other>" in metadata and '"estimated_tokens_after": 7' in metadata
            shown += json.dumps([(s.error, s.tags, s.annotations) for s in fake_span_llmobs.spans], default=str)
        assert reason_secret not in shown and count_secret not in shown
        [report] = _report_files(private_report_dir)
        text = report.read_text()
        assert reason_secret in text and count_secret in text

    def test_known_compaction_reasons_are_kept(self):
        from airedale.harness.pi_spans import COMPACTION_REASONS

        assert {"manual", "threshold", "overflow"} == COMPACTION_REASONS

    def test_the_old_redaction_code_is_gone(self):
        import airedale.harness.pi as pi_module

        for name in ("_redact_secrets", "_redact_bytes", "_drop_cut_secret", "_byte_forms", "_secret_strings_of_json"):
            assert not hasattr(pi_module, name), name
        assert not hasattr(pi_module.PiRunner, "_secret_values")


class TestProjectEnablesCodemode:
    """A repo ``.pi/mcp.json`` with ``autoEnableCodemode: true`` replaces the ``false`` of the scenario: warn once."""

    @staticmethod
    def _project(tmp_path, value):
        (tmp_path / ".pi").mkdir(exist_ok=True)
        text = json.dumps({"autoEnableCodemode": value, "mcpServers": {}})
        (tmp_path / ".pi" / "mcp.json").write_text(text)

    SERVERS = (McpServerSpec(name="docs", url="http://localhost/mcp"),)

    def test_true_in_the_repo_warns_once_with_cause_and_consequence(self, tmp_path, fake_pi, caplog):
        self._project(tmp_path, True)
        command, _ = fake_pi(_events_tool_run())
        with caplog.at_level("WARNING"):
            result = _run(PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=list(self.SERVERS)))
        assert result.error is None
        messages = [r.getMessage() for r in caplog.records if "autoEnableCodemode" in r.getMessage()]
        assert messages == [
            "repository .pi/mcp.json sets autoEnableCodemode=true: pi enables codemode for this run, so the agent can "
            "call other models; results may differ from runs without it"
        ]

    @pytest.mark.parametrize("value", [False, None, "yes", 1])
    def test_other_values_do_not_warn(self, tmp_path, fake_pi, caplog, value):
        self._project(tmp_path, value)
        command, _ = fake_pi(_events_tool_run())
        with caplog.at_level("WARNING"):
            _run(PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=list(self.SERVERS)))
        assert "autoEnableCodemode" not in caplog.text

    def test_no_scenario_server_means_no_change_and_no_warning(self, tmp_path, fake_pi, caplog):
        self._project(tmp_path, True)  # Pi's own default is true: nothing changes.
        command, _ = fake_pi(_events_tool_run())
        with caplog.at_level("WARNING"):
            _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "autoEnableCodemode" not in caplog.text

    def test_no_repo_file_does_not_warn(self, tmp_path, fake_pi, caplog):
        command, _ = fake_pi(_events_tool_run())
        with caplog.at_level("WARNING"):
            _run(PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=list(self.SERVERS)))
        assert "autoEnableCodemode" not in caplog.text

    def test_the_run_still_goes_on_and_the_scenario_file_says_false(self, tmp_path, fake_pi):
        self._project(tmp_path, True)
        command, record = fake_pi(_events_tool_run())
        _run(PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=list(self.SERVERS)))
        assert json.loads(json.loads(record.read_text())["files"]["mcp.json"])["autoEnableCodemode"] is False

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flag_in_the_metadata_of_the_agent_span(self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, enabled):
        import airedale.harness.base as base

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        self._project(tmp_path, enabled)
        command, _ = fake_pi(_events_tool_run())
        _bounded(
            PiRunner(cwd=tmp_path, pi_command=command, mcp_servers=list(self.SERVERS)).run(
                model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
            )
        )
        flags = [
            a["metadata"].get("project_auto_enable_codemode") for a in fake_span_llmobs.annotations if "metadata" in a
        ]
        assert (True in flags) is enabled


@pytest.mark.skipif(os.name == "nt", reason="POSIX owner and mode checks")
class TestPrivateReportDirectory:
    """The directory of the private report is checked, and used through its descriptor (items 90, 92)."""

    @staticmethod
    def _write(tmp_path, env=None):
        import airedale.harness.pi as pi_module

        runner = pi_module.PiRunner(cwd=tmp_path)
        return runner._write_private_report(MODEL, 3, "DUMMY-RAW-0123456789", [])

    @pytest.mark.parametrize(("mode", "written"), [(0o700, True), (0o755, False), (0o777, False), (0o770, False)])
    def test_configured_directory_must_be_closed_to_others(self, tmp_path, monkeypatch, mode, written):
        directory = tmp_path / "out"
        directory.mkdir()
        directory.chmod(mode)
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(directory))
        result = self._write(tmp_path)
        assert (result is not None) is written
        assert bool(list(directory.glob("pi-run-*.log"))) is written

    def test_new_configured_directory_is_created_closed(self, tmp_path, monkeypatch):
        directory = tmp_path / "a" / "b"
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(directory))
        result = self._write(tmp_path)
        assert result is not None and result.parent == directory
        assert directory.stat().st_mode & 0o077 == 0
        assert result.stat().st_mode & 0o777 == 0o600

    def test_configured_directory_that_is_a_link_is_refused(self, tmp_path, monkeypatch):
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(link))
        assert self._write(tmp_path) is None
        assert list(target.iterdir()) == []

    def test_unsafe_default_directory_is_refused(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.delenv("AIREDALE_OUTPUT_DIR")
        monkeypatch.setattr(pi_module.tempfile, "gettempdir", lambda: str(tmp_path))
        directory = tmp_path / f"airedale-{os.getuid()}"
        directory.mkdir()
        directory.chmod(0o755)
        assert self._write(tmp_path) is None

    def test_default_directory_that_is_a_link_is_refused(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        monkeypatch.delenv("AIREDALE_OUTPUT_DIR")
        monkeypatch.setattr(pi_module.tempfile, "gettempdir", lambda: str(tmp_path))
        target = tmp_path / "elsewhere"
        target.mkdir(mode=0o700)
        (tmp_path / f"airedale-{os.getuid()}").symlink_to(target, target_is_directory=True)
        assert self._write(tmp_path) is None
        assert list(target.iterdir()) == []

    def test_existing_file_or_link_of_that_name_is_not_opened(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        directory = tmp_path / "out"
        directory.mkdir(mode=0o700)
        victim = tmp_path / "victim"
        victim.write_text("unchanged")
        (directory / "pi-run-fixed-12345678.log").symlink_to(victim)
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(directory))
        monkeypatch.setattr(pi_module.time, "strftime", lambda *args: "fixed")
        monkeypatch.setattr(pi_module.uuid, "uuid4", lambda: type("U", (), {"hex": "12345678"})())
        assert self._write(tmp_path) is None
        assert victim.read_text() == "unchanged"

    def test_directory_replaced_by_a_link_after_the_check_changes_nothing(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        parent = tmp_path / "parent"
        parent.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        outside.chmod(0o777)
        directory = parent / f"airedale-{os.getuid()}"
        monkeypatch.delenv("AIREDALE_OUTPUT_DIR")
        monkeypatch.setattr(pi_module.tempfile, "gettempdir", lambda: str(parent))
        real_open = os.open

        def swap(path, flags, mode=0o777, **kwargs):
            if Path(path).name.startswith("pi-run-"):  # Between the check of the directory and the file.
                directory.rename(parent / "saved")
                directory.symlink_to(outside, target_is_directory=True)
            return real_open(path, flags, mode, **kwargs)

        monkeypatch.setattr(pi_module.os, "open", swap)
        result = self._write(tmp_path)
        assert list(outside.iterdir()) == []  # Nothing outside the directory that was checked.
        assert result is not None and (parent / "saved").glob("pi-run-*.log")

    def test_system_without_dir_fd_writes_nothing_and_warns(self, tmp_path, monkeypatch, caplog):
        import airedale.harness.pi as pi_module

        monkeypatch.setattr(pi_module, "_OPEN_TAKES_DIR_FD", False)
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(tmp_path / "out"))
        with caplog.at_level("WARNING"):
            assert self._write(tmp_path) is None
        assert not (tmp_path / "out").exists()
        assert "Unable to write the private report of the failed pi run" in caplog.text
        assert "DUMMY-RAW" not in caplog.text

    def test_unknown_user_in_the_directory_keeps_the_run_error(self, tmp_path, fake_pi, monkeypatch, caplog):
        monkeypatch.setenv("AIREDALE_OUTPUT_DIR", "~airedale-user-that-does-not-exist/out")
        command, _ = fake_pi([], FAKE_PI_EXIT="3", FAKE_PI_STDERR="boom DUMMY-RAW-0123456789\n")
        with caplog.at_level("WARNING"):
            result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "status 3" in str(result.error) and type(result.error) is RuntimeError
        assert "private file" not in str(result.error)
        assert "Unable to write the private report" in caplog.text
        assert "DUMMY-RAW" not in caplog.text and "airedale-user-that-does-not-exist" not in caplog.text

    def test_other_exception_types_are_caught_too_but_not_base_exceptions(self, tmp_path, monkeypatch):
        import airedale.harness.pi as pi_module

        def boom(name):
            raise ValueError("odd")

        monkeypatch.setattr(pi_module, "open_private_report", boom)
        assert self._write(tmp_path) is None

        def interrupted(name):
            raise KeyboardInterrupt

        monkeypatch.setattr(pi_module, "open_private_report", interrupted)
        with pytest.raises(KeyboardInterrupt):
            self._write(tmp_path)


class TestRawDiagnosticsAreKept:
    """The raw values are collected when the records arrive, with or without telemetry, whichever error ends the run."""

    FIELDS = ("errorMessage", "responseModel", "responseId", "api", "thinkingLevel", "stopReason", "model", "provider")

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("status", [0, 3])
    def test_every_raw_field_of_a_failed_final_message_reaches_the_report_without_telemetry(
        self, tmp_path, fake_pi, caplog, private_report_dir, field, status
    ):
        marker = f"DUMMY-RAW-{field.upper()}-0123456789"
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "error"
        events[9]["message"][field] = marker
        command, _ = fake_pi(events, FAKE_PI_EXIT=str(status))
        with caplog.at_level("DEBUG"):
            result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is not None
        assert marker not in _all_text(result, caplog)
        [report] = _report_files(private_report_dir)
        assert marker in report.read_text()

    def test_message_of_an_earlier_failed_call_is_kept(self, tmp_path, fake_pi, private_report_dir):
        events = _events_tool_run()
        events[4]["message"]["stopReason"] = "error"
        events[4]["message"]["errorMessage"] = "DUMMY-EARLIER-CALL-0123456789"
        command, _ = fake_pi(events, FAKE_PI_EXIT="3")
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is not None
        [report] = _report_files(private_report_dir)
        assert "DUMMY-EARLIER-CALL-0123456789" in report.read_text()

    def test_unknown_stop_reason_is_kept(self, tmp_path, fake_pi, private_report_dir):
        events = _events_tool_run()
        events[9]["message"]["stopReason"] = "DUMMY-STOP-REASON-0123456789"
        command, _ = fake_pi(events)
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert "DUMMY-STOP-REASON" not in str(result.error)
        [report] = _report_files(private_report_dir)
        assert "DUMMY-STOP-REASON-0123456789" in report.read_text()

    def test_compaction_error_message_is_kept(self, tmp_path, fake_pi, private_report_dir):
        events = _events_tool_run()
        events.insert(
            5, {"type": "compaction_end", "errorMessage": "DUMMY-COMPACT-0123456789", "aborted": False, "result": {}}
        )
        events[10]["message"]["stopReason"] = "error"
        command, _ = fake_pi(events)
        result = _run(PiRunner(cwd=tmp_path, pi_command=command))
        assert result.error is not None
        [report] = _report_files(private_report_dir)
        assert "DUMMY-COMPACT-0123456789" in report.read_text()

    def test_a_long_raw_value_is_cut(self):
        collector = _collect(_events_tool_run())
        message = {"role": "assistant", "stopReason": "error", "errorMessage": "x" * 100_000}
        collector._note_assistant_message(message)
        assert len(collector.private_notes[-1]) < 22_000 and "[cut]" in collector.private_notes[-1]

    @pytest.mark.parametrize("unfinished", [False, True])
    def test_request_id_is_not_in_the_spans_and_is_in_the_report(
        self, tmp_path, fake_pi, monkeypatch, fake_span_llmobs, caplog, private_report_dir, unfinished
    ):
        import airedale.harness.base as base
        from airedale.harness.pi_spans import EXTENSION_REQUEST_ENTRY, EXTENSION_RESPONSE_ENTRY

        monkeypatch.setattr(base, "LLMObs", fake_span_llmobs)
        monkeypatch.setattr(base, "agent", lambda **kwargs: lambda func: func)

        async def _no_catalog(servers, headers):
            return {}

        monkeypatch.setattr(base, "_mcp_tool_metadata_catalog", _no_catalog)
        marker = "DUMMY-HEADER-CREDENTIAL-0123456789"

        def entry(kind, **data):
            return {
                "type": "entry_appended",
                "entry": {"type": "custom", "customType": kind, "data": {"v": 1, "seq": 1, **data}},
            }

        events = _events_tool_run()
        position = len(events) - 1 if unfinished else 9
        events[position:position] = [
            entry(EXTENSION_REQUEST_ENTRY, t=1, prefix=0, appended=[], count=0),
            entry(EXTENSION_RESPONSE_ENTRY, t=2, status=200, requestId=marker),
        ]
        command, _ = fake_pi(events, FAKE_PI_EXIT="3")
        with caplog.at_level("DEBUG"), pytest.raises(RuntimeError) as error:
            _bounded(
                PiRunner(cwd=tmp_path, pi_command=command).run(
                    model=MODEL, system_prompt="S", user_prompt="Q", prompt_version="v", harness="h"
                )
            )
        everything = (
            caplog.text
            + str(error.value)
            + json.dumps(
                [(span.tags, span.annotations) for span in fake_span_llmobs.spans] + fake_span_llmobs.annotations,
                default=str,
            )
        )
        assert marker not in everything
        [report] = _report_files(private_report_dir)
        assert marker in report.read_text()
