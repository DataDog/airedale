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
            # Google: the total of the provider has the thinking tokens, which are not in the parts.
            ("google", {"input": 10, "output": 5, "reasoning": 3, "totalTokens": 18}, 15),
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
        events[4]["message"]["content"].append({"type": "toolCall", "id": "", "name": "read", "arguments": {}})
        events[4]["message"]["content"][2]["id"] = ""
        events[5:7] = [
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "first"},
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "child1", "parentToolCallId": ""},
            {"type": "tool_execution_end", "toolCallId": "", "toolName": "first", "result": {}},
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "second"},
            {"type": "tool_execution_start", "toolCallId": "/1", "toolName": "child2", "parentToolCallId": ""},
            {"type": "tool_execution_end", "toolCallId": "", "toolName": "second", "result": {}},
        ]
        result = _collect(events).result(model=MODEL, returncode=0, stderr_tail="")
        # Each nested call follows the execution that it started under.
        assert [call.name for call in result.tool_calls] == ["ls", "child1", "read", "child2"]

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
                "toolCallId": "",
                "toolName": "t0",
                "result": {"content": [{"type": "text", "text": "r0"}]},
            },
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "t1"},
            {
                "type": "tool_execution_end",
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
                "toolCallId": "",
                "toolName": "inner",
                "parentToolCallId": "outer",
                "result": {"content": [{"type": "text", "text": "nested"}]},
                "durationMs": 9,
            },
            {"type": "tool_execution_start", "toolCallId": "", "toolName": "ls", "args": {"path": "."}},
            {
                "type": "tool_execution_end",
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
            {"input": 1, "output": 1},  # no cost
            {"input": 1, "output": 1, "cost": {}},  # no cost.total
            {"input": 1, "output": 1, "cost": {"input": 0.1, "output": 0.2}},
            {"input": 1, "output": 1, "cost": 5},
            {"input": 1, "output": 1, "cost": {"total": "Infinity"}},
            {"input": 1, "output": 1, "cost": {"total": float("inf")}},
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
        assert reported[0]["env"] == {"TOKEN": "<redacted>"}
        assert reported[0]["args"] == ["<redacted>", "<redacted>"]
        assert reported[1]["url"] == "https://example.test:8443"

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
            {"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "ls", "result": "x", "durationMs": 250},
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
        with pytest.raises(RuntimeError, match="pi executable not found"):
            _run(PiRunner(cwd=tmp_path, pi_command=["pi"]))
        assert seen["start_new_session"] is False


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
