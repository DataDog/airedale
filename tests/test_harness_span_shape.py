# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Span-shape parity tests for the Codex (decorator) harness.

The Codex runner manufactures its LLMObs spans by hand (Codex has no native
ddtrace integration). These tests pin the *shape* of those spans to what the
native ``claude_agent_sdk`` integration produces for the equivalent situation:

- ``llm`` span: structured ``input.messages`` / ``output.messages`` where each
  tool call is its own assistant message carrying a ``tool_calls`` entry, plus a
  final assistant message with the answer.
- ``agent`` span: input/output are role/content **message lists** (LLMObs
  serializes them into ``.value`` for a non-llm span), plus an
  ``agent_manifest`` written under ``metadata._dd.agent_manifest``.
- ``tool`` spans: one per tool call, with raw arguments/result and metadata.

Everything runs offline against captured ``LLMObs.annotate`` calls.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

import airedale.harness.base as base
from airedale.harness.base import (
    AgentThought,
    AgentToolCall,
    _agent_transcript_messages,
    _annotate_agent_loop_span,
    _annotate_llm_span,
    _emit_tool_spans,
    _input_messages,
    _llm_output_messages,
)
from airedale.types import ModelSpec, UsageMetrics

MODEL = ModelSpec.parse("openai/gpt-5.5")


def _tool_call() -> AgentToolCall:
    return AgentToolCall(
        name="mcp.apm.search",
        arguments={"query": "ssi"},
        result={"hits": 3},
        metadata={"source": "openai-codex", "server": "apm", "tool": "search"},
    )


class _CapturingLLMObs:
    """Minimal LLMObs stand-in capturing annotate calls and tool spans."""

    enabled = True

    def __init__(self) -> None:
        self.annotations: list[dict[str, Any]] = []
        self.tool_spans: list[str] = []

    def annotate(self, **kwargs: Any) -> None:
        self.annotations.append(kwargs)

    @contextlib.contextmanager
    def tool(self, *, name: str):
        self.tool_spans.append(name)
        yield object()


@pytest.fixture
def llmobs(monkeypatch):
    fake = _CapturingLLMObs()
    monkeypatch.setattr(base, "LLMObs", fake)
    return fake


# --------------------------------------------------------------------------- #
# Pure message-shape helpers
# --------------------------------------------------------------------------- #
def test_input_messages_keep_system_and_user():
    assert _input_messages("SYS", "USER") == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]


def test_llm_output_messages_interleave_tool_calls_then_answer():
    messages = _llm_output_messages("final", [_tool_call()])
    assert messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "mcp.apm.search", "arguments": {"query": "ssi"}, "type": "tool_use"}],
        },
        {"role": "assistant", "content": "final"},
    ]
    # The llm span never emits tool_results (they live on tool spans / transcript).
    assert all("tool_results" not in message for message in messages)


def test_agent_transcript_pairs_tool_use_and_tool_result():
    messages = _agent_transcript_messages("final", [_tool_call()])
    assert messages == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "mcp.apm.search", "arguments": {"query": "ssi"}, "type": "tool_use"}],
        },
        {
            "role": "user",
            "content": "",
            "tool_results": [{"name": "mcp.apm.search", "result": '{"hits":3}', "type": "tool_result"}],
        },
        {"role": "assistant", "content": "final"},
    ]


def test_llm_output_interleaves_reasoning_and_tool_calls_in_order():
    # Reasoning (and plan) segments surface as bare assistant messages, in item
    # order relative to tool calls, mirroring Claude's ThinkingBlock handling.
    segments = [
        AgentThought(text="thinking about it", kind="reasoning"),
        _tool_call(),
        AgentThought(text="1. do the thing", kind="plan"),
    ]
    messages = _llm_output_messages("final", segments)
    assert messages == [
        {"role": "assistant", "content": "thinking about it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "mcp.apm.search", "arguments": {"query": "ssi"}, "type": "tool_use"}],
        },
        {"role": "assistant", "content": "1. do the thing"},
        {"role": "assistant", "content": "final"},
    ]


def test_agent_transcript_interleaves_reasoning_with_tool_pairs():
    segments = [AgentThought(text="reasoning", kind="reasoning"), _tool_call()]
    messages = _agent_transcript_messages("final", segments)
    assert messages == [
        {"role": "assistant", "content": "reasoning"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "mcp.apm.search", "arguments": {"query": "ssi"}, "type": "tool_use"}],
        },
        {
            "role": "user",
            "content": "",
            "tool_results": [{"name": "mcp.apm.search", "result": '{"hits":3}', "type": "tool_result"}],
        },
        {"role": "assistant", "content": "final"},
    ]


def test_tool_call_count_metadata_ignores_thoughts(llmobs):
    # Only tool calls count toward tool_call_count; thoughts are not tools.
    _annotate_agent_loop_span(
        model=MODEL,
        sdk_name="openai-codex",
        framework="OpenAI Codex",
        prompt_version="s1",
        harness="s1",
        system_prompt="SYS",
        user_prompt="USER",
        answer="final",
        usage=UsageMetrics(input_tokens=1, output_tokens=2, total_tokens=3),
        mcp_servers=[],
        segments=[AgentThought(text="thinking"), _tool_call(), AgentThought(text="plan", kind="plan")],
    )
    (call,) = llmobs.annotations
    assert call["metadata"]["tool_call_count"] == 1


def test_tool_result_text_passes_strings_through():
    call = AgentToolCall(name="t", arguments={}, result="plain text")
    transcript = _agent_transcript_messages("a", [call])
    assert transcript[1]["tool_results"][0]["result"] == "plain text"


def test_tool_use_id_threads_into_entries():
    call = AgentToolCall(name="t", arguments={}, result="", metadata={"tool_use_id": "abc123"})
    transcript = _agent_transcript_messages("a", [call])
    assert transcript[0]["tool_calls"][0]["tool_id"] == "abc123"
    assert transcript[1]["tool_results"][0]["tool_id"] == "abc123"


# --------------------------------------------------------------------------- #
# Span annotation wiring
# --------------------------------------------------------------------------- #
def test_llm_span_uses_structured_messages(llmobs):
    _annotate_llm_span(
        model=MODEL,
        sdk_name="openai-codex",
        prompt_version="s1",
        harness="s1",
        system_prompt="SYS",
        user_prompt="USER",
        answer="final",
        usage=UsageMetrics(input_tokens=1, output_tokens=2, total_tokens=3),
        mcp_servers=[],
        segments=[_tool_call()],
    )
    (call,) = llmobs.annotations
    assert call["input_data"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert call["output_data"][-1] == {"role": "assistant", "content": "final"}
    assert call["output_data"][0]["tool_calls"][0]["name"] == "mcp.apm.search"


def test_agent_span_emits_message_lists_and_manifest(llmobs):
    _annotate_agent_loop_span(
        model=MODEL,
        sdk_name="openai-codex",
        framework="OpenAI Codex",
        prompt_version="s1",
        harness="s1",
        system_prompt="SYS",
        user_prompt="USER",
        answer="final",
        usage=UsageMetrics(input_tokens=1, output_tokens=2, total_tokens=3),
        mcp_servers=[],
        segments=[_tool_call()],
        allowed_builtin_tools=["Read", "Grep"],
        max_turns=42,
    )
    (call,) = llmobs.annotations
    # Input/output are message lists (LLMObs serializes them to .value for agent).
    assert isinstance(call["input_data"], list)
    assert call["input_data"][0]["role"] == "system"
    assert isinstance(call["output_data"], list)
    assert call["output_data"][-1] == {"role": "assistant", "content": "final"}
    # The manifest lands under metadata._dd.agent_manifest, matching the native path.
    manifest = call["metadata"]["_dd"]["agent_manifest"]
    assert manifest["framework"] == "OpenAI Codex"
    assert manifest["model"] == "gpt-5.5"
    assert manifest["max_iterations"] == 42
    assert {"name": "Read"} in manifest["tools"]


def test_codex_tool_id_threads_into_entries_and_span():
    # Codex extraction stores the item id under `tool_id` (vs Claude's
    # `tool_use_id`); both must correlate the llm tool_call, the transcript
    # tool_result, and the tool span.
    call = AgentToolCall(name="mcp.apm.search", arguments={}, result="", metadata={"tool_id": "item_42"})
    transcript = _agent_transcript_messages("a", [call])
    assert transcript[0]["tool_calls"][0]["tool_id"] == "item_42"
    assert transcript[1]["tool_results"][0]["tool_id"] == "item_42"
    assert _llm_output_messages("a", [call])[0]["tool_calls"][0]["tool_id"] == "item_42"


def test_emit_tool_spans_creates_one_span_per_call(llmobs):
    _emit_tool_spans([_tool_call(), _tool_call()])
    assert llmobs.tool_spans == ["mcp.apm.search", "mcp.apm.search"]
    assert len(llmobs.annotations) == 2
    first = llmobs.annotations[0]
    assert first["input_data"] == {"query": "ssi"}
    assert first["output_data"] == {"hits": 3}
    assert first["metadata"]["server"] == "apm"


def test_emit_tool_spans_swallows_telemetry_failures(monkeypatch):
    class _Boom:
        enabled = True

        def tool(self, *, name):  # noqa: ARG002
            raise RuntimeError("span backend down")

    monkeypatch.setattr(base, "LLMObs", _Boom())
    # A telemetry failure must not propagate and abort/mask a completed run.
    _emit_tool_spans([_tool_call()])


def test_span_helpers_are_noops_when_llmobs_disabled(monkeypatch):
    class _Disabled:
        enabled = False

    monkeypatch.setattr(base, "LLMObs", _Disabled)
    # Must not raise and must not attempt to open tool spans.
    _emit_tool_spans([_tool_call()])
    _annotate_agent_loop_span(
        model=MODEL,
        sdk_name="openai-codex",
        framework="OpenAI Codex",
        prompt_version="s1",
        harness="s1",
        system_prompt="SYS",
        user_prompt="USER",
        answer="final",
        usage=UsageMetrics(),
        mcp_servers=[],
        segments=[_tool_call()],
    )
