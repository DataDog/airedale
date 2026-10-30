# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the pi span recorder and the collector extension.

Everything runs offline. The extension test runs the TypeScript file with Node
(when a Node version that can load TypeScript is available) against a fake pi
API.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from typing import Any

import pytest

from airedale.harness.pi_spans import (
    EXTENSION_REQUEST_ENTRY,
    EXTENSION_RESPONSE_ENTRY,
    PI_LLMOBS_EXTENSION_PATH,
    PiSpanRecorder,
    llmobs_input_message,
    llmobs_input_messages,
)
from airedale.types import ModelSpec

MODEL = ModelSpec.parse("anthropic/claude-sonnet-4-6")
MS = 1_000_000
#: Epoch ms of the start of the scripted runs.
T0_MS = 1_800_000_000_000


def _ns(ms: float) -> int:
    return int(ms * MS)


def _recorder(**kwargs: Any) -> PiSpanRecorder:
    return PiSpanRecorder(model=MODEL, sdk_name="pi", harness="h", prompt_version="v", **kwargs)


def _assistant(content: list[dict], *, timestamp: float | None = None, stop: str = "stop", **extra: Any) -> dict:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 15, "cost": {"total": 0.1}},
        "stopReason": stop,
        **extra,
    }
    if timestamp is not None:
        message["timestamp"] = timestamp
    return message


def _input(record) -> list[dict[str, Any]]:
    """The rendered ``llm`` span input of a record."""
    return llmobs_input_messages(record.pi_input_messages)


def _feed(recorder: PiSpanRecorder, events: list[tuple[float, dict[str, Any]]]) -> PiSpanRecorder:
    """Feed ``(arrival ms after T0, event)`` pairs."""
    for at_ms, event in events:
        recorder.feed(event, received_ns=_ns(T0_MS + at_ms))
    return recorder


def _request(seq: int, t_ms: float, prefix: int, appended: list[Any], count: int | None = None, **extra: Any) -> dict:
    data = {
        "v": 1,
        "seq": seq,
        "t": T0_MS + t_ms,
        "prefix": prefix,
        "appended": appended,
        "count": prefix + len(appended) if count is None else count,
        **extra,
    }
    return {"type": "entry_appended", "entry": {"type": "custom", "customType": EXTENSION_REQUEST_ENTRY, "data": data}}


def _response(seq: int, t_ms: float, **extra: Any) -> dict:
    data = {"v": 1, "seq": seq, "t": T0_MS + t_ms, "status": 200, "requestId": "req_1", **extra}
    return {"type": "entry_appended", "entry": {"type": "custom", "customType": EXTENSION_RESPONSE_ENTRY, "data": data}}


SYSTEM = {"role": "system", "content": "", "sections": {"preamble": "You are pi.", "append": "SYS", "gone": None}}
USER = {"role": "user", "content": "Q"}


# --------------------------------------------------------------------------- #
# Message rendering
# --------------------------------------------------------------------------- #
class TestMessageRendering:
    def test_system_message_uses_sections_when_content_is_empty(self):
        assert llmobs_input_message(SYSTEM) == {"role": "system", "content": "You are pi.\n\nSYS"}
        assert llmobs_input_message({"role": "system", "content": "Direct"}) == {"role": "system", "content": "Direct"}

    def test_tool_result_and_assistant(self):
        assistant = _assistant(
            [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Look"},
                {"type": "toolCall", "id": "c1", "name": "ls", "arguments": {"path": "."}},
            ]
        )
        assert llmobs_input_message(assistant) == {
            "role": "assistant",
            "content": "Look",
            "tool_calls": [{"name": "ls", "arguments": {"path": "."}, "type": "tool_use", "tool_id": "c1"}],
        }
        result = {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "ls",
            "content": [{"type": "text", "text": "a"}, {"type": "image", "data": "xx"}],
            "isError": False,
        }
        assert llmobs_input_message(result) == {
            "role": "user",
            "content": "",
            "tool_results": [{"name": "ls", "result": "a\n[image]", "type": "tool_result", "tool_id": "c1"}],
        }

    def test_other_roles(self):
        assert llmobs_input_message({"role": "custom", "content": "note"}) == {"role": "user", "content": "note"}
        assert llmobs_input_message({"role": "compactionSummary", "summary": "S"}) == {"role": "user", "content": "S"}
        assert llmobs_input_message({"role": "bashExecution", "command": "ls", "output": "x"}) == {
            "role": "user",
            "content": "$ ls\nx",
        }
        assert llmobs_input_message({"role": "bashExecution", "command": "ls", "excludeFromContext": True}) is None
        assert llmobs_input_message({"role": "unknown"}) is None
        assert llmobs_input_message("not a message") is None
        assert llmobs_input_messages([USER, None, {"role": "unknown"}]) == [{"role": "user", "content": "Q"}]


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
class TestModelCalls:
    def test_call_times_and_input_from_the_stream(self):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "turn_start"}),
                (1, {"type": "message_end", "message": SYSTEM}),
                (2, {"type": "message_end", "message": USER}),
                (
                    900,
                    {
                        "type": "message_end",
                        "message": _assistant([{"type": "text", "text": "A"}], timestamp=T0_MS + 5),
                    },
                ),
            ],
        )
        [record] = recorder.records
        assert record.kind == "llm"
        assert record.start_ns == _ns(T0_MS + 5)
        assert record.end_ns == _ns(T0_MS + 900)
        assert _input(record) == [
            {"role": "system", "content": "You are pi.\n\nSYS"},
            {"role": "user", "content": "Q"},
        ]
        assert record.output_data == [{"role": "assistant", "content": "A"}]
        assert record.metadata["input_source"] == "transcript"
        assert record.metadata["turn"] == 1
        assert record.metadata["model_name"] == "claude-sonnet-4-6"
        assert record.metrics["input_tokens"] == 10
        assert record.metrics["output_tokens"] == 5
        assert record.error is None

    @pytest.mark.parametrize("timestamp", [None, T0_MS + 10_000, "soon", True])
    def test_untrusted_timestamp_falls_back_to_the_previous_event(self, timestamp):
        message = _assistant([{"type": "text", "text": "A"}])
        message["timestamp"] = timestamp
        recorder = _feed(_recorder(), [(3, {"type": "turn_start"}), (900, {"type": "message_end", "message": message})])
        assert recorder.records[0].start_ns == _ns(T0_MS + 3)

    def test_failed_call_is_an_error_span(self):
        message = _assistant([], stop="error", errorMessage="529 overloaded", timestamp=T0_MS)
        recorder = _feed(_recorder(), [(10, {"type": "message_end", "message": message})])
        assert recorder.records[0].error == "529 overloaded"
        assert recorder.records[0].metadata["stop_reason"] == "error"

    def test_extension_data_sets_time_input_and_response_details(self):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "turn_start"}),
                (1, {"type": "message_end", "message": USER}),
                (2, _request(1, 4, 0, [SYSTEM, USER])),
                (300, _response(1, 300)),
                (
                    900,
                    {
                        "type": "message_end",
                        "message": _assistant([{"type": "text", "text": "A"}], timestamp=T0_MS + 1),
                    },
                ),
            ],
        )
        [record] = recorder.records
        assert record.start_ns == _ns(T0_MS + 4)
        assert record.metadata["input_source"] == "extension"
        assert record.metadata["time_to_first_byte_ms"] == 296
        assert record.metadata["http_status"] == 200
        assert record.metadata["request_id"] == "req_1"
        assert _input(record)[0]["role"] == "system"

    def test_context_changes_rebuild_each_request(self):
        first = _assistant([{"type": "toolCall", "id": "c1", "name": "ls", "arguments": {}}], timestamp=T0_MS + 1)
        result = {"role": "toolResult", "toolCallId": "c1", "toolName": "ls", "content": [], "isError": False}
        summary = {"role": "compactionSummary", "summary": "S"}
        recorder = _feed(
            _recorder(),
            [
                (1, _request(1, 1, 0, [SYSTEM, USER])),
                (2, {"type": "message_end", "message": first}),
                (3, _request(2, 3, 2, [first, result])),
                (4, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 3)}),
                # Compaction replaced everything after the system message.
                (5, _request(3, 5, 1, [summary])),
                (6, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 5)}),
            ],
        )
        roles = [[message["role"] for message in _input(record)] for record in recorder.records]
        assert roles == [["system", "user"], ["system", "user", "assistant", "user"], ["system", "user"]]
        assert _input(recorder.records[2])[1] == {"role": "user", "content": "S"}

    def test_bad_context_change_waits_for_a_full_list(self):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": USER}),
                (1, _request(1, 1, 0, [SYSTEM, USER])),
                (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)}),
                (3, _request(2, 3, 5, [USER])),  # prefix larger than the known list
                (4, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 3)}),
                (5, _request(3, 5, 1, [USER])),  # relative to an unknown list
                (6, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 5)}),
                (7, _request(4, 7, 0, [USER])),  # a full list: used again
                (8, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 7)}),
            ],
        )
        sources = [record.metadata["input_source"] for record in recorder.records]
        assert sources == ["extension", "transcript", "transcript", "extension"]
        # The request time is still used.
        assert recorder.records[1].start_ns == _ns(T0_MS + 3)
        assert recorder.invalid_extension_entries == 2

    def test_skipped_request_uses_the_transcript_then_a_full_list(self):
        skipped = _request(2, 3, 0, [])
        skipped["entry"]["data"].update({"appended": None, "skipped": True, "count": 3})
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": USER}),
                (1, _request(1, 1, 0, [USER])),
                (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)}),
                (3, skipped),
                (4, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 3)}),
                (5, _request(3, 5, 0, [USER, USER])),
                (6, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 5)}),
            ],
        )
        sources = [record.metadata["input_source"] for record in recorder.records]
        assert sources == ["extension", "transcript", "extension"]
        assert recorder.records[1].start_ns == _ns(T0_MS + 3)
        assert recorder.invalid_extension_entries == 0

    @pytest.mark.parametrize(
        "entry",
        [
            _request(1, 1, 0, [USER], v=2),
            _request(1, 1, 0, [USER], count=7),
            _request(True, 1, 0, [USER]),
            {"type": "entry_appended", "entry": {"customType": EXTENSION_REQUEST_ENTRY, "data": "x"}},
        ],
    )
    def test_invalid_extension_entries_are_ignored(self, entry):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": USER}),
                (1, entry),
                (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)}),
            ],
        )
        assert recorder.records[0].metadata["input_source"] == "transcript"
        assert recorder.invalid_extension_entries == 1

    def test_sequence_must_increase_and_response_must_match(self):
        recorder = _feed(
            _recorder(),
            [
                (1, _request(2, 1, 0, [USER])),
                (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)}),
                (3, _request(2, 3, 1, [])),  # same seq again
                (4, _response(9, 4)),  # no matching request
                (5, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 3)}),
            ],
        )
        assert recorder.records[1].metadata["input_source"] == "transcript"
        assert "http_status" not in recorder.records[1].metadata

    def test_retry_after_a_failed_call(self):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": USER}),
                (1, _request(1, 1, 0, [USER])),
                (
                    5,
                    {
                        "type": "message_end",
                        "message": _assistant([], stop="error", errorMessage="529", timestamp=T0_MS + 1),
                    },
                ),
                (6, {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 1}),
                (8, _request(2, 8, 1, [])),
                (9, _response(2, 9)),
                (20, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 8)}),
                (21, {"type": "auto_retry_end", "success": True, "attempt": 2}),
            ],
        )
        failed, retried = recorder.records
        assert (failed.error, failed.start_ns, failed.end_ns) == ("529", _ns(T0_MS + 1), _ns(T0_MS + 5))
        assert "http_status" not in failed.metadata
        assert (retried.error, retried.start_ns) == (None, _ns(T0_MS + 8))
        assert retried.metadata["http_status"] == 200
        assert [message["role"] for message in _input(retried)] == ["user"]

    def test_response_without_request_is_ignored(self):
        recorder = _feed(
            _recorder(),
            [(1, _response(1, 1)), (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)})],
        )
        assert "http_status" not in recorder.records[0].metadata
        assert recorder.records[0].metadata["input_source"] == "transcript"

    def test_a_request_is_used_only_once(self):
        recorder = _feed(
            _recorder(),
            [
                (1, _request(1, 1, 0, [USER])),
                (2, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 1)}),
                (3, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 3)}),
            ],
        )
        assert [record.metadata["input_source"] for record in recorder.records] == ["extension", "transcript"]

    def test_input_budget_stops_extension_data(self, monkeypatch):
        import airedale.harness.pi_spans as pi_spans

        monkeypatch.setattr(pi_spans, "MAX_INPUT_MESSAGE_REFS", 5)
        recorder = _recorder()
        events: list[tuple[float, dict[str, Any]]] = [(0, {"type": "message_end", "message": USER})]
        for seq in range(1, 4):
            # Each request repeats a context of 2 messages with no change: a
            # cheap entry that still copies the context.
            events.append((seq, _request(seq, seq, 0 if seq == 1 else 2, [USER, USER] if seq == 1 else [])))
            events.append((seq + 0.5, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + seq)}))
        _feed(recorder, events)
        sources = [record.metadata["input_source"] for record in recorder.records]
        # 2 + 2 references fit in the budget of 5; the third request does not,
        # and the extension data is not used any more.
        assert sources == ["extension", "extension", "transcript"]
        assert recorder.records[2].metadata["input_truncated"] is True
        assert recorder.invalid_extension_entries == 1

    def test_input_budget_keeps_only_the_new_messages(self, monkeypatch):
        import airedale.harness.pi_spans as pi_spans

        monkeypatch.setattr(pi_spans, "MAX_INPUT_MESSAGE_REFS", 3)
        reply = _assistant([{"type": "text", "text": "A"}], timestamp=T0_MS)
        events: list[tuple[float, dict[str, Any]]] = []
        for index in range(3):
            events.append((index * 10, {"type": "message_end", "message": {"role": "user", "content": f"u{index}"}}))
            events.append((index * 10 + 5, {"type": "message_end", "message": reply}))
        recorder = _feed(_recorder(), events)
        inputs = [[message["content"] for message in _input(record)] for record in recorder.records]
        # Call 1 copies 1 reference and call 2 copies 3 (more than the 2 left).
        assert inputs == [["u0"], ["u1"], ["u2"]]
        assert [record.metadata["input_truncated"] for record in recorder.records] == [False, True, True]

    def test_other_entries_are_ignored(self):
        recorder = _feed(_recorder(), [(1, {"type": "entry_appended", "entry": {"customType": "other", "data": {}}})])
        assert recorder.extension_entries == 0

    def test_feed_never_raises(self):
        recorder = _recorder()
        for event in [
            {"type": "message_end", "message": {"role": "assistant", "content": 5, "usage": "x"}},
            {"type": "tool_execution_end", "toolCallId": None, "durationMs": "x"},
            {"type": "compaction_end", "result": []},
            {"type": "entry_appended", "entry": 3},
            {"type": "turn_start", "unexpected": object()},
        ]:
            recorder.feed(event)


# --------------------------------------------------------------------------- #
# Tools and compaction
# --------------------------------------------------------------------------- #
def _tool_start(tool_id: str, name: str = "bash", parent: str | None = None) -> dict:
    event = {"type": "tool_execution_start", "toolCallId": tool_id, "toolName": name, "args": {"command": "ls"}}
    if parent:
        event["parentToolCallId"] = parent
    return event


def _tool_end(
    tool_id: str, *, duration_ms: float | None = None, error: bool = False, parent: str | None = None
) -> dict:
    event: dict[str, Any] = {
        "type": "tool_execution_end",
        "toolCallId": tool_id,
        "toolName": "bash",
        "result": {"content": [{"type": "text", "text": "out"}]},
        "isError": error,
    }
    if duration_ms is not None:
        event["durationMs"] = duration_ms
    if parent:
        event["parentToolCallId"] = parent
    return event


class TestTools:
    def test_duration_from_pi_is_used_and_bounded(self):
        recorder = _feed(
            _recorder(),
            [
                (10, _tool_start("a")),
                (20, _tool_start("b")),
                (50, _tool_end("a", duration_ms=30)),
                (60, _tool_end("b", duration_ms=500)),  # cannot end after its end record
            ],
        )
        a, b = recorder.records
        assert (a.start_ns, a.end_ns) == (_ns(T0_MS + 10), _ns(T0_MS + 40))
        assert (b.start_ns, b.end_ns) == (_ns(T0_MS + 20), _ns(T0_MS + 60))
        assert a.output_data == "out"
        assert a.metadata["duration_ms"] == 30

    def test_error_and_parallel_tools_are_siblings(self):
        recorder = _feed(
            _recorder(),
            [(1, _tool_start("a")), (2, _tool_start("b")), (3, _tool_end("b", error=True)), (4, _tool_end("a"))],
        )
        assert [record.metadata["tool_id"] for record in recorder.records] == ["b", "a"]
        assert recorder.records[0].error == "out"
        assert all(not record.children for record in recorder.records)

    def test_nested_calls_are_children(self):
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("code", name="codemode")),
                (2, _tool_start("code/1", parent="code")),
                (3, _tool_end("code/1", parent="code")),
                (4, _tool_start("code/2", parent="code")),
                (9, _tool_end("code")),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 10))
        [parent] = recorder.records
        assert parent.name == "codemode"
        assert [child.metadata["tool_id"] for child in parent.children] == ["code/1", "code/2"]
        assert parent.children[0].metadata["parent_tool_id"] == "code"
        # code/2 was still open when its parent ended: close() ends it, it
        # keeps its parent, and it does not end after its parent.
        assert [child.error for child in parent.children] == [None, "tool did not finish"]
        assert parent.end_ns == _ns(T0_MS + 9)
        assert parent.children[1].end_ns == parent.end_ns

    def test_close_marks_unfinished_steps(self):
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("code", name="codemode")),
                (2, _tool_start("code/1", parent="code")),
                (3, {"type": "compaction_start", "reason": "threshold"}),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 10))
        tool = next(record for record in recorder.records if record.kind == "tool")
        assert tool.error == "tool did not finish"
        assert tool.end_ns == _ns(T0_MS + 10)
        assert [child.error for child in tool.children] == ["tool did not finish"]
        compaction = next(record for record in recorder.records if record.kind == "task")
        assert compaction.error == "compaction did not finish"

    def test_end_without_start(self):
        recorder = _feed(_recorder(), [(5, _tool_end("x", duration_ms=1))])
        assert recorder.records[0].start_ns == _ns(T0_MS + 5)
        assert recorder.records[0].end_ns == _ns(T0_MS + 5)

    def test_compaction_span(self):
        result = {"summary": "S", "tokensBefore": 100, "usage": {"input": 1, "output": 2, "totalTokens": 3}}
        recorder = _feed(
            _recorder(),
            [
                (1, {"type": "compaction_start", "reason": "threshold"}),
                (9, {"type": "compaction_end", "result": result, "aborted": False}),
                (10, {"type": "compaction_start", "reason": "overflow"}),
                (11, {"type": "compaction_end", "aborted": True}),
            ],
        )
        done, aborted = recorder.records
        assert (done.kind, done.name, done.output_data) == ("task", "pi.compaction", "S")
        assert done.metadata["tokens_before"] == 100
        assert done.metadata["usage"]["output_tokens"] == 2
        assert (done.start_ns, done.end_ns) == (_ns(T0_MS + 1), _ns(T0_MS + 9))
        assert aborted.error == "compaction aborted"


# --------------------------------------------------------------------------- #
# Emission
# --------------------------------------------------------------------------- #
class TestEmit:
    def test_spans_have_recorded_times_parents_and_errors(self, fake_span_llmobs):
        recorder = _feed(
            _recorder(tool_definitions=[{"name": "bash"}]),
            [
                (1, {"type": "message_end", "message": USER}),
                (100, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 2)}),
                (110, _tool_start("code", name="codemode")),
                (120, _tool_start("code/1", parent="code")),
                (130, _tool_end("code/1", parent="code", error=True)),
                (140, _tool_end("code")),
            ],
        )
        recorder.emit(fake_span_llmobs)
        spans = fake_span_llmobs.spans
        assert [(span.kind, span.name) for span in spans] == [
            ("llm", "eval.anthropic_agent"),
            ("tool", "codemode"),
            ("tool", "bash"),
        ]
        llm_span, parent, child = spans
        assert llm_span.parent is None and parent.parent is None
        assert child.parent is parent
        # ``finish(finish_time=<seconds>)`` is a float: allow 1 microsecond.
        assert llm_span.start_ns == _ns(T0_MS + 2)
        assert llm_span.end_ns == pytest.approx(_ns(T0_MS + 100), abs=1000)
        assert child.start_ns == _ns(T0_MS + 120)
        assert child.end_ns == pytest.approx(_ns(T0_MS + 130), abs=1000)
        assert llm_span.kwargs == {"model_name": "claude-sonnet-4-6", "model_provider": "anthropic"}
        assert llm_span.annotations["tool_definitions"] == [{"name": "bash"}]
        assert llm_span.annotations["prompt"]["version"] == "v"
        assert llm_span.annotations["tags"]["sdk"] == "pi"
        assert child.error == 1
        assert child.tags["error.message"] == "out"
        assert fake_span_llmobs.active == []

    def test_parent_covers_late_children(self, fake_span_llmobs):
        recorder = _feed(
            _recorder(),
            [
                (10, _tool_start("code", name="codemode")),
                # The parent ends (from durationMs) before its child starts.
                (30, _tool_end("code", duration_ms=5)),
                (40, _tool_start("code/1", parent="code")),
                (50, _tool_end("code/1", parent="code")),
            ],
        )
        recorder.emit(fake_span_llmobs)
        parent, child = fake_span_llmobs.spans
        assert child.parent is parent
        assert parent.start_ns <= child.start_ns
        assert parent.end_ns >= child.end_ns - 1000

    def test_emit_is_sorted_by_start_time(self, fake_span_llmobs):
        recorder = _feed(
            _recorder(), [(1, _tool_start("a")), (2, _tool_start("b")), (3, _tool_end("b")), (4, _tool_end("a"))]
        )
        recorder.emit(fake_span_llmobs)
        assert [span.annotations["metadata"]["tool_id"] for span in fake_span_llmobs.spans] == ["a", "b"]

    def test_disabled_llmobs_sends_nothing(self, fake_span_llmobs):
        fake_span_llmobs.enabled = False
        recorder = _feed(_recorder(), [(1, _tool_start("a")), (2, _tool_end("a"))])
        recorder.emit(fake_span_llmobs)
        assert fake_span_llmobs.spans == []

    def test_emit_never_raises(self, fake_span_llmobs):
        def broken(**kwargs):
            raise RuntimeError("annotate failed")

        fake_span_llmobs.annotate = broken
        recorder = _feed(
            _recorder(), [(1, _tool_start("a")), (2, _tool_end("a")), (3, _tool_start("b")), (4, _tool_end("b"))]
        )
        recorder.emit(fake_span_llmobs)
        # Every span is still finished (no span stays active).
        assert len(fake_span_llmobs.spans) == 2
        assert fake_span_llmobs.active == []


# --------------------------------------------------------------------------- #
# Collector extension (TypeScript, run with Node)
# --------------------------------------------------------------------------- #
DRIVER = textwrap.dedent(
    """
    const { default: register } = await import(process.argv[2]);
    const steps = JSON.parse(process.argv[3]);
    const handlers = {};
    const entries = [];
    register({
      on: (name, handler) => { handlers[name] = handler; },
      appendEntry: (customType, data) => entries.push({ customType, data }),
    });
    const returned = [];
    for (const [name, event] of steps) {
      if (handlers[name]) returned.push(await handlers[name]({ type: name, ...event }) ?? null);
    }
    console.log(JSON.stringify({ entries, returned, events: Object.keys(handlers).sort() }));
    """
)


def _node_with_typescript() -> str | None:
    node = shutil.which("node")
    if node is None:
        return None
    try:
        output = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
        major = int(output.strip().lstrip("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    # Node 23.6+ runs TypeScript files (type stripping) by default.
    return node if major >= 24 else None


def _run_extension(tmp_path, steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    node = _node_with_typescript()
    if node is None:
        pytest.skip("needs Node 24+ to run the TypeScript extension")
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER)
    completed = subprocess.run(
        [node, str(driver), str(PI_LLMOBS_EXTENSION_PATH), json.dumps(steps)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(completed.stdout)


class TestCollectorExtension:
    def test_entries_rebuild_the_requests(self, tmp_path):
        image_user = {"role": "user", "content": [{"type": "image", "data": "A" * 100, "mimeType": "image/png"}]}
        assistant = _assistant(
            [{"type": "thinking", "thinking": "t", "thinkingSignature": "opaque"}, {"type": "text", "text": "A"}]
        )
        result = {"role": "toolResult", "toolCallId": "c", "toolName": "ls", "content": [], "details": {"big": 1}}
        first = [SYSTEM, image_user]
        second = [SYSTEM, image_user, assistant, result]
        out = _run_extension(
            tmp_path,
            [
                ("context_with_system", {"messages": first}),
                ("before_provider_request", {"payload": {}}),
                ("before_provider_request", {"payload": {}}),  # a compaction call: ignored
                ("after_provider_response", {"status": 200, "headers": {"request-id": "req_9", "x": "y"}}),
                ("after_provider_response", {"status": 500, "headers": {}}),  # ignored
                ("context_with_system", {"messages": second}),
                ("before_provider_request", {"payload": {}}),
            ],
        )
        assert out["events"] == ["after_provider_response", "before_provider_request", "context_with_system"]
        # The handlers never change the request.
        assert out["returned"] == [None] * 7
        entries = out["entries"]
        assert [entry["customType"] for entry in entries] == [
            EXTENSION_REQUEST_ENTRY,
            EXTENSION_RESPONSE_ENTRY,
            EXTENSION_REQUEST_ENTRY,
        ]
        request_1, response_1, request_2 = (entry["data"] for entry in entries)
        assert (request_1["seq"], request_1["prefix"], request_1["count"]) == (1, 0, 2)
        assert request_1["appended"][1]["content"][0] == {"type": "image", "mimeType": "image/png", "bytes": 100}
        assert (response_1["seq"], response_1["status"], response_1["requestId"]) == (1, 200, "req_9")
        assert (request_2["seq"], request_2["prefix"], request_2["count"]) == (2, 2, 4)
        assert "thinkingSignature" not in json.dumps(request_2)
        assert "details" not in request_2["appended"][1]

        # The recorder rebuilds the same request messages from the entries.
        recorder = _recorder()
        now = T0_MS
        for entry in entries:
            now += 1
            recorder.feed({"type": "entry_appended", "entry": entry}, received_ns=_ns(now))
            if entry["customType"] == EXTENSION_REQUEST_ENTRY and entry["data"]["seq"] == 1:
                continue
            if entry["customType"] == EXTENSION_RESPONSE_ENTRY:
                recorder.feed({"type": "message_end", "message": _assistant([])}, received_ns=_ns(now))
        recorder.feed({"type": "message_end", "message": _assistant([])}, received_ns=_ns(now + 1))
        assert [record.metadata["input_source"] for record in recorder.records] == ["extension", "extension"]
        roles = [message["role"] for message in _input(recorder.records[1])]
        assert roles == ["system", "user", "assistant", "user"]
        assert recorder.records[0].metadata["request_id"] == "req_9"

    def test_change_is_relative_to_the_last_sent_request(self, tmp_path):
        reply = {"role": "assistant", "content": [{"type": "text", "text": "A"}]}
        out = _run_extension(
            tmp_path,
            [
                ("context_with_system", {"messages": [SYSTEM, USER]}),
                ("before_provider_request", {}),
                # A context that is not sent (for example the run was aborted).
                ("context_with_system", {"messages": [SYSTEM, {"role": "user", "content": "other"}]}),
                ("context_with_system", {"messages": [SYSTEM, USER, reply]}),
                ("before_provider_request", {}),
            ],
        )
        first, second = (entry["data"] for entry in out["entries"])
        assert (first["seq"], first["prefix"], first["count"]) == (1, 0, 2)
        # Relative to request 1 (the last sent one), not to the unsent context.
        assert (second["seq"], second["prefix"], second["count"]) == (3, 2, 3)
        assert second["appended"] == [reply]

    def test_too_large_change_is_skipped_then_sent_in_full(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AIREDALE_PI_LLMOBS_MAX_APPENDED_CHARS", "200")
        big = {"role": "user", "content": "x" * 300}
        reply = {"role": "assistant", "content": [{"type": "text", "text": "A"}]}
        out = _run_extension(
            tmp_path,
            [
                ("context_with_system", {"messages": [USER]}),
                ("before_provider_request", {}),
                ("context_with_system", {"messages": [USER, big]}),
                ("before_provider_request", {}),
                ("context_with_system", {"messages": [USER, reply]}),
                ("before_provider_request", {}),
            ],
        )
        first, skipped, full = (entry["data"] for entry in out["entries"])
        assert (first["prefix"], first["count"]) == (0, 1)
        assert skipped["skipped"] is True and skipped["appended"] is None and skipped["count"] == 2
        # After a skipped entry, the next one is a full list.
        assert (full["prefix"], full["count"], len(full["appended"])) == (0, 2, 2)

    def test_handler_errors_are_contained(self, tmp_path):
        out = _run_extension(
            tmp_path,
            [
                ("context_with_system", {"messages": None}),
                ("before_provider_request", {}),
                ("after_provider_response", {"headers": None}),
            ],
        )
        assert out["returned"] == [None, None, None]
        request, response = (entry["data"] for entry in out["entries"])
        assert request["count"] == 0
        assert response["status"] is None and response["requestId"] is None
