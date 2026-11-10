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
        assert llmobs_input_message(SYSTEM) == [{"role": "system", "content": "You are pi.\n\nSYS"}]
        assert llmobs_input_message({"role": "system", "content": "Direct"}) == [
            {"role": "system", "content": "Direct"}
        ]

    def test_tool_result_and_assistant(self):
        assistant = _assistant(
            [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Look"},
                {"type": "toolCall", "id": "c1", "name": "ls", "arguments": {"path": "."}},
            ]
        )
        # The thinking block is kept (pi sends it back to the model), in order.
        assert llmobs_input_message(assistant) == [
            {"role": "assistant", "content": "hmm"},
            {
                "role": "assistant",
                "content": "Look",
                "tool_calls": [{"name": "ls", "arguments": {"path": "."}, "type": "tool_use", "tool_id": "c1"}],
            },
        ]
        result = {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "ls",
            "content": [{"type": "text", "text": "a"}, {"type": "image", "data": "xx"}],
            "isError": False,
        }
        assert llmobs_input_message(result) == [
            {
                "role": "user",
                "content": "",
                "tool_results": [{"name": "ls", "result": "a\n[image]", "type": "tool_result", "tool_id": "c1"}],
            }
        ]

    def test_other_roles(self):
        assert llmobs_input_message({"role": "custom", "content": "note"}) == [{"role": "user", "content": "note"}]
        assert llmobs_input_message({"role": "compactionSummary", "summary": "S"}) == [{"role": "user", "content": "S"}]
        assert llmobs_input_message({"role": "bashExecution", "command": "ls", "output": "x"}) == [
            {"role": "user", "content": "$ ls\nx"}
        ]
        assert llmobs_input_message({"role": "bashExecution", "command": "ls", "excludeFromContext": True}) == []
        assert llmobs_input_message({"role": "unknown"}) == []
        assert llmobs_input_message("not a message") == []

    def test_assistant_blocks_keep_their_order(self):
        message = _assistant(
            [
                {"type": "text", "text": "A"},
                {"type": "text", "text": "B"},
                {"type": "toolCall", "id": "c1", "name": "ls", "arguments": {}},
                {"type": "thinking", "thinking": "T"},
                {"type": "toolCall", "id": "c2", "name": "ls", "arguments": {}},
                {"type": "text", "text": "C"},
                {"type": "thinking", "thinking": " "},
            ]
        )
        rendered = llmobs_input_message(message)
        assert [(item["content"], [call["tool_id"] for call in item.get("tool_calls", [])]) for item in rendered] == [
            ("A\n\nB", ["c1"]),
            ("T", []),
            ("", ["c2"]),
            ("C", []),
        ]
        assert llmobs_input_message(_assistant([])) == [{"role": "assistant", "content": ""}]
        assert llmobs_input_message({"role": "assistant", "content": " plain "}) == [
            {"role": "assistant", "content": "plain"}
        ]
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
        assert record.metrics is not None
        assert record.metrics["input_tokens"] == 10
        assert record.metrics["output_tokens"] == 5
        assert record.error is None

    def test_output_keeps_the_block_order(self):
        message = _assistant(
            [
                {"type": "text", "text": "First"},
                {"type": "toolCall", "id": "c1", "name": "ls", "arguments": {}},
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "B"},
                {"type": "text", "text": "C"},
            ],
            timestamp=T0_MS,
        )
        recorder = _feed(_recorder(), [(1, {"type": "message_end", "message": message})])
        assert recorder.records[0].output_data == [
            {"role": "assistant", "content": "First"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "ls", "arguments": {}, "type": "tool_use", "tool_id": "c1"}],
            },
            {"role": "assistant", "content": "hmm"},
            {"role": "assistant", "content": "B\n\nC"},
        ]
        text_only = _assistant([], timestamp=T0_MS)
        text_only["content"] = "Plain"
        recorder = _feed(_recorder(), [(1, {"type": "message_end", "message": text_only})])
        assert recorder.records[0].output_data == [{"role": "assistant", "content": "Plain"}]
        tool_only = _assistant([{"type": "text", "text": "T"}, {"type": "toolCall", "id": "c2", "name": "ls"}])
        recorder = _feed(_recorder(), [(1, {"type": "message_end", "message": tool_only})])
        assert [message["content"] for message in recorder.records[0].output_data] == ["T", "", ""]

    @pytest.mark.parametrize("timestamp", [None, T0_MS + 10_000, "soon", True])
    def test_untrusted_timestamp_falls_back_to_the_previous_event(self, timestamp):
        message = _assistant([{"type": "text", "text": "A"}])
        message["timestamp"] = timestamp
        recorder = _feed(_recorder(), [(3, {"type": "turn_start"}), (900, {"type": "message_end", "message": message})])
        assert recorder.records[0].start_ns == _ns(T0_MS + 3)

    def test_end_uses_the_pi_duration(self):
        message = _assistant([], timestamp=T0_MS + 5, durationMs=200)
        recorder = _feed(_recorder(), [(900, {"type": "message_end", "message": message})])
        record = recorder.records[0]
        assert (record.start_ns, record.end_ns) == (_ns(T0_MS + 5), _ns(T0_MS + 205))
        assert record.metadata["duration_ms"] == 200

    @pytest.mark.parametrize("duration_ms", [5000, -1, "x", None, float("inf"), float("nan"), 1e300, 10**400])
    def test_end_is_bounded_by_the_arrival(self, duration_ms):
        message = _assistant([], timestamp=T0_MS + 5, durationMs=duration_ms)
        recorder = _feed(_recorder(), [(900, {"type": "message_end", "message": message})])
        assert recorder.records[0].end_ns == _ns(T0_MS + 900)

    @pytest.mark.parametrize("timestamp", [0, float("inf"), T0_MS + 10_000, 1e300, 10**400, -5])
    def test_duration_needs_a_plausible_timestamp(self, timestamp):
        message = _assistant([], timestamp=timestamp, durationMs=1)
        recorder = _feed(_recorder(), [(2, _request(1, 2, 0, [])), (900, {"type": "message_end", "message": message})])
        record = recorder.records[0]
        # The extension start is used; the end is the arrival, not 0 + 1 ms.
        assert (record.start_ns, record.end_ns) == (_ns(T0_MS + 2), _ns(T0_MS + 900))
        assert "duration_ms" not in record.metadata

    def test_duration_with_an_earlier_extension_start(self):
        message = _assistant([], timestamp=T0_MS + 5, durationMs=100)
        recorder = _feed(_recorder(), [(2, _request(1, 3, 0, [])), (900, {"type": "message_end", "message": message})])
        record = recorder.records[0]
        assert (record.start_ns, record.end_ns) == (_ns(T0_MS + 3), _ns(T0_MS + 105))

    @pytest.mark.parametrize("t", [1e300, 10**400, float("nan")])
    def test_unusable_extension_time_is_ignored(self, t):
        entry = _request(1, 0, 0, [])
        entry["entry"]["data"]["t"] = t
        message = _assistant([], timestamp=T0_MS + 5)
        recorder = _feed(_recorder(), [(2, entry), (900, {"type": "message_end", "message": message})])
        assert recorder.records[0].start_ns == _ns(T0_MS + 5)

    def test_unfinished_request_with_an_old_start_uses_the_close_time(self):
        entry = _request(1, 0, 0, [USER])
        entry["entry"]["data"]["t"] = 1  # 1970
        recorder = _feed(_recorder(), [(2, entry)])
        recorder.close(received_ns=_ns(T0_MS + 50))
        assert recorder.records[0].start_ns == _ns(T0_MS + 50)

    def test_unfinished_request_and_open_tool(self):
        recorder = _feed(_recorder(), [(1, _tool_start("a")), (2, _request(1, 2, 0, [USER]))])
        recorder.close(received_ns=_ns(T0_MS + 9))
        assert sorted((record.kind, record.error) for record in recorder.records) == [
            ("llm", "model call did not finish"),
            ("tool", "tool did not finish"),
        ]

    def test_unfinished_request_is_an_error_span(self):
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "turn_start"}),
                (1, {"type": "message_end", "message": USER}),
                (2, _request(1, 2, 0, [USER])),
                (3, _response(1, 3)),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 50))
        [record] = recorder.records
        assert (record.kind, record.error) == ("llm", "model call did not finish")
        assert (record.start_ns, record.end_ns) == (_ns(T0_MS + 2), _ns(T0_MS + 50))
        assert record.metadata["input_source"] == "extension"
        assert record.metadata["http_status"] == 200
        assert record.metrics is None
        assert _input(record) == [{"role": "user", "content": "Q"}]
        # Closing again adds nothing.
        recorder.close(received_ns=_ns(T0_MS + 60))
        assert len(recorder.records) == 1

    def test_unfinished_request_is_sent(self, fake_span_llmobs):
        recorder = _feed(_recorder(), [(2, _request(1, 2, 0, [USER]))])
        recorder.close(received_ns=_ns(T0_MS + 50))
        recorder.emit(fake_span_llmobs)
        [span] = fake_span_llmobs.spans
        assert (span.kind, span.error) == ("llm", 1)
        assert span.tags["error.message"] == "model call did not finish"
        assert "metrics" not in span.annotations

    def test_failed_call_is_an_error_span(self):
        message = _assistant([], stop="error", errorMessage="529 overloaded", timestamp=T0_MS)
        recorder = _feed(_recorder(), [(10, {"type": "message_end", "message": message})])
        assert recorder.records[0].error == "provider error (stopReason=error)"
        assert "529 overloaded" not in repr(recorder.records[0].metadata)
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
        assert "request_id" not in record.metadata  # (The request id is in the private notes only.)
        assert any("req_1" in note for note in recorder.private_notes)
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
        assert (failed.error, failed.start_ns, failed.end_ns) == (
            "provider error (stopReason=error)",
            _ns(T0_MS + 1),
            _ns(T0_MS + 5),
        )
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
        # A truncated input starts at the previous response.
        assert inputs == [["u0"], ["A", "u1"], ["A", "u2"]]
        assert [record.metadata["input_truncated"] for record in recorder.records] == [False, True, True]

    def test_truncated_input_keeps_the_tool_call_of_a_tool_result(self, monkeypatch):
        import airedale.harness.pi_spans as pi_spans

        monkeypatch.setattr(pi_spans, "MAX_INPUT_MESSAGE_REFS", 1)
        call = _assistant([{"type": "toolCall", "id": "c1", "name": "ls", "arguments": {}}], timestamp=T0_MS)
        result = {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "ls",
            "content": [{"type": "text", "text": "a"}],
        }
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": USER}),
                (1, {"type": "message_end", "message": call}),
                (2, {"type": "message_end", "message": result}),
                (3, {"type": "message_end", "message": _assistant([], timestamp=T0_MS + 2)}),
            ],
        )
        second = recorder.records[1]
        assert second.metadata["input_truncated"] is True
        assert [message.get("tool_calls", message.get("tool_results")) for message in _input(second)] == [
            [{"name": "ls", "arguments": {}, "type": "tool_use", "tool_id": "c1"}],
            [{"name": "ls", "result": "a", "type": "tool_result", "tool_id": "c1"}],
        ]

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
#: The name of the last start record of each id: pi writes the same name in the end record.
_TOOL_NAMES: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _reset_tool_names():
    _TOOL_NAMES.clear()


def _tool_start(tool_id: str, name: str = "bash", parent: str | None = None) -> dict:
    _TOOL_NAMES[tool_id] = name
    event = {"type": "tool_execution_start", "toolCallId": tool_id, "toolName": name, "args": {"command": "ls"}}
    if parent is not None:
        event["parentToolCallId"] = parent
    return event


def _tool_end(
    tool_id: str, *, duration_ms: float | None = None, error: bool = False, parent: str | None = None
) -> dict:
    event: dict[str, Any] = {
        "type": "tool_execution_end",
        "toolCallId": tool_id,
        "toolName": _TOOL_NAMES.get(tool_id, "bash"),
        "result": {"content": [{"type": "text", "text": "out"}]},
        "isError": error,
    }
    if duration_ms is not None:
        event["durationMs"] = duration_ms
    if parent is not None:
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
                # Cannot end after its end record: the records were late, so
                # the span starts earlier and keeps the pi duration.
                (60, _tool_end("b", duration_ms=500)),
            ],
        )
        a, b = recorder.records
        assert (a.start_ns, a.end_ns) == (_ns(T0_MS + 10), _ns(T0_MS + 40))
        assert (b.start_ns, b.end_ns) == (_ns(T0_MS - 440), _ns(T0_MS + 60))
        assert a.output_data == "out"
        assert a.metadata["duration_ms"] == 30

    def test_late_records_do_not_start_before_the_model_call_ended(self):
        call = _assistant([{"type": "toolCall", "id": "a", "name": "bash"}], timestamp=T0_MS)
        recorder = _feed(
            _recorder(),
            [
                (100, {"type": "message_end", "message": call}),
                (101, _tool_start("a")),
                (102, _tool_end("a", duration_ms=500)),
            ],
        )
        llm, tool = recorder.records
        assert llm.end_ns == _ns(T0_MS + 100)
        assert (tool.start_ns, tool.end_ns) == (_ns(T0_MS + 100), _ns(T0_MS + 102))

    def test_late_records_use_the_model_call_that_asked_for_the_tool(self):
        first = _assistant([{"type": "toolCall", "id": "a", "name": "bash"}], timestamp=T0_MS)
        later = _assistant([], timestamp=T0_MS + 101, durationMs=19)
        recorder = _feed(
            _recorder(),
            [
                (0, {"type": "message_end", "message": first}),
                (100, _tool_start("a")),
                (120, {"type": "message_end", "message": later}),
                (130, _tool_end("a", duration_ms=100)),
            ],
        )
        tool = next(record for record in recorder.records if record.kind == "tool")
        # The later model call (it ended at 120) does not limit the tool.
        assert (tool.start_ns, tool.end_ns) == (_ns(T0_MS + 30), _ns(T0_MS + 130))

    def test_error_and_parallel_tools_are_siblings(self):
        recorder = _feed(
            _recorder(),
            [(1, _tool_start("a")), (2, _tool_start("b")), (3, _tool_end("b", error=True)), (4, _tool_end("a"))],
        )
        assert [record.metadata["tool_id"] for record in recorder.records] == ["b", "a"]
        assert recorder.records[0].error == "tool call failed"
        assert recorder.records[0].output_data == "out"  # The span output is unchanged.
        assert recorder.private_notes == []  # The event collector has the output for the private report.
        assert "out" not in recorder.records[0].error and "out" not in recorder.records[0].metadata["error"]
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

    def test_nested_call_of_a_tool_with_an_empty_id(self):
        # Pi can use "" as the id of a model call; its nested calls then have
        # parentToolCallId "" and must stay children of that call.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("", name="codemode")),
                (2, _tool_start("/1", parent="")),
                (3, _tool_end("/1", parent="")),
                (9, _tool_end("")),
            ],
        )
        [parent] = recorder.records
        assert parent.name == "codemode"
        assert [child.metadata["tool_id"] for child in parent.children] == ["/1"]
        assert parent.children[0].metadata["parent_tool_id"] == ""

    def test_reused_empty_id_keeps_the_first_parent(self):
        # A late child of the first "" call must not move to the next "" call.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("", name="first")),
                (2, _tool_start("/1", parent="")),
                (3, _tool_end("")),
                (4, _tool_start("", name="second")),
                (5, _tool_end("/1", parent="")),
                (6, _tool_end("")),
            ],
        )
        first, second = recorder.records
        assert (first.name, second.name) == ("first", "second")
        assert [child.metadata["tool_id"] for child in first.children] == ["/1"]
        assert second.children == []
        # The first call already ended: its child does not end after it.
        assert first.children[0].end_ns == first.end_ns

    def test_child_that_ends_before_its_parent_starts_is_attached_to_it(self):
        # start(child) -> end(child) -> start(parent) -> end(parent): the child
        # is a child span of the parent, not a root span.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("code/1", parent="code")),
                (2, _tool_end("code/1", parent="code")),
                (3, _tool_start("code", name="codemode")),
                (4, _tool_end("code")),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 5))
        [parent] = recorder.records
        assert parent.name == "codemode"
        assert [child.metadata["tool_id"] for child in parent.children] == ["code/1"]
        assert parent.children[0].metadata["parent_tool_id"] == "code"
        # The parent was running while its child ran.
        assert parent.start_ns <= parent.children[0].start_ns
        assert parent.children[0].end_ns <= parent.end_ns

    def test_child_whose_parent_never_starts_stays_at_the_root(self):
        recorder = _feed(
            _recorder(),
            [(1, _tool_start("x/1", parent="x")), (2, _tool_end("x/1", parent="x"))],
        )
        recorder.close(received_ns=_ns(T0_MS + 5))
        [child] = recorder.records
        assert child.metadata["tool_id"] == "x/1"

    def test_early_child_goes_to_the_first_parent_with_the_id(self):
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("/1", parent="")),
                (2, _tool_end("/1", parent="")),
                (3, _tool_start("", name="first")),
                (4, _tool_end("")),
                (5, _tool_start("", name="second")),
                (6, _tool_end("")),
            ],
        )
        first, second = recorder.records
        assert [child.metadata["tool_id"] for child in first.children] == ["/1"]
        assert second.children == []

    @pytest.mark.parametrize(
        "events",
        [
            [
                (1, _tool_start("p", "p")),
                (2, _tool_start("a", "a", "p")),
                (3, _tool_start("b", "b", "p")),
                (4, _tool_end("b", parent="p")),
                (5, _tool_end("a", parent="p")),
                (6, _tool_end("p")),
            ],
            [(1, _tool_start("c", "c", "p")), (2, _tool_end("p")), (3, _tool_end("c", parent="p"))],
            [(1, _tool_end("p")), (2, _tool_start("c", "c", "p")), (3, _tool_end("c", parent="p"))],
            [
                (1, _tool_start("c", "c", "p")),
                (2, _tool_end("c", parent="p")),
                (3, _tool_start("p", "p")),
                (4, _tool_end("p")),
            ],
            [
                (1, _tool_start("kid", "kid", "mid")),
                (2, _tool_start("mid", "mid", "gone")),
                (3, _tool_end("kid", parent="mid")),
                (4, _tool_end("mid", parent="gone")),
            ],
            [(1, _tool_start("x", "x", "gone")), (2, _tool_end("x", parent="gone"))],
            [
                (1, _tool_start("g", "g", "c")),
                (2, _tool_start("c", "c", "p")),
                (3, _tool_start("p", "p")),
                (4, _tool_end("g", parent="c")),
                (5, _tool_end("c", parent="p")),
                (6, _tool_end("p")),
            ],
        ],
        ids=["parallel", "end-only parent", "end-only parent first", "early child", "orphan chain", "orphan", "chain"],
    )
    def test_span_tree_matches_the_collector(self, events):
        """The span recorder and the result collector give each call the same parent."""
        from airedale.harness.pi import _PiEventCollector

        collector = _PiEventCollector()
        recorder = _recorder()
        for at_ms, event in events:
            collector.feed(event)
            recorder.feed(event, received_ns=_ns(T0_MS + at_ms))
        recorder.close(received_ns=_ns(T0_MS + 99))
        collector.result(model=MODEL, returncode=0, stderr_tail="")  # Settles the parents that wait for an id.
        expected = {
            execution.tool_id: (execution.parent.tool_id if execution.parent is not None else None)
            for execution in collector.tool_executions
        }
        actual: dict[str, str | None] = {}

        def walk(record, parent_id):
            actual[record.metadata["tool_id"]] = parent_id
            for child in record.children:
                walk(child, record.metadata["tool_id"])

        for root in recorder.records:
            walk(root, None)
        assert actual == expected

    def test_parent_and_child_that_name_each_other_do_not_remove_the_spans(self):
        # start(A, parent=B) -> end(A) -> start(B, parent=A) -> end(B): the ids make a
        # cycle. Both spans are kept, one is the root, the other its child.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("A", name="a", parent="B")),
                (2, _tool_end("A", parent="B")),
                (3, _tool_start("B", name="b", parent="A")),
                (4, _tool_end("B", parent="A")),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 5))
        assert len(recorder.records) == 1
        assert sum(1 + len(child.children) for child in recorder.records) == 2

    def test_early_child_with_the_same_empty_id_as_its_parent(self):
        # The child (id "", parent "") starts and ends before the parent (id "") starts.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("", name="child", parent="")),
                (2, _tool_end("", parent="")),
                (3, _tool_start("", name="parent")),
                (4, _tool_end("")),
            ],
        )
        [parent] = recorder.records
        assert (parent.name, [child.name for child in parent.children]) == ("parent", ["child"])

    def test_parent_that_has_only_an_end_record(self):
        # start(c, parent=p) -> end(p, no start record) -> end(c): c is a child of p.
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("c", name="child", parent="p")),
                (2, _tool_end("p")),
                (3, _tool_end("c", parent="p")),
            ],
        )
        [parent] = recorder.records
        assert parent.metadata["tool_id"] == "p"
        assert [child.metadata["tool_id"] for child in parent.children] == ["c"]
        assert parent.children[0].end_ns <= parent.end_ns

    def test_child_of_a_parent_that_has_only_an_end_record_starts_after_it(self):
        # end(p, no start record) -> start(c, parent=p) -> end(c)
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_end("p")),
                (2, _tool_start("c", parent="p")),
                (3, _tool_end("c", parent="p")),
            ],
        )
        recorder.close(received_ns=_ns(T0_MS + 4))  # No other record with the id came: p is the parent.
        [parent] = recorder.records
        assert [child.metadata["tool_id"] for child in parent.children] == ["c"]

    def test_nested_call_that_comes_before_the_next_use_of_a_parent_id(self):
        """The id "x" is used twice. A nested call names "x" after the first use ended: it waits for the second."""
        from airedale.harness.pi import _PiEventCollector

        events = [
            (1, _tool_start("x", "first")),
            (2, _tool_end("x")),
            (3, _tool_start("kid", "kid", "x")),
            (4, _tool_end("kid", parent="x")),
            (5, _tool_start("x", "second")),
            (6, _tool_end("x")),
        ]
        collector, recorder = _PiEventCollector(), _recorder()
        for at_ms, event in events:
            collector.feed(event)
            recorder.feed(event, received_ns=_ns(T0_MS + at_ms))
        recorder.close(received_ns=_ns(T0_MS + 9))
        collector.result(model=MODEL, returncode=0, stderr_tail="")
        by_name = {execution.tool_name: execution for execution in collector.tool_executions}
        assert by_name["kid"].parent is by_name["second"]
        first, second = recorder.records
        assert (first.name, second.name) == ("first", "second")
        assert first.children == []
        assert [child.name for child in second.children] == ["kid"]

    def test_nested_call_after_the_end_of_a_parent_id_that_is_not_used_again(self):
        from airedale.harness.pi import _PiEventCollector

        events = [
            (1, _tool_start("x", "first")),
            (2, _tool_end("x")),
            (3, _tool_start("kid", "kid", "x")),
            (4, _tool_end("kid", parent="x")),
        ]
        collector, recorder = _PiEventCollector(), _recorder()
        for at_ms, event in events:
            collector.feed(event)
            recorder.feed(event, received_ns=_ns(T0_MS + at_ms))
        recorder.close(received_ns=_ns(T0_MS + 9))
        collector.result(model=MODEL, returncode=0, stderr_tail="")
        by_name = {execution.tool_name: execution for execution in collector.tool_executions}
        assert by_name["kid"].parent is by_name["first"]  # No other use of "x": the ended one is the parent.
        [first] = recorder.records
        assert [child.metadata["tool_id"] for child in first.children] == ["kid"]

    def test_end_only_call_keeps_its_parent_metadata(self):
        recorder = _feed(_recorder(), [(1, _tool_end("x", parent="y"))])
        assert recorder.records[0].metadata["parent_tool_id"] == "y"

    def test_deep_tree_is_sent(self, fake_span_llmobs):
        # 1,100 calls, each one the child of the previous: deeper than the
        # recursion limit of Python. The calls end in reverse order.
        depth = 1100
        events: list[tuple[float, dict[str, Any]]] = []
        for index in range(depth):
            events.append((index, _tool_start(str(index), name=f"t{index}", parent=str(index - 1) if index else None)))
        for index in reversed(range(depth)):
            events.append((depth + (depth - index), _tool_end(str(index), parent=str(index - 1) if index else None)))
        recorder = _feed(_recorder(), events)
        recorder.emit(fake_span_llmobs)
        assert len(fake_span_llmobs.spans) == depth
        assert [span.name for span in fake_span_llmobs.spans[:2]] == ["t0", "t1"]
        assert fake_span_llmobs.spans[1].parent is fake_span_llmobs.spans[0]
        assert fake_span_llmobs.spans[-1].parent is fake_span_llmobs.spans[-2]
        assert not fake_span_llmobs.active

    def test_span_that_cannot_be_sent_does_not_stop_its_siblings(self, fake_span_llmobs):
        recorder = _feed(
            _recorder(),
            [
                (1, _tool_start("p", name="parent")),
                (2, _tool_start("a", name="bad", parent="p")),
                (3, _tool_end("a", parent="p")),
                (4, _tool_start("b", name="good", parent="p")),
                (5, _tool_end("b", parent="p")),
                (6, _tool_end("p")),
            ],
        )
        original = fake_span_llmobs.annotate

        def annotate(span=None, **kwargs):
            if span is not None and span.name == "bad":
                raise RuntimeError("cannot annotate")
            original(span=span, **kwargs)

        fake_span_llmobs.annotate = annotate  # type: ignore[method-assign]
        recorder.emit(fake_span_llmobs)
        assert [span.name for span in fake_span_llmobs.spans] == ["parent", "bad", "good"]
        assert not fake_span_llmobs.active

    @pytest.mark.parametrize("child_ends", [True, False])
    def test_child_that_starts_before_its_parent_keeps_that_parent(self, child_ends):
        events: list[tuple[float, dict[str, Any]]] = [
            (1, _tool_start("/1", parent="")),
            (2, _tool_start("", name="first")),
            (3, _tool_end("")),
            (4, _tool_start("", name="second")),
        ]
        if child_ends:
            events.append((5, _tool_end("/1", parent="")))
        events.append((6, _tool_end("")))
        recorder = _feed(_recorder(), events)
        recorder.close(received_ns=_ns(T0_MS + 7))
        first, second = recorder.records
        assert [child.metadata["tool_id"] for child in first.children] == ["/1"]
        assert second.children == []

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
        # The pi duration is kept.
        assert recorder.records[0].start_ns == _ns(T0_MS + 4)
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
        assert child.tags["error.message"] == "tool call failed"
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
            [
                {"type": "thinking", "thinking": "t", "thinkingSignature": "opaque"},
                {"type": "text", "text": "A", "textSignature": "opaque"},
                # Tool arguments with the names of opaque pi fields are kept.
                {
                    "type": "toolCall",
                    "id": "c",
                    "name": "ls",
                    "arguments": {
                        "details": "d",
                        "textSignature": "s",
                        "nested": {"type": "image", "data": "abc", "thinkingSignature": "s"},
                    },
                    "thoughtSignature": "opaque",
                },
            ]
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
        assert "opaque" not in json.dumps(request_2)
        assert "opaque" not in json.dumps(request_1)
        assert request_2["appended"][0]["content"][2]["arguments"] == {
            "details": "d",
            "textSignature": "s",
            "nested": {"type": "image", "data": "abc", "thinkingSignature": "s"},
        }
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
        # The assistant message has a thinking block, then a text block with a tool call.
        assert roles == ["system", "user", "assistant", "assistant", "user"]
        assert "request_id" not in recorder.records[0].metadata
        assert any("req_9" in note for note in recorder.private_notes)

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


# --------------------------------------------------------------------------- #
# Option S: only known values from pi reach the spans
# --------------------------------------------------------------------------- #
def _one_call(**extra):
    message = _assistant([{"type": "text", "text": "A"}], timestamp=T0_MS + 1, **extra)
    return _feed(_recorder(), [(10, {"type": "message_end", "message": message})])


class TestOnlyKnownValuesInSpans:
    def test_known_values_are_reported(self):
        recorder = _one_call(
            responseModel="claude-sonnet-4-6", responseId="resp-1", api="anthropic-messages", thinkingLevel="high"
        )
        metadata = recorder.records[0].metadata
        assert metadata["response_model"] == "claude-sonnet-4-6"
        assert metadata["api"] == "anthropic-messages"
        assert metadata["thinking_level"] == "high"
        assert metadata["stop_reason"] == "stop"

    def test_response_id_is_never_reported(self):
        recorder = _one_call(responseId="SECRET-RESPONSE-ID-0123")
        assert "response_id" not in recorder.records[0].metadata
        assert "SECRET-RESPONSE-ID-0123" not in json.dumps(recorder.records[0].metadata)

    def test_unknown_values_are_reported_as_other_and_kept_in_the_notes(self):
        recorder = _one_call(
            responseModel="https://u:SECRET-RM-0123@gw/m", api="SECRET-API-0123", thinkingLevel="SECRET-TL-0123"
        )
        metadata = recorder.records[0].metadata
        assert (metadata["response_model"], metadata["api"], metadata["thinking_level"]) == ("<other>",) * 3
        assert "SECRET" not in json.dumps(metadata, default=str)

    def test_other_response_model_than_the_configured_one_is_not_reported(self):
        recorder = _one_call(responseModel="claude-sonnet-4-6-20991231")
        assert recorder.records[0].metadata["response_model"] == "<other>"

    def test_unknown_stop_reason_is_reported_as_other(self):
        recorder = _one_call(stop="SECRET-STOP-0123")
        assert recorder.records[0].metadata["stop_reason"] == "<other>"

    def test_model_and_provider_of_the_call_are_shown_only_when_configured(self):
        recorder = _one_call(model="SECRET-MODEL-0123", provider="SECRET-PROVIDER-0123")
        record = recorder.records[0]
        assert (record.metadata["model_name"], record.metadata["model_provider"]) == ("<other>", "<other>")
        assert "SECRET" not in json.dumps(record.metadata, default=str)
        configured = _one_call(model="claude-sonnet-4-6", provider="anthropic").records[0]
        assert (configured.metadata["model_name"], configured.metadata["model_provider"]) == (
            "claude-sonnet-4-6",
            "anthropic",
        )

    def test_error_tag_of_a_failed_call_has_no_provider_message(self):
        secret = "https://u:SECRET-ERR-0123@gw/v1?key=SECRET-ERR-QUERY-0123"
        recorder = _feed(
            _recorder(),
            [
                (
                    10,
                    {
                        "type": "message_end",
                        "message": _assistant([], stop="error", errorMessage=secret, timestamp=T0_MS),
                    },
                )
            ],
        )
        assert recorder.records[0].error == "provider error (stopReason=error)"
        assert "SECRET" not in json.dumps([recorder.records[0].error, recorder.records[0].metadata], default=str)

    def test_failed_call_without_a_message_keeps_the_stop_reason_text(self):
        recorder = _feed(
            _recorder(), [(10, {"type": "message_end", "message": _assistant([], stop="aborted", timestamp=T0_MS)})]
        )
        assert recorder.records[0].error == "stopReason=aborted"

    def test_compaction_error_message_is_not_in_the_span(self):
        recorder = _feed(
            _recorder(),
            [
                (1, {"type": "compaction_start", "reason": "threshold"}),
                (5, {"type": "compaction_end", "aborted": False, "errorMessage": "SECRET-COMPACT-0123", "result": {}}),
            ],
        )
        [record] = [r for r in recorder.records if r.kind == "task"]
        assert record.error == "compaction failed"
        assert "SECRET-COMPACT" not in json.dumps([record.error, record.metadata], default=str)
