# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""LLMObs spans for each pi model call and tool call, with real times.

:class:`PiSpanRecorder` reads the pi JSON-mode events while pi runs. It records
one span for each model call (``llm``), each tool execution (``tool``) and
each compaction (``task``), with the time when each step started and ended.
After the run, :meth:`PiSpanRecorder.emit` sends these spans as children of the
active ``agent`` span. The spans are sent with their recorded start and end
times, so the time of the emission is not important.

Times
-----
- ``llm`` span: it starts when pi sends the request. This is the
  ``timestamp`` of the assistant message (pi sets it when the request starts),
  or the request time from the collector extension. It ends when the
  ``message_end`` record arrives.
- ``tool`` span: it starts when ``tool_execution_start`` arrives. Its duration
  is the ``durationMs`` that pi measured, or the time until
  ``tool_execution_end`` arrives.
- ``task`` span (compaction): from ``compaction_start`` to ``compaction_end``.

Inputs of an ``llm`` span
-------------------------
The collector extension (``pi_llmobs_extension.ts``, loaded with ``-e``) sends
the pi messages of each model request, after the compaction and the
``context`` / ``context_with_system`` handlers that ran before it. It sends
only the change from the previous request (``prefix`` + ``appended``), so the
stream stays small. Without the extension data, the input is the transcript of
the run until that call (all the ``message_end`` messages before it).

Limits:

- Pi loads ``-e`` extensions before repo extensions. A repo extension that
  changes the context after the collector is not shown.
- Budget: see ``MAX_INPUT_MESSAGE_REFS``. The extension does not send the
  messages of a request when they are too large (``skipped``).
- The messages are pi messages, not the provider payload: pi still converts
  some roles (for example a compaction summary) before it sends them. Opaque
  signatures, tool ``details`` and image data are removed.
- The request time is taken before the provider retries of pi's HTTP client,
  and the response data (HTTP status, time to first byte) exists only for a
  successful response.
- Compaction usage and the usage of model work inside tools are not on the
  ``llm`` spans (the compaction ``task`` span has it as metadata). The agent
  span has the complete usage of the run.

Trust
-----
The extension data comes from the pi process, which runs repo code
(``--approve``). A repo extension can write entries of the same type. The data
is only used for telemetry: it is validated, limited (see
``MAX_INPUT_MESSAGE_REFS``) and can never change the run result. Evaluate only trusted repos.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ddtrace.constants import ERROR_MSG, ERROR_TYPE

from airedale.harness.base import (
    LLMOBS_PROMPT_ID,
    AgentOutputSegment,
    AgentThought,
    AgentToolCall,
    _llm_output_messages,
    json_dumps_compact,
    json_safe,
)
from airedale.types import UsageMetrics

if TYPE_CHECKING:
    from airedale.types import ModelSpec

logger = logging.getLogger(__name__)

SOURCE = "pi"
#: The collector extension file, given to pi with ``-e``.
PI_LLMOBS_EXTENSION_PATH = Path(__file__).with_name("pi_llmobs_extension.ts")
#: ``customType`` of the entries that the collector extension appends.
EXTENSION_REQUEST_ENTRY = "airedale.llmobs.request"
EXTENSION_RESPONSE_ENTRY = "airedale.llmobs.response"
#: Version of the entry data format. Entries with another version are ignored.
EXTENSION_DATA_VERSION = 1
#: Budget for the ``llm`` span inputs of one run: the total number of message
#: references that the recorder copies and keeps (the extension data and the
#: stream come from the pi process, so they are not trusted, and the full
#: context of each call grows with the run). When the budget is used up, the
#: extension data is not used any more, and the input of a call is only the
#: messages since the previous call (``input_truncated`` in the metadata). Each
#: message is in at most one of these inputs, so after the budget the recorder
#: keeps at most one more reference for each message of the transcript.
MAX_INPUT_MESSAGE_REFS = 2_000_000
#: Assistant stop reasons that mark the model call as failed.
PI_ERROR_STOP_REASONS = frozenset({"error", "aborted"})
#: Pi message roles that are sent to the model as user text.
_USER_TEXT_ROLES = frozenset({"user", "custom"})
_NS_PER_MS = 1_000_000
#: A message timestamp is used as the start of a model call only when it is at
#: most this far before the end of the call. Else the clock is not trusted.
_MAX_CALL_DURATION_NS = 6 * 60 * 60 * 1_000_000_000


# --------------------------------------------------------------------------- #
# Pi message rendering
# --------------------------------------------------------------------------- #
def message_text(message: dict[str, Any]) -> str:
    """Return the text blocks of a pi message (``content`` string or block list)."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(part.strip() for part in parts if part.strip())


def tool_result_text(result: Any) -> str:
    """Render a pi tool result (``{"content": [...], "details": ...}``) as text."""
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        parts: list[str] = []
        for block in result["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(f"[{block.get('type') or 'content'}]")
        return "\n".join(parts)
    if isinstance(result, str):
        return result
    return json_dumps_compact(result)


def _content_text(content: Any) -> str:
    """Render a ``content`` value (string or blocks) as text; non-text blocks become ``[type]``."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        else:
            parts.append(f"[{block.get('type') or 'content'}]")
    return "\n".join(part for part in parts if part)


def _system_text(message: dict[str, Any]) -> str:
    """Render a pi system message: its content, else its prompt sections in order."""
    text = _content_text(message.get("content"))
    if text.strip():
        return text
    sections = message.get("sections")
    if isinstance(sections, dict):
        return "\n\n".join(str(value) for value in sections.values() if isinstance(value, str) and value.strip())
    return ""


def assistant_segments(message: dict[str, Any], *, include_text: bool) -> list[AgentOutputSegment]:
    """Return the reasoning, text (when ``include_text``) and tool calls of one assistant message."""
    segments: list[AgentOutputSegment] = []
    content = message.get("content")
    if not isinstance(content, list):
        return segments
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            text = str(block.get("thinking") or "").strip()
            if text:
                segments.append(AgentThought(text=text, kind="reasoning"))
        elif block_type == "text" and include_text:
            text = str(block.get("text") or "").strip()
            if text:
                segments.append(AgentThought(text=text, kind="text"))
        elif block_type == "toolCall":
            segments.append(
                AgentToolCall(
                    name=str(block.get("name") or "unknown"),
                    arguments=block.get("arguments") if block.get("arguments") is not None else {},
                    metadata={"source": SOURCE, "tool_id": str(block.get("id") or "") or None},
                )
            )
    return segments


def _tool_call_entries(message: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    content = message.get("content")
    if not isinstance(content, list):
        return entries
    for block in content:
        if isinstance(block, dict) and block.get("type") == "toolCall":
            arguments = block.get("arguments")
            entry: dict[str, Any] = {
                "name": str(block.get("name") or "unknown"),
                "arguments": json_safe(arguments) if isinstance(arguments, dict) else {"value": json_safe(arguments)},
                "type": "tool_use",
            }
            if block.get("id"):
                entry["tool_id"] = str(block["id"])
            entries.append(entry)
    return entries


def llmobs_input_message(message: Any) -> dict[str, Any] | None:
    """Render one pi message as an LLMObs input message (``None``: not sent to the model)."""
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role == "system":
        return {"role": "system", "content": _system_text(message)}
    if role in _USER_TEXT_ROLES:
        return {"role": "user", "content": _content_text(message.get("content"))}
    if role == "assistant":
        rendered: dict[str, Any] = {"role": "assistant", "content": message_text(message)}
        tool_calls = _tool_call_entries(message)
        if tool_calls:
            rendered["tool_calls"] = tool_calls
        return rendered
    if role == "toolResult":
        result: dict[str, Any] = {
            "name": str(message.get("toolName") or "unknown"),
            "result": _content_text(message.get("content")),
            "type": "tool_result",
        }
        if message.get("toolCallId"):
            result["tool_id"] = str(message["toolCallId"])
        return {"role": "user", "content": "", "tool_results": [result]}
    if role == "bashExecution":
        if message.get("excludeFromContext"):
            return None
        return {"role": "user", "content": f"$ {message.get('command') or ''}\n{message.get('output') or ''}"}
    if role in ("branchSummary", "compactionSummary"):
        return {"role": "user", "content": str(message.get("summary") or "")}
    return None


def llmobs_input_messages(messages: list[Any]) -> list[dict[str, Any]]:
    rendered = (llmobs_input_message(message) for message in messages)
    return [message for message in rendered if message is not None]


# --------------------------------------------------------------------------- #
# Span records
# --------------------------------------------------------------------------- #
@dataclass
class SpanRecord:
    """One span to send: kind, name, times (epoch ns) and LLMObs annotations."""

    kind: str
    name: str
    start_ns: int
    end_ns: int
    input_data: Any = None
    #: Pi messages of an ``llm`` span input, rendered only when the span is sent.
    #: The list shares the message objects with the transcript, so a long run
    #: does not keep one rendered copy of the context for each model call.
    pi_input_messages: list[Any] | None = None
    output_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    tool_definitions: list[dict[str, Any]] | None = None
    prompt: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    children: list[SpanRecord] = field(default_factory=list)


@dataclass
class _OpenTool:
    record: SpanRecord
    parent_id: str | None


@dataclass
class _ExtensionRequest:
    """Data of one model request, from the collector extension."""

    seq: int
    start_ns: int | None
    messages: list[Any] | None
    first_byte_ns: int | None = None
    http_status: int | None = None
    request_id: str | None = None


class PiSpanRecorder:
    """Turn pi JSON-mode events into :class:`SpanRecord` values, then send them.

    :meth:`feed` never raises: a telemetry problem must not change the run.
    """

    def __init__(
        self,
        *,
        model: ModelSpec,
        sdk_name: str,
        harness: str,
        prompt_version: str,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.sdk_name = sdk_name
        self.harness = harness
        self.prompt_version = prompt_version
        self.tool_definitions = tool_definitions
        self.records: list[SpanRecord] = []
        self._transcript: list[Any] = []
        self._turn = 0
        self._last_event_ns: int | None = None
        self._open_tools: dict[str, _OpenTool] = {}
        #: Every top-level or nested tool record by id, also after it ended, so
        #: that a nested call that ends late still gets its parent.
        self._tool_records: dict[str, SpanRecord] = {}
        self._compaction: SpanRecord | None = None
        #: The messages of the last request that the extension reported.
        self._extension_context: list[Any] | None = []
        self._pending_request: _ExtensionRequest | None = None
        self._last_request_seq = 0
        self._input_budget = MAX_INPUT_MESSAGE_REFS
        self._extension_disabled = False
        #: Transcript length at the previous assistant message.
        self._previous_call_index = 0
        self.extension_entries = 0
        self.invalid_extension_entries = 0

    # ------------------------------------------------------------------ #
    # Event intake
    # ------------------------------------------------------------------ #
    def feed(self, event: dict[str, Any], *, received_ns: int | None = None) -> None:
        now = received_ns if received_ns is not None else time.time_ns()
        try:
            self._feed(event, now)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            logger.warning("Ignoring a pi event that the span recorder cannot use", exc_info=True)
        finally:
            self._last_event_ns = now

    def _feed(self, event: dict[str, Any], now: int) -> None:
        event_type = event.get("type")
        if event_type == "turn_start":
            self._turn += 1
        elif event_type == "message_end":
            message = event.get("message")
            if isinstance(message, dict):
                if message.get("role") == "assistant":
                    self._on_assistant(message, now)
                self._transcript.append(message)
        elif event_type == "tool_execution_start":
            self._on_tool_start(event, now)
        elif event_type == "tool_execution_end":
            self._on_tool_end(event, now)
        elif event_type == "compaction_start":
            self._on_compaction_start(event, now)
        elif event_type == "compaction_end":
            self._on_compaction_end(event, now)
        elif event_type == "entry_appended":
            self._on_entry(event.get("entry"), now)

    def close(self, *, received_ns: int | None = None) -> None:
        """Close the steps that did not end (the run stopped while they ran)."""
        now = received_ns if received_ns is not None else time.time_ns()
        # Last started first: a nested call must close before the tool that
        # made it, so that it is attached to it.
        for tool_id in reversed(list(self._open_tools)):
            self.close_tool(tool_id, now, error="tool did not finish")
        if self._compaction is not None:
            self._compaction.end_ns = max(self._compaction.start_ns, now)
            self._compaction.error = "compaction did not finish"
            self._compaction.error_type = "PiCompactionError"
            self.records.append(self._compaction)
            self._compaction = None

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def _on_assistant(self, message: dict[str, Any], now: int) -> None:
        request = self._pending_request
        self._pending_request = None
        start_ns = self._call_start_ns(message, request, now)
        input_truncated = False
        if request is not None and request.messages is not None:
            input_messages = request.messages
            input_source = "extension"
        elif len(self._transcript) <= self._input_budget:
            self._input_budget -= len(self._transcript)
            input_messages = list(self._transcript)
            input_source = "transcript"
        else:
            # Budget used up: only the messages since the previous call.
            input_messages = self._transcript[self._previous_call_index :]
            input_source = "transcript"
            input_truncated = True
        self._previous_call_index = len(self._transcript) + 1
        segments = assistant_segments(message, include_text=False)
        stop_reason = message.get("stopReason")
        metadata: dict[str, Any] = {
            "harness": self.harness,
            "reported_model_name": self.model.label,
            "sdk": self.sdk_name,
            "turn": self._turn,
            "stop_reason": stop_reason,
            "input_source": input_source,
            "input_truncated": input_truncated,
            "tool_call_count": sum(1 for segment in segments if isinstance(segment, AgentToolCall)),
        }
        for key, meta_key in (
            ("responseModel", "response_model"),
            ("responseId", "response_id"),
            ("api", "api"),
            ("thinkingLevel", "thinking_level"),
        ):
            if message.get(key):
                metadata[meta_key] = str(message[key])
        if request is not None:
            if request.first_byte_ns is not None and request.start_ns is not None:
                metadata["time_to_first_byte_ms"] = max(0, request.first_byte_ns - request.start_ns) / _NS_PER_MS
            if request.http_status is not None:
                metadata["http_status"] = request.http_status
            if request.request_id:
                metadata["request_id"] = request.request_id
        error = None
        if stop_reason in PI_ERROR_STOP_REASONS:
            error = str(message.get("errorMessage") or f"stopReason={stop_reason}")
        model_name = str(message.get("model") or self.model.name)
        provider = str(message.get("provider") or self.model.provider)
        self.records.append(
            _llm_record(
                name=f"eval.{self.model.provider}_agent",
                start_ns=start_ns,
                end_ns=now,
                input_messages=input_messages,
                output_messages=_llm_output_messages(message_text(message), segments),
                metadata=metadata,
                metrics=UsageMetrics.from_pi(message.get("usage")).to_llmobs_metrics(),
                tags=self._tags(),
                tool_definitions=self.tool_definitions,
                prompt=self._prompt(),
                error=error,
                provider=provider,
                model_name=model_name,
            )
        )

    def _call_start_ns(self, message: dict[str, Any], request: _ExtensionRequest | None, now: int) -> int:
        candidates: list[int] = []
        if request is not None and request.start_ns is not None:
            candidates.append(request.start_ns)
        timestamp = message.get("timestamp")
        if isinstance(timestamp, int | float) and not isinstance(timestamp, bool):
            candidates.append(int(timestamp * _NS_PER_MS))
        for candidate in candidates:
            if now - _MAX_CALL_DURATION_NS <= candidate <= now:
                return candidate
        return self._last_event_ns if self._last_event_ns is not None and self._last_event_ns <= now else now

    def _tags(self) -> dict[str, str]:
        return {
            "harness": self.harness,
            "prompt_version": self.prompt_version,
            "model_name": self.model.label,
            "sdk": self.sdk_name,
        }

    def _prompt(self) -> dict[str, Any]:
        return {
            "id": LLMOBS_PROMPT_ID,
            "version": self.prompt_version,
            "template": "{{prompt}}",
            "variables": {
                "prompt_version": self.prompt_version,
                "harness": self.harness,
                "model_name": self.model.label,
            },
        }

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def _on_tool_start(self, event: dict[str, Any], now: int) -> None:
        tool_id = str(event.get("toolCallId") or "")
        name = str(event.get("toolName") or "unknown")
        parent_id = str(event["parentToolCallId"]) if event.get("parentToolCallId") else None
        metadata: dict[str, Any] = {"source": SOURCE, "tool_id": tool_id or None, "turn": self._turn}
        if parent_id:
            metadata["parent_tool_id"] = parent_id
        record = SpanRecord(
            kind="tool", name=name, start_ns=now, end_ns=now, input_data=json_safe(event.get("args")), metadata=metadata
        )
        if tool_id in self._open_tools:
            # A repeated id: close the earlier execution so that no data is lost.
            self.close_tool(tool_id, now, error="tool started again before it ended")
        self._open_tools[tool_id] = _OpenTool(record=record, parent_id=parent_id)
        if tool_id:
            self._tool_records[tool_id] = record

    def _on_tool_end(self, event: dict[str, Any], now: int) -> None:
        tool_id = str(event.get("toolCallId") or "")
        open_tool = self._open_tools.get(tool_id)
        if open_tool is None:
            # No start record: make a span that ends now.
            name = str(event.get("toolName") or "unknown")
            parent_id = str(event["parentToolCallId"]) if event.get("parentToolCallId") else None
            record = SpanRecord(
                kind="tool",
                name=name,
                start_ns=now,
                end_ns=now,
                metadata={"source": SOURCE, "tool_id": tool_id or None},
            )
            open_tool = _OpenTool(record=record, parent_id=parent_id)
            self._open_tools[tool_id] = open_tool
        record = open_tool.record
        record.output_data = tool_result_text(event.get("result"))
        duration_ms = event.get("durationMs")
        if isinstance(duration_ms, int | float) and not isinstance(duration_ms, bool) and duration_ms >= 0:
            record.metadata["duration_ms"] = duration_ms
            # The pi duration is measured with a monotonic clock: prefer it, but
            # never end after the end record arrived.
            end_ns = min(record.start_ns + int(duration_ms * _NS_PER_MS), now)
        else:
            end_ns = now
        error = (record.output_data or "tool error") if event.get("isError") else None
        self.close_tool(tool_id, end_ns, error=error)

    def close_tool(self, tool_id: str, end_ns: int, *, error: str | None = None) -> None:
        open_tool = self._open_tools.pop(tool_id, None)
        if open_tool is None:
            return
        record = open_tool.record
        record.end_ns = max(record.start_ns, end_ns)
        if error:
            record.error = error
            record.error_type = "PiToolError"
            record.metadata["error"] = error
        if open_tool.parent_id is not None and self._attach_child(open_tool.parent_id, record):
            parent_open = open_tool.parent_id in self._open_tools
            parent = self._tool_records[open_tool.parent_id]
            if not parent_open:
                # The parent already ended: a child span must not end after it.
                record.end_ns = max(record.start_ns, min(record.end_ns, parent.end_ns))
            return
        self.records.append(record)

    def _attach_child(self, parent_id: str, record: SpanRecord) -> bool:
        """Add ``record`` to the tool ``parent_id``; ``False`` when that tool is not known."""
        parent = self._tool_records.get(parent_id)
        if parent is None or parent is record:
            return False
        parent.children.append(record)
        return True

    # ------------------------------------------------------------------ #
    # Compaction
    # ------------------------------------------------------------------ #
    def _on_compaction_start(self, event: dict[str, Any], now: int) -> None:
        if self._compaction is not None:
            self._compaction.end_ns = max(self._compaction.start_ns, now)
            self._compaction.error = "compaction did not finish"
            self._compaction.error_type = "PiCompactionError"
            self.records.append(self._compaction)
        self._compaction = SpanRecord(
            kind="task",
            name="pi.compaction",
            start_ns=now,
            end_ns=now,
            metadata={"source": SOURCE, "reason": event.get("reason"), "turn": self._turn},
        )

    def _on_compaction_end(self, event: dict[str, Any], now: int) -> None:
        record = self._compaction
        self._compaction = None
        if record is None:
            record = SpanRecord(
                kind="task", name="pi.compaction", start_ns=now, end_ns=now, metadata={"source": SOURCE}
            )
        record.end_ns = max(record.start_ns, now)
        result = event.get("result")
        if isinstance(result, dict):
            record.output_data = str(result.get("summary") or "")
            for key, meta_key in (
                ("tokensBefore", "tokens_before"),
                ("estimatedTokensAfter", "estimated_tokens_after"),
            ):
                if result.get(key) is not None:
                    record.metadata[meta_key] = json_safe(result[key])
            if isinstance(result.get("usage"), dict):
                record.metadata["usage"] = UsageMetrics.from_pi(result["usage"]).to_llmobs_metrics()
        if event.get("aborted"):
            record.error = "compaction aborted"
        elif event.get("errorMessage"):
            record.error = str(event["errorMessage"])
        if record.error:
            record.error_type = "PiCompactionError"
        self.records.append(record)

    # ------------------------------------------------------------------ #
    # Collector extension data
    # ------------------------------------------------------------------ #
    def _on_entry(self, entry: Any, now: int) -> None:
        if not isinstance(entry, dict):
            return
        entry_type = entry.get("customType")
        if entry_type not in (EXTENSION_REQUEST_ENTRY, EXTENSION_RESPONSE_ENTRY):
            return
        self.extension_entries += 1
        data = entry.get("data")
        if not isinstance(data, dict) or data.get("v") != EXTENSION_DATA_VERSION or not _is_int(data.get("seq")):
            self._invalid_entry("bad header")
            return
        if entry_type == EXTENSION_REQUEST_ENTRY:
            self._on_request_entry(data)
        else:
            self._on_response_entry(data)

    def _on_request_entry(self, data: dict[str, Any]) -> None:
        seq = data["seq"]
        if seq <= self._last_request_seq:
            self._invalid_entry("request sequence number did not increase")
            return
        self._last_request_seq = seq
        messages = self._apply_context_delta(data)
        self._pending_request = _ExtensionRequest(seq=seq, start_ns=_epoch_ms_to_ns(data.get("t")), messages=messages)

    def _apply_context_delta(self, data: dict[str, Any]) -> list[Any] | None:
        """Rebuild the request messages from the previous request and the change.

        Return ``None`` (use the transcript instead) when the data does not fit.
        After a bad change, the later changes are relative to a message list
        that is not known: only a full list (``prefix`` 0) can be used again.
        The extension sends a full list after it skipped one (``skipped``).
        """
        if self._extension_disabled:
            return None
        if data.get("skipped") is True:
            # The extension did not send the messages (too large).
            self._extension_context = None
            return None
        prefix, appended, count = data.get("prefix"), data.get("appended"), data.get("count")
        base = self._extension_context if self._extension_context is not None else ([] if prefix == 0 else None)
        if (
            base is None
            or not _is_int(prefix)
            or not _is_int(count)
            or not isinstance(appended, list)
            or not 0 <= prefix <= len(base)
            or prefix + len(appended) != count
        ):
            self._invalid_entry("context change does not fit")
            self._extension_context = None
            return None
        if count > self._input_budget:
            self._invalid_entry("input budget used up")
            self._extension_disabled = True
            self._extension_context = None
            return None
        self._input_budget -= count
        # A new list each time: the list of an earlier request is never changed.
        messages = [*base[:prefix], *appended]
        self._extension_context = messages
        return messages

    def _on_response_entry(self, data: dict[str, Any]) -> None:
        request = self._pending_request
        if request is None or request.seq != data["seq"]:
            return
        request.first_byte_ns = _epoch_ms_to_ns(data.get("t"))
        if _is_int(data.get("status")):
            request.http_status = data["status"]
        if isinstance(data.get("requestId"), str) and data["requestId"]:
            request.request_id = data["requestId"][:256]

    def _invalid_entry(self, reason: str) -> None:
        self.invalid_extension_entries += 1
        logger.debug("Ignoring collector extension data: %s", reason)

    # ------------------------------------------------------------------ #
    # Emission
    # ------------------------------------------------------------------ #
    def emit(self, llmobs: Any) -> None:
        """Send the recorded spans as children of the active span. Never raises."""
        if not getattr(llmobs, "enabled", False):
            return
        for record in sorted(self.records, key=lambda item: item.start_ns):
            with contextlib.suppress(Exception):
                _cover_children(record)
                _emit_record(llmobs, record)


def _llm_record(
    *,
    name: str,
    start_ns: int,
    end_ns: int,
    input_messages: list[Any],
    output_messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str],
    tool_definitions: list[dict[str, Any]] | None,
    prompt: dict[str, Any],
    error: str | None,
    provider: str,
    model_name: str,
) -> SpanRecord:
    record = SpanRecord(
        kind="llm",
        name=name,
        start_ns=start_ns,
        end_ns=max(start_ns, end_ns),
        pi_input_messages=input_messages,
        output_data=output_messages,
        metadata=metadata,
        metrics=metrics,
        tags=tags,
        tool_definitions=tool_definitions,
        prompt=prompt,
        error=error,
        error_type="PiModelCallError" if error else None,
    )
    # The model that pi really called (a mismatch is a run error, but the span
    # must show what pi did).
    record.metadata["model_provider"] = provider
    record.metadata["model_name"] = model_name
    return record


def _cover_children(record: SpanRecord) -> None:
    """Make each span start before and end after its children.

    The times come from records that can arrive late, so a child can be a
    little outside its parent. A trace must show the child inside its parent.
    """
    for child in record.children:
        _cover_children(child)
        record.start_ns = min(record.start_ns, child.start_ns)
        record.end_ns = max(record.end_ns, child.end_ns)


def _emit_record(llmobs: Any, record: SpanRecord) -> None:
    if record.kind == "llm":
        span = llmobs.llm(
            model_name=record.metadata.get("model_name"),
            model_provider=record.metadata.get("model_provider"),
            name=record.name,
        )
    elif record.kind == "tool":
        span = llmobs.tool(name=record.name)
    else:
        span = llmobs.task(name=record.name)
    try:
        span.start_ns = record.start_ns
        annotations: dict[str, Any] = {
            "span": span,
            "input_data": (
                llmobs_input_messages(record.pi_input_messages)
                if record.pi_input_messages is not None
                else record.input_data
            ),
            "output_data": record.output_data,
            "metadata": record.metadata,
        }
        if record.metrics is not None:
            annotations["metrics"] = record.metrics
        if record.tags:
            annotations["tags"] = record.tags
        if record.tool_definitions:
            annotations["tool_definitions"] = record.tool_definitions
        if record.prompt is not None:
            annotations["prompt"] = record.prompt
        llmobs.annotate(**annotations)
        if record.error:
            span.error = 1
            span.set_tag(ERROR_MSG, record.error)
            span.set_tag(ERROR_TYPE, record.error_type or "PiError")
        # Children are sent while this span is active, so it is their parent.
        for child in sorted(record.children, key=lambda item: item.start_ns):
            with contextlib.suppress(Exception):
                _emit_record(llmobs, child)
    finally:
        span.finish(finish_time=record.end_ns / 1e9)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _epoch_ms_to_ns(value: Any) -> int | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return int(value * _NS_PER_MS)
    return None
