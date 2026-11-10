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
  or the request time from the collector extension. It ends at
  ``timestamp + durationMs`` (pi 1.5+ measures ``durationMs`` with a monotonic
  clock), but never after the ``message_end`` record arrives; without
  ``durationMs``, it ends when that record arrives. A request that started but
  did not end (the run stopped) is an error span that ends when the run stops.
- ``tool`` span: it starts when ``tool_execution_start`` arrives. Its duration
  is the ``durationMs`` that pi measured, or the time until
  ``tool_execution_end`` arrives. When that duration ends after the
  ``tool_execution_end`` record arrives (the records were late), the span ends
  when the record arrives and starts earlier, so it keeps the pi duration, but
  not before the end of the previous model call.
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
  signatures, tool ``details`` and image data are removed. A system message
  that changes prompt sections is shown as it is (its new sections only).
  Pi sends these messages in place to models that accept system messages in
  the conversation, and replays them into one system prompt for the other
  models.
- The request time is taken before the provider retries of pi's HTTP client
  and before the ``before_provider_request`` handlers of the extensions loaded
  after the collector (pi has no hook after them: ``before_provider_headers``
  runs before ``before_provider_request``). The response data (HTTP status, time to first byte) exists only for a
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
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

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
    from collections.abc import Iterator

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
#: The values that pi writes in these fields (pi-ai 1.4.0 and 1.5.0: ``StopReason``, ``KnownApi``, and the thinking
#: levels). Anything else (a provider, a gateway or an extension can write any text there) is not reported.
PI_KNOWN_STOP_REASONS = frozenset({"pending", "stop", "length", "toolUse", "error", "aborted", "deferred"})
PI_KNOWN_APIS = frozenset(
    {
        "openai-completions",
        "mistral-conversations",
        "openai-responses",
        "azure-openai-responses",
        "openai-codex-responses",
        "anthropic-messages",
        "bedrock-converse-stream",
        "google-generative-ai",
        "google-vertex",
        "pi-messages",
    }
)
PI_KNOWN_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})
#: What a value outside the allow-list is reported as.
OTHER_VALUE = "<other>"


def _configured_or_other(value: Any, configured: str) -> str:
    """Return ``configured`` for no value or the configured one, else :data:`OTHER_VALUE` (never the raw value)."""
    return configured if value is None or value == "" or value == configured else OTHER_VALUE


def known_value(value: Any, allowed: frozenset[str]) -> str:
    """Return ``value`` when it is one of ``allowed``, else :data:`OTHER_VALUE` (never the value itself)."""
    return value if isinstance(value, str) and value in allowed else OTHER_VALUE


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


def _pop_answer(segments: list[AgentOutputSegment]) -> str:
    """Remove the text segments at the end of ``segments`` and return them as the answer.

    The other text segments stay in their position, so the output keeps the
    order of the blocks (for example a text before a tool call).
    """
    parts: list[str] = []
    while segments:
        last = segments[-1]
        if not (isinstance(last, AgentThought) and last.kind == "text"):
            break
        segments.pop()
        parts.append(last.text)
    return "\n\n".join(reversed(parts))


def _tool_call_entry(block: dict[str, Any]) -> dict[str, Any]:
    arguments = block.get("arguments")
    entry: dict[str, Any] = {
        "name": str(block.get("name") or "unknown"),
        "arguments": json_safe(arguments) if isinstance(arguments, dict) else {"value": json_safe(arguments)},
        "type": "tool_use",
    }
    if block.get("id"):
        entry["tool_id"] = str(block["id"])
    return entry


def _assistant_input_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Render an earlier assistant message in the input of a later call.

    The blocks keep their order. A thinking block becomes its own assistant
    message (as in the span outputs), because pi sends it back to the model.
    Text blocks next to each other become one message; a tool call is added to
    the current message.
    """
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": "assistant", "content": content.strip()}]
    rendered: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            text = str(block.get("thinking") or "").strip()
            if text:
                rendered.append({"role": "assistant", "content": text})
                current = None
        elif block_type == "text":
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            if current is None or current.get("tool_calls"):
                current = {"role": "assistant", "content": text}
                rendered.append(current)
            else:
                current["content"] = f"{current['content']}\n\n{text}" if current["content"] else text
        elif block_type == "toolCall":
            if current is None:
                current = {"role": "assistant", "content": ""}
                rendered.append(current)
            current.setdefault("tool_calls", []).append(_tool_call_entry(block))
    return rendered or [{"role": "assistant", "content": ""}]


def llmobs_input_message(message: Any) -> list[dict[str, Any]]:
    """Render one pi message as LLMObs input messages (empty: not sent to the model)."""
    if not isinstance(message, dict):
        return []
    role = message.get("role")
    if role == "system":
        return [{"role": "system", "content": _system_text(message)}]
    if role in _USER_TEXT_ROLES:
        return [{"role": "user", "content": _content_text(message.get("content"))}]
    if role == "assistant":
        return _assistant_input_messages(message)
    if role == "toolResult":
        result: dict[str, Any] = {
            "name": str(message.get("toolName") or "unknown"),
            "result": _content_text(message.get("content")),
            "type": "tool_result",
        }
        if message.get("toolCallId"):
            result["tool_id"] = str(message["toolCallId"])
        return [{"role": "user", "content": "", "tool_results": [result]}]
    if role == "bashExecution":
        if message.get("excludeFromContext"):
            return []
        return [{"role": "user", "content": f"$ {message.get('command') or ''}\n{message.get('output') or ''}"}]
    if role in ("branchSummary", "compactionSummary"):
        return [{"role": "user", "content": str(message.get("summary") or "")}]
    return []


def llmobs_input_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [rendered for message in messages for rendered in llmobs_input_message(message)]


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
    #: The record of the parent tool when the call started. An id can be used
    #: again (for example "" for every model call without id), so the child
    #: keeps the record it started under, not the latest one with the id.
    parent_record: SpanRecord | None = None
    #: The earlier record with the id of the parent, when it had already ended at the start of the call. Another
    #: record with that id can still come and be the parent (pi reuses ids): this one is the parent only at the
    #: end, when no other record came.
    fallback_parent_record: SpanRecord | None = None
    #: End of the model call that asked for the tool: the tool cannot start
    #: before it (``None`` when not known).
    earliest_start_ns: int | None = None
    #: ``toolName`` as pi wrote it. The span name is "unknown" for an empty name, but a tool that is literally called
    #: "unknown" is another tool: the end record is compared with this value.
    raw_name: str = ""


@dataclass
class _ExtensionRequest:
    """Data of one model request, from the collector extension."""

    seq: int
    start_ns: int | None
    messages: list[Any] | None
    first_byte_ns: int | None = None
    http_status: int | None = None


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
        #: Nested calls that ended before their parent started, with the id of
        #: the parent. They are at the root until the parent starts.
        self._orphan_tools: list[tuple[str, SpanRecord, SpanRecord | None]] = []
        #: The parent span of each nested span that was attached (by ``id()`` of
        #: the record): a nested span is never attached under one of its descendants.
        self._span_parents: dict[int, SpanRecord] = {}
        self._compaction: SpanRecord | None = None
        #: The messages of the last request that the extension reported.
        self._extension_context: list[Any] | None = []
        self._pending_request: _ExtensionRequest | None = None
        self._last_request_seq = 0
        self._input_budget = MAX_INPUT_MESSAGE_REFS
        self._extension_disabled = False
        #: Transcript index of the previous assistant message. A truncated
        #: input starts there, so that it keeps that response (for example the
        #: tool calls of the tool results that follow it).
        self._previous_call_index = 0
        #: End of the last model call (a tool that it asked for starts after it).
        self._last_call_end_ns: int | None = None
        self.extension_entries = 0
        #: Raw values of the collector extension that are kept out of the spans (the request id of a model call). The
        #: raw values of the stdout records (provider messages, response fields) are collected by the event collector,
        #: also without spans. The private report of a failed run has both.
        self.private_notes: list[str] = []
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
        self._close_unfinished_call(now)
        if self._compaction is not None:
            self._compaction.end_ns = max(self._compaction.start_ns, now)
            self._compaction.error = "compaction did not finish"
            self._compaction.error_type = "PiCompactionError"
            self.records.append(self._compaction)
            self._compaction = None
        self._resolve_orphans()

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def _on_assistant(self, message: dict[str, Any], now: int) -> None:
        request = self._pending_request
        self._pending_request = None
        start_ns = self._call_start_ns(message, request, now)
        input_messages, input_source, input_truncated = self._call_input(request)
        # The message is appended to the transcript after this method.
        self._previous_call_index = len(self._transcript)
        segments = assistant_segments(message, include_text=True)
        # ``content`` can also be a string (only text).
        answer = _pop_answer(segments) if isinstance(message.get("content"), list) else message_text(message)
        stop_reason = message.get("stopReason")
        metadata: dict[str, Any] = {
            "harness": self.harness,
            "reported_model_name": self.model.label,
            "sdk": self.sdk_name,
            "turn": self._turn,
            "stop_reason": known_value(stop_reason, PI_KNOWN_STOP_REASONS),
            "input_source": input_source,
            "input_truncated": input_truncated,
            "tool_call_count": sum(1 for segment in segments if isinstance(segment, AgentToolCall)),
        }
        self._add_reported_metadata(metadata, message)
        if request is not None:
            if request.first_byte_ns is not None and request.start_ns is not None:
                metadata["time_to_first_byte_ms"] = max(0, request.first_byte_ns - request.start_ns) / _NS_PER_MS
            if request.http_status is not None:
                metadata["http_status"] = request.http_status
        error = None
        if stop_reason in PI_ERROR_STOP_REASONS:
            # A fixed text: the message of the provider can hold a credential or a URL with one.
            error = (
                f"provider error (stopReason={stop_reason})"
                if message.get("errorMessage")
                else f"stopReason={stop_reason}"
            )
        # The model that pi really called (a mismatch is a run error): shown only when it is the configured one.
        model_name = _configured_or_other(message.get("model"), self.model.name)
        provider = _configured_or_other(message.get("provider"), self.model.provider)
        end_ns = self._call_end_ns(message, start_ns, now, metadata)
        self._last_call_end_ns = end_ns
        self.records.append(
            _llm_record(
                name=f"eval.{self.model.provider}_agent",
                start_ns=start_ns,
                end_ns=end_ns,
                input_messages=input_messages,
                output_messages=_llm_output_messages(answer, segments),
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

    def _add_reported_metadata(self, metadata: dict[str, Any], message: dict[str, Any]) -> None:
        """Add what pi reports about the response to ``metadata``: only values that are known are reported.

        ``responseModel`` only when it is the configured model, ``api`` and ``thinkingLevel`` only from the values that
        pi defines, ``responseId`` never. The raw values that are not reported go to the private notes.
        """
        response_model = message.get("responseModel")
        if response_model:
            if response_model == self.model.name:
                metadata["response_model"] = self.model.name
            else:
                metadata["response_model"] = OTHER_VALUE
        for key, meta_key, allowed in (
            ("api", "api", PI_KNOWN_APIS),
            ("thinkingLevel", "thinking_level", PI_KNOWN_THINKING_LEVELS),
        ):
            value = message.get(key)
            if value:
                metadata[meta_key] = known_value(value, allowed)

    def _call_input(self, request: _ExtensionRequest | None) -> tuple[list[Any], str, bool]:
        """Return the input messages of a model call, their source, and if they are truncated."""
        if request is not None and request.messages is not None:
            return request.messages, "extension", False
        if len(self._transcript) <= self._input_budget:
            self._input_budget -= len(self._transcript)
            return list(self._transcript), "transcript", False
        # Budget used up: only the messages since the previous call.
        return self._transcript[self._previous_call_index :], "transcript", True

    def _close_unfinished_call(self, now: int) -> None:
        """Record the model request that started but did not end (the run stopped)."""
        request = self._pending_request
        self._pending_request = None
        if request is None:
            return
        start_ns = request.start_ns if request.start_ns is not None and _plausible_start(request.start_ns, now) else now
        input_messages, input_source, input_truncated = self._call_input(request)
        metadata: dict[str, Any] = {
            "harness": self.harness,
            "reported_model_name": self.model.label,
            "sdk": self.sdk_name,
            "turn": self._turn,
            "input_source": input_source,
            "input_truncated": input_truncated,
        }
        if request.http_status is not None:
            metadata["http_status"] = request.http_status
        self.records.append(
            _llm_record(
                name=f"eval.{self.model.provider}_agent",
                start_ns=start_ns,
                end_ns=now,
                input_messages=input_messages,
                output_messages=[],
                metadata=metadata,
                metrics=None,
                tags=self._tags(),
                tool_definitions=self.tool_definitions,
                prompt=self._prompt(),
                error="model call did not finish",
                provider=self.model.provider,
                model_name=self.model.name,
            )
        )

    @staticmethod
    def _call_end_ns(message: dict[str, Any], start_ns: int, now: int, metadata: dict[str, Any]) -> int:
        """Return the end of a model call.

        Pi 1.5+ sets ``durationMs``: the time from the message ``timestamp``
        until the response ended, measured with a monotonic clock. Use it when
        it is there, but never after the record arrived. Else the call ends when
        the record arrives (a slow reader makes the call look longer).
        """
        timestamp_ns, duration_ms = _message_timestamp_ns(message, now), message.get("durationMs")
        if timestamp_ns is None or not _is_number(duration_ms) or duration_ms < 0:
            return now
        metadata["duration_ms"] = duration_ms
        end_ns = timestamp_ns + int(min(duration_ms, _MAX_CALL_DURATION_NS / _NS_PER_MS) * _NS_PER_MS)
        return max(start_ns, min(end_ns, now))

    def _call_start_ns(self, message: dict[str, Any], request: _ExtensionRequest | None, now: int) -> int:
        candidates: list[int] = []
        if request is not None and request.start_ns is not None:
            candidates.append(request.start_ns)
        timestamp_ns = _message_timestamp_ns(message, now)
        if timestamp_ns is not None:
            candidates.append(timestamp_ns)
        for candidate in candidates:
            if _plausible_start(candidate, now):
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
        parent_id = _parent_tool_id(event)
        metadata: dict[str, Any] = {"source": SOURCE, "tool_id": tool_id or None, "turn": self._turn}
        if parent_id is not None:
            metadata["parent_tool_id"] = parent_id
        record = SpanRecord(
            kind="tool", name=name, start_ns=now, end_ns=now, input_data=json_safe(event.get("args")), metadata=metadata
        )
        if tool_id in self._open_tools:
            # A repeated id: close the earlier execution so that no data is lost.
            self.close_tool(tool_id, now, error="tool started again before it ended")
        parent_record, fallback = self._parent_at_start(parent_id)
        self._open_tools[tool_id] = _OpenTool(
            raw_name=str(event.get("toolName")),
            record=record,
            parent_id=parent_id,
            parent_record=parent_record,
            fallback_parent_record=fallback,
            earliest_start_ns=self._last_call_end_ns,
        )
        # Also an empty id: pi can use "" as the id of a model call, and then
        # sends ``parentToolCallId`` "" for the calls that it makes.
        self._register_tool_record(tool_id, record)

    def _on_tool_end(self, event: dict[str, Any], now: int) -> None:
        tool_id = str(event.get("toolCallId") or "")
        open_tool = self._open_tools.get(tool_id)
        if open_tool is not None and open_tool.raw_name != event.get("toolName"):
            # Pi repeats the name of the start record in the end record (the run fails on a record that does not):
            # the result is not given to the span of another tool.
            logger.debug("Ignoring the end record of tool call %r: the tool name is not the one of its start", tool_id)
            return
        if open_tool is None:
            # No start record: make a span that ends now.
            name = str(event.get("toolName") or "unknown")
            parent_id = _parent_tool_id(event)
            record = SpanRecord(
                kind="tool",
                name=name,
                start_ns=now,
                end_ns=now,
                metadata={"source": SOURCE, "tool_id": tool_id or None},
            )
            if parent_id is not None:
                record.metadata["parent_tool_id"] = parent_id
            parent_record, fallback = self._parent_at_start(parent_id)
            open_tool = _OpenTool(
                raw_name=str(event.get("toolName")),
                record=record,
                parent_id=parent_id,
                parent_record=parent_record,
                fallback_parent_record=fallback,
                earliest_start_ns=self._last_call_end_ns,
            )
            self._open_tools[tool_id] = open_tool
            # Like a start record: calls that wait for this id get it as their parent.
            self._register_tool_record(tool_id, record)
        record = open_tool.record
        record.output_data = tool_result_text(event.get("result"))
        duration_ms = event.get("durationMs")
        if _is_number(duration_ms) and duration_ms >= 0:
            record.metadata["duration_ms"] = duration_ms
            # The pi duration is measured with a monotonic clock: prefer it.
            # A record arrives at or after the time of its step, so the tool
            # cannot end after the end record arrived. When the duration goes
            # past it, the records were late: start earlier, keep the duration.
            # But do not start before the end of the model call that asked for
            # the tool.
            duration_ns = int(min(duration_ms, _MAX_CALL_DURATION_NS / _NS_PER_MS) * _NS_PER_MS)
            end_ns = min(record.start_ns + duration_ns, now)
            earliest = end_ns - duration_ns
            if open_tool.earliest_start_ns is not None:
                earliest = max(earliest, open_tool.earliest_start_ns)
            record.start_ns = min(record.start_ns, earliest)
        else:
            end_ns = now
        error = None
        if event.get("isError"):
            # The output of a failed tool is not in the error tag (it can hold a credential, a path or a prompt
            # injection): the tag has a fixed text, and the private report has the output.
            error = "tool call failed"
            if record.output_data:
                self.private_notes.append(
                    f"output of the failed tool call {tool_id[:256]!r}: {record.output_data[:8192]}"
                )
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
        parent = open_tool.parent_record
        if parent is not None and parent is not record and not self._is_ancestor(record, parent):
            self._attach(parent, record)
            parent_open = any(other.record is parent for other in self._open_tools.values())
            if not parent_open:
                # The parent already ended: a child span must not end after it.
                record.end_ns = max(record.start_ns, min(record.end_ns, parent.end_ns))
            return
        self.records.append(record)
        if parent is None and open_tool.parent_id is not None:
            # The parent did not start yet (damaged order): wait for it. (The
            # same id as its own is the id of the next call with that id.)
            self._orphan_tools.append((open_tool.parent_id, record, open_tool.fallback_parent_record))

    def _register_tool_record(self, tool_id: str, record: SpanRecord) -> None:
        """Make ``record`` the latest tool record of ``tool_id``, and the parent of the calls waiting for it.

        This is also the case for an id that pi uses again (even ""): a call
        that waits for the id gets the first record that comes after it.
        """
        self._tool_records[tool_id] = record
        # A call that is still open and started before its parent gets it now.
        for other in self._open_tools.values():
            if other.parent_record is None and other.parent_id == tool_id and other.record is not record:
                other.parent_record = record
                other.fallback_parent_record = None
        # A call that already ended is moved from the root under its parent.
        adopted = [orphan for orphan_parent_id, orphan, _ in self._orphan_tools if orphan_parent_id == tool_id]
        if not adopted:
            return
        self._orphan_tools = [entry for entry in self._orphan_tools if entry[0] != tool_id]
        moved = [orphan for orphan in adopted if orphan is not record and not self._is_ancestor(orphan, record)]
        if not moved:
            return
        self.records = [item for item in self.records if not any(item is orphan for orphan in moved)]
        for orphan in moved:
            self._attach(record, orphan)
        # The parent was running while its children ran.
        record.start_ns = min([record.start_ns, *(orphan.start_ns for orphan in moved)])

    def _parent_at_start(self, parent_id: str | None) -> tuple[SpanRecord | None, SpanRecord | None]:
        """Return ``(parent, fallback)`` for a call that names ``parent_id`` when it starts.

        The parent is the latest record with that id while it is still open. If that record has ended, a later
        record with the same id can still be the parent (pi reuses ids, and records can come out of order), so the
        call waits for it, and the ended record is only the fallback for the end of the run.
        """
        if parent_id is None:
            return None, None
        latest = self._tool_records.get(parent_id)
        if latest is None:
            return None, None
        open_tool = self._open_tools.get(parent_id)
        if open_tool is not None and open_tool.record is latest:
            return latest, None
        return None, latest

    def _resolve_orphans(self) -> None:
        """At the end of the run, put the calls that waited for a parent under the ended record they named."""
        waiting, self._orphan_tools = self._orphan_tools, []
        for _parent_id, record, fallback in waiting:
            if fallback is None or fallback is record or self._is_ancestor(record, fallback):
                continue
            self.records = [item for item in self.records if item is not record]
            self._attach(fallback, record)
            fallback.start_ns = min(fallback.start_ns, record.start_ns)

    def _attach(self, parent: SpanRecord, child: SpanRecord) -> None:
        parent.children.append(child)
        self._span_parents[id(child)] = parent

    def _is_ancestor(self, candidate: SpanRecord, record: SpanRecord) -> bool:
        """Return True when ``candidate`` is ``record`` or a span above it (attaching ``candidate`` under ``record`` would make a cycle)."""
        seen: set[int] = set()
        current: SpanRecord | None = record
        while current is not None and id(current) not in seen:
            if current is candidate:
                return True
            seen.add(id(current))
            current = self._span_parents.get(id(current))
        return False

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
            record.error = "compaction failed"
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
            # Not in the spans: a gateway can put a credential in the header that carries it. The private report has it.
            self.private_notes.append(f"requestId of model request {data['seq']}: {data['requestId'][:256]!s}")

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
        self._resolve_orphans()
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
    metrics: dict[str, Any] | None,
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
    (No recursion: the nesting of pi tools has no limit.)
    """
    order = [record]
    for item in order:  # Parents come before their children.
        order.extend(item.children)
    for item in reversed(order):
        for child in item.children:
            item.start_ns = min(item.start_ns, child.start_ns)
            item.end_ns = max(item.end_ns, child.end_ns)


def _open_span(llmobs: Any, record: SpanRecord) -> Any:
    """Start the span of ``record`` (it becomes the active span) and annotate it.

    If this fails, the span is finished and the error is raised.
    """
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
    except BaseException:
        span.finish(finish_time=record.end_ns / 1e9)
        raise
    return span


def _emit_record(llmobs: Any, root: SpanRecord) -> None:
    """Send ``root`` and its children. Children are sent while their parent is the active span.

    No recursion: the nesting of pi tools has no limit. A span that cannot be
    sent is skipped with its children; the other spans are still sent.
    """
    stack: list[tuple[SpanRecord, Any, Iterator[SpanRecord]]] = []
    span = _open_span(llmobs, root)
    stack.append((root, span, iter(sorted(root.children, key=lambda item: item.start_ns))))
    try:
        while stack:
            record, span, children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                span.finish(finish_time=record.end_ns / 1e9)
                continue
            with contextlib.suppress(Exception):
                child_span = _open_span(llmobs, child)
                stack.append((child, child_span, iter(sorted(child.children, key=lambda item: item.start_ns))))
    finally:
        while stack:  # An error that is not suppressed: finish what is open.
            record, span, _ = stack.pop()
            span.finish(finish_time=record.end_ns / 1e9)


def _is_number(value: Any) -> TypeGuard[int | float]:
    """Return ``True`` for an int or a finite float (JSON can carry ``Infinity`` / ``NaN``)."""
    if isinstance(value, bool):
        return False
    # Do not convert an int to a float: a very large int cannot be converted.
    return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))


#: Largest epoch time in ms that is accepted (year 5138); larger values are not
#: times and cannot always be converted to ns.
_MAX_EPOCH_MS = 100_000_000_000_000


def _epoch_ms_to_ns(value: Any) -> int | None:
    """Convert an epoch time in ms to ns; ``None`` when it is not a usable time."""
    if _is_number(value) and 0 < value <= _MAX_EPOCH_MS:
        return int(value * _NS_PER_MS)
    return None


def _plausible_start(start_ns: int, now: int) -> bool:
    """A start time is used only when it is at most ``_MAX_CALL_DURATION_NS`` before ``now``."""
    return now - _MAX_CALL_DURATION_NS <= start_ns <= now


def _message_timestamp_ns(message: dict[str, Any], now: int) -> int | None:
    """Return the plausible ``timestamp`` (request start) of a message in ns, else ``None``."""
    timestamp_ns = _epoch_ms_to_ns(message.get("timestamp"))
    return timestamp_ns if timestamp_ns is not None and _plausible_start(timestamp_ns, now) else None


def _parent_tool_id(event: dict[str, Any]) -> str | None:
    """Return ``parentToolCallId`` (can be ""); ``None`` when the model made the call."""
    parent = event.get("parentToolCallId")
    return None if parent is None else str(parent)


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)
