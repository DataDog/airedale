# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Shared pytest fixtures for airedale unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def clear_credential_cache():
    """Clear the gateway credential cache before and after each test.

    Import is deferred so only gateway tests pay the import cost.
    """
    from airedale.gateway import _credential_cache

    _credential_cache.clear()
    yield
    _credential_cache.clear()


class FakeSpan:
    """A span stand-in: records its kind, name, parent, times, annotations and error."""

    def __init__(self, tracer: FakeSpanLLMObs, kind: str, name: str | None, **kwargs) -> None:
        self.tracer = tracer
        self.kind = kind
        self.name = name
        self.kwargs = kwargs
        self.parent = tracer.active[-1] if tracer.active else None
        self.start_ns = 0
        self.end_ns: int | None = None
        self.error = 0
        self.tags: dict = {}
        self.annotations: dict = {}

    def set_tag(self, key, value) -> None:
        self.tags[key] = value

    def finish(self, finish_time: float | None = None) -> None:
        assert self.tracer.active and self.tracer.active[-1] is self, "spans must finish in order"
        self.tracer.active.pop()
        self.end_ns = round(finish_time * 1e9) if finish_time is not None else self.start_ns


class FakeSpanLLMObs:
    """LLMObs stand-in for manual spans: ``llm`` / ``tool`` / ``task`` + ``annotate``."""

    enabled = True

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.active: list[FakeSpan] = []
        self.annotations: list[dict] = []

    def _start(self, kind: str, name: str | None, **kwargs) -> FakeSpan:
        span = FakeSpan(self, kind, name, **kwargs)
        self.spans.append(span)
        self.active.append(span)
        return span

    def llm(self, *, model_name=None, model_provider=None, name=None):
        return self._start("llm", name, model_name=model_name, model_provider=model_provider)

    def tool(self, *, name=None):
        return self._start("tool", name)

    def task(self, *, name=None):
        return self._start("task", name)

    def annotate(self, span=None, **kwargs) -> None:
        self.annotations.append(kwargs)
        if span is not None:
            span.annotations.update(kwargs)


@pytest.fixture
def fake_span_llmobs() -> FakeSpanLLMObs:
    return FakeSpanLLMObs()
