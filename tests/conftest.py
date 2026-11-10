# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Shared pytest fixtures for airedale unit tests."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def private_report_dir(tmp_path_factory, monkeypatch) -> Path:
    """Send the private reports of failed pi runs to a temporary directory (and give the tests its path)."""
    directory = tmp_path_factory.mktemp("airedale-output")
    monkeypatch.setenv("AIREDALE_OUTPUT_DIR", str(directory))
    return directory


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


PI_REFERENCE_SCRIPT = Path(__file__).parent / "support" / "pi_reference.mjs"
PI_REFERENCE_FIXTURES = Path(__file__).parent / "fixtures" / "pi_reference"
PI_REFERENCE_VERSIONS = ("1.4.0", "1.5.0")


def _canonical(request: dict) -> str:
    return json.dumps(request, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _live_pi_packages() -> dict[str, Path] | None:
    """Return the package directory of each installed pi release, or ``None`` without ``node`` or the releases."""
    if os.environ.get("AIREDALE_PI_REFERENCE") == "fixtures" or shutil.which("node") is None:
        return None
    releases = Path(os.environ.get("PI_RELEASES", Path.home() / ".pi" / "agent" / "install" / "releases"))
    packages = {
        version: releases / version / "node_modules" / "@earendil-works" / "pi-coding-agent"
        for version in PI_REFERENCE_VERSIONS
    }
    return packages if all(package.is_dir() for package in packages.values()) else None


@pytest.fixture(scope="session")
def pi_reference():
    """Return ``run(request)``: what pi 1.4.0 and 1.5.0 answer to ``request``, by version.

    The tests that use it compare airedale with the code of pi. The answers are
    recorded in ``tests/fixtures/pi_reference/<action>.json``, so that the tests
    run everywhere, without ``node``. When ``node`` and the releases are installed
    (``$PI_RELEASES``, default ``~/.pi/agent/install/releases``), each answer is
    also computed live and must be the recorded one (set
    ``AIREDALE_PI_REFERENCE=fixtures`` to skip that check).

    To record the answers of new requests (or after a new pi release), run the
    tests with ``AIREDALE_RECORD_PI_REFERENCE=1``: the answers of the live run are
    written to the fixture files.
    """
    packages = _live_pi_packages()
    record = os.environ.get("AIREDALE_RECORD_PI_REFERENCE") == "1"
    if record and packages is None:
        pytest.fail("AIREDALE_RECORD_PI_REFERENCE=1 needs node and the pi releases 1.4.0 and 1.5.0")
    loaded: dict[str, dict] = {}

    def fixtures_of(action: str) -> dict:
        if action not in loaded:
            path = PI_REFERENCE_FIXTURES / f"{action}.json"
            loaded[action] = json.loads(path.read_text()) if path.is_file() else {"cases": {}}
        return loaded[action]

    def live(request: dict) -> dict[str, dict]:
        assert packages is not None
        answers = {}
        for version, package in packages.items():
            process = subprocess.run(
                ["node", str(PI_REFERENCE_SCRIPT), str(package)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            answers[version] = json.loads(process.stdout)
        return answers

    def run(request: dict) -> dict[str, dict]:
        key, cases = _canonical(request), fixtures_of(request["action"])["cases"]
        if record:
            cases[key] = live(request)
            return cases[key]
        if key not in cases:
            pytest.fail(
                "no recorded answer of pi for this request; record it with "
                "AIREDALE_RECORD_PI_REFERENCE=1 (needs node and pi 1.4.0 and 1.5.0): " + key[:200]
            )
        if packages is not None:
            assert live(request) == cases[key], "the recorded answer of pi is out of date; record it again"
        return cases[key]

    yield run
    if record:
        PI_REFERENCE_FIXTURES.mkdir(parents=True, exist_ok=True)
        node_version = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True).stdout.strip()
        for action, data in loaded.items():
            document = {
                "_comment": "Answers of pi to the requests of the tests. Record again: see tests/conftest.py (pi_reference).",
                "pi_versions": list(PI_REFERENCE_VERSIONS),
                "node": node_version,
                "cases": dict(sorted(data["cases"].items())),
            }
            (PI_REFERENCE_FIXTURES / f"{action}.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
