# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.types — ModelSpec, UsageMetrics, slugify, coerce_int."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from airedale.types import (
    HarnessResult,
    ModelSpec,
    UsageMetrics,
    coerce_int,
    slugify,
)

# ---------------------------------------------------------------------------
# coerce_int
# ---------------------------------------------------------------------------


class TestCoerceInt:
    def test_positive_int(self):
        assert coerce_int(42) == 42

    def test_zero(self):
        assert coerce_int(0) == 0

    def test_negative_clamped_to_zero(self):
        assert coerce_int(-5) == 0

    def test_string_int(self):
        assert coerce_int("10") == 10

    def test_none_returns_zero(self):
        assert coerce_int(None) == 0

    def test_invalid_string_returns_zero(self):
        assert coerce_int("abc") == 0

    def test_float_truncated(self):
        assert coerce_int(3.9) == 3

    def test_overflow_returns_zero(self):
        assert coerce_int(float("inf")) == 0


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_lowercase(self):
        assert slugify("Hello World") == "hello-world"

    def test_slash_becomes_dash(self):
        assert slugify("anthropic/claude-sonnet-4-6") == "anthropic-claude-sonnet-4-6"

    def test_pipe_becomes_dash(self):
        assert slugify("a|b") == "a-b"

    def test_multiple_specials_collapsed(self):
        assert slugify("a  b") == "a-b"

    def test_leading_trailing_stripped(self):
        assert slugify("  --hello--  ") == "hello"

    def test_dots_and_underscores_preserved(self):
        result = slugify("gpt-4.1_mini")
        assert result == "gpt-4.1_mini"

    def test_truncates_at_120(self):
        result = slugify("a" * 200)
        assert len(result) == 120

    def test_empty_string_returns_scenario(self):
        assert slugify("") == "scenario"

    def test_all_special_chars_returns_scenario(self):
        assert slugify("---") == "scenario"

    def test_empty_after_strip_returns_scenario(self):
        assert slugify("   ") == "scenario"


# ---------------------------------------------------------------------------
# ModelSpec.parse
# ---------------------------------------------------------------------------


class TestModelSpecParse:
    def test_anthropic_valid(self):
        spec = ModelSpec.parse("anthropic/claude-sonnet-4-6")
        assert spec.provider == "anthropic"
        assert spec.name == "claude-sonnet-4-6"
        assert spec.label == "anthropic/claude-sonnet-4-6"

    def test_openai_valid(self):
        spec = ModelSpec.parse("openai/gpt-5.5")
        assert spec.provider == "openai"
        assert spec.name == "gpt-5.5"
        assert spec.label == "openai/gpt-5.5"

    def test_rejects_bare_model_no_slash(self):
        with pytest.raises(ValueError, match="provider-qualified"):
            ModelSpec.parse("gpt-4")

    def test_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="Unsupported model provider"):
            ModelSpec.parse("gemini/pro")

    def test_rejects_empty_name_after_slash(self):
        with pytest.raises(ValueError, match="empty"):
            ModelSpec.parse("openai/")

    def test_whitespace_stripped(self):
        spec = ModelSpec.parse("  openai/gpt-4o  ")
        assert spec.provider == "openai"
        assert spec.name == "gpt-4o"
        assert spec.label == "openai/gpt-4o"

    def test_frozen_dataclass(self):
        spec = ModelSpec.parse("anthropic/claude-3-haiku-20240307")
        with pytest.raises((AttributeError, TypeError)):
            spec.provider = "openai"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UsageMetrics.add
# ---------------------------------------------------------------------------


class TestUsageMetricsAdd:
    def test_accumulates_all_fields(self):
        a = UsageMetrics(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cache_write_input_tokens=2,
            cache_read_input_tokens=3,
            reasoning_output_tokens=1,
            estimated_cost_usd=0.01,
        )
        b = UsageMetrics(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            cache_write_input_tokens=1,
            cache_read_input_tokens=1,
            reasoning_output_tokens=0,
            estimated_cost_usd=0.02,
        )
        a.add(b)
        assert a.input_tokens == 13
        assert a.output_tokens == 7
        assert a.total_tokens == 20
        assert a.cache_write_input_tokens == 3
        assert a.cache_read_input_tokens == 4
        assert a.reasoning_output_tokens == 1
        assert abs(a.estimated_cost_usd - 0.03) < 1e-9

    def test_add_zero_usage(self):
        a = UsageMetrics(input_tokens=5, output_tokens=5, total_tokens=10)
        a.add(UsageMetrics())
        assert a.input_tokens == 5
        assert a.total_tokens == 10


# ---------------------------------------------------------------------------
# UsageMetrics.to_llmobs_metrics
# ---------------------------------------------------------------------------


class TestToLlmobsMetrics:
    def test_all_zero_returns_empty(self):
        assert UsageMetrics().to_llmobs_metrics() == {}

    def test_basic_tokens_present(self):
        m = UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15)
        result = m.to_llmobs_metrics()
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["total_tokens"] == 15

    def test_total_computed_from_input_output_when_missing(self):
        m = UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=0)
        result = m.to_llmobs_metrics()
        assert result["total_tokens"] == 15

    def test_cache_tokens_included(self):
        m = UsageMetrics(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cache_write_input_tokens=20,
            cache_read_input_tokens=30,
        )
        result = m.to_llmobs_metrics()
        assert result["cache_write_input_tokens"] == 20
        assert result["cache_read_input_tokens"] == 30

    def test_reasoning_tokens_included(self):
        m = UsageMetrics(input_tokens=100, output_tokens=50, total_tokens=150, reasoning_output_tokens=10)
        result = m.to_llmobs_metrics()
        assert result["reasoning_output_tokens"] == 10

    def test_cost_included(self):
        m = UsageMetrics(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.005)
        result = m.to_llmobs_metrics()
        assert result["estimated_cost_usd"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# UsageMetrics.from_anthropic
# ---------------------------------------------------------------------------


class TestFromAnthropic:
    def test_basic(self):
        fake = SimpleNamespace(input_tokens=100, output_tokens=50)
        m = UsageMetrics.from_anthropic(fake)
        assert m.input_tokens == 100
        assert m.output_tokens == 50
        assert m.total_tokens == 150

    def test_with_cache_tokens(self):
        fake = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )
        m = UsageMetrics.from_anthropic(fake)
        # effective_input = 10 + 20 + 30 = 60
        assert m.input_tokens == 60
        assert m.cache_write_input_tokens == 20
        assert m.cache_read_input_tokens == 30
        assert m.total_tokens == 65

    def test_missing_attrs_default_zero(self):
        fake = SimpleNamespace()
        m = UsageMetrics.from_anthropic(fake)
        assert m.input_tokens == 0
        assert m.output_tokens == 0


# ---------------------------------------------------------------------------
# UsageMetrics.from_openai
# ---------------------------------------------------------------------------


class TestFromOpenAI:
    def test_basic(self):
        fake = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        m = UsageMetrics.from_openai(fake)
        assert m.input_tokens == 100
        assert m.output_tokens == 50
        assert m.total_tokens == 150

    def test_total_computed_when_zero(self):
        fake = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=0)
        m = UsageMetrics.from_openai(fake)
        assert m.total_tokens == 150


# ---------------------------------------------------------------------------
# UsageMetrics.from_claude_sdk
# ---------------------------------------------------------------------------


class TestFromClaudeSDK:
    def test_dict_input(self):
        usage = {"input_tokens": 10, "output_tokens": 5}
        m = UsageMetrics.from_claude_sdk(usage)
        assert m.input_tokens == 10
        assert m.output_tokens == 5
        assert m.total_tokens == 15

    def test_with_cost(self):
        usage = {"input_tokens": 10, "output_tokens": 5}
        m = UsageMetrics.from_claude_sdk(usage, total_cost_usd=0.05)
        assert m.estimated_cost_usd == pytest.approx(0.05)

    def test_with_cache_tokens(self):
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
        }
        m = UsageMetrics.from_claude_sdk(usage)
        assert m.cache_write_input_tokens == 20
        assert m.cache_read_input_tokens == 30
        # effective_input = 10 + 20 + 30 = 60
        assert m.input_tokens == 60

    def test_non_dict_input_returns_zeros(self):
        m = UsageMetrics.from_claude_sdk(None)
        assert m.input_tokens == 0
        assert m.total_tokens == 0

    def test_non_dict_object_returns_zeros(self):
        m = UsageMetrics.from_claude_sdk(SimpleNamespace(input_tokens=5))
        assert m.input_tokens == 0


# ---------------------------------------------------------------------------
# UsageMetrics.from_codex
# ---------------------------------------------------------------------------


class TestFromCodex:
    def test_basic(self):
        total = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_input_tokens=0,
            reasoning_output_tokens=0,
        )
        m = UsageMetrics.from_codex(SimpleNamespace(total=total))
        assert m.input_tokens == 100
        assert m.output_tokens == 50
        assert m.total_tokens == 150

    def test_with_cache_and_reasoning(self):
        total = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_input_tokens=40,
            reasoning_output_tokens=20,
        )
        m = UsageMetrics.from_codex(SimpleNamespace(total=total))
        assert m.cache_read_input_tokens == 40
        assert m.reasoning_output_tokens == 20

    def test_total_computed_when_zero(self):
        total = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=0,
            cached_input_tokens=0,
            reasoning_output_tokens=0,
        )
        m = UsageMetrics.from_codex(SimpleNamespace(total=total))
        assert m.total_tokens == 150


# ---------------------------------------------------------------------------
# HarnessResult
# ---------------------------------------------------------------------------


class TestHarnessResult:
    def test_to_output_data_shape(self):
        usage = UsageMetrics(input_tokens=10, output_tokens=5, total_tokens=15)
        result = HarnessResult(answer="The answer", usage=usage, tool_calls=[{"tool": "search"}], harness="fat-mcp")
        out = result.to_output_data()
        assert out["answer"] == "The answer"
        assert out["harness"] == "fat-mcp"
        assert out["tool_calls"] == [{"tool": "search"}]
        assert "input_tokens" in out["usage"]

    def test_defaults(self):
        result = HarnessResult(answer="hi")
        out = result.to_output_data()
        assert out["answer"] == "hi"
        assert out["tool_calls"] == []
        assert out["harness"] == ""
        assert out["usage"] == {}
