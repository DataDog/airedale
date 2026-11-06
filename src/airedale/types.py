# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Shared value objects for LLMObs experiment evaluation runs."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal


def coerce_int(value: Any) -> int:
    """Return a non-negative int from any value, 0 on failure."""
    try:
        result = int(value) if value is not None else 0
        return max(0, result)
    except (ValueError, TypeError, OverflowError):
        return 0


ModelProvider = Literal["anthropic", "openai"]


def slugify(value: str) -> str:
    """Return a stable slug suitable for LLMObs names and scenario identifiers."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-._")
    return slug[:120] or "scenario"


@dataclass(frozen=True)
class ModelSpec:
    """Provider-qualified model name used by the eval runner.

    The experiment ``config["model_name"]`` keeps ``label`` exactly as supplied
    (for example ``anthropic/claude-sonnet-4-6``), while manual LLMObs spans use
    ``provider`` and provider-native ``name`` so backend-side estimated cost can
    be computed from the supported provider/model fields.
    """

    provider: ModelProvider
    name: str
    label: str

    @classmethod
    def parse(cls, value: str) -> ModelSpec:
        """Parse ``provider/model`` strings such as ``openai/gpt-5.5``."""
        raw = value.strip()
        if "/" not in raw:
            raise ValueError(f"Model must be provider-qualified as '<provider>/<model>': {value!r}")
        provider, name = raw.split("/", 1)
        provider = provider.strip().lower()
        name = name.strip()
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"Unsupported model provider {provider!r}; expected 'anthropic' or 'openai'")
        if not name:
            raise ValueError(f"Model name is empty in {value!r}")
        return cls(provider=provider, name=name, label=f"{provider}/{name}")  # type: ignore[arg-type]


@dataclass
class UsageMetrics:
    """Provider-agnostic token usage collected from model responses."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: UsageMetrics) -> None:
        """Accumulate another usage object into this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cache_write_input_tokens += other.cache_write_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.estimated_cost_usd += other.estimated_cost_usd

    def to_llmobs_metrics(self) -> dict[str, int | float]:
        """Return metrics with LLMObs token names."""
        metrics: dict[str, int | float] = {}
        if self.input_tokens:
            metrics["input_tokens"] = self.input_tokens
        if self.output_tokens:
            metrics["output_tokens"] = self.output_tokens
        total = self.total_tokens or self.input_tokens + self.output_tokens
        if total:
            metrics["total_tokens"] = total
        if self.cache_write_input_tokens:
            metrics["cache_write_input_tokens"] = self.cache_write_input_tokens
        if self.cache_read_input_tokens:
            metrics["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.reasoning_output_tokens:
            metrics["reasoning_output_tokens"] = self.reasoning_output_tokens
        if self.estimated_cost_usd:
            metrics["estimated_cost_usd"] = self.estimated_cost_usd
        return metrics

    @classmethod
    def from_anthropic(cls, usage: Any) -> UsageMetrics:
        """Build usage metrics from an Anthropic SDK usage object."""
        input_tokens = coerce_int(getattr(usage, "input_tokens", 0))
        output_tokens = coerce_int(getattr(usage, "output_tokens", 0))
        cache_write = coerce_int(getattr(usage, "cache_creation_input_tokens", 0))
        cache_read = coerce_int(getattr(usage, "cache_read_input_tokens", 0))
        effective_input = input_tokens + cache_write + cache_read
        return cls(
            input_tokens=effective_input,
            output_tokens=output_tokens,
            total_tokens=effective_input + output_tokens,
            cache_write_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )

    @classmethod
    def from_openai(cls, usage: Any) -> UsageMetrics:
        """Build usage metrics from an OpenAI SDK usage object."""
        input_tokens = coerce_int(getattr(usage, "prompt_tokens", 0))
        output_tokens = coerce_int(getattr(usage, "completion_tokens", 0))
        total_tokens = coerce_int(getattr(usage, "total_tokens", 0)) or input_tokens + output_tokens
        return cls(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)

    @classmethod
    def from_claude_sdk(cls, usage: Any, *, total_cost_usd: float | None = None) -> UsageMetrics:
        """Build usage metrics from a Claude Agent SDK usage dictionary."""
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = coerce_int(usage_dict.get("input_tokens"))
        output_tokens = coerce_int(usage_dict.get("output_tokens"))
        cache_write = coerce_int(usage_dict.get("cache_creation_input_tokens"))
        cache_read = coerce_int(usage_dict.get("cache_read_input_tokens"))
        effective_input = input_tokens + cache_write + cache_read
        return cls(
            input_tokens=effective_input,
            output_tokens=output_tokens,
            total_tokens=effective_input + output_tokens,
            cache_write_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            estimated_cost_usd=float(total_cost_usd or 0.0),
        )

    @classmethod
    def from_codex(cls, usage: Any) -> UsageMetrics:
        """Build usage metrics from an OpenAI Codex SDK ``ThreadTokenUsage`` object."""
        total = getattr(usage, "total", None)
        input_tokens = coerce_int(getattr(total, "input_tokens", 0))
        output_tokens = coerce_int(getattr(total, "output_tokens", 0))
        total_tokens = coerce_int(getattr(total, "total_tokens", 0)) or input_tokens + output_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_input_tokens=coerce_int(getattr(total, "cached_input_tokens", 0)),
            reasoning_output_tokens=coerce_int(getattr(total, "reasoning_output_tokens", 0)),
        )

    @classmethod
    def from_pi(cls, usage: Any) -> UsageMetrics:
        """Build usage metrics from one pi assistant-message ``usage`` dictionary.

        Pi reports ``input`` without the cached tokens (for every provider API),
        so the effective input is ``input + cacheRead + cacheWrite``, as for the
        Claude Agent SDK. ``reasoning`` is a subset of ``output``. ``cost.total``
        is the cost pi computes from its model catalog. ``total_tokens`` is the sum
        of the parts, whatever ``totalTokens`` says.
        """
        usage_dict = usage if isinstance(usage, dict) else {}
        input_tokens = coerce_int(usage_dict.get("input"))
        output_tokens = coerce_int(usage_dict.get("output"))
        cache_write = coerce_int(usage_dict.get("cacheWrite"))
        cache_read = coerce_int(usage_dict.get("cacheRead"))
        effective_input = input_tokens + cache_write + cache_read
        # The total is the sum of the parts. The ``totalTokens`` that pi writes is not always that sum: Anthropic
        # (the sum), OpenAI Responses (the total of the provider), Bedrock (without the cache) and Google (the
        # total of the provider; its thinking tokens are already in ``output``) differ. A damaged value must not change the total,
        # and a different value is not an error, so the field is not used (``is_valid_pi_usage`` checks its form).
        total_tokens = effective_input + output_tokens
        cost = usage_dict.get("cost")
        try:
            cost_total = float(cost.get("total") or 0.0) if isinstance(cost, dict) else 0.0
        except (TypeError, ValueError, OverflowError):
            cost_total = 0.0
        if not math.isfinite(cost_total):
            cost_total = 0.0  # "Infinity" or "NaN": not a cost, and not valid JSON in the metrics.
        return cls(
            input_tokens=effective_input,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_write_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
            reasoning_output_tokens=coerce_int(usage_dict.get("reasoning")),
            estimated_cost_usd=max(0.0, cost_total),
        )


#: The token counts of a pi ``usage`` record.
PI_USAGE_COUNT_KEYS = ("input", "output", "cacheRead", "cacheWrite", "totalTokens", "reasoning")


def is_valid_pi_usage(usage: Any) -> bool:
    """Return True when ``usage`` is a complete pi usage record with usable values.

    Pi writes ``input``, ``output`` and ``cost.total`` in every record (pi 1.4.0
    and 1.5.0 start each usage with a cost of zero and add the catalog price, so a
    model with an unknown price has ``total`` 0: zeros are data). A record is
    damaged when it is not an object, lacks ``input`` or ``output``, has a count
    that is not a finite non-negative whole number (a missing optional count is
    fine, a null, text, boolean, fraction or nested value is not), or has no
    ``cost`` object with a finite non-negative ``total``: without it the cost of
    the run would be recorded as zero.
    :meth:`UsageMetrics.from_pi` turns such values into zero, which hides the
    damage, so callers check this first.
    """
    if not isinstance(usage, dict) or "input" not in usage or "output" not in usage:
        return False
    if not all(key not in usage or _is_count(usage[key]) for key in PI_USAGE_COUNT_KEYS):
        return False
    cost = usage.get("cost")
    return isinstance(cost, dict) and _is_cost(cost.get("total"))


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    # (``math.isfinite`` converts to a float: an integer above 1.8e308 raises OverflowError, and it is finite.)
    return value >= 0 if isinstance(value, int) else math.isfinite(value) and value >= 0


def _is_cost(value: Any) -> bool:
    """A cost total: a finite number >= 0 that a float can hold (``float()`` of a bigger integer raises)."""
    return _is_finite_number(value) and value <= sys.float_info.max


def _is_count(value: Any) -> bool:
    return _is_finite_number(value) and (isinstance(value, int) or value.is_integer())


@dataclass
class HarnessResult:
    """Structured task output returned from a harness run."""

    answer: str
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    harness: str = ""

    def to_output_data(self) -> dict[str, Any]:
        """Return JSON-serializable output for LLMObs Experiments."""
        return {
            "answer": self.answer,
            "usage": self.usage.to_llmobs_metrics(),
            "tool_calls": self.tool_calls,
            "harness": self.harness,
        }
