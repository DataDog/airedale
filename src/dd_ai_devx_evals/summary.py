"""Post-run summary table for the evaluation matrix.

One :class:`ExperimentSummary` is accumulated per ``(model, scenario, task)``
cell from the experiment rows. :func:`print_summary` renders a per-cell table of
token usage (input/output/total, cache read/write), latency, and tool-call
counts plus an aggregate row, keyed to :class:`UsageMetrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dd_ai_devx_evals.types import UsageMetrics, coerce_int


@dataclass
class ExperimentSummary:
    """Aggregated stats for one ``(model, scenario, task)`` cell."""

    experiment_name: str
    model: str
    scenario: str
    task: str
    experiment_url: str = ""
    num_runs: int = 0
    usage: UsageMetrics = field(default_factory=UsageMetrics)
    total_latency_ms: float = 0.0
    tool_calls_total: int = 0

    def add_run(self, *, usage: UsageMetrics, tool_calls: int = 0, latency_ms: float = 0.0) -> None:
        """Incorporate one experiment run (one dataset row) into the cell."""
        self.num_runs += 1
        self.usage.add(usage)
        self.tool_calls_total += max(tool_calls, 0)
        self.total_latency_ms += max(latency_ms, 0.0)

    def add_row(self, row: dict[str, Any]) -> None:
        """Incorporate one LLMObs experiment result row.

        The output payload is located defensively (``output``/``output_data``)
        because the row shape varies across LLMObs versions.
        """
        output = _row_output(row)
        usage = _usage_from_metrics(output.get("usage") if isinstance(output, dict) else None)
        tool_calls = output.get("tool_calls") if isinstance(output, dict) else None
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        self.add_run(usage=usage, tool_calls=tool_call_count, latency_ms=_row_latency_ms(row))


def _row_output(row: dict[str, Any]) -> dict[str, Any]:
    """Return the task output payload from an experiment row, if any."""
    if not isinstance(row, dict):
        return {}
    for key in ("output", "output_data"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _row_latency_ms(row: dict[str, Any]) -> float:
    """Return the latency of a row in milliseconds, best-effort."""
    if not isinstance(row, dict):
        return 0.0
    for key in ("latency_ms", "duration_ms"):
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
    duration = row.get("duration")
    if isinstance(duration, int | float):
        # ``duration`` is conventionally nanoseconds on LLMObs spans.
        return float(duration) / 1e6
    return 0.0


def _usage_from_metrics(metrics: Any) -> UsageMetrics:
    """Rebuild a :class:`UsageMetrics` from a ``to_llmobs_metrics`` dict."""
    if not isinstance(metrics, dict):
        return UsageMetrics()
    return UsageMetrics(
        input_tokens=coerce_int(metrics.get("input_tokens")),
        output_tokens=coerce_int(metrics.get("output_tokens")),
        total_tokens=coerce_int(metrics.get("total_tokens")),
        cache_write_input_tokens=coerce_int(metrics.get("cache_write_input_tokens")),
        cache_read_input_tokens=coerce_int(metrics.get("cache_read_input_tokens")),
        reasoning_output_tokens=coerce_int(metrics.get("reasoning_output_tokens")),
        estimated_cost_usd=float(metrics.get("estimated_cost_usd") or 0.0),
    )


def _fmt_tokens(n: int) -> str:
    """Format a token count with a compact thousands/millions suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_duration(ms: float) -> str:
    """Format a duration in human-readable form."""
    if ms >= 60_000:
        return f"{ms / 60_000:.1f}m"
    if ms >= 1_000:
        return f"{ms / 1_000:.1f}s"
    return f"{ms:.0f}ms"


def print_summary(summaries: list[ExperimentSummary]) -> None:
    """Print a per-cell and aggregate token/latency/tool-call table."""
    if not summaries:
        return

    columns = (
        ("Model", 24),
        ("Scenario", 16),
        ("Task", 18),
        ("Runs", 5),
        ("Input", 9),
        ("Output", 9),
        ("Total", 9),
        ("CacheR", 9),
        ("CacheW", 9),
        ("Tools", 6),
        ("Latency", 9),
    )
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    sep = "=" * len(header)
    thin = "-" * len(header)

    print(f"\n{sep}")
    print("  EXPERIMENT SUMMARY - Token Usage / Latency / Tool Calls")
    print(sep)
    print(header)
    print(thin)

    aggregate = UsageMetrics()
    total_runs = 0
    total_tool_calls = 0
    total_latency_ms = 0.0
    for summary in summaries:
        print(_format_row(columns, summary))
        aggregate.add(summary.usage)
        total_runs += summary.num_runs
        total_tool_calls += summary.tool_calls_total
        total_latency_ms += summary.total_latency_ms

    print(thin)
    total_row = ExperimentSummary(
        experiment_name="TOTAL",
        model="TOTAL",
        scenario="",
        task="",
        num_runs=total_runs,
        usage=aggregate,
        total_latency_ms=total_latency_ms,
        tool_calls_total=total_tool_calls,
    )
    print(_format_row(columns, total_row))
    print(f"{sep}\n")


def _format_row(columns: tuple[tuple[str, int], ...], summary: ExperimentSummary) -> str:
    usage = summary.usage
    total = usage.total_tokens or usage.input_tokens + usage.output_tokens
    values = (
        summary.model,
        summary.scenario,
        summary.task,
        str(summary.num_runs),
        _fmt_tokens(usage.input_tokens),
        _fmt_tokens(usage.output_tokens),
        _fmt_tokens(total),
        _fmt_tokens(usage.cache_read_input_tokens),
        _fmt_tokens(usage.cache_write_input_tokens),
        str(summary.tool_calls_total),
        _fmt_duration(summary.total_latency_ms),
    )
    return "  ".join(f"{str(value)[:width]:<{width}}" for value, (_, width) in zip(values, columns))
