"""Command-line entry point for dd-ai-devx-evals."""

from __future__ import annotations

import argparse
import asyncio
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="dd-ai-devx-evals",
        description="Run a config-driven evaluation matrix against Datadog LLM Observability Experiments.",
    )

    parser.add_argument(
        "config",
        metavar="CONFIG",
        help="experiment TOML file",
    )
    parser.add_argument(
        "--gateway-config",
        metavar="PATH",
        default=None,
        help="gateway TOML file; when omitted, provider default APIs and env-var keys are used",
    )

    parser.add_argument(
        "--model",
        metavar="M",
        action="append",
        default=None,
        help="run only these models (repeatable / comma-separated)",
    )
    parser.add_argument(
        "--scenario",
        metavar="S",
        action="append",
        default=None,
        help="run only these scenarios (repeatable / comma-separated)",
    )
    parser.add_argument(
        "--task",
        metavar="T",
        action="append",
        default=None,
        help="run only these task ids (repeatable / comma-separated)",
    )

    parser.add_argument(
        "--runs",
        metavar="N",
        type=int,
        default=None,
        help="override runs per cell",
    )
    parser.add_argument(
        "--judge-model",
        metavar="M",
        default=None,
        help="override the judge model",
    )
    parser.add_argument(
        "--jobs",
        metavar="N",
        type=int,
        default=1,
        help="total number of cells run concurrently across the whole matrix (default 1 = sequential)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="print the resolved matrix and exit without running",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="disable the live progress display",
    )
    parser.add_argument(
        "--agentless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLMObs submission mode (default: agentless)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="stop on the first task/evaluator error",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    from dd_ai_devx_evals.config import ConfigError
    from dd_ai_devx_evals.config.experiment import load_experiment
    from dd_ai_devx_evals.config.gateway import load_gateway
    from dd_ai_devx_evals.experiment import run_experiments

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load experiment config.
    try:
        config = load_experiment(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Resolve gateway config (only when a gateway file is supplied).
    gateway = None
    if args.gateway_config is not None:
        try:
            gateway = load_gateway(args.gateway_config)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        summaries = asyncio.run(
            run_experiments(
                config,
                gateway=gateway,
                models=args.model,
                scenarios=args.scenario,
                tasks=args.task,
                runs=args.runs,
                judge_model=args.judge_model,
                jobs=args.jobs,
                dry_run=args.dry_run,
                show_progress=not args.no_progress,
                fail_fast=args.fail_fast,
                agentless=args.agentless,
            )
        )
    except KeyboardInterrupt:
        # Force-quit (a second Ctrl+C) or an interrupt outside the matrix run.
        # 130 is the conventional "terminated by SIGINT" exit status.
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        for s in summaries:
            print(f"{s.experiment_name}: {s.model} / {s.scenario} / {s.task} -> {s.experiment_url}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
