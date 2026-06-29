"""Run the ``model × scenario × task`` evaluation matrix through LLMObs.

Each cell becomes one :class:`LLMObs.async_experiment` over a single-record
dataset view: a provider-native :class:`AgentRunner` answers the prompt, the
:class:`RubricEvaluator` scores it, and the experiment span is annotated with the
run's token usage. A single global semaphore caps how many cells run
concurrently (``jobs``); each cell runs its ``runs`` repetitions sequentially, so
``jobs`` is the total number of in-flight agent runs at any time. Per-cell
results are collected into :class:`ExperimentRunSummary` objects and rendered by
:mod:`summary`.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs

from dd_ai_devx_evals import dataset as dataset_module
from dd_ai_devx_evals.harness import create_runner, mcp_system_prompt
from dd_ai_devx_evals.mcp import McpServerSpec, managed_servers
from dd_ai_devx_evals.observability import enable_llmobs
from dd_ai_devx_evals.progress import ProgressReporter
from dd_ai_devx_evals.scoring import RubricEvaluator
from dd_ai_devx_evals.summary import ExperimentSummary, print_summary
from dd_ai_devx_evals.types import ModelSpec, slugify

if TYPE_CHECKING:
    from dd_ai_devx_evals.config.experiment import ExperimentConfig, ScenarioConfig, TaskConfig
    from dd_ai_devx_evals.config.gateway import GatewayConfig


@dataclass
class ExperimentRunSummary:
    """Local summary for one ``(model, scenario, task)`` LLMObs experiment."""

    experiment_name: str
    experiment_url: str
    model: str
    scenario: str
    task: str
    rows: list[dict[str, Any]] = field(default_factory=list)


async def run_experiments(
    config: ExperimentConfig,
    *,
    gateway: GatewayConfig | None,
    models: list[str] | None = None,
    scenarios: list[str] | None = None,
    tasks: list[str] | None = None,
    runs: int | None = None,
    judge_model: str | None = None,
    jobs: int = 1,
    dry_run: bool = False,
    show_progress: bool = True,
    fail_fast: bool = False,
    agentless: bool = True,
) -> list[ExperimentRunSummary]:
    """Run the filtered evaluation matrix and return per-cell summaries."""
    selected_models = _select_models(config, models)
    selected_scenarios = _select_scenarios(config, scenarios)
    selected_tasks = _select_tasks(config, tasks)

    matrix = [
        (model, scenario, task)
        for model in selected_models
        for scenario in selected_scenarios
        for task in selected_tasks
    ]

    if dry_run:
        _print_matrix(matrix)
        return []

    if not matrix:
        return []

    effective_runs = runs if runs is not None else config.runs
    effective_judge_model = judge_model or config.judge_model
    dataset_name = config.dataset_name or config.project

    enable_llmobs(config.project, agentless=agentless)

    dataset = dataset_module.get_or_create_dataset(config.tasks, dataset_name=dataset_name, project=config.project)

    progress = ProgressReporter(enabled=show_progress)
    progress.start(len(matrix))

    summaries: list[ExperimentRunSummary] = []
    managed_specs = _managed_server_specs(selected_scenarios)

    try:
        async with managed_servers(managed_specs):
            # A single semaphore bounds concurrent cells across the whole matrix.
            # ``jobs == 1`` therefore runs everything sequentially.
            semaphore = asyncio.Semaphore(max(1, jobs))

            async def _bounded(cell: tuple[ModelSpec, ScenarioConfig, TaskConfig]) -> ExperimentRunSummary:
                async with semaphore:
                    return await _run_cell(
                        cell,
                        config=config,
                        gateway=gateway,
                        dataset=dataset,
                        judge_model=effective_judge_model,
                        runs=effective_runs,
                        fail_fast=fail_fast,
                        progress=progress,
                    )

            summaries = list(await asyncio.gather(*(_bounded(cell) for cell in matrix)))
    finally:
        progress.stop()

    print_summary(_to_experiment_summaries(summaries))
    return summaries


async def _run_cell(
    cell: tuple[ModelSpec, ScenarioConfig, TaskConfig],
    *,
    config: ExperimentConfig,
    gateway: GatewayConfig | None,
    dataset: Any,
    judge_model: str,
    runs: int,
    fail_fast: bool,
    progress: ProgressReporter,
) -> ExperimentRunSummary:
    """Run one matrix cell as a single LLMObs experiment.

    The cell's ``runs`` repetitions execute sequentially (``jobs=1`` on the
    LLMObs experiment); cross-cell concurrency is governed by the caller's global
    semaphore so that the total number of in-flight agent runs stays within the
    configured ``jobs``.
    """
    model, scenario, task = cell
    experiment_name = _experiment_name(scenario, model, task)

    system_prompt = mcp_system_prompt()
    if scenario.system_prompt:
        system_prompt = f"{system_prompt}\n\n{scenario.system_prompt}"

    cell_mcp_servers = [McpServerSpec.from_config(server) for server in scenario.mcp_servers]

    with tempfile.TemporaryDirectory(prefix="dd-ai-devx-eval-") as cwd:
        runner = create_runner(model, scenario=scenario, gateway=gateway, cwd=cwd)

        async def experiment_task(input_data: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
            await progress.message(f"running {model.label} / {scenario.name} / {task.id}")
            result = await runner.run(
                model=model,
                system_prompt=system_prompt,
                user_prompt=str(input_data.get("prompt") or ""),
                prompt_version=scenario.name,
                harness=scenario.name,
                progress=progress.message,
            )
            _annotate_experiment_usage(result, scenario=scenario.name, model=model.label, task=task.id)
            return result.to_output_data()

        experiment_task.__name__ = "dd_ai_devx_eval_task"

        experiment = LLMObs.async_experiment(
            name=experiment_name,
            task=experiment_task,
            dataset=dataset_module.dataset_for_task(dataset, task),
            evaluators=[RubricEvaluator(judge_model=judge_model, gateway=gateway)],
            project_name=config.project,
            description=f"{scenario.name} / {model.label} / {task.id}",
            tags={
                "scenario": scenario.name,
                "model_name": model.label,
                "task_slug": task.id,
                "judge_model": judge_model,
            },
            config={
                "model_name": model.label,
                "scenario": scenario.name,
                "task": task.id,
                "judge_model": judge_model,
                "mcp_servers": [server.to_safe_dict() for server in cell_mcp_servers],
                "gateway_enabled": gateway is not None,
            },
            runs=runs,
        )
        await progress.message(f"starting {experiment_name}")
        result = await experiment.run(jobs=1, raise_errors=fail_fast)
        LLMObs.flush()
        progress.advance(f"finished {model.label} / {scenario.name} / {task.id}")

    return ExperimentRunSummary(
        experiment_name=experiment_name,
        experiment_url=str(getattr(experiment, "url", "") or ""),
        model=model.label,
        scenario=scenario.name,
        task=task.id,
        rows=list(result.get("rows", [])) if isinstance(result, dict) else [],
    )


def _annotate_experiment_usage(result: Any, *, scenario: str, model: str, task: str) -> None:
    """Annotate the active experiment span with the run's token usage."""
    if not LLMObs.enabled:
        return
    LLMObs.annotate(
        metrics=result.usage.to_llmobs_metrics(),
        tags={"scenario": scenario, "model_name": model, "task_slug": task},
    )


def _select_models(config: ExperimentConfig, models: list[str] | None) -> list[ModelSpec]:
    requested = _normalize_filter(models)
    labels = list(config.models)
    if requested:
        labels = [label for label in labels if label in requested]
    return [ModelSpec.parse(label) for label in labels]


def _select_scenarios(config: ExperimentConfig, scenarios: list[str] | None) -> list[ScenarioConfig]:
    requested = _normalize_filter(scenarios)
    selected = list(config.scenarios)
    if requested:
        selected = [scenario for scenario in selected if scenario.name in requested]
    return selected


def _select_tasks(config: ExperimentConfig, tasks: list[str] | None) -> list[TaskConfig]:
    requested = _normalize_filter(tasks)
    selected = list(config.tasks)
    if requested:
        selected = [task for task in selected if task.id in requested]
    return selected


def _normalize_filter(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    result: set[str] = set()
    for value in values:
        result.update(part.strip() for part in value.split(",") if part.strip())
    return result


def _managed_server_specs(scenarios: list[ScenarioConfig]) -> list[McpServerSpec]:
    """Return deduplicated auto-start MCP specs across the selected scenarios."""
    specs: list[McpServerSpec] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for scenario in scenarios:
        for server in scenario.mcp_servers:
            if not server.is_managed:
                continue
            spec = McpServerSpec.from_config(server)
            key = (spec.name, spec.url, spec.command)
            if key in seen:
                continue
            seen.add(key)
            specs.append(spec)
    return specs


def _to_experiment_summaries(summaries: list[ExperimentRunSummary]) -> list[ExperimentSummary]:
    """Fold per-cell run summaries into renderable :class:`ExperimentSummary`."""
    rendered: list[ExperimentSummary] = []
    for summary in summaries:
        cell = ExperimentSummary(
            experiment_name=summary.experiment_name,
            model=summary.model,
            scenario=summary.scenario,
            task=summary.task,
            experiment_url=summary.experiment_url,
        )
        for row in summary.rows:
            if isinstance(row, dict):
                cell.add_row(row)
        rendered.append(cell)
    return rendered


def _print_matrix(matrix: list[tuple[ModelSpec, ScenarioConfig, TaskConfig]]) -> None:
    """Print the resolved matrix for a dry run."""
    print(f"Matrix: {len(matrix)} cell(s)")
    for model, scenario, task in matrix:
        print(f"  {model.label} | {scenario.name} | {task.id}")


def _experiment_name(scenario: ScenarioConfig, model: ModelSpec, task: TaskConfig) -> str:
    """Return the stable, slugified ``<scenario>|<model.label>|<task.id>`` name."""
    parts = [_experiment_name_part(scenario.name), _experiment_name_part(model.label), _experiment_name_part(task.id)]
    return "|".join(parts)[:180]


def _experiment_name_part(value: str) -> str:
    return slugify(value).replace("|", "-") or "unknown"
