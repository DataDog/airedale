"""Build and sync LLMObs datasets from experiment tasks.

Each :class:`TaskConfig` becomes one dataset record keyed by its ``id`` (the
record ``slug``). The dataset is reused across the whole matrix: it is pulled by
name and synced (add/update/delete) to match the configured tasks, then sliced
into single-record views — one per cell — for the per-task experiments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ddtrace.llmobs import LLMObs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from airedale.config.experiment import TaskConfig


def task_to_record(task: TaskConfig) -> dict[str, Any]:
    """Return the LLMObs dataset record for one task.

    ``task.context`` is appended to the prompt when present so the judge and the
    scenario runner see the same fully-formed prompt.
    """
    prompt = task.prompt
    if task.context:
        prompt = f"{prompt}\n\n{task.context}"
    return {
        "input_data": {
            "slug": task.id,
            "description": task.description or "",
            "prompt": prompt,
        },
        "expected_output": {"criteria": list(task.criteria)},
        "metadata": {
            "scenario_slug": task.id,
            "id": task.id,
            "description": task.description or "",
            "latency_threshold_ms": task.latency_threshold_ms,
        },
    }


def get_or_create_dataset(tasks: Iterable[TaskConfig], *, dataset_name: str, project: str) -> Any:
    """Pull an existing dataset by name or create it, then sync it to ``tasks``."""
    task_list = list(tasks)
    records = [task_to_record(task) for task in task_list]
    try:
        dataset = LLMObs.pull_dataset(dataset_name, project_name=project)
    except Exception:
        return LLMObs.create_dataset(
            dataset_name=dataset_name,
            project_name=project,
            description=_dataset_description(task_list),
            records=records,
        )

    sync_dataset_records(dataset, task_list)
    return dataset


def sync_dataset_records(dataset: Any, tasks: Iterable[TaskConfig]) -> None:
    """Add/update/delete records so the dataset matches the configured tasks."""
    desired_records_by_slug = {task.id: task_to_record(task) for task in tasks}
    existing_records = list(dataset)
    existing_by_slug: dict[str, tuple[int, dict[str, Any]]] = {}
    stale_indices: list[int] = []

    for index, record in enumerate(existing_records):
        slug = _dataset_record_slug(record)
        if slug in desired_records_by_slug and slug not in existing_by_slug:
            existing_by_slug[slug] = (index, record)
        else:
            stale_indices.append(index)

    changed = False
    for slug, desired_record in desired_records_by_slug.items():
        existing = existing_by_slug.get(slug)
        if existing is None:
            dataset.append(desired_record)
            changed = True
            continue

        index, existing_record = existing
        update = _dataset_record_update(existing_record, desired_record)
        if update:
            dataset.update(index, update)
            changed = True

    for index in reversed(stale_indices):
        dataset.delete(index)
        changed = True

    if changed:
        dataset.push()


def dataset_for_task(dataset: Any, task: TaskConfig) -> Any:
    """Return a single-record view of ``dataset`` containing just ``task``."""
    try:
        records = list(dataset)
    except TypeError:
        return dataset

    records_by_slug = {_dataset_record_slug(record): record for record in records}
    selected_record = records_by_slug.get(task.id)
    if selected_record is None:
        raise RuntimeError(f"Dataset is missing a record for task id: {task.id}")

    current_slugs = [_dataset_record_slug(record) for record in records]
    if current_slugs == [task.id]:
        return dataset

    return type(dataset)(
        name=dataset.name,
        project=dataset.project,
        dataset_id=dataset._id,
        records=[deepcopy(selected_record)],
        description=dataset.description,
        latest_version=dataset.latest_version,
        version=dataset.version,
        _dne_client=dataset._dne_client,
    )


def _dataset_description(tasks: list[TaskConfig]) -> str:
    return f"Evaluation tasks ({len(tasks)} records)."


def _dataset_record_slug(record: dict[str, Any]) -> str | None:
    input_data = record.get("input_data")
    if isinstance(input_data, dict) and isinstance(input_data.get("slug"), str):
        return input_data["slug"]

    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("scenario_slug"), str):
        return metadata["scenario_slug"]
    if isinstance(metadata, dict) and isinstance(metadata.get("id"), str):
        return metadata["id"]

    legacy_id = record.get("id")
    return legacy_id if isinstance(legacy_id, str) else None


def _dataset_record_update(existing_record: dict[str, Any], desired_record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: desired_record[key]
        for key in ("input_data", "expected_output", "metadata")
        if existing_record.get(key) != desired_record[key]
    }
