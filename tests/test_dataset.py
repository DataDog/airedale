# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.dataset — task_to_record shape and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from airedale import dataset as dataset_module
from airedale.config.experiment import TaskConfig
from airedale.dataset import get_or_create_dataset, task_to_record

if TYPE_CHECKING:
    import pytest


class TestTaskToRecord:
    def test_basic_shape(self):
        task = TaskConfig(id="ssi_overview", prompt="What is SSI?", criteria=("c1", "c2"))
        record = task_to_record(task)
        assert "input_data" in record
        assert "expected_output" in record
        assert "metadata" in record

    def test_input_data_contains_prompt_and_slug(self):
        task = TaskConfig(id="my-task", prompt="Explain X", criteria=("c",))
        record = task_to_record(task)
        assert record["input_data"]["slug"] == "my-task"
        assert record["input_data"]["prompt"] == "Explain X"

    def test_context_appended_to_prompt(self):
        task = TaskConfig(id="t", prompt="What is Y?", criteria=("c",), context="Background: Y is Z.")
        record = task_to_record(task)
        expected_prompt = "What is Y?\n\nBackground: Y is Z."
        assert record["input_data"]["prompt"] == expected_prompt

    def test_no_context_prompt_unchanged(self):
        task = TaskConfig(id="t", prompt="What is Y?", criteria=("c",))
        record = task_to_record(task)
        assert record["input_data"]["prompt"] == "What is Y?"

    def test_expected_output_contains_criteria(self):
        task = TaskConfig(id="t", prompt="Q", criteria=("c1", "c2", "c3"))
        record = task_to_record(task)
        assert record["expected_output"]["criteria"] == ["c1", "c2", "c3"]

    def test_metadata_slug_and_id(self):
        task = TaskConfig(id="ssi_overview", prompt="Q", criteria=("c",))
        record = task_to_record(task)
        meta = record["metadata"]
        assert meta["scenario_slug"] == "ssi_overview"
        assert meta["id"] == "ssi_overview"

    def test_metadata_description(self):
        task = TaskConfig(id="t", prompt="Q", criteria=("c",), description="My description")
        record = task_to_record(task)
        assert record["input_data"]["description"] == "My description"
        assert record["metadata"]["description"] == "My description"

    def test_metadata_latency_threshold(self):
        task = TaskConfig(id="t", prompt="Q", criteria=("c",), latency_threshold_ms=5000)
        record = task_to_record(task)
        assert record["metadata"]["latency_threshold_ms"] == 5000

    def test_no_description_defaults_to_empty_string(self):
        task = TaskConfig(id="t", prompt="Q", criteria=("c",))
        record = task_to_record(task)
        assert record["input_data"]["description"] == ""
        assert record["metadata"]["description"] == ""


class TestGetOrCreateDataset:
    def test_new_dataset_gets_every_task_record_field(self, monkeypatch: pytest.MonkeyPatch):
        # `get_or_create_dataset` copies each record into the SDK record type.
        # This test fails if that copy drops a field that `task_to_record` sets.
        tasks = [
            TaskConfig(id="t1", prompt="Q1", criteria=("c1",), description="d1", latency_threshold_ms=10),
            TaskConfig(id="t2", prompt="Q2", criteria=("c2", "c3"), context="ctx"),
        ]
        created: dict[str, Any] = {}

        def pull_dataset(*_args: Any, **_kwargs: Any) -> Any:
            raise LookupError("dataset not found")

        def create_dataset(**kwargs: Any) -> str:
            created.update(kwargs)
            return "created"

        monkeypatch.setattr(dataset_module.LLMObs, "pull_dataset", pull_dataset)
        monkeypatch.setattr(dataset_module.LLMObs, "create_dataset", create_dataset)

        result = get_or_create_dataset(tasks, dataset_name="ds", project="proj")

        assert result == "created"
        assert created["records"] == [task_to_record(task) for task in tasks]
