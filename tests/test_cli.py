# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Tests for airedale.cli — argument parser and main entry point."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from airedale.cli import build_arg_parser, main


def write_experiment_toml(tmp_path: Path) -> Path:
    content = textwrap.dedent("""
    project = "test-project"
    models = ["anthropic/claude-3-haiku-20240307"]

    [scenarios.default]
    description = "Default scenario"

    [tasks.task1]
    prompt = "What is X?"
    criteria = ["Defines X correctly"]
    """)
    p = tmp_path / "experiment.toml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    def setup_method(self):
        self.parser = build_arg_parser()

    def test_config_required(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args([])

    def test_config_positional(self):
        args = self.parser.parse_args(["exp.toml"])
        assert args.config == "exp.toml"

    def test_gateway_config_optional(self):
        args = self.parser.parse_args(["e.toml"])
        assert args.gateway_config is None

    def test_gateway_config_path(self):
        args = self.parser.parse_args(["e.toml", "--gateway-config", "gw.toml"])
        assert args.gateway_config == "gw.toml"

    def test_model_filter(self):
        args = self.parser.parse_args(["e.toml", "--model", "anthropic/claude-3-haiku-20240307"])
        assert args.model == ["anthropic/claude-3-haiku-20240307"]

    def test_model_repeatable(self):
        args = self.parser.parse_args(
            [
                "e.toml",
                "--model",
                "anthropic/claude-3-haiku-20240307",
                "--model",
                "openai/gpt-4o",
            ]
        )
        assert len(args.model) == 2

    def test_scenario_filter(self):
        args = self.parser.parse_args(["e.toml", "--scenario", "default"])
        assert args.scenario == ["default"]

    def test_task_filter(self):
        args = self.parser.parse_args(["e.toml", "--task", "task1"])
        assert args.task == ["task1"]

    def test_runs_override(self):
        args = self.parser.parse_args(["e.toml", "--runs", "3"])
        assert args.runs == 3

    def test_judge_model_override(self):
        args = self.parser.parse_args(["e.toml", "--judge-model", "openai/gpt-4o"])
        assert args.judge_model == "openai/gpt-4o"

    def test_jobs_default(self):
        args = self.parser.parse_args(["e.toml"])
        assert args.jobs == 1

    def test_jobs_override(self):
        args = self.parser.parse_args(["e.toml", "--jobs", "4"])
        assert args.jobs == 4

    def test_fail_fast_flag(self):
        args = self.parser.parse_args(["e.toml", "--fail-fast"])
        assert args.fail_fast is True

    def test_fail_fast_default_false(self):
        args = self.parser.parse_args(["e.toml"])
        assert args.fail_fast is False

    def test_dry_run_flag(self):
        args = self.parser.parse_args(["e.toml", "--dry-run"])
        assert args.dry_run is True

    def test_dry_run_default_false(self):
        args = self.parser.parse_args(["e.toml"])
        assert args.dry_run is False

    def test_agentless_default_true(self):
        args = self.parser.parse_args(["e.toml"])
        assert args.agentless is True

    def test_no_agentless_flag(self):
        args = self.parser.parse_args(["e.toml", "--no-agentless"])
        assert args.agentless is False

    def test_no_progress_flag(self):
        args = self.parser.parse_args(["e.toml", "--no-progress"])
        assert args.no_progress is True


# ---------------------------------------------------------------------------
# main — dry-run and error paths
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def test_dry_run_returns_zero(self, tmp_path, capsys):
        config = write_experiment_toml(tmp_path)
        exit_code = main([str(config), "--dry-run"])
        assert exit_code == 0

    def test_dry_run_prints_matrix(self, tmp_path, capsys):
        config = write_experiment_toml(tmp_path)
        main([str(config), "--dry-run"])
        out = capsys.readouterr().out
        assert "Matrix:" in out
        assert "task1" in out

    def test_dry_run_without_gateway(self, tmp_path, capsys):
        config = write_experiment_toml(tmp_path)
        exit_code = main([str(config), "--dry-run"])
        assert exit_code == 0

    def test_bad_config_returns_exit_2(self, tmp_path):
        bad_config = tmp_path / "bad.toml"
        bad_config.write_text('project = "p"\n')  # missing models/scenarios/tasks
        exit_code = main([str(bad_config), "--dry-run"])
        assert exit_code == 2

    def test_missing_config_file_returns_exit_2(self, tmp_path):
        exit_code = main([str(tmp_path / "nonexistent.toml"), "--dry-run"])
        assert exit_code == 2

    def test_invalid_toml_returns_exit_2(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not [ valid toml !!!")
        exit_code = main([str(bad), "--dry-run"])
        assert exit_code == 2
