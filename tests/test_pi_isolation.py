# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""The boundary of a pi run: the project configuration is part of the experiment, the user configuration is not.

The live tests start the installed ``pi`` in RPC mode (no model request, no network) with the command line and the
environment that the runner builds, in a fake home directory that has user-level extensions, skills, prompt
templates and an ``mcp.json`` server, and a project (``cwd``) that has its own. They observe extension commands, skill
and prompt template names, and extension and MCP marker files. They do not observe settings, themes, packages,
``auth.json`` or the content of context files: context files are checked only through the ``--no-context-files``
flag (``test_command_line_and_environment_cut_the_user_level``). They are skipped when ``pi`` or ``node`` is not
installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from airedale.harness.pi import PiRunner, _make_private_agent_dir
from airedale.skills import pi_skill_dirs
from airedale.types import ModelSpec

MODEL = ModelSpec.parse("openai/gpt-5.5")

EXTENSION = """
import { writeFileSync } from "node:fs";
export default function (pi: any) {
  writeFileSync(%(marker)r, "loaded");
  pi.registerCommand(%(command)r, { description: %(command)r, handler: async () => {} });
}
"""


def _pi_or_none() -> str | None:
    return shutil.which("pi") if shutil.which("node") else None


def _pi() -> str:
    pi = _pi_or_none()
    assert pi is not None
    return pi


needs_pi = pytest.mark.skipif(_pi_or_none() is None, reason="needs node and a pi executable")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake home with a user extension, skills (``~/.pi/agent`` and ``~/.agents``), a prompt template, an MCP server,
    a context file and a settings file, and a project with its own extension, skill and MCP server.

    The test observes the extension, the skills, the prompt template and the MCP server only (see the module docstring).
    """
    home, cwd, markers = tmp_path / "home", tmp_path / "cwd", tmp_path / "markers"
    markers.mkdir()
    user = home / ".pi" / "agent"
    _write(
        user / "extensions" / "user-ext.ts", EXTENSION % {"marker": str(markers / "user-ext"), "command": "user-cmd"}
    )
    _write(user / "skills" / "user-skill" / "SKILL.md", "---\nname: user-skill\ndescription: user\n---\nx\n")
    _write(user / "prompts" / "user-prompt.md", "---\ndescription: user prompt\n---\nhello\n")
    # Not observed by the test (it lists commands and marker files only); kept as realistic user-level files.
    _write(user / "AGENTS.md", "USER CONTEXT FILE\n")
    _write(user / "settings.json", json.dumps({"defaultProvider": "anthropic"}))
    _write(
        user / "mcp.json",
        json.dumps({"mcpServers": {"usersrv": {"command": "touch", "args": [str(markers / "user-mcp")]}}}),
    )
    _write(home / ".agents" / "skills" / "user-agents-skill" / "SKILL.md", "---\nname: ua\ndescription: user\n---\nx\n")
    _write(
        cwd / ".pi" / "extensions" / "proj-ext.ts",
        EXTENSION % {"marker": str(markers / "proj-ext"), "command": "proj-cmd"},
    )
    _write(cwd / ".pi" / "skills" / "proj-skill" / "SKILL.md", "---\nname: proj-skill\ndescription: project\n---\nx\n")
    _write(
        cwd / ".pi" / "mcp.json",
        json.dumps({"mcpServers": {"projsrv": {"command": "touch", "args": [str(markers / "proj-mcp")]}}}),
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    return {"home": home, "cwd": cwd, "markers": markers}


def _rpc(command: list[str], env: dict[str, str], cwd: Path) -> dict[str, list[str]]:
    requests = "\n".join(json.dumps(r) for r in [{"id": "1", "type": "get_commands"}]) + "\n"
    result = subprocess.run(command, input=requests, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
    responses = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    [commands] = [r for r in responses if r.get("id") == "1"]
    assert commands["success"], result.stderr[-500:]
    listed = commands["data"]["commands"]
    return {
        "names": sorted(c["name"] for c in listed),
        "scopes": sorted({c.get("sourceInfo", {}).get("scope", "") for c in listed}),
    }


def _runner_command(cwd: Path, pi: str) -> tuple[list[str], dict[str, str], Path]:
    runner = PiRunner(cwd=cwd, pi_command=[pi])
    agent_dir = _make_private_agent_dir(cwd)
    prompt = runner._write_agent_dir(agent_dir, model=MODEL, system_prompt="S", trace_headers={})
    command = runner._command(model=MODEL, system_prompt_path=prompt, skill_dirs=pi_skill_dirs([], str(cwd)))
    command[command.index("--mode") + 1] = "rpc"
    env = runner._env(agent_dir, model=MODEL)
    env["OPENAI_API_KEY"] = "dummy-key-for-the-test"
    return command, env, agent_dir


@needs_pi
def test_user_configuration_is_not_loaded_and_project_configuration_is(world):
    command, env, agent_dir = _runner_command(world["cwd"], _pi())
    try:
        seen = _rpc(command, env, world["cwd"])
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)
    names = seen["names"]
    # The project extension and the project skill are part of the experiment.
    assert "proj-cmd" in names and "skill:proj-skill" in names
    # Nothing of the user level that the test observes: extension commands, skills (``~/.pi/agent`` and ``~/.agents``), prompt templates.
    assert not {"user-cmd", "user-prompt", "skill:user-skill", "skill:ua"} & set(names), names
    markers = sorted(p.name for p in world["markers"].iterdir())
    assert "proj-ext" in markers and "proj-mcp" in markers
    assert "user-ext" not in markers and "user-mcp" not in markers
    assert "user" not in seen["scopes"]


@needs_pi
def test_the_check_can_see_a_leak(world):
    """Control: pi started as a user would start it (no isolation) does load the user level."""
    env = {**__import__("os").environ, "OPENAI_API_KEY": "dummy", "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1"}
    seen = _rpc([_pi(), "--mode", "rpc", "--no-session", "--approve"], env, world["cwd"])
    assert {"user-cmd", "user-prompt", "skill:user-skill", "skill:ua", "proj-cmd"} <= set(seen["names"])
    assert (world["markers"] / "user-ext").exists()


def test_command_line_and_environment_cut_the_user_level(tmp_path):
    """No pi needed: the flags and the agent directory that cut the user-level sources."""
    runner = PiRunner(cwd=tmp_path, pi_command=["pi"])
    agent_dir = _make_private_agent_dir(tmp_path)
    try:
        prompt = runner._write_agent_dir(agent_dir, model=MODEL, system_prompt="S", trace_headers={})
        command = runner._command(model=MODEL, system_prompt_path=prompt, skill_dirs=[])
        env = runner._env(agent_dir, model=MODEL)
        # User skills (``~/.pi/agent/skills``, ``~/.agents/skills``, ancestors), prompts, themes and context files.
        for flag in ("--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-session"):
            assert flag in command
        # Everything else of the user level lives in the agent directory, which is private to the run.
        assert env["PI_CODING_AGENT_DIR"] == str(agent_dir)
        assert agent_dir != Path.home() / ".pi" / "agent"
        allowed = {"airedale-system-prompt.md", "auth.json", "mcp.json", "models.json", "settings.json"}
        assert {p.name for p in agent_dir.iterdir()} <= allowed
        assert json.loads((agent_dir / "settings.json").read_text()) == {}
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)
