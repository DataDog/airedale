# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Which MCP servers pi runs when the repo has a ``.pi/mcp.json``.

Airedale gives its scenario MCP servers to pi in the ``mcp.json`` of a private
agent directory (the "global" servers of pi). Pi also reads the repo file
``<cwd>/.pi/mcp.json`` and merges it over them. The merge and the validation of
each entry are in :mod:`airedale.pi_mcp_config` (a translation of the code of
pi 1.4.0 and 1.5.0). This module uses them to know:

* which scenario servers a repo file would replace or change (the run must fail:
  the scenario must win, as for the other harnesses, and pi cannot do that);
* which servers the repo file adds (for the span metadata and the agent
  manifest).

Where pi 1.4.0 and 1.5.0 differ, the more permissive rule is used (an entry is
valid when one of the two versions accepts it): a wrong "valid" only stops a run
that could have run, and a wrong "invalid" would let a repo file change a
scenario server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from airedale.pi_mcp_config import (
    OVERRIDE_KEYS,
    PiMcpEntry,
    apply_project_config,
    is_http_config,
    mcp_namespace,
    validate_server_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from airedale.mcp import McpServerSpec

__all__ = [
    "OVERRIDE_KEYS",
    "PiMcpEntry",
    "PiMcpMerge",
    "discover_pi_project_mcp_servers",
    "mcp_namespace",
    "merge_project_mcp_config",
    "validate_server_config",
]


@dataclass
class PiMcpMerge:
    """Result of the merge of a repo ``.pi/mcp.json`` over the scenario servers."""

    servers: dict[str, PiMcpEntry] = field(default_factory=dict)
    #: Pi's diagnostics for the repo file (entries that pi does not load).
    errors: list[str] = field(default_factory=list)
    #: Scenario servers that the repo file replaces or changes.
    changed: list[str] = field(default_factory=list)

    @property
    def project_servers(self) -> dict[str, dict[str, Any]]:
        """Servers defined by the repo file (not the scenario servers), by name."""
        return {name: entry.config for name, entry in self.servers.items() if entry.scope == "project"}


def merge_project_mcp_config(project_file: Path, scenario: Mapping[str, Mapping[str, Any]]) -> PiMcpMerge:
    """Merge the repo file ``project_file`` over the scenario servers, as pi does.

    ``scenario`` has the ``mcp.json`` entry of each scenario server. A missing
    file gives the scenario servers unchanged.
    """
    merge = PiMcpMerge(servers={name: PiMcpEntry(dict(config), "global") for name, config in scenario.items()})
    before = {name: dict(config) for name, config in scenario.items()}
    apply_project_config(project_file, merge.servers, merge.errors)
    # A scenario server is changed when its config differs (an override, a replacement) or when the repo
    # file defines it again, even with the same config: the entry is then a repo entry for pi.
    merge.changed = [
        name
        for name, config in before.items()
        if merge.servers[name].config != config or merge.servers[name].scope != "global"
    ]
    return merge


_PLAIN_COMMAND = re.compile(r"[A-Za-z0-9._+@-]+")


def _reportable_command(command: Any) -> str:
    """Return the file name of ``command`` for reports, or ``<redacted>``.

    The repo file can write a credential in a directory of the path (``/opt/<token>/bin/server``) or in a
    command line (``node --token=abc``); only a plain file name is kept.
    """
    from airedale.mcp import REDACTED

    name = str(command).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return name if _PLAIN_COMMAND.fullmatch(name) else REDACTED


def discover_pi_project_mcp_servers(
    cwd: str | Path, scenario: Mapping[str, Mapping[str, Any]] | None = None
) -> list[McpServerSpec]:
    """Return the servers that the repo file ``<cwd>/.pi/mcp.json`` adds, for reporting.

    Pi reads this file itself (the runner does not give these servers to pi).
    The specs describe the project servers in the span metadata and the agent
    manifest. Only the servers that pi loads and starts are listed: an entry
    that pi rejects (invalid, or a name that differs from a scenario server only
    in ``-``/``_``) and an entry with ``"enabled": false`` are left out. The repo
    file can hold credentials, and these specs go to telemetry, so ``env``
    values, ``args``, the directories of ``command`` and everything in ``url`` after the
    host and port are redacted (headers are redacted by :meth:`McpServerSpec.to_safe_dict`).
    The other values of the spec are the server name (limited by pi to letters, digits,
    ``_`` and ``-``), the transport and the fixed fields.
    """
    from airedale.mcp import REDACTED, McpServerSpec, url_without_secrets

    merge = merge_project_mcp_config(Path(cwd) / ".pi" / "mcp.json", scenario or {})
    specs = []
    for name, config in merge.project_servers.items():
        if config.get("enabled") is False:
            continue
        if is_http_config(config):
            specs.append(
                McpServerSpec(
                    name=name,
                    type="http",
                    url=url_without_secrets(config["url"]),
                    headers=dict.fromkeys(config.get("headers") or {}, REDACTED),
                )
            )
        else:
            specs.append(
                McpServerSpec(
                    name=name,
                    type="stdio",
                    command=_reportable_command(config["command"]),
                    args=tuple(REDACTED for _ in config.get("args") or []),
                    env=dict.fromkeys(config.get("env") or {}, REDACTED),
                )
            )
    return specs
