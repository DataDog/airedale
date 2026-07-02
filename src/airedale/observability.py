# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""LLMObs initialization for evaluation experiments."""

from __future__ import annotations

import os

from ddtrace.llmobs import LLMObs


def enable_llmobs(project: str, *, agentless: bool = True, integrations_enabled: bool = True) -> None:
    """Enable LLMObs for experiments with the specified project name.

    Evals default to agentless LLMObs submission so experiment row spans and
    child LLM spans reach the same LLMObs intake as the experiment metadata,
    without depending on a local Datadog Agent's LLMObs proxy configuration.
    Built-in integrations are enabled so supported SDKs (notably Claude Agent
    SDK and MCP) own their native spans. Custom spans are kept only for
    components without a native integration, such as the Codex app-server
    subprocess boundary.

    Args:
        project: The project name used for ml_app and project_name
        agentless: Whether to use agentless mode (default True)
        integrations_enabled: Whether to enable built-in integrations (default True)
    """
    if LLMObs.enabled:
        return

    LLMObs.enable(
        api_key=os.environ.get("DD_API_KEY"),
        app_key=os.environ.get("DD_APP_KEY"),
        site=os.environ.get("DD_SITE"),
        project_name=project,
        ml_app=os.environ.get("DD_LLMOBS_ML_APP", project),
        integrations_enabled=integrations_enabled,
        agentless_enabled=agentless,
        service=os.environ.get("DD_SERVICE", project),
        env=os.environ.get("DD_ENV") or os.environ.get("USER") or "local",
    )
