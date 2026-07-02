# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Distributed tracing utilities for evaluation experiments."""

from __future__ import annotations

import logging
from typing import Any

from ddtrace.llmobs import LLMObs

logger = logging.getLogger(__name__)


def current_trace_headers() -> dict[str, str]:
    """Return LLMObs distributed headers for the active eval span.

    Returns empty dict if LLMObs is disabled or header injection fails.
    This is used to propagate trace context to MCP HTTP servers so their
    spans link to the experiment span for complete token counting.
    """
    if not LLMObs.enabled:
        return {}

    request_headers: dict[str, Any] = {}
    try:
        injected_headers = LLMObs.inject_distributed_headers(request_headers)
    except Exception:
        logger.debug("Failed to inject LLMObs distributed headers", exc_info=True)
        return {}

    headers = injected_headers if injected_headers is not None else request_headers
    return {str(key): str(value) for key, value in headers.items()}
