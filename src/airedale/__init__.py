# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""airedale: config-driven evaluation harness for agentic LLM runs.

Runs an evaluation matrix of model x scenario x task through provider-native
agentic SDKs and reports findings to Datadog LLM Observability Experiments.
"""

from __future__ import annotations

__version__ = "0.1.0"
