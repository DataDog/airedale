# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Shared pytest fixtures for airedale unit tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def clear_credential_cache():
    """Clear the gateway credential cache before and after each test.

    Import is deferred so only gateway tests pay the import cost.
    """
    from airedale.gateway import _credential_cache

    _credential_cache.clear()
    yield
    _credential_cache.clear()
