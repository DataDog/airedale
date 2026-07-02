# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Enable ``python -m airedale``."""

from __future__ import annotations

from airedale.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
