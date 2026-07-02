# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Configuration parsing and validation for experiment and gateway configs."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(ValueError):
    """Raised when configuration files have invalid or missing data."""

    pass


def read_toml_file(path: str | Path) -> dict:
    """Read and parse a TOML file, raising ConfigError on failure."""
    from pathlib import Path

    path = Path(path)
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e
    except Exception as e:
        raise ConfigError(f"Failed to read {path}: {e}") from e
