# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Verify that every source file carries the Datadog license header.

Scans ``src/``, ``tests/`` and ``examples/`` for ``*.py`` files, plus ``*.yml`` /
``*.yaml`` files across the repository (e.g. ``.github/workflows``), and checks
that each one begins (after an optional ``#!`` shebang) with the required header
block::

    # Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
    #
    # This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright <year> Datadog, Inc.

The ``<year>`` may be any 4-digit year (files introduced in different years keep
their original year). Exits non-zero and lists offenders when any file is
missing or has a malformed header.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Directories scanned for Python sources.
PYTHON_SCAN_DIRS = ("src", "tests", "examples")

# YAML files use the same ``#``-comment header; they are scanned repo-wide, minus
# these directories.
YAML_EXCLUDE_DIRS = {".git", ".jj", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

LINE1 = "# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License."
LINE2 = "#"
LINE3_RE = re.compile(
    r"^# This product includes software developed at Datadog \(https://www\.datadoghq\.com/\) "
    r"Copyright \d{4} Datadog, Inc\.$"
)
# PEP 263 source-encoding declaration, allowed on line 1 or 2 before the header.
ENCODING_RE = re.compile(r"^[ \t\f]*#.*coding[:=][ \t]*[-\w.]+")


def has_valid_header(path: Path) -> bool:
    """Return True when ``path`` opens with a valid license header block."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False

    # Allow a shebang and/or a PEP 263 encoding cookie before the header.
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    if lines and ENCODING_RE.match(lines[0]):
        lines = lines[1:]

    if len(lines) < 3:
        return False

    return lines[0] == LINE1 and lines[1] == LINE2 and bool(LINE3_RE.match(lines[2]))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[Path] = []
    seen: set[Path] = set()

    def check(path: Path) -> None:
        if path in seen:
            return
        seen.add(path)
        if not has_valid_header(path):
            offenders.append(path.relative_to(repo_root))

    for scan_dir in PYTHON_SCAN_DIRS:
        base = repo_root / scan_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            check(path)

    for pattern in ("*.yml", "*.yaml"):
        for path in sorted(repo_root.rglob(pattern)):
            if YAML_EXCLUDE_DIRS.intersection(path.relative_to(repo_root).parts):
                continue
            check(path)

    if offenders:
        print("Missing or malformed license header in the following files:", file=sys.stderr)
        for path in offenders:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nEach source file must begin with:\n\n"
            f"{LINE1}\n{LINE2}\n"
            "# This product includes software developed at Datadog "
            "(https://www.datadoghq.com/) Copyright <year> Datadog, Inc.",
            file=sys.stderr,
        )
        return 1

    print("All source files carry a valid license header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
