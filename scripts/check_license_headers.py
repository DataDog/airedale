# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026 Datadog, Inc.

"""Verify that every Python source file carries the Datadog license header.

Scans ``src/``, ``tests/`` and ``examples/`` for ``*.py`` files and checks that
each one begins (after an optional ``#!`` shebang) with the required header
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

SCAN_DIRS = ("src", "tests", "examples")

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

    for scan_dir in SCAN_DIRS:
        base = repo_root / scan_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if not has_valid_header(path):
                offenders.append(path.relative_to(repo_root))

    if offenders:
        print("Missing or malformed license header in the following files:", file=sys.stderr)
        for path in offenders:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nEach Python source file must begin with:\n\n"
            f"{LINE1}\n{LINE2}\n"
            "# This product includes software developed at Datadog "
            "(https://www.datadoghq.com/) Copyright <year> Datadog, Inc.",
            file=sys.stderr,
        )
        return 1

    print("All Python source files carry a valid license header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
