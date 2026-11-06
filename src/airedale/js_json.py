# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Parse JSON like ``JSON.parse`` of JavaScript does, for what pi writes and reads (its stdout records, ``mcp.json``,
``settings.json``).

Python's parser differs in three ways that matter for a guard that has to read what pi reads:

* it accepts ``NaN`` and ``Infinity``, which are not JSON (``JSON.parse`` rejects them);
* it keeps an integer exactly where ``JSON.parse`` gives the nearest double;
* it raises ``ValueError`` for an integer of more than 4,300 digits (a limit on the conversion of a string), where
  ``JSON.parse`` returns a number (``Infinity``, as the value is above the range of a double). A file with such a
  number would be taken for an unreadable file, which pi reads without a problem.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

#: Longer integers are above 1.8e308 (a double has at most 309 digits): ``Infinity`` for JavaScript, and no
#: conversion of such a long text.
_MAX_INT_DIGITS = 400


def loads(text: str) -> Any:
    """Parse ``text`` as JSON like ``JSON.parse``. Raises :class:`ValueError` for text that JavaScript rejects."""
    return json.loads(text, parse_constant=_reject_constant, parse_int=_parse_int)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Unexpected token {name} in JSON")


def _parse_int(text: str) -> int | float:
    """An integer token as ``JSON.parse`` reads it: the nearest double (9007199254740993 is 9007199254740992)."""
    if len(text.lstrip("-")) <= _MAX_INT_DIGITS:
        value = int(text)
        if abs(value) <= sys.float_info.max:
            return int(float(value))  # float() of an int rounds to the nearest double, as JavaScript does.
    return -math.inf if text.startswith("-") else math.inf
