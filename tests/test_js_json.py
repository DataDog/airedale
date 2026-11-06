# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the JSON reader that follows ``JSON.parse`` (``airedale.js_json``)."""

from __future__ import annotations

import math

import pytest

from airedale import js_json


def test_a_huge_integer_is_infinity_not_an_error():
    # Python raises ValueError above 4,300 digits; JSON.parse gives Infinity.
    assert js_json.loads("[" + "9" * 5000 + "]") == [math.inf]
    assert js_json.loads("-" + "9" * 5000) == -math.inf
    assert js_json.loads("1" + "0" * 400) == math.inf  # Above the range of a double, below the limit of Python.
    assert js_json.loads("1" + "0" * 308) == 10**308  # Inside the range of a double.


def test_ordinary_numbers_are_unchanged():
    assert js_json.loads('{"a": 1, "b": -2, "c": 1.5, "d": 1e3}') == {"a": 1, "b": -2, "c": 1.5, "d": 1000.0}


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "[NaN]", "{bad", ""])
def test_text_that_javascript_rejects_raises_value_error(text):
    with pytest.raises(ValueError):  # noqa: PT011
        js_json.loads(text)
