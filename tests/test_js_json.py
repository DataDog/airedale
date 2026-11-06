# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the JSON reader that follows ``JSON.parse`` (``airedale.js_json``)."""

from __future__ import annotations

import math
import sys

import pytest

from airedale import js_json


def test_a_huge_integer_is_infinity_not_an_error():
    # Python raises ValueError above 4,300 digits; JSON.parse gives Infinity.
    assert js_json.loads("[" + "9" * 5000 + "]") == [math.inf]
    assert js_json.loads("-" + "9" * 5000) == -math.inf
    assert js_json.loads("1" + "0" * 400) == math.inf  # Above the range of a double, below the limit of Python.
    assert js_json.loads("1" + "0" * 308) == int(float(10**308))  # Inside the range of a double: rounded.


def test_ordinary_numbers_are_unchanged():
    assert js_json.loads('{"a": 1, "b": -2, "c": 1.5, "d": 1e3}') == {"a": 1, "b": -2, "c": 1.5, "d": 1000.0}


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "[NaN]", "{bad", ""])
def test_text_that_javascript_rejects_raises_value_error(text):
    with pytest.raises(ValueError):  # noqa: PT011
        js_json.loads(text)


MAX_DOUBLE = int(sys.float_info.max)
NUMBERS = [
    "0", "7", "-7", "123456789", "9007199254740991", "9007199254740992", "9007199254740993", "9007199254740995",
    "-9007199254740993", "1000000000000000100", "123456789012345678901", "99999999999999999999999",
    "1" + "0" * 308, "1" + "0" * 309, "9" * 400, "9" * 5000, "-" + "9" * 5000,
    str(MAX_DOUBLE), str(MAX_DOUBLE + 1), str(-MAX_DOUBLE), str(-MAX_DOUBLE - 1),
    str(2**1024 - 2**970 - 1), str(2**1024 - 2**970), str(2**1024 - 2**970 + 1),
    str(-(2**1024 - 2**970)), str(-(2**1024 - 2**970) - 1),
    "NaN", "Infinity", "-Infinity", "1.5", "1e3", "1E400", "[1]", "{",
]  # fmt: skip


def test_numbers_are_read_like_json_parse(pi_reference):
    """Same value as ``JSON.parse`` (checked with Node, recorded): the integers are rounded to a double."""
    answers = pi_reference({"action": "json", "texts": NUMBERS})
    for version, native_values in answers.items():
        for text, native in zip(NUMBERS, native_values, strict=True):
            try:
                ours = js_json.loads(text)
            except ValueError:
                assert native == "error", (version, text[:30])
                continue
            if native == "other":
                assert isinstance(ours, list | dict), (version, text[:30])
                continue
            assert native != "error", (version, text[:30])
            assert float(ours) == float(native.replace("Infinity", "inf")), (version, text[:30], ours, native)
            if native.lstrip("-").isdigit():
                # (JavaScript prints the shortest digits that give the double: its exact value is int(float(...)).)
                assert ours == int(float(native)), (version, text[:30], ours, native)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("9007199254740993", 9007199254740992),
        ("1000000000000000100", 1000000000000000128),
        ("-9007199254740993", -9007199254740992),
    ],
)
def test_integers_are_the_nearest_double(text, value):
    assert js_json.loads(text) == value
    assert isinstance(js_json.loads(text), int)


def test_the_largest_double_plus_one_is_not_infinity():
    largest = int(sys.float_info.max)
    assert js_json.loads(str(largest + 1)) == largest
    assert js_json.loads(str(-largest - 1)) == -largest
    assert js_json.loads(str(2**1024 - 2**970)) == math.inf  # Rounds above the largest double.
    assert js_json.loads(str(-(2**1024 - 2**970))) == -math.inf
