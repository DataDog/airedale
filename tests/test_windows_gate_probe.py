"""Throw-away probe (not for merge): fails on Windows only, to see what a red Windows job does to the checks."""

import os


def test_fails_on_windows_only():
    assert os.name != "nt"
