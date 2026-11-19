# Portable invalid-UTF-8 helper test (ROMAIN-66)

Status: draft. Scope: one test in `tests/test_gateway.py`. No production code change.

## Problem

`TestResolveCredentialsHelper::test_invalid_utf8_output_of_a_real_helper_does_not_leak_through_the_context`
fails in the `test-windows` job with `Failed: DID NOT RAISE RuntimeError`
([job log](https://github.com/DataDog/airedale/actions/runs/44319474629/job/134854312210)).

The test runs the shell command `printf '\377<secret>'` through
`_run_credentials_helper`. The function calls
`subprocess.run(command, shell=True, capture_output=True, text=True, ...)`.

## Root cause (checked against the CI log)

The first idea in the issue was: "`printf` is POSIX-only, so the command fails on
Windows". The log does **not** show this.

- The captured log has **one** line: `Running credentials helper (attempt 1/4)`.
  A failed helper would log `Credentials helper failed (exit …) … Retrying` and
  make more attempts. So the helper ran and exited with code 0.
  (The Windows runner has Git for Windows, which has a `printf.exe`.)
- No `UnicodeDecodeError` was raised. So Python could decode the output.

The cause is `text=True`. Without an `encoding`, `subprocess` decodes the output
with the **locale code page**. On Linux and macOS this is UTF-8, and the byte
`0xFF` is invalid, so the test raises. On Windows runners this is `cp1252`, and
`0xFF` is a valid character (`ÿ`). So the output decodes without error and the
"invalid UTF-8" code path never runs.

Conclusion: the test depends on the locale, not on the shell. A different shell
command for Windows would not fix it. The input bytes must be invalid in the
code page that `subprocess` uses.

Note: the log cannot prove the exact bytes that `printf.exe` wrote. The analysis
above is the only explanation that fits all the facts (exit 0, one attempt, no
decode error). The acceptance criteria below check it again.

## Approach

### Preferred: portable Python helper, with a byte that is invalid in UTF-8 and in `cp1252`

Replace the `printf` command with a command that runs the current interpreter:

```python
command = f'"{sys.executable}" -c "import sys; sys.stdout.buffer.write(bytes([0x81]) + sys.argv[1].encode())" {secret}'
```

- `sys.executable` is the same on all platforms. No dependency on `printf`.
- `sys.stdout.buffer` writes raw bytes. No newline or encoding change.
- Byte `0x81` is invalid in UTF-8 (lone continuation byte) **and** is one of the
  five bytes that `cp1252` does not define (`0x81 0x8D 0x8F 0x90 0x9D`). So it
  fails to decode on Linux, macOS and Windows.
- The command uses only double quotes and no shell meta characters, so it parses
  the same in `/bin/sh` and in `cmd.exe`. The secret is plain ASCII.
- The command is still run by `shell=True` through `_run_credentials_helper`, so
  the real code path (a real child process, a real pipe, a real decode) stays.

Add a **precondition** to the test, so it cannot pass for the wrong reason:
run the same command with `subprocess.run(..., capture_output=True)` (bytes), and
assert that the bytes fail to decode with both `"utf-8"` and
`locale.getpreferredencoding(False)`. If a future runner changes the code page,
the test fails at this line with a clear message, not with `DID NOT RAISE`.

### Where the decode fails on Windows (found in planning)

`subprocess` does not decode in the same place on all platforms. (Checked in the
CPython source of `subprocess.py`, 3.11 and 3.14: `Popen._readerthread` and the two
`_communicate` methods. The CI job uses 3.13.)

- **Linux and macOS:** `communicate()` decodes in the calling thread.
  `subprocess.run` raises `UnicodeDecodeError`, and `_run_credentials_helper`
  raises `RuntimeError("… UnicodeDecodeError at subprocess.py:…")`.
- **Windows:** `communicate()` reads each pipe in a reader thread, and the decode
  runs in that thread. The `UnicodeDecodeError` stops the thread and goes to
  `threading.excepthook`. `subprocess.run` then returns `stdout=None` with exit
  code 0, `result.stdout.strip()` raises `AttributeError`, and
  `_run_credentials_helper` raises `RuntimeError("… AttributeError at gateway.py:62")`.

So on Windows the test passes through a different path. This gives three rules for
the test:

1. Do not assert the error type in the `RuntimeError` message. It is not the same
   on all platforms.
2. Replace `threading.excepthook` with `monkeypatch` (a list `append`), and assert
   that the secret is not in the formatted traceback of a captured thread
   exception. On Windows, this is the text that the default hook writes to stderr.
   It also stops the `PytestUnhandledThreadExceptionWarning` that pytest shows for
   this test.
3. Do not assert the number of captured thread exceptions. It is a CPython detail
   (0 on Linux and macOS, 1 on Windows).

Production note (out of scope, for the follow-up issue): on Windows, a real helper
with output that is not valid in the code page writes a thread traceback to
stderr, and the error says `AttributeError`. The traceback has no secret: the text
of `UnicodeDecodeError` gives only the byte and its position.

### Fallback: `skipif` on Windows

```python
@pytest.mark.skipif(os.name == "nt", reason="printf is POSIX; cp1252 decodes 0xFF")
```

### Trade-offs

| | Portable helper (preferred) | `skipif` (fallback) |
|---|---|---|
| Windows keeps coverage | Yes | No |
| Linux and macOS coverage | Same | Same |
| Test code size | ~12 more lines (helper, precondition, thread hook) | 1 line |
| Risk | Quoting in `cmd.exe`; code page change | None, but a silent gap on Windows |
| Matches the issue preference | Yes | Only if the helper is not practical |

Use the fallback only if the portable helper fails in the `test-windows` job and
the cause is not a bug in the helper.

### Rejected

- **Use `PYTHONUTF8=1` or `encoding="utf-8"` in production code.** This changes
  the decoding of real helpers (a behavior change) to fix a test. Out of scope.
  Related note for a separate issue: on Windows, a real helper that writes
  non-ASCII bytes is decoded as `cp1252`, not UTF-8.
- **Mock `subprocess.run` to raise `UnicodeDecodeError`.** Already covered by other
  tests. The purpose of this test is a real helper.

## Constraints

1. The test still uses a real child process and `_run_credentials_helper` (no mock
   of `subprocess.run`).
2. The test still shows that invalid output does not leak: `RuntimeError` is
   raised, `__cause__` and `__context__` are `None`, and the secret is not in
   `repr(error)` or in the log records.
3. The test passes on Linux, macOS and Windows with the default locale.
4. Only `tests/test_gateway.py` changes. No production file changes.
5. Other `test-windows` failures, if any, are out of scope. Record them in a new issue.

## Acceptance criteria

Items 7 and 8 were changed, and items 9 to 11 were added, in planning. The
ROMAIN-66 plan gives the check for each item.

1. `[unit test]` The test uses a helper command that runs `sys.executable`, and not `printf`.
2. `[unit test]` The test has a precondition: the raw helper bytes fail to decode with
   `utf-8` and with `locale.getpreferredencoding(False)`.
3. `[unit test]` `uv run pytest tests/test_gateway.py -k invalid_utf8 -v` passes on the developer
   machine (macOS).
4. `[CI]` The Linux jobs pass for the changed test.
5. `[CI]` The `test-windows` job passes for the changed test (not skipped).
6. `[CI]` The `test-windows` job is green on the PR. (Note: `ci.yml` on `main` has no
   Windows job yet. It comes with PR #64. Check this PR after #64 is merged, or rebased
   on it.)
7. `[review]` The PR changes only `tests/test_gateway.py` and this design doc
   (`gh pr view 69 --json files`). No production file changes.
8. `[review]` The PR description shows a one-line check of the byte choice: `0xFF`
   decodes with `cp1252` (this is the bug) and `0x81` does not.
9. `[unit test]` The test captures `threading.excepthook` and asserts that the secret is
   not in the formatted traceback of a captured thread exception.
10. `[review]` The test does not assert the error type in the `RuntimeError` message, and
    does not assert the number of captured thread exceptions.
11. `[CI]` After PR #64 and this PR are both merged, the `test-windows` job of the `push`
    run on `main` is green (definition of done of ROMAIN-66).

## Review requirements

- Classification: **trivial**. One test file changes. No production code, public API,
  or security boundary changes.
- Reviewers: the standard Reviewer only. No Adversarial Reviewer.
- The Reviewer checks each acceptance criterion above with the check that the ROMAIN-66
  plan gives, and records pass or fail for each item.
- Focus: the test cannot pass for the wrong reason (precondition, no skip marker), and
  no secret goes out through the error, the logs, or a thread exception.
- If the review finds that a production change is necessary (for example, for the
  Windows thread traceback), stop and escalate to Romain. It is out of scope here.

## Estimate

About 40 minutes of work: 15 minutes to change the test, 5 minutes to run it
locally, 20 minutes to read the CI results. The `test-windows` result also waits
for PR #64 (not included in the 40 minutes).
