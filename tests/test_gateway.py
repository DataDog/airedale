# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.gateway — credential resolution and caching."""

from __future__ import annotations

import subprocess
import sys
import threading
from unittest.mock import MagicMock

import pytest

from airedale.config.gateway import GatewayConfig, ProviderGatewayConfig
from airedale.gateway import ResolvedGatewayConfig, _run_credentials_helper, resolve_provider_config

_NOT_UTF8 = "Credentials helper output is not valid UTF-8 (UnicodeDecodeError). Make the helper write UTF-8."


def _make_gateway(provider: str, **kwargs) -> GatewayConfig:
    return GatewayConfig(providers={provider: ProviderGatewayConfig(provider=provider, **kwargs)})


# ---------------------------------------------------------------------------
# resolve_provider_config — None gateway
# ---------------------------------------------------------------------------


class TestResolveNoGateway:
    def test_none_gateway_returns_empty_override(self):
        result = resolve_provider_config("anthropic", None)
        assert isinstance(result, ResolvedGatewayConfig)
        assert result.base_url is None
        assert result.bearer_token is None
        assert result.api_key is None
        assert result.headers == {}

    def test_unknown_provider_returns_empty_override(self):
        gw = _make_gateway("anthropic", base_url="https://x.com", credentials_helper="h")
        result = resolve_provider_config("openai", gw)
        assert result.base_url is None
        assert result.bearer_token is None


# ---------------------------------------------------------------------------
# resolve_provider_config — api_key_env path
# ---------------------------------------------------------------------------


class TestResolveApiKeyEnv:
    def test_api_key_env_reads_env_var(self, monkeypatch, clear_credential_cache):
        monkeypatch.setenv("MY_OPENAI_KEY", "sk-test-1234")
        gw = _make_gateway("openai", base_url="https://gw.example.com", api_key_env="MY_OPENAI_KEY")
        result = resolve_provider_config("openai", gw)
        assert result.api_key == "sk-test-1234"
        assert result.bearer_token is None
        assert result.base_url == "https://gw.example.com"

    def test_api_key_env_unset_returns_none(self, monkeypatch, clear_credential_cache):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        gw = _make_gateway("openai", base_url="https://gw.example.com", api_key_env="MISSING_KEY")
        result = resolve_provider_config("openai", gw)
        assert result.api_key is None

    def test_headers_forwarded(self, monkeypatch, clear_credential_cache):
        monkeypatch.setenv("MY_KEY", "sk-abc")
        gw = _make_gateway(
            "anthropic",
            base_url="https://gw.example.com",
            api_key_env="MY_KEY",
            headers={"source": "evals", "org-id": "42"},
        )
        result = resolve_provider_config("anthropic", gw)
        assert result.headers == {"source": "evals", "org-id": "42"}


# ---------------------------------------------------------------------------
# resolve_provider_config — credentials_helper path
# ---------------------------------------------------------------------------


class TestResolveCredentialsHelper:
    def test_helper_output_used_as_bearer_token(self, mocker, clear_credential_cache):
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout=b"raw-token-123\n", returncode=0)
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper="my-helper get-token")
        result = resolve_provider_config("anthropic", gw)
        assert result.bearer_token == "raw-token-123"
        assert result.api_key is None

    def test_jwt_preferred_over_plain_line(self, mocker, clear_credential_cache):
        # A JWT-shaped token has three base64url segments separated by dots.
        jwt = "aaaa.bbbb.cccc"
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout=f"some info line\n{jwt}\nother output\n".encode(), returncode=0)
        token = _run_credentials_helper("any-command")
        assert token == jwt

    def test_last_line_used_when_no_jwt(self, mocker, clear_credential_cache):
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout=b"line one\nthe-actual-token\n", returncode=0)
        token = _run_credentials_helper("any-command")
        assert token == "the-actual-token"

    def test_helper_cached_second_call_skips_subprocess(self, mocker, clear_credential_cache):
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout=b"cached-token\n", returncode=0)
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper="my-helper")
        resolve_provider_config("anthropic", gw)
        resolve_provider_config("anthropic", gw)
        assert mock_run.call_count == 1

    def test_failed_helper_raises_runtime_error(self, mocker, clear_credential_cache):
        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, "helper", stderr="auth failed")
        gw = _make_gateway("anthropic", credentials_helper="bad-helper")
        with pytest.raises(RuntimeError, match="failed"):
            resolve_provider_config("anthropic", gw)

    def test_stderr_and_error_text_of_the_helper_are_not_logged_or_raised(self, mocker, clear_credential_cache, caplog):
        import logging

        secret = "Bearer sk-HELPER-SECRET-0123456789"
        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(7, "helper", stderr=f"auth failed for {secret}")
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper="bad-helper")
        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
            resolve_provider_config("anthropic", gw)
        assert secret not in caplog.text and secret not in str(raised.value)
        assert "exit 7" in caplog.text and "exit 7" in str(raised.value)
        assert "Failed to get credentials for anthropic (RuntimeError" in caplog.text
        assert all(record.exc_info is None for record in caplog.records)

    @pytest.mark.parametrize("failure", ["exit", "timeout", "oserror", "decode"])
    def test_command_of_the_helper_is_not_logged_or_raised(self, mocker, clear_credential_cache, caplog, failure):
        import logging
        import traceback

        secret = "sk-INLINE-COMMAND-SECRET-0123456789"
        command = f"helper --api-key {secret}"
        errors = {
            "exit": subprocess.CalledProcessError(7, command, stderr=f"denied for {secret}"),
            "timeout": subprocess.TimeoutExpired(command, 30),
            "oserror": OSError(f"cannot run {command}"),
        }
        mocker.patch("airedale.gateway.time.sleep")
        if failure == "decode":
            bad_output = MagicMock(stdout=b"\xff" + secret.encode(), returncode=0)
            mocker.patch("airedale.gateway.subprocess.run", return_value=bad_output)
        else:
            mocker.patch("airedale.gateway.subprocess.run", side_effect=errors[failure])
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper=command)
        for call in (
            lambda: _run_credentials_helper(command),
            lambda: resolve_provider_config("anthropic", gw),
        ):
            with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
                call()
            error = raised.value
            # The raw error of the helper must not stay anywhere: not in a printed chain, not in the attributes.
            assert error.__cause__ is None
            assert error.__context__ is None
            assert secret not in "".join(traceback.format_exception(error))
            assert secret not in str(error) and secret not in repr(error)
        assert secret not in caplog.text
        assert "Running credentials helper (attempt 1/" in caplog.text

    def test_invalid_utf8_output_of_a_real_helper_does_not_leak_through_the_context(
        self, clear_credential_cache, caplog, thread_errors
    ):
        import logging
        import traceback

        secret = "sk-REAL-HELPER-SECRET-0123456789"

        def helper_command(prefix: list[int]) -> str:
            code = f"import sys; sys.stdout.buffer.write(bytes({prefix}) + sys.argv[1].encode())"
            return f'"{sys.executable}" -c "{code}" {secret}'

        # Positive control: the same helper with ASCII output only works, and gives the secret.
        assert _run_credentials_helper(helper_command([])) == secret
        # 0x81 is not valid in UTF-8 and is not defined in cp1252 (the code page of the Windows runners).
        bad_byte = 0x81
        command = helper_command([bad_byte])
        raw = subprocess.run(command, shell=True, capture_output=True, check=True, timeout=30).stdout
        assert raw == bytes([bad_byte]) + secret.encode()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            pytest.fail("precondition: the helper output decodes with utf-8, so this test proves nothing here")
        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
            _run_credentials_helper(command)
        error = raised.value
        assert str(error) == _NOT_UTF8
        assert error.__cause__ is None and error.__context__ is None
        assert secret not in repr(error) and secret not in caplog.text
        assert secret not in "".join(traceback.format_exception(error))
        assert thread_errors == []

    def test_other_errors_of_the_helper_are_summarized_not_quoted(self, mocker, clear_credential_cache):
        secret = "sk-OSERROR-SECRET-0123456789"
        mocker.patch("airedale.gateway.subprocess.run", side_effect=OSError(f"cannot run {secret}"))
        with pytest.raises(RuntimeError) as raised:
            _run_credentials_helper("odd-helper")
        assert secret not in str(raised.value) and "OSError" in str(raised.value)
        assert raised.value.__context__ is None and raised.value.__cause__ is None

    def test_empty_output_keeps_its_fixed_message(self, mocker, clear_credential_cache):
        mocker.patch("airedale.gateway.subprocess.run", return_value=MagicMock(stdout=b"\n", returncode=0))
        with pytest.raises(RuntimeError, match="returned empty output"):
            _run_credentials_helper("empty-helper")

    def test_failed_helper_retries_up_to_max_attempts(self, mocker, clear_credential_cache):
        from airedale.gateway import _CREDENTIALS_HELPER_MAX_ATTEMPTS

        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, "helper", stderr="auth failed")
        with pytest.raises(RuntimeError, match="after .* attempts"):
            _run_credentials_helper("bad-helper")
        assert mock_run.call_count == _CREDENTIALS_HELPER_MAX_ATTEMPTS

    def test_helper_recovers_after_transient_failure(self, mocker, clear_credential_cache):
        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "helper", stderr="transient"),
            MagicMock(stdout=b"good-token\n", returncode=0),
        ]
        token = _run_credentials_helper("flaky-helper")
        assert token == "good-token"
        assert mock_run.call_count == 2

    def test_timeout_is_not_retried(self, mocker, clear_credential_cache):
        mock_sleep = mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired("helper", 30)
        with pytest.raises(RuntimeError, match="timed out"):
            _run_credentials_helper("slow-helper")
        assert mock_run.call_count == 1
        mock_sleep.assert_not_called()


@pytest.fixture
def thread_errors(monkeypatch) -> list[threading.ExceptHookArgs]:
    """Collect thread errors: on Windows, a decode error in a `subprocess` reader thread goes to `threading.excepthook`."""
    errors: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    return errors


def _python_helper(code: str, *args: str) -> str:
    assert '"' not in code and "%" not in code, "the code must be safe inside double quotes in cmd.exe"
    return " ".join([f'"{sys.executable}" -c "{code}"', *args])


class TestCredentialsHelperDecode:
    def test_subprocess_run_gets_bytes(self, mocker, clear_credential_cache):
        mock_run = mocker.patch(
            "airedale.gateway.subprocess.run", return_value=MagicMock(stdout=b"tok\n", returncode=0)
        )
        assert _run_credentials_helper("any-command") == "tok"
        kwargs = mock_run.call_args.kwargs
        # Each of these makes `subprocess` decode (on Windows, in a reader thread).
        assert not {"text", "universal_newlines", "encoding", "errors"} & kwargs.keys()
        assert kwargs["capture_output"] is True

    def test_non_ascii_utf8_output_is_decoded_as_utf8(self, mocker, clear_credential_cache):
        output = MagicMock(stdout="tökén-é\n".encode(), returncode=0)
        mocker.patch("airedale.gateway.subprocess.run", return_value=output)
        assert _run_credentials_helper("any-command") == "tökén-é"

    def test_real_helper_utf8_bytes_give_the_exact_text(self, clear_credential_cache, thread_errors):
        command = _python_helper("import sys; sys.stdout.buffer.write(bytes([0xC3, 0xA9]))")
        assert _run_credentials_helper(command) == "\u00e9"
        assert thread_errors == []

    def test_real_helper_cp1252_byte_fails_once(self, mocker, clear_credential_cache, thread_errors, caplog):
        import logging

        # 0xE9 is "é" in cp1252 but is not valid UTF-8: a fallback to the locale code page would accept it.
        command = _python_helper("import sys; sys.stdout.buffer.write(bytes([0xE9]))")
        spy = mocker.spy(subprocess, "run")
        sleep = mocker.patch("airedale.gateway.time.sleep")
        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as raised:
            _run_credentials_helper(command)
        assert str(raised.value) == _NOT_UTF8
        assert spy.call_count == 1
        sleep.assert_not_called()
        assert raised.value.__cause__ is None and raised.value.__context__ is None
        assert thread_errors == []

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            pytest.param(b"line\r\ntoken\r\n", "token", id="crlf"),
            pytest.param(b"info\rtoken\r", "token", id="lone-cr"),
            pytest.param(b"\xef\xbb\xbfaaaa.bbbb.cccc\r\nok\r\n", "aaaa.bbbb.cccc", id="bom-jwt"),
            pytest.param(b"\xef\xbb\xbfplain-token\n", "plain-token", id="bom-plain"),
        ],
    )
    def test_line_ends_and_bom(self, mocker, clear_credential_cache, output, expected):
        mocker.patch("airedale.gateway.subprocess.run", return_value=MagicMock(stdout=output, returncode=0))
        assert _run_credentials_helper("any-command") == expected

    def test_real_helper_invalid_stderr_is_not_read(self, clear_credential_cache, thread_errors, caplog, capfd):
        import logging

        marker = "STDERR-MARKER-0123456789"
        code = f"import sys; sys.stderr.buffer.write(bytes([0x81]) + b'{marker}'); sys.stdout.buffer.write(b'ok-token')"
        with caplog.at_level(logging.DEBUG):
            token = _run_credentials_helper(_python_helper(code))
        assert token == "ok-token"
        assert thread_errors == []
        assert marker not in caplog.text
        assert marker not in capfd.readouterr().err
