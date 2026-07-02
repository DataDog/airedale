# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.gateway — credential resolution and caching."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from airedale.config.gateway import GatewayConfig, ProviderGatewayConfig
from airedale.gateway import ResolvedGatewayConfig, _run_credentials_helper, resolve_provider_config


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
        mock_run.return_value = MagicMock(stdout="raw-token-123\n", returncode=0)
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper="my-helper get-token")
        result = resolve_provider_config("anthropic", gw)
        assert result.bearer_token == "raw-token-123"
        assert result.api_key is None

    def test_jwt_preferred_over_plain_line(self, mocker, clear_credential_cache):
        # A JWT-shaped token has three base64url segments separated by dots.
        jwt = "aaaa.bbbb.cccc"
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout=f"some info line\n{jwt}\nother output\n", returncode=0)
        token = _run_credentials_helper("any-command")
        assert token == jwt

    def test_last_line_used_when_no_jwt(self, mocker, clear_credential_cache):
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout="line one\nthe-actual-token\n", returncode=0)
        token = _run_credentials_helper("any-command")
        assert token == "the-actual-token"

    def test_helper_cached_second_call_skips_subprocess(self, mocker, clear_credential_cache):
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.return_value = MagicMock(stdout="cached-token\n", returncode=0)
        gw = _make_gateway("anthropic", base_url="https://gw.example.com", credentials_helper="my-helper")
        resolve_provider_config("anthropic", gw)
        resolve_provider_config("anthropic", gw)
        assert mock_run.call_count == 1

    def test_failed_helper_raises_runtime_error(self, mocker, clear_credential_cache):
        import subprocess

        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, "helper", stderr="auth failed")
        gw = _make_gateway("anthropic", credentials_helper="bad-helper")
        with pytest.raises(RuntimeError, match="failed"):
            resolve_provider_config("anthropic", gw)

    def test_failed_helper_retries_up_to_max_attempts(self, mocker, clear_credential_cache):
        import subprocess

        from airedale.gateway import _CREDENTIALS_HELPER_MAX_ATTEMPTS

        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(1, "helper", stderr="auth failed")
        with pytest.raises(RuntimeError, match="after .* attempts"):
            _run_credentials_helper("bad-helper")
        assert mock_run.call_count == _CREDENTIALS_HELPER_MAX_ATTEMPTS

    def test_helper_recovers_after_transient_failure(self, mocker, clear_credential_cache):
        import subprocess

        mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "helper", stderr="transient"),
            MagicMock(stdout="good-token\n", returncode=0),
        ]
        token = _run_credentials_helper("flaky-helper")
        assert token == "good-token"
        assert mock_run.call_count == 2

    def test_timeout_is_not_retried(self, mocker, clear_credential_cache):
        import subprocess

        mock_sleep = mocker.patch("airedale.gateway.time.sleep")
        mock_run = mocker.patch("airedale.gateway.subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired("helper", 30)
        with pytest.raises(RuntimeError, match="timed out"):
            _run_credentials_helper("slow-helper")
        assert mock_run.call_count == 1
        mock_sleep.assert_not_called()
