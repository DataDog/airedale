# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Runtime gateway credential resolution and provider configuration."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from airedale.safe_log import exception_summary

if TYPE_CHECKING:
    from airedale.config.gateway import GatewayConfig

logger = logging.getLogger(__name__)

# Cache for credential helper results
_credential_cache: dict[str, tuple[str, float]] = {}
_credential_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 1800  # 30 minutes
_CREDENTIALS_HELPER_MAX_ATTEMPTS = 4  # total tries on non-zero exit before giving up
_CREDENTIALS_HELPER_RETRY_DELAY_SECONDS = 1.0  # base backoff between retries


@dataclass
class ResolvedGatewayConfig:
    """Resolved gateway configuration for a provider."""

    base_url: str | None
    bearer_token: str | None
    api_key: str | None
    headers: dict[str, str]


class _HelperOutputError(ValueError):
    """The output of the helper is not usable (the message is a fixed text of Airedale)."""


def _run_credentials_helper(command: str) -> str:
    """Run a credentials helper command and return the output."""
    with _credential_cache_lock:
        now = time.time()
        if command in _credential_cache:
            cached_token, cached_time = _credential_cache[command]
            if now - cached_time < _CACHE_TTL_SECONDS:
                return cached_token

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, _CREDENTIALS_HELPER_MAX_ATTEMPTS + 1):
        try:
            # (The command is never logged or put in an error: it can hold an inline secret.)
            logger.debug("Running credentials helper (attempt %d/%d)", attempt, _CREDENTIALS_HELPER_MAX_ATTEMPTS)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, check=True)
            output = result.stdout.strip()

            if not output:
                raise _HelperOutputError("Credentials helper returned empty output")

            # Parse output - prefer JWT-shaped tokens, otherwise use last non-empty line
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if not lines:
                raise _HelperOutputError("Credentials helper returned no usable output")

            # Look for JWT pattern (three base64-like segments separated by dots)
            jwt_pattern = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
            for line in lines:
                if jwt_pattern.match(line):
                    token = line
                    break
            else:
                # Use the last non-empty line if no JWT found
                token = lines[-1]

            # Cache the result
            with _credential_cache_lock:
                _credential_cache[command] = (token, now)

            return token

        except subprocess.TimeoutExpired:
            # (``from None``: the message of the timeout error holds the command.)
            raise RuntimeError("Credentials helper timed out") from None
        except subprocess.CalledProcessError as e:
            # Non-zero exit: retry a bounded number of times before giving up.
            last_error = e
            # (The stderr of the helper is never logged or put in an error: it can hold a credential.)
            if attempt < _CREDENTIALS_HELPER_MAX_ATTEMPTS:
                logger.warning(
                    "Credentials helper failed (exit %s, attempt %d/%d). Retrying...",
                    e.returncode,
                    attempt,
                    _CREDENTIALS_HELPER_MAX_ATTEMPTS,
                )
                time.sleep(_CREDENTIALS_HELPER_RETRY_DELAY_SECONDS * attempt)
                continue
            break
        except _HelperOutputError as e:
            raise RuntimeError(f"Failed to run credentials helper\nError: {e}") from e
        except Exception as e:
            # (Not the message of the error: it can hold data of the helper.)
            raise RuntimeError(f"Failed to run credentials helper\nError: {exception_summary(e)}") from None

    # Exhausted all attempts on non-zero exit.
    assert last_error is not None
    raise RuntimeError(
        f"Credentials helper failed after {_CREDENTIALS_HELPER_MAX_ATTEMPTS} attempts "
        f"(exit {last_error.returncode}). Run the command yourself to see its error output."
    ) from None


def resolve_provider_config(provider: str, gateway_config: GatewayConfig | None = None) -> ResolvedGatewayConfig:
    """Resolve gateway configuration for a provider.

    Returns resolved base_url, bearer token/api key, and headers.
    When gateway_config is None or provider not configured, returns "no override"
    so callers can fall back to provider SDK defaults.
    """
    if gateway_config is None:
        return ResolvedGatewayConfig(base_url=None, bearer_token=None, api_key=None, headers={})

    provider_config = gateway_config.get(provider)
    if provider_config is None:
        return ResolvedGatewayConfig(base_url=None, bearer_token=None, api_key=None, headers={})

    # Resolve credentials
    bearer_token = None
    api_key = None

    if provider_config.credentials_helper:
        try:
            bearer_token = _run_credentials_helper(provider_config.credentials_helper)
        except Exception as e:
            logger.error("Failed to get credentials for %s (%s)", provider, exception_summary(e))
            raise
    elif provider_config.api_key_env:
        api_key = os.environ.get(provider_config.api_key_env)
        if not api_key:
            logger.warning(f"Environment variable {provider_config.api_key_env} not set for {provider}")

    return ResolvedGatewayConfig(
        base_url=provider_config.base_url,
        bearer_token=bearer_token,
        api_key=api_key,
        headers=provider_config.headers or {},
    )
