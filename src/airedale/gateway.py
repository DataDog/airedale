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
            logger.debug(
                f"Running credentials helper (attempt {attempt}/{_CREDENTIALS_HELPER_MAX_ATTEMPTS}): {command}"
            )
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, check=True)
            output = result.stdout.strip()

            if not output:
                raise ValueError("Credentials helper returned empty output")

            # Parse output - prefer JWT-shaped tokens, otherwise use last non-empty line
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if not lines:
                raise ValueError("Credentials helper returned no usable output")

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

        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Credentials helper timed out: {command}") from e
        except subprocess.CalledProcessError as e:
            # Non-zero exit: retry a bounded number of times before giving up.
            last_error = e
            stderr = e.stderr.strip() if e.stderr else "no error output"
            if attempt < _CREDENTIALS_HELPER_MAX_ATTEMPTS:
                logger.warning(
                    f"Credentials helper failed (exit {e.returncode}, attempt "
                    f"{attempt}/{_CREDENTIALS_HELPER_MAX_ATTEMPTS}): {command}\nError: {stderr}\nRetrying..."
                )
                time.sleep(_CREDENTIALS_HELPER_RETRY_DELAY_SECONDS * attempt)
                continue
            break
        except Exception as e:
            raise RuntimeError(f"Failed to run credentials helper: {command}\nError: {e}") from e

    # Exhausted all attempts on non-zero exit.
    assert last_error is not None
    stderr = last_error.stderr.strip() if last_error.stderr else "no error output"
    raise RuntimeError(
        f"Credentials helper failed after {_CREDENTIALS_HELPER_MAX_ATTEMPTS} attempts "
        f"(exit {last_error.returncode}): {command}\nError: {stderr}"
    ) from last_error


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
            logger.error(f"Failed to get credentials for {provider}: {e}")
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
