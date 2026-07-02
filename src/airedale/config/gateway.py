# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Gateway configuration parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from airedale.config import ConfigError, read_toml_file


@dataclass(frozen=True)
class ProviderGatewayConfig:
    """Gateway configuration for a specific provider."""

    provider: str
    base_url: str | None = None
    credentials_helper: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate that headers is not None."""
        if self.headers is None:
            object.__setattr__(self, "headers", {})


@dataclass(frozen=True)
class GatewayConfig:
    """Complete gateway configuration."""

    providers: dict[str, ProviderGatewayConfig]

    def get(self, provider: str) -> ProviderGatewayConfig | None:
        """Get gateway config for a provider, or None if not configured."""
        return self.providers.get(provider)


def _parse_provider(name: str, config: dict[str, Any]) -> ProviderGatewayConfig:
    """Parse provider gateway config from TOML data."""
    return ProviderGatewayConfig(
        provider=name,
        base_url=config.get("base_url"),
        credentials_helper=config.get("credentials_helper"),
        api_key_env=config.get("api_key_env"),
        headers=config.get("headers") or {},
    )


def load_gateway(path: str | Path) -> GatewayConfig:
    """Load and validate a gateway configuration from a TOML file."""
    data = read_toml_file(path)

    # Check for unknown top-level keys
    known_keys = {"providers"}
    unknown_keys = set(data.keys()) - known_keys
    if unknown_keys:
        raise ConfigError(f"Unknown top-level keys: {', '.join(sorted(unknown_keys))}")

    providers = {}
    if "providers" in data:
        for provider_name, provider_config in data["providers"].items():
            # Check for unknown provider keys
            known_provider_keys = {"base_url", "credentials_helper", "api_key_env", "headers"}
            unknown_provider_keys = set(provider_config.keys()) - known_provider_keys
            if unknown_provider_keys:
                raise ConfigError(
                    f"Unknown keys in provider '{provider_name}': {', '.join(sorted(unknown_provider_keys))}"
                )

            providers[provider_name] = _parse_provider(provider_name, provider_config)

    return GatewayConfig(providers=providers)
