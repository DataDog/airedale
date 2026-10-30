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
    #: Extra pi model definitions for this provider (pi ``models.json``
    #: ``models`` entries, passed through as-is). Pi already knows the details
    #: (context window, output limit, cost, ...) of the models in its built-in
    #: catalog; use this only for a model the catalog does not have, or to
    #: replace a catalog entry. Only the pi harness reads it.
    pi_models: tuple[dict[str, Any], ...] = ()

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


def _parse_pi_models(provider: str, raw: Any) -> tuple[dict[str, Any], ...]:
    """Validate ``pi_models``: a list of tables, each with a non-empty string ``id``."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError(
            f"Provider '{provider}' 'pi_models' must be an array of tables ([[providers.{provider}.pi_models]])"
        )
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"Provider '{provider}' 'pi_models[{index}]' must be a table")
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigError(f"Provider '{provider}' 'pi_models[{index}]' must set a non-empty string 'id'")
        if model_id in seen:
            raise ConfigError(f"Provider '{provider}' 'pi_models' defines model '{model_id}' more than once")
        seen.add(model_id)
        models.append(dict(entry))
    return tuple(models)


def _parse_provider(name: str, config: dict[str, Any]) -> ProviderGatewayConfig:
    """Parse provider gateway config from TOML data."""
    return ProviderGatewayConfig(
        provider=name,
        base_url=config.get("base_url"),
        credentials_helper=config.get("credentials_helper"),
        api_key_env=config.get("api_key_env"),
        headers=config.get("headers") or {},
        pi_models=_parse_pi_models(name, config.get("pi_models")),
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
            known_provider_keys = {"base_url", "credentials_helper", "api_key_env", "headers", "pi_models"}
            unknown_provider_keys = set(provider_config.keys()) - known_provider_keys
            if unknown_provider_keys:
                raise ConfigError(
                    f"Unknown keys in provider '{provider_name}': {', '.join(sorted(unknown_provider_keys))}"
                )

            providers[provider_name] = _parse_provider(provider_name, provider_config)

    return GatewayConfig(providers=providers)
