# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Gateway configuration parsing and validation."""

from __future__ import annotations

import math
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
        _check_json_value(entry, f"Provider '{provider}' 'pi_models[{index}]'")
        models.append(dict(entry))
    return tuple(models)


def _check_json_value(value: Any, where: str) -> None:
    """Raise :class:`ConfigError` unless ``value`` is made only of values that JSON can hold.

    A ``pi_models`` entry is written as it is to the ``models.json`` of pi. TOML also has dates and times, which
    JSON has not (and ``inf`` and ``nan``, which are not valid JSON).
    """
    stack: list[tuple[Any, str]] = [(value, where)]
    while stack:
        item, path = stack.pop()
        if isinstance(item, bool | str | int):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ConfigError(f"{path} must be a finite number for JSON, not {item!r}")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ConfigError(f"{path} has a key that is not a string: {key!r}")
                stack.append((child, f"{path}.{key}"))
        elif isinstance(item, list):
            stack.extend((child, f"{path}[{index}]") for index, child in enumerate(item))
        else:
            raise ConfigError(
                f"{path} has a {type(item).__name__} value, which JSON cannot hold "
                "(TOML dates and times are not allowed here); use a string or a number"
            )


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


#: The variable that a provider reads by itself: naming it in ``api_key_env`` changes nothing.
_NATIVE_API_KEY_ENVS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def _check_gateway_keys_have_a_base_url(name: str, config: dict[str, Any]) -> None:
    """Raise when a provider sets gateway keys without ``base_url``.

    ``headers``, ``credentials_helper`` and a custom ``api_key_env`` apply to requests that go to the gateway. Without
    a ``base_url`` no harness sends them (the provider's own API is used), and pi falls back to the credentials of the
    operator: a run could silently use another account or other headers than the file says. The load fails instead.
    """
    if config.get("base_url"):
        return
    ignored = []
    if config.get("headers"):
        ignored.append("headers")
    if config.get("credentials_helper"):
        ignored.append("credentials_helper")
    api_key_env = config.get("api_key_env")
    if api_key_env and api_key_env != _NATIVE_API_KEY_ENVS.get(name):
        ignored.append("api_key_env")
    if ignored:
        keys = ", ".join(f"'{key}'" for key in ignored)
        raise ConfigError(
            f"Provider '{name}' sets {keys} without 'base_url'. They only apply to a gateway and would be ignored "
            "(and pi would use your own credentials). Set 'base_url', or remove them."
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

            _check_gateway_keys_have_a_base_url(provider_name, provider_config)
            providers[provider_name] = _parse_provider(provider_name, provider_config)

    return GatewayConfig(providers=providers)
