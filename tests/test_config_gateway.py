# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.config.gateway — load_gateway parsing and validation."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from airedale.config import ConfigError
from airedale.config.gateway import GatewayConfig, ProviderGatewayConfig, load_gateway


def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "gateway.toml"
    p.write_text(textwrap.dedent(content))
    return p


class TestLoadGatewayValid:
    def test_empty_providers(self, tmp_path):
        config = load_gateway(write_toml(tmp_path, "[providers]\n"))
        assert config.providers == {}

    def test_single_provider_credentials_helper(self, tmp_path):
        toml = """
        [providers.anthropic]
        base_url = "https://gateway.example.com"
        credentials_helper = "mytool auth token"
        headers = { source = "evals", provider = "anthropic" }
        """
        config = load_gateway(write_toml(tmp_path, toml))
        p = config.get("anthropic")
        assert p is not None
        assert p.provider == "anthropic"
        assert p.base_url == "https://gateway.example.com"
        assert p.credentials_helper == "mytool auth token"
        assert p.headers == {"source": "evals", "provider": "anthropic"}
        assert p.api_key_env is None

    def test_single_provider_api_key_env(self, tmp_path):
        toml = """
        [providers.openai]
        base_url = "https://gateway.example.com/v1"
        api_key_env = "MY_OPENAI_KEY"
        headers = { source = "evals" }
        """
        config = load_gateway(write_toml(tmp_path, toml))
        p = config.get("openai")
        assert p is not None
        assert p.api_key_env == "MY_OPENAI_KEY"
        assert p.credentials_helper is None

    def test_multiple_providers(self, tmp_path):
        toml = """
        [providers.anthropic]
        base_url = "https://gateway.example.com"
        credentials_helper = "helper --provider anthropic"

        [providers.openai]
        base_url = "https://gateway.example.com/v1"
        credentials_helper = "helper --provider openai"
        """
        config = load_gateway(write_toml(tmp_path, toml))
        assert config.get("anthropic") is not None
        assert config.get("openai") is not None

    def test_get_unknown_provider_returns_none(self, tmp_path):
        toml = """
        [providers.anthropic]
        base_url = "https://gateway.example.com"
        credentials_helper = "helper"
        """
        config = load_gateway(write_toml(tmp_path, toml))
        assert config.get("cohere") is None

    def test_headers_default_to_empty_dict(self, tmp_path):
        toml = """
        [providers.anthropic]
        base_url = "https://gateway.example.com"
        credentials_helper = "helper"
        """
        config = load_gateway(write_toml(tmp_path, toml))
        p = config.get("anthropic")
        assert p is not None
        assert p.headers == {}

    def test_empty_file(self, tmp_path):
        p = tmp_path / "gateway.toml"
        p.write_text("")
        config = load_gateway(p)
        assert isinstance(config, GatewayConfig)
        assert config.providers == {}


class TestLoadGatewayErrors:
    def test_unknown_top_level_key(self, tmp_path):
        # The unknown key must appear before [providers] to land at the top level
        toml = """
        extra_section = "boom"

        [providers.anthropic]
        base_url = "https://x.com"
        credentials_helper = "h"
        """
        with pytest.raises(ConfigError, match="Unknown top-level keys"):
            load_gateway(write_toml(tmp_path, toml))

    def test_unknown_provider_key(self, tmp_path):
        toml = """
        [providers.anthropic]
        base_url = "https://x.com"
        credentials_helper = "h"
        unknown_key = "value"
        """
        with pytest.raises(ConfigError, match="Unknown keys in provider"):
            load_gateway(write_toml(tmp_path, toml))

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_gateway(tmp_path / "missing.toml")


class TestProviderGatewayConfig:
    def test_frozen_dataclass(self):
        p = ProviderGatewayConfig(provider="anthropic", base_url="https://x.com")
        with pytest.raises((AttributeError, TypeError)):
            p.provider = "openai"  # type: ignore[misc]

    def test_headers_none_becomes_empty_dict(self):
        p = ProviderGatewayConfig(provider="anthropic", headers=None)
        assert p.headers == {}


class TestPiModels:
    def test_pi_models_parsed(self, tmp_path):
        config = load_gateway(
            write_toml(
                tmp_path,
                """
                [providers.anthropic]
                base_url = "https://gw"

                [[providers.anthropic.pi_models]]
                id = "claude-next"
                contextWindow = 200000
                cost = { input = 3, output = 15, cacheRead = 0.3, cacheWrite = 3.75 }
                """,
            )
        )
        (model,) = config.get("anthropic").pi_models
        assert model["id"] == "claude-next"
        assert model["contextWindow"] == 200000
        assert config.get("anthropic").pi_models

    def test_default_is_empty(self, tmp_path):
        config = load_gateway(write_toml(tmp_path, '[providers.openai]\nbase_url = "https://gw"\n'))
        assert config.get("openai").pi_models == ()

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ('pi_models = "x"', "array of tables"),
            ("pi_models = [1]", "must be a table"),
            ("[[providers.openai.pi_models]]\nname = 'no id'", "non-empty string 'id'"),
            ("[[providers.openai.pi_models]]\nid = 'a'\n[[providers.openai.pi_models]]\nid = 'a'", "more than once"),
        ],
    )
    def test_invalid_pi_models(self, tmp_path, body, message):
        with pytest.raises(ConfigError, match=message):
            load_gateway(write_toml(tmp_path, "[providers.openai]\n" + body + "\n"))
