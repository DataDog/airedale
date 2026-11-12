# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for airedale.config.gateway — load_gateway parsing and validation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

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

    def test_example_sets_eval_target_account_header_for_every_provider(self):
        example = Path(__file__).parent.parent / "examples" / "gateway.toml"
        config = load_gateway(example)

        assert config is not None
        assert config.providers
        for provider in config.providers.values():
            assert provider.headers is not None
            assert provider.headers["x-target-account"] == "eval"


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
        anthropic = config.get("anthropic")
        assert anthropic is not None
        (model,) = anthropic.pi_models
        assert model["id"] == "claude-next"
        assert model["contextWindow"] == 200000
        assert anthropic.pi_models

    def test_json_values_of_every_kind_are_accepted(self, tmp_path):
        config = load_gateway(
            write_toml(
                tmp_path,
                """
                [[providers.openai.pi_models]]
                id = "a"
                reasoning = true
                contextWindow = 1000
                cost = { input = 1.5, output = 2 }
                input = ["text", "image"]
                compat = { nested = { list = [1, "x", false] } }
                """,
            )
        )
        provider = config.get("openai")
        assert provider is not None
        json.dumps(provider.pi_models)  # JSON can hold the whole entry.

    def test_default_is_empty(self, tmp_path):
        config = load_gateway(write_toml(tmp_path, '[providers.openai]\nbase_url = "https://gw"\n'))
        openai = config.get("openai")
        assert openai is not None
        assert openai.pi_models == ()

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ('pi_models = "x"', "array of tables"),
            ("pi_models = [1]", "must be a table"),
            ("[[providers.openai.pi_models]]\nname = 'no id'", "non-empty string 'id'"),
            ("[[providers.openai.pi_models]]\nid = 'a'\n[[providers.openai.pi_models]]\nid = 'a'", "more than once"),
            # TOML dates and times, anywhere in the entry, and numbers that JSON does not have.
            ("[[providers.openai.pi_models]]\nid = 'a'\nreleased = 2026-01-02", "date value"),
            ("[[providers.openai.pi_models]]\nid = 'a'\nat = 07:30:00", "time value"),
            ("[[providers.openai.pi_models]]\nid = 'a'\nat = 2026-01-02T07:30:00Z", "datetime value"),
            ("[[providers.openai.pi_models]]\nid = 'a'\ncost = { input = 1, when = 2026-01-02 }", r"cost\.when"),
            ("[[providers.openai.pi_models]]\nid = 'a'\nlist = [1, [2026-01-02]]", r"list\[1\]\[0\]"),
            ("[[providers.openai.pi_models]]\nid = 'a'\ncost = { input = inf }", "finite number"),
            ("[[providers.openai.pi_models]]\nid = 'a'\ncost = { input = nan }", "finite number"),
        ],
    )
    def test_invalid_pi_models(self, tmp_path, body, message):
        with pytest.raises(ConfigError, match=message):
            load_gateway(write_toml(tmp_path, "[providers.openai]\n" + body + "\n"))


class TestGatewayKeysNeedABaseUrl:
    @pytest.mark.parametrize(
        ("lines", "keys"),
        [
            ('headers = { source = "evals" }', "'headers'"),
            ('credentials_helper = "mytool auth token"', "'credentials_helper'"),
            ('api_key_env = "MY_KEY"', "'api_key_env'"),
            (
                'headers = { a = "b" }\ncredentials_helper = "h"\napi_key_env = "MY_KEY"',
                "'headers', 'credentials_helper', 'api_key_env'",
            ),
        ],
    )
    def test_gateway_keys_without_base_url_are_refused(self, tmp_path, lines, keys):
        toml = f"[providers.anthropic]\n{lines}\n"
        with pytest.raises(ConfigError) as error:
            load_gateway(write_toml(tmp_path, toml))
        message = str(error.value)
        assert "Provider 'anthropic'" in message and keys in message
        assert "base_url" in message and "remove them" in message

    def test_the_second_provider_is_named(self, tmp_path):
        toml = '[providers.anthropic]\nbase_url = "https://x"\n[providers.openai]\nheaders = { a = "b" }\n'
        with pytest.raises(ConfigError, match="Provider 'openai'"):
            load_gateway(write_toml(tmp_path, toml))

    @pytest.mark.parametrize(
        "lines",
        [
            'base_url = "https://x"\nheaders = { a = "b" }\ncredentials_helper = "h"\napi_key_env = "MY_KEY"',
            "headers = {}",  # An empty table says nothing.
            'api_key_env = "ANTHROPIC_API_KEY"',  # The variable that the provider reads by itself.
            "",
            'pi_models = [{ id = "m" }]',  # Not a gateway key.
        ],
    )
    def test_other_files_load(self, tmp_path, lines):
        assert load_gateway(write_toml(tmp_path, f"[providers.anthropic]\n{lines}\n")).get("anthropic") is not None

    def test_the_variable_of_another_provider_is_custom(self, tmp_path):
        with pytest.raises(ConfigError, match="'api_key_env'"):
            load_gateway(write_toml(tmp_path, '[providers.openai]\napi_key_env = "ANTHROPIC_API_KEY"\n'))
