# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Tests for the URL reader (``airedale.whatwg_url``): it must read what JavaScript's ``URL`` reads."""

from __future__ import annotations

import pytest

from airedale.whatwg_url import is_loopback_host, parse_url

URLS = [
    "http://host:" + "9" * 5000 + "/",
    "http://host:" + "0" * 5000 + "80/",
    "http://host:" + "0" * 5000 + "/",
    "http://host:" + "1" * 6 + "/",
    "http://host:" + "0" * 50 + "8080/x",
    "https://%C2%AD127.1/mcp",
    "https://\u00ad127.1/mcp",
    "https://127\u200b.1/mcp",
    "http://127.\u200b1/",
    "http://\u200b/",
    "http://\u00ad.\u00ad/",
    "http://1\u00ad.2.3.4/",
    "http://999\u00ad.1/",
    "http://%C2%AD/",
    "http://a\u00ad b/",
    "https://example.test/mcp",
    "http://localhost:80/x?y#z",
    "ht\ttps://example.invalid/mcp",
    "\x00https://example.invalid/mcp",
    "  https://x.test  ",
    "http://127.1/meta",
    "http://0x7f.1/",
    "http://2130706433/",
    "http://0177.0.0.1/",
    "http://[::1]/",
    "http://[0:0:0:0:0:0:0:1]:8080/p",
    "http://[::ffff:127.0.0.1]/",
    "http://LOCALHOST/",
    "http://localhost./",
    "http://%6Cocalhost/",
    "http://ｌｏｃａｌｈｏｓｔ/",
    "http:///x",
    "http://",
    "http:",
    "http:example.test",
    "http:\\\\example.test\\x",
    "https:/\\example.test",
    "http://user@/x",
    "http://@host/",
    "http://u:p@host:8080/a/b",
    "http://host:/",
    "http://host:abc/",
    "http://host:65536/",
    "http://host:65535/",
    "http://host:0080/",
    "https://host:443/",
    "https://host:444/",
    "http://a b/",
    "http://a<b/",
    "http://a%25b/",
    "http://a%00b/",
    "http://1.2.3.4.5/",
    "http://1.2.3.256/",
    "http://256/",
    "http://4294967296/",
    "http://0x100000000/",
    "http://09/",
    "http://1.2.3/",
    "http://1.2.3./",
    "http://a.1/",
    "http://a.0x/",
    "http://xn--nxasmq6b/",
    "http://exämple.com/",
    "http://127.0.0.1:8080/callback",
    "http://localhost/callback/.",
    "http://localhost/x/../callback",
    "http://localhost/%2e/callback",
    "http://localhost\\callback",
    "http://localhost/callback?",
    "http://localhost/callback#",
    "http://localhost/callback?a#b",
    "http://localhost/callback ",
    "http://[::1",
    "http://[::1]x/",
    "http://[1::2::3]/",
    "http://[::1%lo0]/mcp",
    "http://[::1%25lo0]/mcp",
    "http://[fe80::1%eth0]:8080/",
    "http://[::1%]/",
    "http://[%]/",
    "http://[::1]%/",
    "http://x%/",
    "ftp://x/",
    "file:///x",
    "mailto:x",
    "foo",
    "//x",
    "/x",
    "?x",
    "HTTP://X.TEST/",
    "https://x.test:/",
    "http://x.test:0/",
    "http://1.1.1.1.",
    "http://0/",
    "http://..",
    "http://.",
    "http://a..b/",
    "http://-x/",
    "https://[::1]:443/",
    "http://[::]/",
    "http://[v1.x]/",
    "http://x.test/a b?c d#e f",
    "http://x.test/%7e",
    "http://x\u200by/",
    "http://x\u00ady/",
    "http://\u00ad/",
]

#: Hosts that the reader does not check in every detail (IDNA, exact IPv6 syntax): it accepts them (``exact=False``).
INEXACT = {"http://[1::2::3]/", "http://[v1.x]/", "http://\xad/"}


def test_reader_matches_javascript_url(pi_reference):
    """Same answer as ``URL`` for each string, or an answer that accepts more (never one that rejects more)."""
    answers = pi_reference({"action": "urls", "urls": URLS})
    for version, expected in answers.items():
        for value, node in zip(URLS, expected, strict=True):
            ours = parse_url(value)
            if node is None:
                assert ours is None or not ours.exact, (version, value)
                continue
            assert ours is not None, (version, value)
            assert ours.scheme == node["scheme"], (version, value)
            if node["scheme"] not in ("http", "https"):
                continue
            assert (ours.port, ours.pathname) == (node["port"], node["pathname"]), (version, value)
            if ours.exact:
                assert ours.hostname == node["hostname"], (version, value)
            assert ("?" + ours.search if ours.search else "") == node["search"], (version, value)
            assert ("#" + ours.hash if ours.hash else "") == node["hash"], (version, value)


@pytest.mark.parametrize("value", sorted(INEXACT))
def test_uncertain_hosts_are_accepted(value):
    url = parse_url(value)
    assert url is not None
    assert not url.exact
    assert is_loopback_host(url, ("localhost",))


@pytest.mark.parametrize(
    ("value", "loopback"),
    [
        ("http://127.1/meta", True),
        ("http://0x7f.1/", True),
        ("http://2130706433/", True),
        ("http://[0:0:0:0:0:0:0:1]/", True),
        ("http://LOCALHOST/", True),
        ("http://%6Cocalhost/", True),
        ("http://localhost./", False),
        ("http://example.test/", False),
        ("http://[::ffff:127.0.0.1]/", False),
    ],
)
def test_loopback_hosts(value, loopback):
    url = parse_url(value)
    assert url is not None
    assert is_loopback_host(url, ("localhost", "127.0.0.1", "[::1]")) is loopback


@pytest.mark.parametrize("value", ["ht\ttps://example.test/mcp", "\x00https://example.test/", "\nhttp://example.test"])
def test_control_characters_are_removed(value):
    url = parse_url(value)
    assert url is not None
    assert (url.scheme, url.hostname) == (("https" if "https" in value.replace("\t", "") else "http"), "example.test")


@pytest.mark.parametrize(
    "value", ["", "x", "//x", "/x", "http:", "http://", "http:///", "http://@/", "http://h:99999", "http://a b/"]
)
def test_urls_that_fail(value):
    assert parse_url(value) is None


@pytest.mark.parametrize(
    "value", ["https://%C2%AD127.1/mcp", "https://\u00ad127.1/mcp", "https://127\u200b.1/mcp", "http://999\u00ad.1/"]
)
def test_a_host_with_characters_that_idna_removes_is_never_a_sure_failure(value):
    # Pi removes these characters from the host (127.1 is then 127.0.0.1): the reader does not copy that step,
    # so it must not say that the URL fails.
    url = parse_url(value)
    assert url is not None
    assert not url.exact
    assert is_loopback_host(url, ("127.0.0.1",))


def test_an_ascii_forbidden_character_is_still_a_sure_failure():
    assert parse_url("http://a\u00ad b/") is None


@pytest.mark.parametrize("digits", ["9" * 5000, "9" * 5, "1" * 6, "65536"])
def test_a_port_that_is_too_long_or_too_large_fails_without_an_exception(digits):
    assert parse_url(f"http://host:{digits}/") is None


def test_a_port_with_many_leading_zeros_is_read():
    url = parse_url("http://host:" + "0" * 5000 + "8080/")
    assert url is not None
    assert url.port == "8080"
    default = parse_url("http://host:" + "0" * 5000 + "80/")
    assert default is not None
    assert default.port == ""
