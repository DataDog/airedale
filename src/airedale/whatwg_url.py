# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""A reader for the URLs that pi reads with ``new URL(value)`` (JavaScript, WHATWG URL Standard).

Only what the guard for the MCP config of pi needs: does ``URL.canParse(value)``
succeed, and what are the scheme, host name, port, path, query and fragment of
the result. The steps are those of the URL Standard (https://url.spec.whatwg.org/)
for an input with no base URL:

* leading and trailing C0 controls and spaces are removed, and tab and newlines
  are removed everywhere (``"ht\\ttps://x"`` is ``"https://x"``);
* the scheme is read; a ``http`` or ``https`` URL has slashes (any number of ``/``
  and ``\\``) and then a host that is not empty, with an optional port;
* the host is percent-decoded and mapped to ASCII; an IPv4 address can have 1 to
  4 parts in decimal, octal or hexadecimal (``127.1`` is ``127.0.0.1``); an IPv6
  address is in brackets.

Where the result depends on a rule that is not copied here (the IDNA mapping of a
non-ASCII host name, the exact syntax of an IPv6 address), the answer is marked
``exact=False``. The users of this module treat such an answer as the one that
makes pi accept the URL: this module is part of a check that has to fail closed.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote_to_bytes

_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
#: ASCII tab and newlines: removed from the whole input.
_STRIP_ALL = str.maketrans("", "", "\t\n\r")
#: Code points that a host cannot have after it was percent-decoded (the "forbidden domain code points" of the Standard).
_FORBIDDEN_HOST_CHARS = frozenset("\x00\t\n\r #/:<>?@[\\]^|%") | frozenset(chr(code) for code in range(0x20)) | {"\x7f"}
#: Characters that the URL parser percent-encodes in the path, the query (of a special URL) and the fragment.
_FRAGMENT_ENCODE = frozenset(' "<>`')
_QUERY_ENCODE = frozenset(" \"#<>'")
_PATH_ENCODE = frozenset(' "#<>?^`{}')
_DEFAULT_PORTS = {"http": 80, "https": 443}
_IPV4_NUMBER_RE = re.compile(r"[0-9]+|0[xX][0-9a-fA-F]*")


@dataclass(frozen=True)
class ParsedUrl:
    """The parts of a URL, as ``URL`` of JavaScript gives them."""

    scheme: str
    #: ``URL.hostname``: IPv6 addresses are in brackets, IPv4 addresses are dotted.
    hostname: str
    #: ``URL.port``: empty for the default port of the scheme.
    port: str
    pathname: str
    #: ``URL.search`` and ``URL.hash``: empty when there is nothing after ``?`` or ``#``.
    search: str
    hash: str
    #: False when a rule that this module does not copy decides the answer (see the module doc).
    exact: bool = True


def parse_url(value: str) -> ParsedUrl | None:
    """Parse ``value`` like ``new URL(value)``. ``None``: ``URL.canParse(value)`` is false."""
    text = value.strip("".join(chr(code) for code in range(0x21))).translate(_STRIP_ALL)
    scheme_match = _SCHEME_RE.match(text)
    if scheme_match is None:
        return None  # No scheme, and no base URL: a relative URL fails.
    scheme = scheme_match.group(0)[:-1].lower()
    rest = text[scheme_match.end() :]
    if scheme not in _DEFAULT_PORTS:
        # Another scheme: the callers only ask for http and https. The rest of the URL is not read.
        return ParsedUrl(scheme, "", "", "", "", "", exact=True)
    rest = rest.lstrip("/\\")
    end = next((index for index, char in enumerate(rest) if char in "/\\?#"), len(rest))
    authority, remainder = rest[:end], rest[end:]
    host_text = authority
    if "@" in authority:
        credentials, _, host_text = authority.rpartition("@")
        if not host_text:
            return None  # Credentials and no host.
    host_part, port_text = _split_host_port(host_text)
    if port_text is None:
        return None
    if port_text and (not port_text.isascii() or not port_text.isdigit() or int(port_text) > 65535):
        return None
    port = "" if port_text and int(port_text) == _DEFAULT_PORTS[scheme] else str(int(port_text)) if port_text else ""
    host = _parse_host(host_part)
    if host is None:
        return None
    hostname, exact = host
    path, search, fragment = _split_path(remainder)
    return ParsedUrl(
        scheme,
        hostname,
        port,
        _percent_encode(_normalize_path(path), _PATH_ENCODE),
        _percent_encode(search, _QUERY_ENCODE),
        _percent_encode(fragment, _FRAGMENT_ENCODE),
        exact=exact,
    )


def _percent_encode(text: str, encode: frozenset[str]) -> str:
    return "".join(
        "".join(f"%{byte:02X}" for byte in char.encode("utf-8", "replace"))
        if char in encode or ord(char) < 0x20 or ord(char) > 0x7E
        else char
        for char in text
    )


def is_loopback_host(url: ParsedUrl, loopback_hosts: tuple[str, ...]) -> bool:
    """Return True when the host of ``url`` is one of ``loopback_hosts``, or when it may be (``exact=False``)."""
    return url.hostname in loopback_hosts or not url.exact


def _split_host_port(text: str) -> tuple[str, str | None]:
    """Split ``host[:port]``. The port is ``None`` when the text after a bracketed host is not a port."""
    if text.startswith("["):
        close = text.find("]")
        if close == -1:
            return text, ""  # The host parser fails on it.
        after = text[close + 1 :]
        if not after:
            return text[: close + 1], ""
        if not after.startswith(":"):
            return text, ""  # Not a valid host: the host parser fails on it.
        return text[: close + 1], after[1:]
    host, colon, port = text.partition(":")
    return host, (port if colon else "")


def _split_path(remainder: str) -> tuple[str, str, str]:
    """Split what follows the authority into path, query (without ``?``) and fragment (without ``#``)."""
    before_fragment, _, fragment = remainder.partition("#")
    path, _, query = before_fragment.partition("?")
    return path, query, fragment


def _normalize_path(path: str) -> str:
    """Return the path of a special URL: ``\\`` is ``/``, and ``.`` and ``..`` segments are resolved."""
    if not path:
        return "/"
    segments = path.replace("\\", "/").split("/")[1:]
    resolved: list[str] = []
    for index, segment in enumerate(segments):
        lowered = segment.lower()
        last = index == len(segments) - 1
        if lowered in ("..", ".%2e", "%2e.", "%2e%2e"):
            if resolved:
                resolved.pop()
            if last:
                resolved.append("")
        elif lowered in (".", "%2e"):
            if last:
                resolved.append("")
        else:
            resolved.append(segment)
    return "/" + "/".join(resolved)


def _parse_host(text: str) -> tuple[str, bool] | None:
    """Parse the host of a special URL. Return ``(serialized host, exact)``, or ``None`` when the URL fails."""
    if text.startswith("["):
        if not text.endswith("]"):
            return None
        try:
            address = ipaddress.IPv6Address(text[1:-1])
        except ValueError:
            # The syntax of the Standard is not the same as the one of ``ipaddress``: not sure.
            return text.lower(), False
        return f"[{_serialize_ipv6(address)}]", True
    decoded = unquote_to_bytes(text).decode("utf-8", errors="replace")
    if not decoded.isascii():
        # IDNA processing (UTS #46) maps and removes characters (U+00AD, U+200B, full-width forms...), and this module
        # does not copy it. What the host is after that, and if the URL fails, is not known. Only the characters that
        # no mapping removes decide a failure; any other outcome counts as accepted by pi (fail closed).
        if any(char in _FORBIDDEN_HOST_CHARS for char in decoded if char.isascii()):
            return None
        mapped = unicodedata.normalize("NFKC", decoded).lower()
        return mapped, False
    domain = decoded.lower()
    if not domain or any(char in _FORBIDDEN_HOST_CHARS for char in domain):
        return None
    if _ends_in_number(domain):
        address_text = _parse_ipv4(domain)
        return (address_text, True) if address_text is not None else None
    return domain, True


def _serialize_ipv6(address: ipaddress.IPv6Address) -> str:
    """The IPv6 serializer of the Standard: hexadecimal pieces, the first longest run of two or more zeros is ``::``."""
    pieces = [int(address.packed[index : index + 2].hex(), 16) for index in range(0, 16, 2)]
    best_start, best_length, start = -1, 1, -1
    for index, piece in enumerate([*pieces, 1]):
        if piece == 0:
            start = index if start == -1 else start
        elif start != -1:
            if index - start > best_length:
                best_start, best_length = start, index - start
            start = -1
    parts = [f"{piece:x}" for piece in pieces]
    if best_start == -1:
        return ":".join(parts)
    head, tail = ":".join(parts[:best_start]), ":".join(parts[best_start + best_length :])
    return f"{head}::{tail}"


def _ends_in_number(domain: str) -> bool:
    parts = domain.split(".")
    if parts[-1] == "":
        if len(parts) == 1:
            return False
        parts.pop()
    last = parts[-1]
    return bool(last) and (last.isascii() and last.isdigit() or _IPV4_NUMBER_RE.fullmatch(last) is not None)


def _parse_ipv4(domain: str) -> str | None:
    """The IPv4 parser of the Standard. ``None``: failure."""
    parts = domain.split(".")
    if parts[-1] == "" and len(parts) > 1:
        parts.pop()
    if len(parts) > 4:
        return None
    numbers: list[int] = []
    for part in parts:
        number = _parse_ipv4_number(part)
        if number is None:
            return None
        numbers.append(number)
    if any(number > 255 for number in numbers[:-1]) or numbers[-1] >= 256 ** (5 - len(numbers)):
        return None
    address = numbers[-1] + sum(number * 256 ** (3 - index) for index, number in enumerate(numbers[:-1]))
    return ".".join(str((address >> shift) & 255) for shift in (24, 16, 8, 0))


def _parse_ipv4_number(text: str) -> int | None:
    if not text:
        return None
    base = 10
    if len(text) >= 2 and text[:2].lower() == "0x":
        text, base = text[2:], 16
    elif len(text) >= 2 and text[0] == "0":
        text, base = text[1:], 8
    if not text:
        return 0
    try:
        return int(text, base) if text.isascii() and all(char in "0123456789abcdefABCDEF" for char in text) else None
    except ValueError:
        return None
