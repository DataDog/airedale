# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Read the YAML frontmatter of a skill file like pi does.

Pi (``utils/frontmatter.js``) cuts the frontmatter out of the file and parses it
with the ``yaml`` package (YAML 1.2, core schema, unique keys). A skill that pi
cannot parse is not loaded. The collision check for skills has to know the
``name`` and ``description`` that pi reads, so this module follows the same steps:

* remove one BOM, turn CRLF and CR into LF; the frontmatter starts at ``---`` and
  ends at the first ``\\n---``;
* parse the text as one YAML 1.2 document. ``ruamel.yaml`` reads the syntax; the
  scalars are resolved from its parser events with the core schema of the ``yaml``
  package (``ruamel.yaml`` alone reads ``1_000`` and ``0b11`` as numbers and a date
  as a timestamp);
* check the whole document as the ``yaml`` package does: a key that is repeated
  at any depth is an error, and so are ``!!set`` with a value that is not null and
  ``!!omap`` / ``!!pairs`` with an item that is not a single pair (or, for ``!!omap``,
  a key that is repeated). Two numbers are the same key when they are the same
  JavaScript number (``9007199254740992`` and ``9007199254740993`` are);
* a tab is read exactly or not at all (see :class:`UnsupportedFrontmatterError`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError, YAMLError
from ruamel.yaml.events import (
    AliasEvent,
    DocumentStartEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
)
from ruamel.yaml.tokens import BlockEntryToken, KeyToken

if TYPE_CHECKING:
    from pathlib import Path

_YAML_INT_RE = re.compile(r"[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+")
_YAML_FLOAT_RE = re.compile(r"[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+|[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)")
_YAML_INF_RE = re.compile(r"[-+]?\.(?:inf|Inf|INF)")
_YAML_NAN = frozenset({".nan", ".NaN", ".NAN"})
_YAML_NULL = frozenset({"", "~", "null", "Null", "NULL"})
_YAML_BOOL = {"true": True, "True": True, "TRUE": True, "false": False, "False": False, "FALSE": False}
_YAML_PREFIX = "tag:yaml.org,2002:"
_YAML_STR_TAG = _YAML_PREFIX + "str"
#: Tags of scalars that are not strings and are not in the core schema (pi knows them).
_YAML_NON_STRING_TAGS = frozenset(_YAML_PREFIX + name for name in ("binary", "timestamp", "merge"))
_SET_TAG, _OMAP_TAG, _PAIRS_TAG = _YAML_PREFIX + "set", _YAML_PREFIX + "omap", _YAML_PREFIX + "pairs"
#: How many tabs the reader places before it gives up.
_MAX_TAB_REPAIRS = 1000

#: ``(kind, value)`` of a node. Kinds: str, int, float, bool, null, other (a scalar with no value that
#: can be equal to another), mapping, sequence.
Node = tuple[str, Any]


class UnsupportedFrontmatterError(ValueError):
    """The frontmatter has a tab that the reader can not place, so it can not tell what pi reads.

    YAML 1.2 allows a tab as white space between tokens, inside a plain value, and
    in a few other places. ``ruamel.yaml`` allows it in fewer places. The reader
    handles a tab after a ``:``, before a ``#`` comment or a ``:``, after a ``-`` or ``?`` indicator, and at the end of a
    line (the tab is then not part of any value, and it is replaced by a space of the
    same width). For any other tab that the parser rejects it can not tell if
    the tab is part of a value, so it does not guess: a changed name could hide a
    collision with a scenario skill.
    """


def read_pi_frontmatter(path: Path) -> dict[str, str] | None:
    """Return the string values of the top-level frontmatter mapping of ``path``, as pi reads them.

    A ``name`` or ``description`` that is not a string is left out (pi ignores it).
    ``None`` when pi does not load a skill from the file: no frontmatter, invalid
    YAML (also a repeated key), or a frontmatter that is not a mapping.
    Raises :class:`UnsupportedFrontmatterError` when the answer depends on a tab
    that the reader can not place.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    yaml_text = text[4:end]
    if not yaml_text:
        return {}
    try:
        return _parse_with_separator_tabs(yaml_text)
    except UnsupportedFrontmatterError as error:
        raise UnsupportedFrontmatterError(f"Cannot read the frontmatter of {path}: {error}") from None
    except (YAMLError, ValueError, RecursionError):
        return None


def _parse_with_separator_tabs(text: str) -> dict[str, str]:
    """Parse ``text``; place the tabs that the parser rejects when they are plainly white space between tokens."""
    for _ in range(_MAX_TAB_REPAIRS):
        try:
            return _top_level_strings(text)
        except MarkedYAMLError as error:
            index = getattr(error.problem_mark, "index", None) if error.problem_mark is not None else None
            if index is None or index >= len(text) or text[index] != "\t":
                raise
            if not _is_separator_tab(text, index):
                raise UnsupportedFrontmatterError(
                    f"a tab at line {text.count(chr(10), 0, index) + 1} is not plainly white space between tokens, "
                    "and the parser does not accept it"
                ) from None
            text = text[:index] + " " + text[index + 1 :]
    raise UnsupportedFrontmatterError("too many tabs")


def _is_separator_tab(text: str, index: int) -> bool:
    """Return True when the tab at ``index`` can only be white space between tokens (not part of a value).

    The rules do not depend on where the tab is in the document: a tab at the end of a line, before a
    ``#`` comment, before the ``:`` of a key, or after a ``:`` (a ``:`` followed by white space is never
    part of a plain value). A tab after ``-`` or ``?`` is a separator only when the scanner read that
    character as an indicator: in the continuation line of a plain value (``name: a`` then ``  -<TAB>b``)
    it is part of the value.
    """
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    line_end = len(text) if line_end == -1 else line_end
    before, after = text[line_start:index], text[index + 1 : line_end].lstrip(" \t")
    previous = before.rstrip(" \t")
    if not previous:
        return not after  # A line with only white space; a tab as indentation is an error for pi too.
    if not after or after.startswith("#"):
        return True  # The end of the line, or a comment.
    if previous.endswith(":"):
        return True  # After the ``:`` of a key.
    if after.startswith(":") and (len(after) == 1 or after[1] in " \t"):
        return True  # Before the ``:`` of a key.
    if previous.endswith(("-", "?")):
        # (Another indicator after the tab, ``-<TAB>- a``, makes a nested entry; pi rejects it.)
        nested = after[:1] in ("-", "?", ":") and (len(after) == 1 or after[1] in " \t")
        return not nested and _indicator_ends_at(text, line_start + len(previous))
    return False


def _indicator_ends_at(text: str, end: int) -> bool:
    """Return True when the scanner reads the characters that end at ``end`` as a ``-`` or ``?`` indicator."""
    loader = YAML(typ="safe", pure=True)
    loader.version = (1, 2)
    last: tuple[type, int] | None = None
    try:
        for token in loader.scan(text):
            if token.end_mark.index <= end:
                last = (type(token), token.end_mark.index)
    except YAMLError:
        pass
    return last is not None and last[1] == end and last[0] in (BlockEntryToken, KeyToken)


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #
@dataclass
class _Frame:
    kind: str  # "mapping" or "sequence"
    tag: str | None
    anchor: str | None
    keys: set[Any] = field(default_factory=set)
    pending_key: Node | None = None
    pairs: int = 0
    first_key: Node | None = None
    all_values_null: bool = True
    items: list[Node] = field(default_factory=list)
    strings: dict[str, str] = field(default_factory=dict)


def _top_level_strings(text: str) -> dict[str, str]:
    """Parse ``text`` (one YAML 1.2 document) and check it; return the string pairs of its top-level mapping.

    Raises :class:`YAMLError` (also when a key is repeated, or a tag rule is broken) or
    :class:`ValueError` (several documents, an alias with no anchor).
    """
    loader = YAML(typ="safe", pure=True)
    loader.version = (1, 2)
    anchors: dict[str, Node] = {}
    stack: list[_Frame] = []
    root: Node | None = None
    root_strings: dict[str, str] = {}
    documents = 0
    for event in loader.parse(text):
        if isinstance(event, DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise ValueError("Source contains multiple documents")
            continue
        node: Node
        is_alias = False
        if isinstance(event, MappingStartEvent | SequenceStartEvent):
            kind = "mapping" if isinstance(event, MappingStartEvent) else "sequence"
            stack.append(_Frame(kind, event.tag, event.anchor))
            if event.anchor:
                # An alias inside the collection can refer to it (``&m [*m]``): the anchor exists from the start.
                anchors[event.anchor] = ("mapping", (0, None)) if kind == "mapping" else ("sequence", None)
            continue
        if isinstance(event, MappingEndEvent | SequenceEndEvent):
            frame = stack.pop()
            node = _finish_collection(frame)
            if len(stack) == 0:
                root_strings = frame.strings if frame.kind == "mapping" else {}
            if frame.anchor:
                anchors[frame.anchor] = node
        elif isinstance(event, AliasEvent):
            if event.anchor not in anchors:
                raise ValueError(f"Unresolved alias *{event.anchor}")
            node = anchors[event.anchor]
            is_alias = True
        elif isinstance(event, ScalarEvent):
            node = _resolve_scalar(event)
            if event.anchor:
                anchors[event.anchor] = node
        else:
            continue
        if stack:
            _add_child(stack[-1], node, is_alias=is_alias)
        else:
            root = node
    return root_strings if root is not None and root[0] == "mapping" else {}


def _add_child(frame: _Frame, node: Node, *, is_alias: bool = False) -> None:
    if frame.kind == "sequence":
        frame.items.append(node)
        return
    if frame.pending_key is None:
        # (An alias is a node of its own in the ``yaml`` package: it is never the same key as another node.)
        identity = None if is_alias else _key_identity(node)
        if identity is not None:
            if identity in frame.keys:
                raise ValueError("Map keys must be unique")
            frame.keys.add(identity)
        frame.pending_key = node
        if frame.first_key is None:
            frame.first_key = node
        return
    key = frame.pending_key
    frame.pending_key = None
    frame.pairs += 1
    if node[0] != "null":
        frame.all_values_null = False
    if key[0] == "str" and node[0] == "str":
        frame.strings[key[1]] = node[1]


def _finish_collection(frame: _Frame) -> Node:
    """Check the rules of the tag of a collection (a tag for the other kind of collection is ignored by pi)."""
    if frame.pending_key is not None:  # A key with no value: its value is null.
        _add_child(frame, ("null", None))
    if frame.kind == "mapping":
        if frame.tag == _SET_TAG and not frame.all_values_null:
            raise YAMLError("Set items must all have null values")
        return ("mapping", (frame.pairs, frame.first_key))
    if frame.tag in (_OMAP_TAG, _PAIRS_TAG):
        seen: set[Any] = set()
        for item in frame.items:
            key: Node | None = item
            if item[0] == "mapping":
                pairs, first_key = item[1]
                if pairs > 1:
                    raise YAMLError("Each pair must have its own sequence indicator")
                key = first_key if first_key is not None else ("null", None)
            if frame.tag == _OMAP_TAG and key is not None and key[0] not in ("mapping", "sequence"):
                identity = _omap_key_identity(key)
                if identity in seen:
                    raise YAMLError("Ordered maps must not include duplicate keys")
                seen.add(identity)
    return ("sequence", None)


def _key_identity(node: Node) -> Any:
    """What makes two mapping keys the same for pi (``a.value === b.value``); ``None``: never the same."""
    kind, value = node
    if kind == "other" or (kind == "float" and value != value):
        return None
    if kind in ("mapping", "sequence"):
        return None  # A collection is a node of its own.
    return ("number", value) if kind in ("int", "float") else node


def _omap_key_identity(node: Node) -> Any:
    """Like :func:`_key_identity` for ``!!omap`` (``Array.includes``: NaN is the same as NaN)."""
    if node[0] == "float" and node[1] != node[1]:
        return ("nan",)
    identity = _key_identity(node)
    return identity if identity is not None else ("other", id(node))


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #
def _resolve_scalar(event: ScalarEvent) -> Node:
    """Return ``(kind, value)`` of a scalar by the core schema of YAML 1.2 (and the tag, if there is one)."""
    text = event.value
    if event.tag is not None:
        return _resolve_tagged_scalar(event.tag, text)
    if event.style:  # Quoted or block scalars are strings.
        return ("str", text)
    return _resolve_plain_scalar(text) or ("str", text)


def _js_number(value: int | float) -> float:
    """The JavaScript number for a YAML number (a number above 2**53 loses digits, above 1e308 is infinity)."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _resolve_plain_scalar(text: str) -> Node | None:
    """Return ``(kind, value)`` when ``text`` is a null, bool, int or float of the core schema, else ``None``."""
    if text in _YAML_NULL:
        return ("null", None)
    if text in _YAML_BOOL:
        return ("bool", _YAML_BOOL[text])
    if _YAML_INT_RE.fullmatch(text):
        prefix = text[:2]
        number = int(text[2:], 8 if prefix == "0o" else 16) if prefix in ("0o", "0x") else int(text)
        return ("int", _js_number(number))
    if _YAML_INF_RE.fullmatch(text) or text in _YAML_NAN:
        return ("float", float(text.replace(".", "").replace("NaN", "nan").lower()))
    if _YAML_FLOAT_RE.fullmatch(text):
        return ("float", float(text))
    return None


def _resolve_tagged_scalar(tag: str, text: str) -> Node:
    """Return ``(kind, value)`` of a scalar with an explicit tag, as the ``yaml`` package of pi does.

    A core tag (``!!int``, ``!!float``, ``!!bool``, ``!!null``) gives that type
    when the text has the form of it, and a string if not (``!!int x`` is the
    string ``x``). The types that the package knows besides the core schema
    (``!!binary``, ``!!timestamp``) are not strings. Any other tag leaves the
    text as a string (pi only warns).
    """
    if tag in ("!", _YAML_STR_TAG):
        return ("str", text)
    if tag in ("!!", _YAML_PREFIX):
        raise ValueError("Invalid tag")
    if tag in _YAML_NON_STRING_TAGS:
        return ("other", None)
    kind = tag.removeprefix(_YAML_PREFIX) if tag.startswith(_YAML_PREFIX) else None
    plain = _resolve_plain_scalar(text)
    if kind in ("int", "float", "bool", "null") and plain is not None and plain[0] == kind:
        return plain
    return ("str", text)
