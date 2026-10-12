"""``json.loads`` whose strings are always writable as UTF-8.

Standard library only, so text-only modules (rewrite.py) can use it without
importing the HTTP layer (sidebar.py -> fetch.py -> requests).

A JSON escape such as ``"\\ud83d"`` (an emoji cut in half by a JavaScript
substring) is valid JSON but decodes to a lone UTF-16 surrogate, which cannot
be encoded as UTF-8: writing such a record used to abort the crawl.
"""
from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["has_lone_surrogate_escape", "load_json", "scrub_surrogates"]

# Scanned left to right, so an escaped backslash ("\\\\") is consumed first
# and the "u" after it is never mistaken for the start of an escape (JSON
# "\\uD83D\uDE00" is a backslash, the letters "uD83D", then a lone low half).
# A high+low pair decodes to one character; any other surrogate escape is lone.
_SURROGATE_ESCAPE_RE = re.compile(
    r'\\\\'
    r'|\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}'
    r'|(?P<lone>\\u[dD][89a-fA-F][0-9a-fA-F]{2})'
)
_LONE_SURROGATE_RE = re.compile('[\ud800-\udfff]')


def has_lone_surrogate_escape(text: str) -> bool:
    """True if JSON ``text`` has a ``\\uXXXX`` escape that decodes to a lone surrogate."""
    return any(m["lone"] for m in _SURROGATE_ESCAPE_RE.finditer(text))


def scrub_surrogates(value: Any) -> Any:
    """``value`` with every lone UTF-16 surrogate replaced by U+FFFD.

    After ``json.loads`` valid pairs are already combined into one character,
    so any code point left in the surrogate range is a lone half.
    """
    if isinstance(value, str):
        return _LONE_SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, list):
        return [scrub_surrogates(v) for v in value]
    if isinstance(value, dict):
        return {scrub_surrogates(k): scrub_surrogates(v) for k, v in value.items()}
    return value


def load_json(text: str) -> Any:
    """``json.loads`` whose strings are always writable as UTF-8: a lone
    surrogate escape (``"\\ud83d"`` without its pair) becomes U+FFFD.
    Used for ssr-props and for the JSON inside legacy ``[block:*]`` bodies."""
    data = json.loads(text)
    if has_lone_surrogate_escape(text):
        data = scrub_surrogates(data)
    return data
