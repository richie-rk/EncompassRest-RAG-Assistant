"""HTML -> Markdown for ReadMe HTML content, without losing visible text.

Why this exists: some of the richest Developer Connect content is stored as raw
HTML - whole guides live inside a single ``[block:html]`` and custom pages
(api-key, saml-sso, lender-mfa, ...) are HTML documents. The first crawler
simply deleted those bodies, which emptied several guides. markdownify does
the heavy lifting; this module adds the few tweaks needed for clean,
retrieval-friendly Markdown:

  - script/style/svg/noscript/iframe/template, the head-only elements
    (title/meta/link/base) and form controls (button, input, select/option,
    textarea) are removed together with their contents: their text is code or
    UI chrome, not documentation. <head> itself is only unwrapped: HTML5 lets
    a document omit ``</head>``, and html.parser then nests the whole body
    inside it;
  - <pre> becomes a fenced block whose fence is longer than any backtick run
    inside the code and whose info string carries the language from
    ``language-*`` / ``lang-*`` class names;
  - table cells escape "|" and keep line breaks as ``<br>`` so every GFM row
    stays on one line; images inside cells keep their URL; tables without a
    header row use their first row as the header instead of an empty one;
  - ``data:`` image URIs (inline base64) are reduced to their alt text;
  - text that looks like a tag (``&lt;Loan/&gt;`` in the HTML) is
    backslash-escaped so a later pass cannot mistake it for a ReadMe
    component placeholder and strip it;
  - runs of blank lines are collapsed, but never inside fenced code.

The fenced-code scanner lives here as well because rewrite.py needs the exact
same notion of "inside code" for its own rewrites, and it already depends on
this module (the dependency only goes one way).
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

__all__ = ["collapse_blank_lines", "fence_for", "html_to_markdown", "split_fenced_code"]


# --- Fenced code ------------------------------------------------------------

# Opening fence: optional container prefix (indentation and any mix of
# blockquote markers and list markers, e.g. "- > " or "- - " for a fence
# opening a quoted or nested list item) then 3+ backticks or tildes. Real
# bodies have fences indented inside list items and quoted inside
# blockquotes ("> ```json"), and html_to_markdown emits "- > ```" and
# "- - ```" for <pre> at the start of a nested <li>/<blockquote>.
_FENCE_OPEN_RE = re.compile(
    r"^(?:[ \t]*(?:>|(?:[-*+]|\d{1,9}[.)])[ \t]))*[ \t]*"
    r"(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$"
)
_FENCE_CLOSE_RE = re.compile(r"^(?:[ \t]*>)*[ \t]*(?P<fence>`{3,}|~{3,})[ \t\r]*$")


def split_fenced_code(text: str) -> List[Tuple[bool, str]]:
    """Split Markdown into ``(is_code, chunk)`` pieces; ``"".join`` restores it.

    A code chunk runs from the first character of its opening fence line to
    the last character of its closing fence line (the newlines around it stay
    in the neighbouring text chunks, so blank-line handling there sees them).
    The closing fence must use the same character and be at least as long as
    the opening one; an unclosed fence runs to the end of the text, as in
    CommonMark. Fences are only closed by a fence line - never by container
    heuristics - because wrongly ending code early is what corrupts it.
    """
    out: List[Tuple[bool, str]] = []
    pos = 0  # start of the pending chunk
    offset = 0  # start of the current line
    fence_char = ""
    fence_len = 0
    for line in text.split("\n"):
        end = offset + len(line)
        if not fence_char:
            m = _FENCE_OPEN_RE.match(line)
            # A backtick fence's info string may not contain backticks,
            # otherwise "```x```" (an inline span) would open a block.
            if m and not (m["fence"][0] == "`" and "`" in m["info"]):
                if offset > pos:
                    out.append((False, text[pos:offset]))
                pos = offset
                fence_char, fence_len = m["fence"][0], len(m["fence"])
        else:
            m = _FENCE_CLOSE_RE.match(line)
            if m and m["fence"][0] == fence_char and len(m["fence"]) >= fence_len:
                out.append((True, text[pos:end]))
                pos = end
                fence_char, fence_len = "", 0
        offset = end + 1
    if pos < len(text):
        out.append((bool(fence_char), text[pos:]))
    return out


# Two or more blank (or whitespace-only) lines in a row -> one blank line.
_BLANK_RUN_RE = re.compile(r"\n(?:[ \t]*\n){2,}")


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines to a single blank line outside fenced code."""
    return "".join(
        chunk if is_code else _BLANK_RUN_RE.sub("\n\n", chunk)
        for is_code, chunk in split_fenced_code(text)
    )


def fence_for(code: str, char: str = "`") -> str:
    """Return a fence longer than any run of ``char`` inside ``code`` (min 3)."""
    longest = max((len(r) for r in re.findall(re.escape(char) + "+", code)), default=0)
    return char * max(3, longest + 1)


# --- HTML -> Markdown -------------------------------------------------------

# Removed with their contents. Form controls are UI chrome; title/meta/link/
# base are not visible page text; the rest is code or embedded media.
# <head> is NOT here: see html_to_markdown.
_DROP_TAGS = (
    "script", "style", "svg", "noscript", "iframe", "template", "object",
    "title", "meta", "link", "base",
    "button", "input", "select", "option", "optgroup", "textarea", "datalist",
)

_LANG_CLASS_RE = re.compile(r"^(?:language|lang)-(?P<lang>[\w#+.-]+)$")
_TAG_LIKE_LT_RE = re.compile(r"<(?=[A-Za-z/!?])")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_CELL_NEWLINES_RE = re.compile(r"[ \t]*\n+[ \t]*")
_WS_ONLY_LINE_RE = re.compile(r"^[ \t]+$", re.M)


def _code_language(el: Any) -> str:
    """Language from ``class="language-json"`` on <pre> or its <code> child."""
    candidates = [el]
    code = el.find("code")
    if code is not None:
        candidates.append(code)
    for node in candidates:
        for cls in node.get("class") or []:
            m = _LANG_CLASS_RE.match(cls)
            if m:
                return m["lang"]
    return ""


class _Converter(MarkdownConverter):
    """markdownify with the tweaks described in the module docstring."""

    def escape(self, text: str, parent_tags: Set[str]) -> str:
        text = super().escape(text, parent_tags)
        return _TAG_LIKE_LT_RE.sub(r"\\<", text)

    def convert_pre(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        code = text.strip("\n")
        if not code.strip():
            return ""
        fence = fence_for(code)
        return f"\n\n{fence}{_code_language(el)}\n{code}\n{fence}\n\n"

    def convert_br(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        if "pre" in parent_tags:
            return "\n"
        if "td" in parent_tags or "th" in parent_tags:
            return "<br>"
        return super().convert_br(el, text, parent_tags)

    def _cell(self, el: Any, text: str) -> str:
        colspan = 1
        span = str(el.get("colspan") or "")
        if span.isdigit():
            colspan = max(1, min(1000, int(span)))
        text = _CELL_NEWLINES_RE.sub("<br>", text.strip())
        text = _UNESCAPED_PIPE_RE.sub(r"\\|", text)
        return " " + text + " |" * colspan

    def convert_td(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        return self._cell(el, text)

    def convert_th(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        return self._cell(el, text)

    def convert_img(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        alt = " ".join((el.get("alt") or "").split())
        src = (el.get("src") or "").strip()
        if not src or src.startswith("data:"):
            return alt
        # markdownify reduces images in "inline" contexts (headings and table
        # cells) to their alt text; only headings should lose the URL.
        if "_inline" in parent_tags and not ({"td", "th"} & parent_tags):
            return alt
        title = el.get("title") or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        alt = alt.replace("[", r"\[").replace("]", r"\]")
        return f"![{alt}]({src.replace(' ', '%20')}{title_part})"

    def convert_a(self, el: Any, text: str, parent_tags: Set[str]) -> str:
        href = (el.get("href") or "").strip()
        if href.lower().startswith("javascript:"):
            return text
        return super().convert_a(el, text, parent_tags)


_OPTIONS = dict(
    heading_style="ATX",
    bullets="-",
    escape_underscores=False,  # field names like loan_id must stay searchable
    escape_asterisks=True,
    table_infer_header=True,
)


def html_to_markdown(html: Optional[str]) -> str:
    """Convert an HTML fragment or document to Markdown, keeping visible text."""
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(list(_DROP_TAGS)):
        if not getattr(tag, "decomposed", False):  # already gone with a parent
            tag.decompose()
    # Head-only elements are gone now. Without a "</head>" html.parser puts
    # <body> and everything after it inside <head>, so keep what is left.
    for head in soup.find_all("head"):
        head.unwrap()
    md = _Converter(**_OPTIONS).convert_soup(soup)
    md = "".join(
        chunk if is_code else _WS_ONLY_LINE_RE.sub("", chunk)
        for is_code, chunk in split_fenced_code(md)
    )
    return collapse_blank_lines(md).strip()
