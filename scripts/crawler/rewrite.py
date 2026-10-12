"""Normalize ReadMe-flavored Markdown so it stands on its own outside the hub.

Why: ReadMe bodies mix CommonMark with ReadMe-only syntax that means nothing
outside the hub - link macros, legacy JSON "magic blocks" and reusable-content
placeholders. The first version of this module deleted ``[block:html]`` bodies
(emptying whole guides), left every other legacy block as raw JSON, and ran
its regexes inside code samples (stripping XML such as ``<Loan/>``). This
version turns all of it into plain Markdown and never touches code.

Transforms, applied only outside fenced code blocks and inline code spans:

  - Link macros ``(doc|ref|changelog|page):slug[#anchor]`` become absolute
    URLs under ``<base_url>/<section>/{docs,reference,changelog,page}/``. They
    are handled in Markdown links (including the ``(<ref: slug>)`` and
    escaped ``]\\(ref: slug)`` spellings found in real bodies), in link
    reference definitions and in ``href="doc:..."`` attributes of HTML.
  - Root-relative link targets (``[x](/developer-connect/docs/y)``,
    ``[x]: /path``, ``href="/path"``, ``src="/path"``), which converted HTML
    is full of, are made absolute against ``base_url``: a chunk read outside
    the site cannot resolve them.
  - Legacy blocks ``[block:TYPE]{json}[/block]`` become Markdown: parameters
    -> GFM table, callout -> blockquote, api-header -> ``## title``, image ->
    ``![caption](url)``, code -> fenced blocks, html -> html_to_markdown,
    embed -> link (+ converted html). Unknown types keep their human-readable
    strings and malformed JSON keeps its inner text: nothing is dropped.
  - Self-closing component placeholders ``<Name />`` are replaced by the
    reusable content the caller supplies for ``Name`` (rewritten recursively,
    at most MAX_REUSABLE_DEPTH levels, cycle-safe) and removed otherwise.
    ReadMe's MDX built-ins ``<Image src=... />`` and ``<Embed url=... />``
    become an image / a link instead of being removed.
  - Runs of blank lines collapse to one blank line; the result is stripped.

``rewrite_body`` is idempotent: running it on its own output changes nothing.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .htmlmd import collapse_blank_lines, fence_for, html_to_markdown, split_fenced_code

__all__ = [
    "MAX_REUSABLE_DEPTH",
    "find_component_placeholders",
    "format_table_cell",
    "rewrite_body",
]

# Nesting limit for reusable content that itself contains placeholders.
MAX_REUSABLE_DEPTH = 3

KIND_TO_SEGMENT = {"doc": "docs", "ref": "reference", "changelog": "changelog", "page": "page"}

_KIND = r"(?P<kind>doc|ref|changelog|page)"
_SLUG = r"(?P<slug>[A-Za-z0-9][A-Za-z0-9._~-]*)"
_ANCHOR = r"(?P<anchor>#[^\s()<>\"']*)?"

# "(doc:slug#a)", "(<ref: slug>)", "]\(ref: slug)" (escaped by ReadMe's
# editor), optionally with a link title: "(doc:slug "Title")".
LINK_MACRO_RE = re.compile(
    r"(?:(?<=\])\\)?\(\s*(?P<lt><)?\s*" + _KIND + r":\s*" + _SLUG + _ANCHOR
    + r"\s*(?(lt)>)\s*(?P<title>\"[^\"\n]*\"|'[^'\n]*')?\s*\)"
)
# "[label]: doc:slug" link reference definitions.
_REF_DEF_MACRO_RE = re.compile(
    r"^(?P<lead>[ \t]{0,3}\[[^\]\n]+\]:[ \t]*)(?P<lt><)?" + _KIND + r":" + _SLUG + _ANCHOR
    + r"(?(lt)>)",
    re.M,
)
# href="doc:slug" in inline HTML and in [block:html] bodies.
_HREF_MACRO_RE = re.compile(
    r"(?P<attr>\bhref\s*=\s*)(?P<q>[\"'])\s*" + _KIND + r":\s*" + _SLUG + _ANCHOR + r"\s*(?P=q)"
)
# Root-relative targets ("/x", never protocol-relative "//host/x") in Markdown
# links/images, link reference definitions and href/src attributes.
_ROOT_REL_LINK_RE = re.compile(r"(?P<lead>\]\(\s*<?)(?=/(?![/\\]))")
_ROOT_REL_DEF_RE = re.compile(r"^(?P<lead>[ \t]{0,3}\[[^\]\n]+\]:[ \t]*<?)(?=/(?![/\\]))", re.M)
_ROOT_REL_ATTR_RE = re.compile(r"(?P<lead>\b(?:href|src)\s*=\s*[\"'])(?=/(?![/\\]))")

# Self-closing PascalCase component, with optional JSX-style attributes.
_ATTR = r"[A-Za-z_][\w.:-]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|\{[^{}]*\}|[^\s\"'=<>`{}/]+))?"
COMPONENT_RE = re.compile(
    r"(?<!\\)<(?P<name>[A-Z][A-Za-z0-9_]*)(?P<attrs>(?:\s+" + _ATTR + r")*)\s*/>"
)
_ATTR_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w.:-]*)\s*=\s*(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'"
    r"|\{\s*(?:\"(?P<edq>[^\"]*)\"|'(?P<esq>[^']*)'|`(?P<ebq>[^`]*)`)\s*\})"
)
# ReadMe MDX built-ins rendered in place rather than treated as reusable content.
_BUILTIN_COMPONENTS = frozenset({"Image", "Embed"})

LEGACY_BLOCK_RE = re.compile(
    r"\[block:(?P<type>[A-Za-z0-9_-]+)\](?P<inner>.*?)\[/block\]", re.S
)
# Container prefix of a line: blockquote markers, indentation, one list marker.
_CONTAINER_PREFIX_RE = re.compile(
    r"^(?P<quote>(?:[ \t]*>[ \t]?)*)(?P<indent>[ \t]*)(?P<marker>(?:[-*+]|\d{1,9}[.)])[ \t]+)?$"
)
_QUOTE_MARKERS_RE = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.M)

_BACKTICKS_RE = re.compile(r"`+")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_CELL_NEWLINES_RE = re.compile(r"[ \t]*\r?\n[ \t]*")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_CELL_KEY_RE = re.compile(r"^(?:h|(?P<row>\d+))-(?P<col>\d+)$")

_ALIGN = {"left": ":---", "center": ":---:", "right": "---:"}

# Keys and values that are presentation, not prose, in unknown blocks.
_SKIP_STRING_KEYS = frozenset(
    {"align", "sizing", "border", "theme", "icon", "emoji", "id", "_id", "typeofembed", "favicon"}
)
_NON_PROSE_RE = re.compile(r"^(?:#[0-9A-Fa-f]{3,8}|-?\d+(?:\.\d+)?(?:px|%|em|rem)?)$")


@dataclass(frozen=True)
class _Ctx:
    base: str  # <base_url>/<section>, for link macros
    reusable: Mapping[str, str] = field(default_factory=dict)
    site: str = ""  # <base_url>, for root-relative links ("" = leave them alone)


# --- Public API -------------------------------------------------------------


def rewrite_body(
    body: str,
    *,
    base_url: str,
    section: str,
    reusable: Optional[Mapping[str, str]] = None,
) -> str:
    """Return ``body`` as self-contained Markdown (see the module docstring)."""
    site = base_url.rstrip("/")
    ctx = _Ctx(base=f"{site}/{section}", reusable=reusable or {}, site=site)
    return _rewrite(body, ctx, depth=0, active=frozenset())


def find_component_placeholders(body: str) -> List[str]:
    """Names of ``<Name />`` placeholders outside code, in order, de-duplicated.

    Placeholders inside legacy blocks count the way rewrite_body sees them
    (after the blocks are converted), so a name inside a ``[block:code]``
    sample is not reported. MDX built-ins (Image, Embed) are not reusable
    content and are not reported either.
    """
    if not body:
        return []
    text = _convert_legacy_blocks(body, _Ctx(base=""))
    names: List[str] = []
    for piece in _prose_pieces(text):
        for m in COMPONENT_RE.finditer(piece):
            name = m["name"]
            if name not in _BUILTIN_COMPONENTS and name not in names:
                names.append(name)
    return names


def format_table_cell(value: Any) -> str:
    """Render a ``[block:parameters]`` cell for a one-line GFM table cell.

    Strips the value, turns line breaks into ``<br>`` and escapes unescaped
    pipes. Markdown inside the cell is kept as-is.
    """
    if value is None or isinstance(value, bool):
        return "" if value is None else str(value).lower()
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = _CELL_NEWLINES_RE.sub("<br>", str(value).strip())
    return _UNESCAPED_PIPE_RE.sub(r"\\|", text)


# --- Driver -----------------------------------------------------------------


def _rewrite(body: Optional[str], ctx: _Ctx, *, depth: int, active: FrozenSet[str]) -> str:
    if not body or not body.strip():
        return ""
    text = _convert_legacy_blocks(body, ctx)
    text = _rewrite_prose(text, ctx, depth=depth, active=active)
    return collapse_blank_lines(text).strip()


def _split_inline_code(text: str) -> List[Tuple[bool, str]]:
    """Split text into ``(is_code, piece)`` around CommonMark inline code spans.

    A run of N backticks opens a span that closes at the next run of exactly
    N backticks; spans never cross a blank line. Unmatched runs are literal.
    """
    if "`" not in text:
        return [(False, text)]
    # Index runs by width so finding a closer is a bisect, not a rescan:
    # bodies can hold thousands of stray backticks (minified samples).
    runs = [(m.start(), m.end()) for m in _BACKTICKS_RE.finditer(text)]
    by_width: Dict[int, List[int]] = {}
    for idx, (start, end) in enumerate(runs):
        by_width.setdefault(end - start, []).append(idx)
    blanks = [m.start() for m in _BLANK_LINE_RE.finditer(text)]

    out: List[Tuple[bool, str]] = []
    pos = i = 0
    while i < len(runs):
        start, end = runs[i]
        if start and text[start - 1] == "\\":  # escaped backtick: literal
            i += 1
            continue
        b = bisect_left(blanks, end)
        limit = blanks[b] if b < len(blanks) else len(text)
        same = by_width[end - start]
        k = bisect_right(same, i)
        if k == len(same) or runs[same[k]][0] >= limit:
            i += 1  # unmatched run is literal text
            continue
        close_end = runs[same[k]][1]
        if start > pos:
            out.append((False, text[pos:start]))
        out.append((True, text[start:close_end]))
        pos = close_end
        i = same[k] + 1
    if pos < len(text):
        out.append((False, text[pos:]))
    return out


def _prose_pieces(text: str) -> List[str]:
    """Text outside fenced code and inline code (for read-only scans)."""
    return [
        piece
        for is_code, chunk in split_fenced_code(text)
        if not is_code
        for in_code, piece in _split_inline_code(chunk)
        if not in_code
    ]


def _continuation(prefix: str) -> Optional[str]:
    """Prefix that keeps later lines inside the container ``prefix`` opened.

    ``"> "`` -> ``"> "``, ``"   "`` -> ``"   "``, ``"1. "`` -> ``"   "``.
    Returns None when the prefix contains prose (the insertion is mid-line).
    """
    m = _CONTAINER_PREFIX_RE.match(prefix)
    if m is None:
        return None
    return m["quote"] + m["indent"] + " " * len(m["marker"] or "")


def _continue_lines(md: str, cont: str) -> str:
    if not cont:
        return md
    lines = md.split("\n")
    blank = cont.rstrip()
    return "\n".join([lines[0]] + [cont + ln if ln.strip() else blank for ln in lines[1:]])


def _line_context(s: str, start: int, end: int) -> Tuple[str, str]:
    """Text before ``start`` and after ``end`` on the same line of ``s``."""
    line_start = s.rfind("\n", 0, start) + 1
    line_end = s.find("\n", end)
    return s[line_start:start], s[end:len(s) if line_end < 0 else line_end]


def _place_block(md: str, s: str, start: int, end: int) -> str:
    """Replacement for ``s[start:end]`` that keeps multi-line ``md`` well-formed.

    Alone on its line, the content follows the line's container (list item,
    blockquote) and is separated from neighbouring paragraphs by blank lines.
    Next to other text, a one-line result stays inline and a multi-line one
    is split out into its own paragraph.
    """
    prefix, suffix = _line_context(s, start, end)
    cont = _continuation(prefix)
    if "\n" not in md and (cont is None or suffix.strip()):
        return md
    if cont is None:
        return f"\n\n{md}\n\n"
    trailer = "\n" + cont if suffix.strip() else ""
    if not cont:
        return f"\n{md}\n{trailer}"
    return _continue_lines(md, cont) + trailer


# --- Link macros and placeholders ---------------------------------------------


def _macro_url(base: str, m: "re.Match[str]") -> str:
    return f"{base}/{KIND_TO_SEGMENT[m['kind']]}/{m['slug']}{m['anchor'] or ''}"


def _rewrite_link_macros(text: str, base: str) -> str:
    if ":" not in text:
        return text

    def link(m: "re.Match[str]") -> str:
        title = f" {m['title']}" if m["title"] else ""
        return f"({_macro_url(base, m)}{title})"

    text = LINK_MACRO_RE.sub(link, text)
    text = _REF_DEF_MACRO_RE.sub(lambda m: m["lead"] + _macro_url(base, m), text)
    return _rewrite_href_macros(text, base)


def _absolutize_root_relative(text: str, site: str) -> str:
    """Prefix root-relative link targets with ``site`` (already absolute ones,
    macros and protocol-relative ``//host`` URLs are left alone)."""
    if not site or "/" not in text:
        return text

    def lead(m: "re.Match[str]") -> str:
        return m["lead"] + site

    text = _ROOT_REL_LINK_RE.sub(lead, text)
    text = _ROOT_REL_DEF_RE.sub(lead, text)
    return _ROOT_REL_ATTR_RE.sub(lead, text)


def _rewrite_href_macros(html: str, base: str) -> str:
    return _HREF_MACRO_RE.sub(
        lambda m: f"{m['attr']}{m['q']}{_macro_url(base, m)}{m['q']}", html
    )


def _parse_attrs(attrs: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _ATTR_VALUE_RE.finditer(attrs or ""):
        value = next((v for v in m.group("dq", "sq", "edq", "esq", "ebq") if v is not None), "")
        out[m["key"]] = value
    return out


def _builtin_component(name: str, attrs: str) -> str:
    a = _parse_attrs(attrs)
    if name == "Image" and a.get("src"):
        alt = a.get("alt") or a.get("caption") or a.get("title") or ""
        return _image_md(a["src"], alt)
    if name == "Embed" and (a.get("url") or a.get("href")):
        url = a.get("url") or a.get("href") or ""
        return f"[{_escape_label(_one_line(a.get('title') or url))}]({_url(url)})"
    return ""


def _rewrite_prose(text: str, ctx: _Ctx, *, depth: int, active: FrozenSet[str]) -> str:
    """Rewrite macros and placeholders outside code; insert reusable content.

    Expanded content is already fully rewritten, so it is parked behind a
    token while the surrounding text is processed and placed afterwards with
    its line context known (it must not be rewritten a second time).
    """
    inserts: List[str] = []
    token = ""

    def component(m: "re.Match[str]") -> str:
        nonlocal token
        name = m["name"]
        content = ""
        nested = ctx.reusable.get(name)
        if isinstance(nested, str) and name not in active and depth < MAX_REUSABLE_DEPTH:
            content = _rewrite(nested, ctx, depth=depth + 1, active=active | {name})
        elif name in _BUILTIN_COMPONENTS:
            content = _builtin_component(name, m["attrs"])
        if not content:
            return ""
        if not token:
            token = _unused_char(text)
        inserts.append(content)
        return f"{token}{len(inserts) - 1}{token}"

    out: List[str] = []
    for is_code, chunk in split_fenced_code(text):
        if is_code:
            out.append(chunk)
            continue
        for in_code, piece in _split_inline_code(chunk):
            if not in_code:
                piece = _rewrite_link_macros(piece, ctx.base)
                piece = _absolutize_root_relative(piece, ctx.site)
                piece = COMPONENT_RE.sub(component, piece)
            out.append(piece)
    result = "".join(out)
    if inserts:
        token_re = re.compile(re.escape(token) + r"(\d+)" + re.escape(token))
        result = token_re.sub(
            lambda m: _place_block(inserts[int(m[1])], m.string, m.start(), m.end()), result
        )
    return result


def _unused_char(text: str) -> str:
    """A private-use character absent from ``text``, usable as a token."""
    for cp in range(0xE000, 0xF900):
        if chr(cp) not in text:
            return chr(cp)
    raise ValueError("no free private-use character for placeholder tokens")


# --- Legacy blocks --------------------------------------------------------------


def _convert_legacy_blocks(text: str, ctx: _Ctx) -> str:
    if "[block:" not in text:
        return text
    return "".join(
        chunk
        if is_code or "[block:" not in chunk
        else LEGACY_BLOCK_RE.sub(lambda m: _replace_block(m, ctx), chunk)
        for is_code, chunk in split_fenced_code(text)
    )


def _replace_block(m: "re.Match[str]", ctx: _Ctx) -> str:
    inner = m["inner"]
    prefix, _ = _line_context(m.string, m.start(), m.end())
    cont = _continuation(prefix)
    if cont and ">" in cont:  # block inside a blockquote: unquote its JSON
        inner = _QUOTE_MARKERS_RE.sub("", inner)
    md = _render_block(m["type"], inner, ctx)
    return _place_block(md, m.string, m.start(), m.end())


def _render_block(block_type: str, inner: str, ctx: _Ctx) -> str:
    try:
        data = json.loads(inner)
    except ValueError:
        return inner.strip()
    renderer = _BLOCK_RENDERERS.get(block_type.lower())
    md: Optional[str] = None
    if renderer is not None:
        try:
            md = renderer(data, ctx)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            md = None
    if md is None:  # unknown type or unexpected shape: keep the prose
        md = _render_strings(data)
    return md.strip()


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _escape_label(text: str) -> str:
    return text.replace("[", r"\[").replace("]", r"\]")


def _url(url: str) -> str:
    return url.strip().replace(" ", "%20")


def _image_md(url: str, alt: str) -> str:
    return f"![{_escape_label(_one_line(alt))}]({_url(url)})"


def _as_count(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _render_parameters(data: Any, ctx: _Ctx) -> Optional[str]:
    cells = data.get("data")
    if not isinstance(cells, dict):
        return None
    max_row = max_col = -1
    for key in cells:
        km = _CELL_KEY_RE.match(str(key))
        if km:
            max_col = max(max_col, int(km["col"]))
            if km["row"] is not None:
                max_row = max(max_row, int(km["row"]))
    cols = _as_count(data.get("cols"))
    rows = _as_count(data.get("rows"))
    cols = max_col + 1 if cols is None else cols
    rows = max_row + 1 if rows is None else rows
    if cols <= 0:
        return None
    align = data.get("align") if isinstance(data.get("align"), list) else []

    def row(values: List[str]) -> str:
        return "| " + " | ".join(values) + " |"

    lines = [
        row([format_table_cell(cells.get(f"h-{c}")) for c in range(cols)]),
        row([_ALIGN.get(align[c] if c < len(align) else "", "---") for c in range(cols)]),
    ]
    for r in range(max(rows, 0)):
        lines.append(row([format_table_cell(cells.get(f"{r}-{c}")) for c in range(cols)]))
    return "\n".join(lines)


def _render_callout(data: Any, ctx: _Ctx) -> Optional[str]:
    kind = _one_line(_text(data.get("type")))
    title = _one_line(_text(data.get("title")))
    body = _text(data.get("body"))
    if not (title or body):
        return None
    head = ": ".join(x for x in (kind[:1].upper() + kind[1:], title) if x)
    lines: List[str] = [f"**{head}**"] if head else []
    if body:
        if lines:
            lines.append("")
        lines.extend(body.split("\n"))
    return "\n".join("> " + ln if ln.strip() else ">" for ln in lines)


def _render_api_header(data: Any, ctx: _Ctx) -> Optional[str]:
    title = _one_line(_text(data.get("title")))
    return f"## {title}" if title else None


def _render_image(data: Any, ctx: _Ctx) -> Optional[str]:
    images = data.get("images")
    if not isinstance(images, list):
        return None
    out: List[str] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        img = item.get("image")
        url = name = img_alt = ""
        if isinstance(img, list):
            url = _text(img[0]) if img else ""
            name = _text(img[1]) if len(img) > 1 else ""
            # [url, file name, alt text] in newer bodies; older ones put
            # width, height and colour there ([url, name, 993, 403, "#fff"]).
            third = img[2] if len(img) > 2 and isinstance(img[2], str) else ""
            if third.strip() and not _NON_PROSE_RE.match(third.strip()):
                img_alt = third.strip()
        elif isinstance(img, str):
            url = img.strip()
        url = url or _text(item.get("url")) or _text(item.get("src"))
        alt = _text(item.get("caption")) or _text(item.get("alt")) or img_alt or name
        if url:
            out.append(_image_md(url, alt))
        elif alt:
            out.append(_one_line(alt))
    return "\n\n".join(out)


def _render_code(data: Any, ctx: _Ctx) -> Optional[str]:
    codes = data.get("codes")
    if not isinstance(codes, list):
        return None
    out: List[str] = []
    for entry in codes:
        if isinstance(entry, str):
            code, language, name = entry, "", ""
        elif isinstance(entry, dict):
            code = entry.get("code") if isinstance(entry.get("code"), str) else _text(entry.get("code"))
            language, name = _text(entry.get("language")), _one_line(_text(entry.get("name")))
        else:
            continue
        code = code.strip("\n")
        if not code.strip() and not name:
            continue
        fence = fence_for(code)
        lang = (language.split() or [""])[0].replace("`", "")
        block = f"{fence}{lang}\n{code}\n{fence}"
        out.append(f"**{name}**\n\n{block}" if name else block)
    return "\n\n".join(out)


def _render_html(data: Any, ctx: _Ctx) -> Optional[str]:
    html = data.get("html")
    if not isinstance(html, str):
        return None
    return html_to_markdown(_rewrite_href_macros(html, ctx.base))


def _render_embed(data: Any, ctx: _Ctx) -> Optional[str]:
    url = _text(data.get("url")) or _text(data.get("href"))
    title = _one_line(_text(data.get("title")))
    parts: List[str] = []
    if url:
        parts.append(f"[{_escape_label(title or url)}]({_url(url)})")
    elif title:
        parts.append(title)
    html = data.get("html")
    if isinstance(html, str) and html.strip():
        converted = html_to_markdown(_rewrite_href_macros(html, ctx.base))
        if converted:
            parts.append(converted)
    return "\n\n".join(parts) if parts else None


def _render_strings(data: Any) -> str:
    """Human-readable string values of arbitrary JSON, as paragraphs."""
    found: List[str] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                k = str(k)
                if k.lower() not in _SKIP_STRING_KEYS and not k.lower().endswith("color"):
                    walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str):
            text = value.strip()
            if text and not _NON_PROSE_RE.match(text) and text not in found:
                found.append(text)

    walk(data)
    return "\n\n".join(found)


_BLOCK_RENDERERS: Dict[str, Callable[[Any, _Ctx], Optional[str]]] = {
    "parameters": _render_parameters,
    "callout": _render_callout,
    "api-header": _render_api_header,
    "image": _render_image,
    "code": _render_code,
    "html": _render_html,
    "embed": _render_embed,
}
