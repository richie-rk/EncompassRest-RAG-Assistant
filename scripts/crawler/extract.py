"""Build a normalized page record (plus discovery hints) from one page's ssr-props.

Input shapes encountered on Developer Connect:
  guide       -> ssr.doc            (type=basic, isReference=False)
  reference   -> ssr.doc            + ssr.oasDefinition when ReadMe has it
  changelog   -> ssr.changelogs[*]  on the index page (list shape)
                 ssr.changelog       on per-entry pages   (singular shape)
  custompage  -> ssr.custompage      (markdown ``body`` and/or raw ``html``)

Pages that return the ICE login form have no ssr-props at all; that case is
classified from the raw HTML (``auth_required``).

``extract_page`` parses ssr-props ONCE and returns a ``PageExtraction`` with
everything the crawler needs from the page: the record, the page's OpenAPI
document (stored once per distinct document in a registry, keyed by
``oas_id``, instead of being embedded in every record: ReadMe attaches the
whole API group's OAS to every reference page), the operationIds it lists,
the sidebars it carries and the internal links it contains.

Reusable content: ReadMe pages can embed shared snippets as ``<Name />``
components (``doc.reusableContent`` lists the names). Where ReadMe keeps the
snippet bodies inside ssr-props has not been verified against a live page, so
``find_reusable_content`` searches the whole ssr-props tree generically (see
its docstring). Names that cannot be located are listed per record in
``reusable_content_unresolved`` so the gap stays visible.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

from .htmlmd import html_to_markdown
from .oas import find_page_operation, iter_operations, oas_fingerprint
from .rewrite import MAX_REUSABLE_DEPTH, find_component_placeholders, rewrite_body
from .sidebar import (
    FrontierEntry,
    SLUG_RE,
    classify_internal_url,
    page_url,
    parse_ssr_props,
)

# Error classes for pages that yield no record.
ERR_AUTH_REQUIRED = "auth_required"
ERR_NO_SSR_PROPS = "no_ssr_props"
ERR_NO_DOC = "no_doc_or_custompage"
ERR_CHANGELOG_MISMATCH = "changelog_slug_mismatch"


@dataclass
class PageExtraction:
    """Everything the crawler takes from one fetched page."""

    record: Optional[Dict]
    error: Optional[str] = None          # error class when ``record`` is None
    detail: Optional[str] = None         # human-readable detail for the manifest
    has_ssr: bool = False
    oas_id: Optional[str] = None
    oas_doc: Optional[Dict] = None       # summarized OAS, for the registry
    operation_ids: List[str] = field(default_factory=list)
    sidebars: Optional[Dict] = None
    links: List[FrontierEntry] = field(default_factory=list)
    reusable_unresolved: List[str] = field(default_factory=list)
    html_blocks: int = 0                 # ``[block:html]`` blocks converted
    body_from_html: bool = False         # body came from html_to_markdown(cp.html)


def classify_without_ssr(html: str) -> str:
    """Why a page with no ssr-props yields nothing."""
    if "<title>Login</title>" in html or 'class="loginForm"' in html:
        return ERR_AUTH_REQUIRED
    return ERR_NO_SSR_PROPS


def _category_summary(doc: Dict) -> Dict[str, Optional[str]]:
    cat = doc.get("category") or {}
    if not isinstance(cat, dict):
        cat = {}
    return {
        "title": cat.get("title"),
        "slug": cat.get("slug"),
        "type": cat.get("type"),
        "id": cat.get("_id") or cat.get("id"),
    }


def _summarize_oas(oas: Any) -> Optional[Dict]:
    """Strip ReadMe-internal noise but keep the full OpenAPI document.
    Returns None if no real paths or schemas are present.
    """
    if not isinstance(oas, dict):
        return None
    paths = oas.get("paths") or {}
    schemas = (oas.get("components") or {}).get("schemas") or {}
    if not paths and not schemas:
        return None
    return {
        "openapi": oas.get("openapi"),
        "info": oas.get("info"),
        "servers": oas.get("servers"),
        "paths": paths,
        "components": oas.get("components") or {},
    }


def _operation_info(oas: Dict, *, slug: str, alt_slug: str, title: str,
                    doc_api: Optional[Dict]) -> Optional[Dict]:
    hit = find_page_operation(oas, slug=slug, title=title, doc_api=doc_api)
    if hit is None and alt_slug and alt_slug != slug:
        hit = find_page_operation(oas, slug=alt_slug, title=title, doc_api=doc_api)
    if hit is None:
        return None
    method, path = hit
    for m, p, op in iter_operations(oas):
        if m == method and p == path:
            return {"method": method, "path": path,
                    "operationId": op.get("operationId"), "summary": op.get("summary")}
    return {"method": method, "path": path, "operationId": None, "summary": None}


# --- reusable content -----------------------------------------------------------

#: Keys whose value may hold component bodies, as ``{name: body|{...}}`` or a
#: list of ``{name, body}`` objects.
REUSABLE_CONTAINER_KEYS = ("reusableContent", "reusable_content", "reusableContents",
                           "customBlocks", "components")
REUSABLE_NAME_KEYS = ("name", "tag", "title", "slug")
REUSABLE_BODY_KEYS = ("body", "source", "content", "markdown", "mdx")
# Huge and irrelevant (OpenAPI schemas have "title"/"content" keys too).
_REUSABLE_SKIP_KEYS = frozenset({"oasDefinition"})


def _iter_dicts(root: Any) -> Iterator[Dict]:
    """Breadth-first over every dict in a JSON tree (skipping noisy subtrees)."""
    queue: Deque[Any] = deque([root])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            yield node
            for k, v in node.items():
                if k not in _REUSABLE_SKIP_KEYS and isinstance(v, (dict, list)):
                    queue.append(v)
        elif isinstance(node, list):
            queue.extend(v for v in node if isinstance(v, (dict, list)))


def _body_of(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item if item.strip() else None
    if isinstance(item, dict):
        for key in REUSABLE_BODY_KEYS:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return None


def _name_of(item: Dict) -> Optional[str]:
    for key in REUSABLE_NAME_KEYS:
        val = item.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def find_reusable_content(
    ssr: Any,
    names: Iterable[str],
    *,
    exclude: Iterable[Any] = (),
) -> Dict[str, str]:
    """Locate reusable-content bodies for ``names`` anywhere in ``ssr``.

    Search order (first hit per name wins, nearest the root first):
      1. container keys (``reusableContent``, ``customBlocks``, ...) holding a
         mapping ``{Name: "body"}`` / ``{Name: {"body": ...}}`` or a list of
         ``{"name"|"tag"|"title"|"slug": Name, "body"|"source"|"content"|
         "markdown"|"mdx": "..."}`` objects;
      2. any object anywhere whose name key equals ``Name`` and that carries a
         non-empty string body key.
    ``oasDefinition`` is never searched; objects in ``exclude`` (the page's own
    doc) are never taken as a component. Returns only the names found.
    """
    wanted = [n for n in dict.fromkeys(names) if n]
    if not wanted or not isinstance(ssr, (dict, list)):
        return {}
    excluded = {id(o) for o in exclude if o is not None}
    found: Dict[str, str] = {}

    for node in _iter_dicts(ssr):
        for ckey in REUSABLE_CONTAINER_KEYS:
            container = node.get(ckey)
            if isinstance(container, dict):
                for name in wanted:
                    if name not in found:
                        body = _body_of(container.get(name))
                        if body is not None:
                            found[name] = body
            elif isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        name = _name_of(item)
                        if name in wanted and name not in found:
                            body = _body_of(item)
                            if body is not None:
                                found[name] = body
        if len(found) == len(wanted):
            return found

    wanted_set = set(wanted)
    for node in _iter_dicts(ssr):
        if id(node) in excluded:
            continue
        for key in REUSABLE_NAME_KEYS:
            name = node.get(key)
            if isinstance(name, str) and name in wanted_set and name not in found:
                body = _body_of(node)
                if body is not None:
                    found[name] = body
                    break
        if len(found) == len(wanted):
            break
    return found


def _reusable_names(declared: Any, raw_body: str) -> List[str]:
    names: List[str] = []
    for item in declared or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = _name_of(item)
            if name:
                names.append(name)
    names.extend(find_component_placeholders(raw_body or ""))
    return list(dict.fromkeys(n for n in names if n))


@dataclass
class _Body:
    md: str
    raw_chars: int
    unresolved: List[str]
    html_blocks: int
    from_html: bool = False
    resolved_bodies: List[str] = field(default_factory=list)


def _render_body(
    raw_body: str,
    *,
    ssr: Dict,
    declared_reusable: Any,
    exclude: Iterable[Any],
    base_url: str,
    section: str,
) -> _Body:
    reusable, unresolved = _resolve_reusable(
        ssr, _reusable_names(declared_reusable, raw_body), exclude=exclude)
    md = rewrite_body(raw_body, base_url=base_url, section=section, reusable=reusable or None)
    return _Body(
        md=md, raw_chars=len(raw_body),
        unresolved=unresolved,
        html_blocks=raw_body.count("[block:html]"),
        resolved_bodies=list(reusable.values()),
    )


def _resolve_reusable(
    ssr: Dict, names: List[str], *, exclude: Iterable[Any],
) -> Tuple[Dict[str, str], List[str]]:
    """Look up ``names`` and, transitively, the placeholders inside the bodies
    found, level by level as deep as rewrite_body expands them
    (MAX_REUSABLE_DEPTH). Returns ``(name -> body, unresolved names)``; a
    nested name that cannot be found, or sits deeper than rewrite_body
    expands, is unresolved too, because rewrite_body removes it."""
    exclude = tuple(exclude)
    reusable: Dict[str, str] = {}
    unresolved: List[str] = []
    pending = list(names)
    for _ in range(MAX_REUSABLE_DEPTH):
        if not pending:
            break
        found = find_reusable_content(ssr, pending, exclude=exclude)
        reusable.update(found)
        unresolved.extend(n for n in pending if n not in found)
        nested: List[str] = []
        for name in pending:
            for inner in find_component_placeholders(found.get(name, "")):
                if inner not in reusable and inner not in unresolved and inner not in nested:
                    nested.append(inner)
        pending = nested
    unresolved.extend(n for n in pending if n not in unresolved)
    return reusable, unresolved


# --- record builders ------------------------------------------------------------


def _doc_record(ssr: Dict, doc: Dict, entry: FrontierEntry, *, base_url: str,
                section: str) -> Tuple[Dict, _Body]:
    raw_body = doc.get("body") or ""
    body = _render_body(raw_body, ssr=ssr, declared_reusable=doc.get("reusableContent"),
                        exclude=(doc,), base_url=base_url, section=section)
    kind = entry.kind
    if kind not in ("guide", "reference"):
        kind = "reference" if doc.get("isReference") else "guide"

    record: Dict[str, Any] = {
        "source": "developer_connect",
        "section": section,
        "kind": kind,
        "slug": doc.get("slug") or entry.slug,
        "title": doc.get("title") or entry.title,
        "url": entry.url,
        "category": _category_summary(doc),
        "parent_slug": entry.parent_slug,
        "breadcrumb": entry.breadcrumb,
        "hidden": bool(doc.get("hidden", entry.hidden)),
        "deprecated": bool(doc.get("deprecated", entry.deprecated)),
        "type": doc.get("type"),
        "is_api": bool(doc.get("isApi")),
        "is_reference": bool(doc.get("isReference")),
        "excerpt": doc.get("excerpt") or "",
        "body_md": body.md,
        "body_md_chars": len(body.md),
        "raw_body_chars": body.raw_chars,
        "updated_at": doc.get("updatedAt"),
        "created_at": doc.get("createdAt"),
        "revision": doc.get("revision"),
        "discovered_via": entry.discovered_via,
        "doc_id": doc.get("_id") or doc.get("id"),
    }

    # ReadMe surfaces some content via fields outside `doc.body` and `oasDefinition`.
    # `doc.api` is the high-value one: for high-traffic endpoints (e.g., `get-loan`)
    # it carries rendered cURL/Python/JS code samples that the OAS spec alone does not.
    if doc.get("api"):
        record["doc_api"] = doc["api"]
    if doc.get("reusableContent"):
        record["reusable_content"] = doc["reusableContent"]
    if doc.get("tutorials"):
        record["doc_tutorials"] = doc["tutorials"]
    if body.unresolved:
        record["reusable_content_unresolved"] = body.unresolved
    return record, body


def _custompage_record(ssr: Dict, cp: Dict, entry: FrontierEntry, *, base_url: str,
                       section: str) -> Tuple[Dict, _Body]:
    raw_body = cp.get("body") or ""
    html = cp.get("html") or ""
    body = _render_body(raw_body, ssr=ssr, declared_reusable=cp.get("reusableContent"),
                        exclude=(cp,), base_url=base_url, section=section)
    # ReadMe custom pages are either markdown (`body`) or raw HTML (`html`,
    # selected by `htmlmode`). Use the HTML when it is the selected mode or
    # when the markdown renders to nothing. The converted Markdown goes through
    # rewrite_body too, so `href="doc:..."` macros and root-relative links
    # become absolute URLs like everywhere else.
    if html.strip() and (cp.get("htmlmode") or not body.md.strip()):
        converted = rewrite_body(html_to_markdown(html), base_url=base_url, section=section)
        if converted.strip():
            body.md = converted
            body.from_html = True

    record = {
        "source": "developer_connect",
        "section": section,
        "kind": "custompage",
        "slug": cp.get("slug") or entry.slug,
        "title": cp.get("title") or entry.title,
        "url": entry.url,
        "category": {"title": entry.category_title, "slug": entry.category_slug,
                     "type": "custompage", "id": None},
        "parent_slug": entry.parent_slug,
        "breadcrumb": entry.breadcrumb,
        "hidden": bool(cp.get("hidden", entry.hidden)),
        "deprecated": False,
        "type": "custompage",
        "is_api": False,
        "is_reference": False,
        "excerpt": "",
        "body_md": body.md,
        "body_md_chars": len(body.md),
        "raw_body_chars": len(raw_body),
        "html_chars": len(html),
        "fullscreen": bool(cp.get("fullscreen")),
        "updated_at": cp.get("updatedAt"),
        "created_at": cp.get("createdAt"),
        "revision": cp.get("revision"),
        "discovered_via": entry.discovered_via,
        "doc_id": cp.get("_id") or cp.get("id"),
        "body_source": "html" if body.from_html else "markdown",
    }
    if body.unresolved:
        record["reusable_content_unresolved"] = body.unresolved
    return record, body


def _pick_changelog(
    ssr: Dict,
    slug: str,
    changelog_index: Optional[Mapping[str, Dict]],
) -> Tuple[Optional[Dict], str, Optional[str]]:
    """Find the changelog entry for ``slug``: ``(entry, source, served_slug)``.

    The /changelog index page puts its newest entry at ``ssr.changelog`` too,
    so a singular entry whose slug differs from the requested one must NOT be
    attached to this URL. ``served_slug`` reports such a mismatch.
    """
    single = ssr.get("changelog")
    served: Optional[str] = None
    if isinstance(single, dict) and single:
        served = single.get("slug")
        if served in (None, slug):
            return single, "page", None
    for item in ssr.get("changelogs") or []:
        if isinstance(item, dict) and item.get("slug") == slug:
            return item, "page_list", None
    if changelog_index and isinstance(changelog_index.get(slug), dict):
        return changelog_index[slug], "index", None
    doc = ssr.get("doc")
    if isinstance(doc, dict) and doc and doc.get("slug") in (None, slug):
        return doc, "doc", None
    return None, "", served


def _changelog_record(entry: FrontierEntry, match: Dict, ssr: Dict, source: str, *,
                      base_url: str, section: str) -> Tuple[Dict, _Body]:
    raw_body = match.get("body") or ""
    body = _render_body(raw_body, ssr=ssr, declared_reusable=match.get("reusableContent"),
                        exclude=(match,), base_url=base_url, section=section)
    title = match.get("title") or entry.title
    # Entries found through a reference or a link only know their slug; their
    # placeholder breadcrumb (["Release Notes", "<slug>"]) would end up in
    # every chunk header. Navigation breadcrumbs are kept.
    breadcrumb = list(entry.breadcrumb)
    if not breadcrumb or breadcrumb[-1] == entry.slug:
        breadcrumb = ["Release Notes", title]
    record = {
        "source": "developer_connect",
        "section": section,
        "kind": "changelog",
        "slug": match.get("slug") or entry.slug,
        "title": title,
        "url": entry.url,
        "category": {"title": "Release Notes", "slug": "changelog", "type": "changelog",
                     "id": None},
        "parent_slug": None,
        "breadcrumb": breadcrumb,
        "hidden": bool(match.get("hidden", entry.hidden)),
        "deprecated": False,
        "type": match.get("type"),
        "is_api": False,
        "is_reference": False,
        "excerpt": match.get("bodyPreview") or "",
        "body_md": body.md,
        "body_md_chars": len(body.md),
        "raw_body_chars": body.raw_chars,
        "updated_at": match.get("updatedAt"),
        "created_at": match.get("createdAt"),
        "revision": match.get("revision"),
        "discovered_via": entry.discovered_via,
        "doc_id": match.get("_id") or match.get("id"),
    }
    if source == "index":
        # The per-entry page did not carry this entry; the text comes from the
        # changelog index list instead.
        record["changelog_source"] = "index"
    if body.unresolved:
        record["reusable_content_unresolved"] = body.unresolved
    return record, body


# --- main entry point -----------------------------------------------------------


def extract_page(
    html: str,
    entry: FrontierEntry,
    *,
    base_url: str,
    section: str,
    changelog_index: Optional[Mapping[str, Dict]] = None,
    discover: bool = True,
) -> PageExtraction:
    """Parse a fetched page once and return its record and discovery hints."""
    base_url = base_url.rstrip("/")
    ssr = parse_ssr_props(html)
    if ssr is None:
        return PageExtraction(record=None, error=classify_without_ssr(html or ""))

    ext = PageExtraction(record=None, has_ssr=True)
    sidebars = ssr.get("sidebars")
    ext.sidebars = sidebars if isinstance(sidebars, dict) else None

    summary = _summarize_oas(ssr.get("oasDefinition"))
    if summary is not None:
        ext.oas_doc = summary
        ext.oas_id = oas_fingerprint(summary)
        ext.operation_ids = [op["operationId"] for _, _, op in iter_operations(summary)
                             if isinstance(op.get("operationId"), str) and op["operationId"]]

    body: Optional[_Body] = None
    extra_links: List[FrontierEntry] = []
    if entry.kind == "changelog":
        match, source, served = _pick_changelog(ssr, entry.slug, changelog_index)
        if match is None:
            if served:
                ext.error = ERR_CHANGELOG_MISMATCH
                ext.detail = f"requested {entry.slug!r}, page served {served!r}"
                if SLUG_RE.match(str(served)):
                    extra_links.append(FrontierEntry(
                        slug=served, url=page_url(base_url, section, "changelog", served),
                        kind="changelog", title=served,
                        category_title="Release Notes", category_slug="changelog",
                        breadcrumb=["Release Notes", served],
                        discovered_via="changelog_ref", source_url=entry.url,
                    ))
            else:
                ext.error = ERR_NO_DOC
        else:
            ext.record, body = _changelog_record(entry, match, ssr, source,
                                                 base_url=base_url, section=section)
    else:
        doc = ssr.get("doc")
        cp = ssr.get("custompage")
        if isinstance(doc, dict) and doc:
            ext.record, body = _doc_record(ssr, doc, entry, base_url=base_url, section=section)
        elif isinstance(cp, dict) and cp:
            ext.record, body = _custompage_record(ssr, cp, entry, base_url=base_url,
                                                  section=section)
        else:
            ext.error = ERR_NO_DOC

    record = ext.record
    if record is not None and record["kind"] in ("guide", "reference") and (
        summary is not None or record["kind"] == "reference"
    ):
        record["oas_id"] = ext.oas_id
        record["operation"] = (
            _operation_info(summary, slug=record["slug"], alt_slug=entry.slug,
                            title=record["title"], doc_api=record.get("doc_api"))
            if summary is not None else None
        )
        record["oas_public_url"] = ssr.get("oasPublicUrl")

    if body is not None:
        ext.reusable_unresolved = body.unresolved
        ext.html_blocks = body.html_blocks
        ext.body_from_html = body.from_html

    if discover:
        ext.links = _discover_links(
            html, ssr, base_url=base_url, section=section, source_url=entry.url,
            extra_bodies=body.resolved_bodies if body else (),
        )
    ext.links.extend(extra_links)
    return ext


def extract_record(
    html: str,
    entry: FrontierEntry,
    *,
    base_url: str,
    section: str,
) -> Optional[Dict]:
    """Backward-compatible wrapper: just the record (or None)."""
    return extract_page(html, entry, base_url=base_url, section=section,
                        discover=False).record


# --- Link discovery -------------------------------------------------------------

# Markdown link `[text](url)` / `[text](url "title")`; the lookbehind keeps
# image links `![alt](url)` out.
_LINK_RE = re.compile(r'(?<!!)\[[^\]]*?\]\(\s*<?([^)\s>]+)>?(?:\s+["\'][^"\']*["\'])?\s*\)')

# HTML href attribute, also inside JSON-escaped HTML (`href=\"...\"`) as found in
# legacy `[block:html]` bodies. ReadMe surfaces some nav links only in the DOM.
_HREF_RE = re.compile(r'\bhref=\\?["\']([^"\'\\\s<>]+)')


def _collect_link_source_body(ssr: Dict) -> str:
    """Concatenate every body markdown a page might carry, so we can scan once."""
    parts: List[str] = []
    for key in ("doc", "changelog", "custompage"):
        obj = ssr.get(key)
        if isinstance(obj, dict):
            for field_name in ("body", "html"):
                if isinstance(obj.get(field_name), str) and obj[field_name]:
                    parts.append(obj[field_name])
    for c in ssr.get("changelogs") or []:
        if isinstance(c, dict) and isinstance(c.get("body"), str):
            parts.append(c["body"])
    return "\n\n".join(parts)


def _discover_links(
    html: str,
    ssr: Optional[Dict],
    *,
    base_url: str,
    section: str,
    source_url: str,
    extra_bodies: Iterable[str] = (),
) -> List[FrontierEntry]:
    seen_in_page: Set[str] = set()
    discovered: List[FrontierEntry] = []

    def _emit(raw: str, *, from_body: bool) -> None:
        c = classify_internal_url(raw, base_url=base_url, section=section,
                                  source_url=source_url)
        if c is None:
            return
        kind, slug, url = c
        if url in seen_in_page:
            return
        seen_in_page.add(url)
        via = "changelog_ref" if (from_body and kind == "changelog"
                                  and raw.lower().startswith("changelog:")) else "link_bfs"
        discovered.append(FrontierEntry(
            slug=slug, url=url, kind=kind, title=slug,
            breadcrumb=[], hidden=False, deprecated=False,
            sidebar_meta={"discovered_via": via},
            discovered_via=via, source_url=source_url,
        ))

    # 1) author-written cross-refs in every body the page carries
    body = _collect_link_source_body(ssr) if ssr else ""
    body = "\n\n".join([body, *extra_bodies])
    for m in _LINK_RE.finditer(body):
        _emit(m.group(1), from_body=True)
    for m in _HREF_RE.finditer(body):
        _emit(m.group(1), from_body=True)

    # 2) hrefs across the whole rendered page (nav/breadcrumbs/footer).
    for m in _HREF_RE.finditer(html or ""):
        _emit(m.group(1), from_body=False)
    return discovered


def extract_internal_links(
    html: str,
    *,
    base_url: str,
    section: str,
    source_url: str,
) -> List[FrontierEntry]:
    """Discover internal cross-references in a fetched page.

    Recognizes:
      - ReadMe macros: [X](doc:slug), [X](ref:slug), [X](changelog:slug), [X](page:slug)
      - Absolute URLs: https://.../<section>/(docs|reference|changelog|page)/<slug>
      - Relative URLs: /<section>/(docs|reference|changelog|page)/<slug>
      - Section-relative links resolved against source_url

    Caller is responsible for deduping against the existing frontier.
    """
    return _discover_links(html, parse_ssr_props(html), base_url=base_url.rstrip("/"),
                           section=section, source_url=source_url)
