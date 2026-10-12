"""Frontier discovery: every way a Developer Connect page can be found.

Relying on one source lost pages: the welcome page's sidebar JSON only listed
219 of 939 URLs, and the other 720 were found by following links, by luck. So
the frontier is filled from several sources, and each entry remembers where it
came from (``discovered_via``):

  sidebar            docs (and refs) trees in a page's ``ssr-props.sidebars``;
                     the welcome page seeds the crawl
  reference_sidebar  the refs tree harvested from *any* fetched page. A
                     reference page carries the full reference sidebar, which
                     the welcome page does not.
  sitemap            ``/sitemap.xml`` and ``/<section>/sitemap.xml`` (and their
                     ``sitemapindex`` children)
  changelog_index    ``/changelog``, ``?page=2``, ``?page=3``... until a page
                     adds nothing new
  changelog_ref      ``changelog:slug`` references in bodies
  oas_operation      an operationId in a page's OpenAPI document; ReadMe usually
                     uses it as the page slug, so a miss (404) is expected
  link_bfs           links in bodies and hrefs in the rendered HTML

``Frontier`` de-duplicates by URL, upgrades an entry's ``discovered_via`` when a
more authoritative source finds it again, and counts (never silently drops)
URLs refused by the ``max_pages`` cap.

Sidebar nodes that are links (``type: "link"`` or a ``link_url``) are not
fetched as pages: an internal target is enqueued instead, anything else is
reported as an external link.

Hidden pages are included unless ``skip_hidden`` is set; a hidden node's
children are still walked because the decision is made per page, which is the
only thing known for link-discovered pages too.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse

from .fetch import OFFLINE_CACHE_MISS, Fetcher

SSR_PROPS_RE = re.compile(
    r'<script\b[^>]*\bid=["\']ssr-props["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
CHANGELOG_REF_RE = re.compile(r'\(changelog:([a-z0-9][a-z0-9._-]*)\)', re.IGNORECASE)
SLUG_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
# A JSON escape such as "\ud83d" (an emoji cut in half) decodes to a lone
# surrogate, which cannot be written as UTF-8 and used to abort the crawl.
# A high surrogate escape not followed by a low one, or a low one without a
# high one before it (valid pairs decode to one character and are fine).
_LONE_SURROGATE_ESCAPE_RE = re.compile(
    r'\\u[dD][89abAB][0-9a-fA-F]{2}(?!\\u[dD][c-fC-F][0-9a-fA-F]{2})'
    r'|(?<!\\u[dD][89abAB][0-9a-fA-F]{2})\\u[dD][c-fC-F][0-9a-fA-F]{2}'
)
_LONE_SURROGATE_RE = re.compile('[\ud800-\udfff]')

#: Every ``discovered_via`` value, most authoritative first.
DISCOVERY_SOURCES: Tuple[str, ...] = (
    "sidebar", "reference_sidebar", "changelog_index", "sitemap",
    "changelog_ref", "oas_operation", "link_bfs",
)
SOURCE_PRIORITY: Dict[str, int] = {s: i for i, s in enumerate(DISCOVERY_SOURCES)}
#: Sources that are the site's own navigation; a failure there is a real gap.
NAVIGATION_SOURCES = frozenset({"sidebar", "reference_sidebar", "sitemap", "changelog_index"})

KIND_TO_SEGMENT = {"guide": "docs", "reference": "reference", "changelog": "changelog",
                   "custompage": "page"}
SEGMENT_TO_KIND = {v: k for k, v in KIND_TO_SEGMENT.items()}
_MACRO_TO_KIND = {"doc": "guide", "ref": "reference", "changelog": "changelog",
                  "page": "custompage"}
_MACRO_RE = re.compile(r'^(doc|ref|changelog|page):([A-Za-z0-9][A-Za-z0-9._-]*)(?:#.*)?$',
                       re.IGNORECASE)
_SIDEBAR_KIND = {"docs": "guide", "refs": "reference", "reference": "reference",
                 "custompages": "custompage"}

DEFAULT_CHANGELOG_MAX_PAGES = 100
DEFAULT_SITEMAP_MAX_DOCUMENTS = 50
DEFAULT_SITEMAP_MAX_DEPTH = 3


@dataclass
class FrontierEntry:
    slug: str
    url: str
    kind: str  # "guide" | "reference" | "changelog" | "custompage"
    title: str
    category_title: Optional[str] = None
    category_slug: Optional[str] = None
    parent_slug: Optional[str] = None
    breadcrumb: List[str] = field(default_factory=list)
    hidden: bool = False
    deprecated: bool = False
    sidebar_meta: Dict = field(default_factory=dict)  # raw node, minus children
    discovered_via: str = "sidebar"
    source_url: Optional[str] = None  # page/document the entry was found on


# --- helpers ------------------------------------------------------------------


def _scrub_surrogates(value: Any) -> Any:
    """``value`` with every lone UTF-16 surrogate replaced by U+FFFD.

    After ``json.loads`` valid pairs are already combined into one character,
    so any code point left in the surrogate range is a lone half.
    """
    if isinstance(value, str):
        return _LONE_SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, list):
        return [_scrub_surrogates(v) for v in value]
    if isinstance(value, dict):
        return {_scrub_surrogates(k): _scrub_surrogates(v) for k, v in value.items()}
    return value


def load_json(text: str) -> Any:
    """``json.loads`` whose strings are always writable as UTF-8: a lone
    surrogate escape (``"\\ud83d"`` without its pair) becomes U+FFFD.
    Used for ssr-props and for the JSON inside legacy ``[block:*]`` bodies."""
    data = json.loads(text)
    if _LONE_SURROGATE_ESCAPE_RE.search(text):
        data = _scrub_surrogates(data)
    return data


def parse_ssr_props(html: str) -> Optional[Dict]:
    m = SSR_PROPS_RE.search(html or "")
    if not m:
        return None
    try:
        data = load_json(m.group(1))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def page_url(base_url: str, section: str, kind: str, slug: str) -> str:
    return f"{base_url.rstrip('/')}/{section}/{KIND_TO_SEGMENT[kind]}/{slug}"


def url_key(url: str) -> str:
    """Frontier identity of a URL: no fragment, query or trailing slash."""
    return url.split("#", 1)[0].split("?", 1)[0].rstrip("/")


def classify_internal_url(
    raw: str,
    *,
    base_url: str,
    section: str,
    source_url: Optional[str] = None,
) -> Optional[Tuple[str, str, str]]:
    """Return ``(kind, slug, absolute_url)`` if ``raw`` points at a section page.

    Recognizes ReadMe macros (``doc:``, ``ref:``, ``changelog:``, ``page:``),
    absolute URLs, root-relative URLs and URLs relative to ``source_url``,
    for the ``docs``, ``reference``, ``changelog`` and ``page`` segments.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    macro = _MACRO_RE.match(raw)
    if macro:
        kind = _MACRO_TO_KIND[macro.group(1).lower()]
        slug = macro.group(2)
        return kind, slug, page_url(base_url, section, kind, slug)

    section_prefix = f"{base_url.rstrip('/')}/{section}/"
    absolute = urljoin(source_url or section_prefix, raw)
    absolute = url_key(absolute)
    if not absolute.startswith(section_prefix):
        return None
    parts = absolute[len(section_prefix):].split("/")
    if len(parts) != 2:
        return None
    segment, slug = parts
    kind = SEGMENT_TO_KIND.get(segment)
    if kind is None or not SLUG_RE.match(slug):
        return None
    return kind, slug, absolute


# --- frontier -----------------------------------------------------------------


class Frontier:
    """Ordered, URL-de-duplicated crawl queue with an explicit size cap.

    ``max_pages=0`` means unlimited. URLs refused by the cap are counted per
    source (``not_enqueued``) so the caller can report them loudly.
    """

    def __init__(self, max_pages: int = 0):
        self.max_pages = max_pages
        self.entries: List[FrontierEntry] = []
        self._by_key: Dict[str, FrontierEntry] = {}
        self.not_enqueued: Counter = Counter()
        self.not_enqueued_examples: List[str] = []
        self.upgraded = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[FrontierEntry]:
        return iter(self.entries)

    def __contains__(self, url: str) -> bool:
        return url_key(url) in self._by_key

    def get(self, url: str) -> Optional[FrontierEntry]:
        return self._by_key.get(url_key(url))

    @property
    def cap_hit(self) -> bool:
        return bool(self.not_enqueued)

    def add(self, entry: FrontierEntry) -> bool:
        """Enqueue ``entry``; return True only if it is new and fits the cap."""
        key = url_key(entry.url)
        existing = self._by_key.get(key)
        if existing is not None:
            self._maybe_upgrade(existing, entry)
            return False
        if self.max_pages and len(self.entries) >= self.max_pages:
            self.not_enqueued[entry.discovered_via] += 1
            if len(self.not_enqueued_examples) < 50:
                self.not_enqueued_examples.append(entry.url)
            return False
        entry.url = key
        self.entries.append(entry)
        self._by_key[key] = entry
        return True

    def add_many(self, entries: Iterable[FrontierEntry]) -> int:
        return sum(1 for e in entries if self.add(e))

    def _maybe_upgrade(self, existing: FrontierEntry, new: FrontierEntry) -> None:
        old_rank = SOURCE_PRIORITY.get(existing.discovered_via, len(SOURCE_PRIORITY))
        new_rank = SOURCE_PRIORITY.get(new.discovered_via, len(SOURCE_PRIORITY))
        if new_rank >= old_rank:
            return
        self.upgraded += 1
        existing.discovered_via = new.discovered_via
        existing.source_url = new.source_url
        # Navigation sources carry real metadata; link discovery only has the slug.
        if new.breadcrumb:
            existing.breadcrumb = list(new.breadcrumb)
            existing.category_title = new.category_title
            existing.category_slug = new.category_slug
            existing.parent_slug = new.parent_slug
        if new.title and new.title != new.slug:
            existing.title = new.title
        if new.sidebar_meta:
            existing.sidebar_meta = dict(new.sidebar_meta)
        existing.hidden = existing.hidden or new.hidden
        existing.deprecated = existing.deprecated or new.deprecated


# --- sidebars -----------------------------------------------------------------


@dataclass
class SidebarWalk:
    entries: List[FrontierEntry] = field(default_factory=list)
    external_links: List[Dict] = field(default_factory=list)
    skipped_hidden: int = 0


def _url_slug(slug: str) -> Optional[str]:
    """The URL path segment for a sidebar slug (percent-quoted if unusual)."""
    if SLUG_RE.match(slug):
        return slug
    if not slug.strip() or "/" in slug or "\\" in slug or slug.strip(".") == "":
        return None
    return quote(slug, safe="._-~")


def _node_link(node: Dict) -> Tuple[bool, Optional[str]]:
    """``(is_link_node, target)`` for a sidebar node."""
    target = node.get("link_url") or node.get("linkUrl")
    if not target and node.get("type") == "link":
        target = node.get("url") or None
    is_link = node.get("type") == "link" or bool(target)
    return is_link, (str(target) if target else None)


def _walk_nodes(
    nodes: Iterable[Dict],
    *,
    kind: str,
    base_url: str,
    section: str,
    category_title: Optional[str],
    category_slug: Optional[str],
    breadcrumb: List[str],
    parent_slug: Optional[str],
    skip_hidden: bool,
    discovered_via: str,
    source_url: Optional[str],
    out: SidebarWalk,
) -> None:
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        slug = node.get("slug")
        title = node.get("title") or slug or ""
        crumb = [*breadcrumb, title]
        hidden = bool(node.get("hidden", False))
        is_link, target = _node_link(node)

        if slug and is_link:
            internal = (classify_internal_url(target, base_url=base_url, section=section,
                                              source_url=source_url)
                        if target else None)
            if internal is None:
                out.external_links.append({
                    "title": title, "slug": slug, "url": target,
                    "category": category_title, "source_url": source_url,
                })
            elif hidden and skip_hidden:
                out.skipped_hidden += 1
            else:
                t_kind, t_slug, t_url = internal
                out.entries.append(FrontierEntry(
                    slug=t_slug, url=t_url, kind=t_kind, title=title,
                    category_title=category_title, category_slug=category_slug,
                    parent_slug=parent_slug, breadcrumb=crumb, hidden=hidden,
                    deprecated=bool(node.get("deprecated", False)),
                    sidebar_meta={k: v for k, v in node.items() if k != "pages"},
                    discovered_via=discovered_via, source_url=source_url,
                ))
        elif slug and _url_slug(str(slug)) is None:
            # Never silently dropped: a slug we cannot turn into a URL is reported.
            out.external_links.append({
                "title": title, "slug": slug, "url": None, "category": category_title,
                "source_url": source_url, "note": "sidebar slug is not a usable URL segment",
            })
        elif slug:
            node_kind = "custompage" if node.get("type") == "custompage" else kind
            url_slug = _url_slug(str(slug))
            if hidden and skip_hidden:
                out.skipped_hidden += 1
            else:
                out.entries.append(FrontierEntry(
                    slug=url_slug, url=page_url(base_url, section, node_kind, url_slug),
                    kind=node_kind, title=title,
                    category_title=category_title, category_slug=category_slug,
                    parent_slug=parent_slug, breadcrumb=crumb, hidden=hidden,
                    deprecated=bool(node.get("deprecated", False)),
                    sidebar_meta={k: v for k, v in node.items() if k != "pages"},
                    discovered_via=discovered_via, source_url=source_url,
                ))

        children = node.get("pages") or []
        if children:
            _walk_nodes(
                children, kind=kind, base_url=base_url, section=section,
                category_title=category_title, category_slug=category_slug,
                breadcrumb=crumb, parent_slug=slug or parent_slug,
                skip_hidden=skip_hidden, discovered_via=discovered_via,
                source_url=source_url, out=out,
            )


def walk_sidebars(
    sidebars: Optional[Dict],
    *,
    base_url: str,
    section: str,
    skip_hidden: bool = False,
    docs_via: str = "sidebar",
    refs_via: str = "sidebar",
    source_url: Optional[str] = None,
) -> SidebarWalk:
    """Turn an ``ssr-props.sidebars`` object into frontier entries.

    ``docs_via``/``refs_via`` label where the trees came from: the welcome page
    seeds both as ``sidebar``; trees harvested from other pages label refs as
    ``reference_sidebar``.
    """
    out = SidebarWalk()
    if not isinstance(sidebars, dict):
        return out
    base_url = base_url.rstrip("/")
    for key, kind in _SIDEBAR_KIND.items():
        via = docs_via if kind != "reference" else refs_via
        for cat in sidebars.get(key) or []:
            if not isinstance(cat, dict):
                continue
            _walk_nodes(
                cat.get("pages") or [], kind=kind, base_url=base_url, section=section,
                category_title=cat.get("title"), category_slug=cat.get("slug"),
                breadcrumb=[cat.get("title") or ""], parent_slug=None,
                skip_hidden=skip_hidden, discovered_via=via, source_url=source_url,
                out=out,
            )
    return out


def enumerate_welcome(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str,
    skip_hidden: bool = False,
) -> SidebarWalk:
    """Seed the frontier from the welcome page's docs + refs sidebars."""
    base_url = base_url.rstrip("/")
    welcome_url = f"{base_url}/{section}/docs/welcome"
    res = fetcher.fetch(welcome_url)
    if res is None or res.http_status != 200:
        detail = getattr(res, "error", None) or getattr(res, "http_status", None)
        raise RuntimeError(f"Could not fetch welcome page: {welcome_url} ({detail})")
    ssr = parse_ssr_props(res.html)
    if ssr is None:
        raise RuntimeError(f"No ssr-props in welcome page: {welcome_url}")
    return walk_sidebars(
        ssr.get("sidebars"), base_url=base_url, section=section,
        skip_hidden=skip_hidden, docs_via="sidebar", refs_via="sidebar",
        source_url=welcome_url,
    )


# --- changelog index ----------------------------------------------------------


@dataclass
class ChangelogIndex:
    entries: List[FrontierEntry] = field(default_factory=list)
    #: slug -> raw list item (title/body/...), used when a per-entry page
    #: does not carry its own entry.
    by_slug: Dict[str, Dict] = field(default_factory=dict)
    pages: List[Dict] = field(default_factory=list)  # {url, http_status, new_slugs, error, unexpected}
    truncated: bool = False
    max_pages: int = DEFAULT_CHANGELOG_MAX_PAGES
    skipped_hidden: int = 0
    #: Pagination stopped on a failed fetch, not on the end of the list.
    incomplete: bool = False
    #: slug -> ``changelog_ref`` entry for slugs referenced in bodies but not
    #: (yet) listed by the index; replaced if a later index page lists them.
    ref_entries: Dict[str, FrontierEntry] = field(default_factory=dict)


def changelog_entries_from_ssr(
    ssr: Dict,
    *,
    base_url: str,
    section: str,
    source_url: str,
    seen: set,
    skip_hidden: bool,
    out: ChangelogIndex,
) -> int:
    """Add the ``changelogs`` list of one index page to ``out``; return new slugs.

    ``seen`` holds the slugs the index itself has listed so far. The page's
    items are processed before the ``(changelog:slug)`` references in their
    bodies, and a reference never counts as a listing: newer notes reference
    older ones, and treating those as "seen" mislabelled the older entries,
    left them out of ``by_slug`` and stopped pagination early. A slug first
    seen as a reference and listed later is upgraded to ``changelog_index``.
    """
    items = [item for item in ssr.get("changelogs") or [] if isinstance(item, dict)]
    new = 0
    for item in items:
        slug = item.get("slug")
        if not slug or slug in seen or not SLUG_RE.match(str(slug)):
            continue
        seen.add(slug)
        new += 1
        out.by_slug[slug] = item
        title = item.get("title") or slug
        ref = out.ref_entries.pop(slug, None)
        pos = None if ref is None else next(i for i, e in enumerate(out.entries) if e is ref)
        if pos is not None:
            del out.entries[pos]
        if item.get("hidden") and skip_hidden:
            out.skipped_hidden += 1
            continue
        entry = FrontierEntry(
            slug=slug, url=page_url(base_url, section, "changelog", slug),
            kind="changelog", title=title,
            category_title="Release Notes", category_slug="changelog",
            breadcrumb=["Release Notes", title],
            hidden=bool(item.get("hidden", False)),
            sidebar_meta={"type": item.get("type"), "updatedAt": item.get("updatedAt")},
            discovered_via="changelog_index", source_url=source_url,
        )
        if pos is None:
            out.entries.append(entry)
        else:  # keep the crawl position the reference gave it
            out.entries.insert(pos, entry)
    for item in items:
        for ref_slug in CHANGELOG_REF_RE.findall(item.get("body") or ""):
            if ref_slug in seen or ref_slug in out.ref_entries:
                continue
            ref = FrontierEntry(
                slug=ref_slug, url=page_url(base_url, section, "changelog", ref_slug),
                kind="changelog", title=ref_slug,  # real title comes from the page
                category_title="Release Notes", category_slug="changelog",
                breadcrumb=["Release Notes", ref_slug],
                discovered_via="changelog_ref", source_url=source_url,
            )
            out.ref_entries[ref_slug] = ref
            out.entries.append(ref)
    return new


def enumerate_changelog(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str,
    skip_hidden: bool = False,
    max_pages: int = DEFAULT_CHANGELOG_MAX_PAGES,
) -> ChangelogIndex:
    """Walk ``/changelog``, ``?page=2``, ... until a page adds no new slug,
    is not a 200, has no ssr-props, or ``max_pages`` is reached (flagged as
    ``truncated`` when the last page still produced new slugs).

    Only a 404/410 (or, offline, a cache miss: 404s are never cached) past
    page 1 is the normal end of the list. Any other failed fetch - a 5xx or
    429 after retries, a network error, page 1 missing - loses every entry
    after it, so the page is marked ``unexpected`` and the index
    ``incomplete``; the crawler turns that into an unexpected failure.
    """
    base_url = base_url.rstrip("/")
    index_url = f"{base_url}/{section}/changelog"
    out = ChangelogIndex(max_pages=max_pages)
    seen: set = set()
    for n in range(1, max_pages + 1):
        url = index_url if n == 1 else f"{index_url}?page={n}"
        res = fetcher.fetch(url)
        status = getattr(res, "http_status", None)
        page_info = {"url": url, "http_status": status, "new_slugs": 0,
                     "error": getattr(res, "error", None)}
        out.pages.append(page_info)
        if res is None or status != 200:
            offline_miss = getattr(res, "error", None) == OFFLINE_CACHE_MISS
            if not (n > 1 and (status in (404, 410) or offline_miss)):
                page_info["unexpected"] = True
                out.incomplete = True
            break
        ssr = parse_ssr_props(res.html)
        if ssr is None:
            page_info["error"] = "no_ssr_props"
            break
        new = changelog_entries_from_ssr(
            ssr, base_url=base_url, section=section, source_url=url,
            seen=seen, skip_hidden=skip_hidden, out=out,
        )
        page_info["new_slugs"] = new
        if new == 0:
            break
        if n == max_pages:
            out.truncated = True
    return out


# --- sitemap ------------------------------------------------------------------


@dataclass
class SitemapResult:
    entries: List[FrontierEntry] = field(default_factory=list)
    documents: List[Dict] = field(default_factory=list)  # {url, http_status, kind, locs, error}
    ignored_urls: int = 0  # <loc>s outside the section's page URL space
    truncated: bool = False
    truncated_documents: List[str] = field(default_factory=list)
    #: A sitemap document failed with a server/network error (not a 404).
    incomplete: bool = False


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def parse_sitemap(text: str) -> Tuple[Optional[str], List[str]]:
    """Return ``("urlset"|"sitemapindex", [loc, ...])`` or ``(None, [])``."""
    text = (text or "").lstrip("﻿").strip()
    if not text.startswith("<"):
        return None, []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None, []
    kind = _local(root.tag)
    if kind not in ("urlset", "sitemapindex"):
        return None, []
    locs: List[str] = []
    for child in root:
        for sub in child:
            if _local(sub.tag) == "loc" and sub.text and sub.text.strip():
                locs.append(sub.text.strip())
    return kind, locs


def enumerate_sitemap(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str,
    max_documents: int = DEFAULT_SITEMAP_MAX_DOCUMENTS,
    max_depth: int = DEFAULT_SITEMAP_MAX_DEPTH,
) -> SitemapResult:
    """Collect section page URLs from the site and section sitemaps.

    Missing sitemaps (a 4xx such as 404, HTML instead of XML, or offline a
    cache miss) are tolerated and recorded. A server error (5xx, 429 after
    retries) or a network error is not: the document is marked ``unexpected``
    and the result ``incomplete`` so the crawler can fail the run.
    ``sitemapindex`` children on the same host are followed up to
    ``max_depth``/``max_documents``; anything left over is reported as
    truncated rather than silently dropped.
    """
    base_url = base_url.rstrip("/")
    host = urlparse(base_url).netloc
    page_re = re.compile(
        rf"^{re.escape(base_url)}/{re.escape(section)}/(docs|reference|changelog|page)/"
        r"([A-Za-z0-9][A-Za-z0-9._-]*)/?$"
    )
    out = SitemapResult()
    queue: List[Tuple[str, int]] = [
        (f"{base_url}/sitemap.xml", 0),
        (f"{base_url}/{section}/sitemap.xml", 0),
    ]
    seen_docs: set = set()
    seen_pages: set = set()
    while queue:
        doc_url, depth = queue.pop(0)
        if doc_url in seen_docs:
            continue
        if len(seen_docs) >= max_documents:
            out.truncated = True
            out.truncated_documents.append(doc_url)
            continue
        seen_docs.add(doc_url)
        res = fetcher.fetch_resource(doc_url)
        status = getattr(res, "http_status", None)
        info: Dict = {"url": doc_url, "http_status": status, "kind": None, "locs": 0,
                      "error": getattr(res, "error", None)}
        out.documents.append(info)
        if res is None or status != 200:
            transient = status is None or status == 429 or status >= 500
            if transient and info["error"] != OFFLINE_CACHE_MISS:
                info["unexpected"] = True
                out.incomplete = True
            continue
        kind, locs = parse_sitemap(res.html)
        info["kind"], info["locs"] = kind, len(locs)
        if kind is None:
            info["error"] = "not a sitemap XML document"
            continue
        if kind == "sitemapindex":
            for loc in locs:
                if urlparse(loc).netloc != host:
                    continue
                if depth + 1 > max_depth:
                    out.truncated = True
                    out.truncated_documents.append(loc)
                    continue
                queue.append((loc, depth + 1))
            continue
        for loc in locs:
            m = page_re.match(url_key(loc))
            if not m:
                out.ignored_urls += 1
                continue
            kind_name = SEGMENT_TO_KIND[m.group(1)]
            slug = m.group(2)
            url = page_url(base_url, section, kind_name, slug)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            out.entries.append(FrontierEntry(
                slug=slug, url=url, kind=kind_name, title=slug,
                discovered_via="sitemap", source_url=doc_url,
            ))
    return out


# --- seeds --------------------------------------------------------------------


@dataclass
class SeedResult:
    entries: List[FrontierEntry]
    welcome: SidebarWalk
    changelog: ChangelogIndex
    sitemap: Optional[SitemapResult]


def enumerate_seeds(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str = "developer-connect",
    skip_hidden: bool = False,
    include_sitemap: bool = True,
    changelog_max_pages: int = DEFAULT_CHANGELOG_MAX_PAGES,
    sitemap_max_documents: int = DEFAULT_SITEMAP_MAX_DOCUMENTS,
) -> SeedResult:
    """Initial frontier: welcome sidebars, paginated changelog index, sitemaps."""
    base_url = base_url.rstrip("/")
    welcome = enumerate_welcome(fetcher, base_url=base_url, section=section,
                                skip_hidden=skip_hidden)
    changelog = enumerate_changelog(fetcher, base_url=base_url, section=section,
                                    skip_hidden=skip_hidden, max_pages=changelog_max_pages)
    sitemap = (enumerate_sitemap(fetcher, base_url=base_url, section=section,
                                 max_documents=sitemap_max_documents)
               if include_sitemap else None)
    entries = [*welcome.entries, *changelog.entries, *(sitemap.entries if sitemap else [])]
    return SeedResult(entries=entries, welcome=welcome, changelog=changelog, sitemap=sitemap)


def enumerate_section(
    fetcher: Fetcher,
    *,
    base_url: str,
    section: str = "developer-connect",
    skip_hidden: bool = False,
    include_sitemap: bool = False,
) -> List[FrontierEntry]:
    """Backward-compatible wrapper: welcome sidebars + changelog index entries,
    de-duplicated by URL (sitemap only when asked)."""
    seeds = enumerate_seeds(fetcher, base_url=base_url, section=section,
                            skip_hidden=skip_hidden, include_sitemap=include_sitemap)
    frontier = Frontier()
    frontier.add_many(seeds.entries)
    return list(frontier)
