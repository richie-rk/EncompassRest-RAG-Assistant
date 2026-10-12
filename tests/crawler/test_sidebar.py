"""Frontier discovery: sidebars, frontier de-dup/caps, changelog pagination, sitemaps."""
from __future__ import annotations

import json

import pytest

from scripts.crawler.fetch import OFFLINE_CACHE_MISS
from scripts.crawler.sidebar import (
    Frontier,
    FrontierEntry,
    classify_internal_url,
    enumerate_changelog,
    enumerate_section,
    enumerate_sitemap,
    enumerate_welcome,
    parse_sitemap,
    parse_ssr_props,
    walk_sidebars,
)
from tests.crawler.fixtures.fake_site import FakeFetcher, ssr_html

BASE = "https://dev.example.com"
SEC = "dc"
P = f"{BASE}/{SEC}"


SIDEBARS = {
    "docs": [{
        "title": "Get Started", "slug": "get-started",
        "pages": [
            {"slug": "welcome", "title": "Welcome", "type": "basic", "pages": []},
            {"slug": "secret", "title": "Secret", "hidden": True, "pages": [
                {"slug": "child-of-secret", "title": "Child", "pages": []},
            ]},
            {"slug": "partner-pdf", "title": "Partner PDF", "type": "link",
             "link_url": "https://cdn.example.com/guide.pdf", "pages": []},
            {"slug": "go-auth", "title": "Auth (moved)", "type": "link",
             "link_url": f"/{SEC}/docs/authentication", "pages": []},
            {"slug": "mystery-link", "title": "Mystery", "type": "link", "pages": []},
        ],
    }],
    "refs": [{
        "title": "Loans", "slug": "loans",
        "pages": [{"slug": "get-loan", "title": "Get Loan", "type": "endpoint",
                   "pages": [{"slug": "get-loan-v2", "title": "Get Loan v2"}]}],
    }],
}


def test_parse_ssr_props_tolerates_attribute_order_and_bad_json():
    assert parse_ssr_props(ssr_html({"a": 1})) == {"a": 1}
    html = '<script type="application/json" id="ssr-props">{"b": 2}</script>'
    assert parse_ssr_props(html) == {"b": 2}
    assert parse_ssr_props('<script id="ssr-props" type="application/json">{oops</script>') is None
    assert parse_ssr_props("<html></html>") is None


def test_walk_sidebars_includes_hidden_by_default_and_labels_sources():
    walk = walk_sidebars(SIDEBARS, base_url=BASE, section=SEC, docs_via="sidebar",
                         refs_via="reference_sidebar", source_url=f"{P}/reference/x")
    by_slug = {e.slug: e for e in walk.entries}
    assert set(by_slug) == {"welcome", "secret", "child-of-secret", "authentication",
                            "get-loan", "get-loan-v2"}
    assert by_slug["secret"].hidden is True
    assert by_slug["welcome"].url == f"{P}/docs/welcome"
    assert by_slug["welcome"].discovered_via == "sidebar"
    assert by_slug["get-loan"].kind == "reference"
    assert by_slug["get-loan"].discovered_via == "reference_sidebar"
    assert by_slug["get-loan-v2"].parent_slug == "get-loan"
    assert by_slug["get-loan-v2"].breadcrumb == ["Loans", "Get Loan", "Get Loan v2"]
    assert by_slug["child-of-secret"].source_url == f"{P}/reference/x"
    # internal link node -> its target, under the node's title
    assert by_slug["authentication"].url == f"{P}/docs/authentication"
    assert by_slug["authentication"].title == "Auth (moved)"
    # external / target-less link nodes are reported, never fetched
    ext = {link["slug"]: link for link in walk.external_links}
    assert ext["partner-pdf"]["url"] == "https://cdn.example.com/guide.pdf"
    assert ext["mystery-link"]["url"] is None
    assert all("partner-pdf" not in e.url for e in walk.entries)


def test_walk_sidebars_skip_hidden_skips_only_hidden_nodes():
    walk = walk_sidebars(SIDEBARS, base_url=BASE, section=SEC, skip_hidden=True)
    slugs = {e.slug for e in walk.entries}
    assert "secret" not in slugs
    assert "child-of-secret" in slugs  # decided per page, like link-discovered pages
    assert walk.skipped_hidden == 1


def test_classify_internal_url():
    def c(raw, src=None):
        return classify_internal_url(raw, base_url=BASE, section=SEC, source_url=src)

    assert c("doc:intro") == ("guide", "intro", f"{P}/docs/intro")
    assert c("ref:get-loan#query") == ("reference", "get-loan", f"{P}/reference/get-loan")
    assert c("changelog:rel-1") == ("changelog", "rel-1", f"{P}/changelog/rel-1")
    assert c("page:api-key") == ("custompage", "api-key", f"{P}/page/api-key")
    assert c(f"/{SEC}/page/saml-sso") == ("custompage", "saml-sso", f"{P}/page/saml-sso")
    assert c(f"{P}/docs/intro/?x=1#y") == ("guide", "intro", f"{P}/docs/intro")
    assert c("../reference/abc", src=f"{P}/docs/intro") == ("reference", "abc",
                                                            f"{P}/reference/abc")
    assert c("https://other.example.com/dc/docs/intro") is None
    assert c(f"{BASE}/other-section/docs/intro") is None
    assert c(f"/{SEC}/docs/a/b") is None
    assert c(f"/{SEC}/blog/a") is None
    assert c("mailto:x@example.com") is None
    assert c("") is None


def _e(slug, via, **kw):
    return FrontierEntry(slug=slug, url=f"{P}/docs/{slug}", kind="guide", title=slug,
                         discovered_via=via, **kw)


def test_frontier_dedupes_upgrades_and_counts_cap():
    fr = Frontier(max_pages=2)
    assert fr.add(_e("a", "link_bfs", source_url=f"{P}/docs/z"))
    assert not fr.add(FrontierEntry(slug="a", url=f"{P}/docs/a/#frag", kind="guide",
                                    title="A", discovered_via="link_bfs"))
    assert len(fr) == 1
    # better source upgrades the existing entry with its metadata
    fr.add(_e("a", "sidebar", breadcrumb=["Cat", "A"], category_title="Cat"))
    a = fr.get(f"{P}/docs/a")
    assert a.discovered_via == "sidebar" and a.breadcrumb == ["Cat", "A"]
    # a worse source never downgrades
    fr.add(_e("a", "oas_operation"))
    assert fr.get(f"{P}/docs/a").discovered_via == "sidebar"

    assert fr.add(_e("b", "sitemap"))
    assert not fr.add(_e("c", "link_bfs"))
    assert not fr.add(_e("d", "oas_operation"))
    assert fr.cap_hit and dict(fr.not_enqueued) == {"link_bfs": 1, "oas_operation": 1}
    assert fr.not_enqueued_examples == [f"{P}/docs/c", f"{P}/docs/d"]
    assert f"{P}/docs/b" in fr and f"{P}/docs/c" not in fr


def test_enumerate_welcome_requires_the_welcome_page():
    with pytest.raises(RuntimeError):
        enumerate_welcome(FakeFetcher({}), base_url=BASE, section=SEC)
    with pytest.raises(RuntimeError):
        enumerate_welcome(FakeFetcher({f"{P}/docs/welcome": "<html></html>"}),
                          base_url=BASE, section=SEC)
    walk = enumerate_welcome(
        FakeFetcher({f"{P}/docs/welcome": ssr_html({"sidebars": SIDEBARS})}),
        base_url=BASE + "/", section=SEC)
    assert {e.discovered_via for e in walk.entries} == {"sidebar"}


# --- changelog ----------------------------------------------------------------


def _cl(*slugs, hidden=(), bodies=None):
    bodies = bodies or {}
    return ssr_html({"changelogs": [
        {"slug": s, "title": s.upper(), "hidden": s in hidden, "body": bodies.get(s, "")}
        for s in slugs
    ]})


def test_changelog_index_paginates_until_no_new_slugs():
    fetcher = FakeFetcher({
        f"{P}/changelog": _cl("r5", "r4", bodies={"r4": "see [old](changelog:r0-old)"}),
        f"{P}/changelog?page=2": _cl("r3", "r2", hidden={"r2"}),
        f"{P}/changelog?page=3": _cl("r1"),
        f"{P}/changelog?page=4": _cl("r1"),  # ReadMe repeating the last page
    })
    idx = enumerate_changelog(fetcher, base_url=BASE, section=SEC)
    slugs = [e.slug for e in idx.entries]
    assert slugs == ["r5", "r4", "r0-old", "r3", "r2", "r1"]
    assert {e.slug: e.discovered_via for e in idx.entries}["r0-old"] == "changelog_ref"
    assert {e.discovered_via for e in idx.entries if e.slug != "r0-old"} == {"changelog_index"}
    assert idx.entries[3].source_url == f"{P}/changelog?page=2"
    assert [p["new_slugs"] for p in idx.pages] == [2, 2, 1, 0]
    assert set(idx.by_slug) == {"r5", "r4", "r3", "r2", "r1"}
    assert not idx.truncated
    assert fetcher.calls[-1] == f"{P}/changelog?page=4"
    assert f"{P}/changelog?page=5" not in fetcher.calls


def test_changelog_index_stops_on_non_200_and_skip_hidden():
    fetcher = FakeFetcher({f"{P}/changelog": _cl("r2", "r1", hidden={"r1"}),
                           f"{P}/changelog?page=2": 500})
    idx = enumerate_changelog(fetcher, base_url=BASE, section=SEC, skip_hidden=True)
    assert [e.slug for e in idx.entries] == ["r2"]
    assert idx.skipped_hidden == 1
    assert idx.pages[-1]["http_status"] == 500


def test_changelog_index_cap_is_reported():
    pages = {f"{P}/changelog": _cl("a")}
    for n, s in ((2, "b"), (3, "c")):
        pages[f"{P}/changelog?page={n}"] = _cl(s)
    idx = enumerate_changelog(FakeFetcher(pages), base_url=BASE, section=SEC, max_pages=2)
    assert [e.slug for e in idx.entries] == ["a", "b"]
    assert idx.truncated is True


# --- sitemap ------------------------------------------------------------------

NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'


def _urlset(*locs):
    return (f'<?xml version="1.0" encoding="UTF-8"?><urlset {NS}>'
            + "".join(f"<url><loc>{u}</loc><lastmod>2026-01-01</lastmod></url>" for u in locs)
            + "</urlset>")


def _index(*locs):
    return (f'<?xml version="1.0"?><sitemapindex {NS}>'
            + "".join(f"<sitemap><loc>{u}</loc></sitemap>" for u in locs)
            + "</sitemapindex>")


def test_parse_sitemap():
    assert parse_sitemap(_urlset("https://a/x", " https://a/y ")) == (
        "urlset", ["https://a/x", "https://a/y"])
    assert parse_sitemap("﻿" + _index("https://a/s1.xml")) == (
        "sitemapindex", ["https://a/s1.xml"])
    assert parse_sitemap("<html><body>not xml") == (None, [])
    assert parse_sitemap("<rss></rss>") == (None, [])
    assert parse_sitemap("") == (None, [])


def test_enumerate_sitemap_follows_index_and_filters_urls():
    resources = {
        f"{BASE}/sitemap.xml": 404,
        f"{P}/sitemap.xml": _index(f"{P}/sitemap-pages.xml", "https://evil.example.com/s.xml",
                                   f"{P}/sitemap-html.xml"),
        f"{P}/sitemap-pages.xml": _urlset(
            f"{P}/docs/welcome", f"{P}/reference/get-loan/", f"{P}/page/api-key",
            f"{P}/changelog/rel-1?utm=x", f"{P}/docs/a/b", f"{BASE}/other/docs/x",
            f"{P}/docs/welcome",
        ),
        f"{P}/sitemap-html.xml": "<html>soft 404</html>",
    }
    fetcher = FakeFetcher({}, resources)
    res = enumerate_sitemap(fetcher, base_url=BASE, section=SEC)
    got = {(e.kind, e.url) for e in res.entries}
    assert got == {
        ("guide", f"{P}/docs/welcome"),
        ("reference", f"{P}/reference/get-loan"),
        ("custompage", f"{P}/page/api-key"),
        ("changelog", f"{P}/changelog/rel-1"),
    }
    assert {e.discovered_via for e in res.entries} == {"sitemap"}
    assert res.entries[0].source_url == f"{P}/sitemap-pages.xml"
    assert res.ignored_urls == 2
    assert "https://evil.example.com/s.xml" not in fetcher.resource_calls
    statuses = {d["url"]: (d["http_status"], d["kind"], d["error"]) for d in res.documents}
    assert statuses[f"{BASE}/sitemap.xml"][0] == 404
    assert statuses[f"{P}/sitemap-html.xml"][2] == "not a sitemap XML document"
    assert not res.truncated


def test_enumerate_sitemap_caps_are_reported():
    resources = {
        f"{BASE}/sitemap.xml": _index(f"{P}/s1.xml"),
        f"{P}/s1.xml": _index(f"{P}/s2.xml"),
        f"{P}/s2.xml": _urlset(f"{P}/docs/deep"),
    }
    res = enumerate_sitemap(FakeFetcher({}, resources), base_url=BASE, section=SEC,
                            max_depth=1)
    assert res.truncated and res.truncated_documents == [f"{P}/s2.xml"]
    assert res.entries == []

    res = enumerate_sitemap(FakeFetcher({}, resources), base_url=BASE, section=SEC,
                            max_documents=2)
    assert res.truncated and len(res.documents) == 2


def test_enumerate_section_backward_compatible_wrapper():
    fetcher = FakeFetcher({
        f"{P}/docs/welcome": ssr_html({"sidebars": SIDEBARS}),
        f"{P}/changelog": _cl("r1"),
    })
    entries = enumerate_section(fetcher, base_url=BASE, section=SEC)
    kinds = {e.kind for e in entries}
    assert kinds == {"guide", "reference", "changelog"}
    assert fetcher.resource_calls == []  # sitemap only on request


def test_unusual_sidebar_slugs_are_quoted_or_reported():
    sidebars = {"docs": [{"title": "C", "slug": "c", "pages": [
        {"slug": "odd slug", "title": "Odd"},
        {"slug": "a/b", "title": "Slashed"},
        {"slug": "..", "title": "Dots"},
    ]}]}
    walk = walk_sidebars(sidebars, base_url=BASE, section=SEC)
    assert [e.url for e in walk.entries] == [f"{P}/docs/odd%20slug"]
    reported = {link["slug"] for link in walk.external_links}
    assert reported == {"a/b", ".."}
    assert all(link["url"] is None for link in walk.external_links)


# --- regressions --------------------------------------------------------------


def test_changelog_references_do_not_hide_later_index_entries():
    """Newer notes reference older ones. A reference must not count as the
    index listing that slug: the entry stays changelog_index (with its real
    title), lands in by_slug, and pagination is not cut short when a page
    only lists already-referenced slugs."""
    fetcher = FakeFetcher({
        f"{P}/changelog": _cl("r3", "r2", bodies={"r3": "see [r2](changelog:r2) "
                                                        "and [r0](changelog:r0)"}),
        f"{P}/changelog?page=2": _cl("r1", bodies={"r1": "[r0](changelog:r0)"}),
        f"{P}/changelog?page=3": _cl("r0"),  # every item here was referenced before
        f"{P}/changelog?page=4": _cl("rm1"),
        f"{P}/changelog?page=5": _cl(),
    })
    idx = enumerate_changelog(fetcher, base_url=BASE, section=SEC)
    assert [(e.slug, e.discovered_via) for e in idx.entries] == [
        ("r3", "changelog_index"), ("r2", "changelog_index"),
        ("r0", "changelog_index"),  # upgraded in place of its reference
        ("r1", "changelog_index"), ("rm1", "changelog_index")]
    r0 = idx.entries[2]
    assert r0.title == "R0" and r0.breadcrumb == ["Release Notes", "R0"]
    assert r0.source_url == f"{P}/changelog?page=3"
    assert set(idx.by_slug) == {"r3", "r2", "r1", "r0", "rm1"}
    assert [p["new_slugs"] for p in idx.pages] == [2, 1, 1, 1, 0]
    assert f"{P}/changelog?page=4" in fetcher.calls
    assert idx.ref_entries == {}


def test_changelog_reference_to_hidden_entry_is_dropped_with_skip_hidden():
    fetcher = FakeFetcher({
        f"{P}/changelog": _cl("r2", bodies={"r2": "[r1](changelog:r1)"}),
        f"{P}/changelog?page=2": _cl("r1", hidden={"r1"}),
        f"{P}/changelog?page=3": _cl(),
    })
    idx = enumerate_changelog(fetcher, base_url=BASE, section=SEC, skip_hidden=True)
    assert [e.slug for e in idx.entries] == ["r2"]
    assert idx.skipped_hidden == 1 and "r1" in idx.by_slug


@pytest.mark.parametrize("pages,unexpected", [
    ({f"{P}/changelog": _cl("r2"), f"{P}/changelog?page=2": 503}, True),
    ({f"{P}/changelog": 503}, True),
    ({}, True),  # the index itself is gone (404 on page 1)
    ({f"{P}/changelog": _cl("r2"), f"{P}/changelog?page=2": 429}, True),
    ({f"{P}/changelog": _cl("r2"), f"{P}/changelog?page=2": 404}, False),  # end of the list
    ({f"{P}/changelog": _cl("r2"), f"{P}/changelog?page=2": 410}, False),
])
def test_changelog_index_failure_is_flagged_not_mistaken_for_the_end(pages, unexpected):
    idx = enumerate_changelog(FakeFetcher(pages), base_url=BASE, section=SEC)
    assert idx.incomplete is unexpected
    assert bool(idx.pages[-1].get("unexpected")) is unexpected


def test_changelog_offline_cache_miss_past_page_one_ends_the_list():
    class Offline(FakeFetcher):
        def fetch(self, url):
            res = super().fetch(url)
            if res.http_status == 404:
                res.http_status, res.error = None, OFFLINE_CACHE_MISS
            return res

    idx = enumerate_changelog(Offline({f"{P}/changelog": _cl("r1")}), base_url=BASE, section=SEC)
    assert not idx.incomplete and [e.slug for e in idx.entries] == ["r1"]
    idx = enumerate_changelog(Offline({}), base_url=BASE, section=SEC)
    assert idx.incomplete and idx.pages[0]["unexpected"]


def test_sitemap_server_error_is_flagged_but_missing_sitemap_is_not():
    resources = {
        f"{BASE}/sitemap.xml": 404,
        f"{P}/sitemap.xml": _index(f"{P}/s1.xml", f"{P}/s2.xml", f"{P}/s3.xml"),
        f"{P}/s1.xml": 503,
        f"{P}/s2.xml": 403,
        f"{P}/s3.xml": _urlset(f"{P}/docs/a"),
    }
    res = enumerate_sitemap(FakeFetcher({}, resources), base_url=BASE, section=SEC)
    assert res.incomplete
    flagged = {d["url"] for d in res.documents if d.get("unexpected")}
    assert flagged == {f"{P}/s1.xml"}
    assert [e.url for e in res.entries] == [f"{P}/docs/a"]

    resources[f"{P}/s1.xml"] = 404
    res = enumerate_sitemap(FakeFetcher({}, resources), base_url=BASE, section=SEC)
    assert not res.incomplete


def test_parse_ssr_props_replaces_lone_surrogates():
    # an emoji cut in half by a JS substring: "\ud83d" alone is valid JSON
    html = ('<script id="ssr-props" type="application/json">'
            '{"doc": {"body": "Great job \\ud83d", "ok": "\\ud83d\\ude00", "\\udc00k": 1}}'
            '</script>')
    data = parse_ssr_props(html)
    assert data["doc"]["body"] == "Great job �"
    assert data["doc"]["ok"] == "\U0001F600"  # a valid pair is untouched
    assert data["doc"]["�k"] == 1
    json.dumps(data, ensure_ascii=False).encode("utf-8")  # writable as UTF-8
