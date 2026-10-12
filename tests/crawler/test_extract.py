"""Record extraction: new data contract, reusable content, HTML bodies, changelogs."""
from __future__ import annotations

import json

from scripts.crawler import extract as extract_mod
from scripts.crawler.extract import (
    ERR_AUTH_REQUIRED,
    ERR_CHANGELOG_MISMATCH,
    ERR_NO_DOC,
    ERR_NO_SSR_PROPS,
    _summarize_oas,
    extract_internal_links,
    extract_page,
    extract_record,
    find_reusable_content,
)
from scripts.crawler.oas import oas_fingerprint
from scripts.crawler.sidebar import FrontierEntry
from tests.crawler.fixtures.fake_site import LOGIN_HTML, ssr_html

BASE = "https://dev.example.com"
SEC = "dc"
P = f"{BASE}/{SEC}"

OAS = {
    "openapi": "3.0.1",
    "info": {"title": "Loans", "version": "1"},
    "servers": [{"url": "https://api.example.com"}],
    "x-readme": {"internal": True},
    "paths": {
        "/loans/{id}": {
            "get": {"operationId": "get-loan", "summary": "Get Loan"},
            "delete": {"operationId": "delete-loan", "summary": "Delete Loan"},
            "parameters": [{"name": "id", "in": "path"}],
        },
        "/loans": {"post": {"operationId": "create-loan", "summary": "Create Loan"}},
    },
    "components": {"schemas": {"Loan": {"type": "object", "title": "Loan"}}},
}


def entry(slug, kind="guide", via="sidebar", **kw):
    seg = {"guide": "docs", "reference": "reference", "changelog": "changelog",
           "custompage": "page"}[kind]
    return FrontierEntry(slug=slug, url=f"{P}/{seg}/{slug}", kind=kind, title=slug,
                         discovered_via=via, **kw)


def run(props, e, **kw):
    return extract_page(ssr_html(props), e, base_url=BASE, section=SEC, **kw)


# --- doc pages ----------------------------------------------------------------


def test_guide_record_keeps_fields_and_gains_discovered_via():
    doc = {"_id": "d1", "slug": "intro", "title": "Intro", "type": "basic",
           "body": "Hello [next](doc:next-page).", "hidden": False, "revision": 3,
           "category": {"title": "Start", "slug": "start", "type": "guide", "_id": "c1"},
           "api": {"method": "get", "url": ""}, "tutorials": ["t"]}
    e = entry("intro", via="link_bfs", breadcrumb=["Start", "Intro"])
    ext = run({"doc": doc}, e)
    r = ext.record
    assert r["discovered_via"] == "link_bfs"
    assert r["doc_id"] == "d1" and r["kind"] == "guide" and r["revision"] == 3
    assert r["body_md"] == f"Hello [next]({P}/docs/next-page)."
    assert r["body_md_chars"] == len(r["body_md"]) and r["raw_body_chars"] == len(doc["body"])
    assert r["category"] == {"title": "Start", "slug": "start", "type": "guide", "id": "c1"}
    assert r["doc_api"] == doc["api"] and r["doc_tutorials"] == ["t"]
    assert r["breadcrumb"] == ["Start", "Intro"]
    assert "oas" not in r and "oas_id" not in r and "operation" not in r
    assert ext.oas_id is None and ext.error is None
    assert "reusable_content_unresolved" not in r


def test_reference_page_stores_oas_by_id_not_inline():
    doc = {"slug": "get-loan", "title": "Get Loan", "type": "endpoint", "body": "",
           "isReference": True, "api": {"method": "get", "url": ""}}
    ext = run({"doc": doc, "oasDefinition": OAS, "oasPublicUrl": "https://x/oas.json"},
              entry("get-loan", "reference"))
    r = ext.record
    summary = _summarize_oas(OAS)
    assert "x-readme" not in summary
    assert ext.oas_doc == summary
    assert ext.oas_id == oas_fingerprint(summary) == r["oas_id"]
    assert r["operation"] == {"method": "get", "path": "/loans/{id}",
                              "operationId": "get-loan", "summary": "Get Loan"}
    assert r["oas_public_url"] == "https://x/oas.json"
    assert "oas" not in r
    assert sorted(ext.operation_ids) == ["create-loan", "delete-loan", "get-loan"]
    assert len(json.dumps(r)) < 2000  # no OpenAPI payload in the record


def test_reference_operation_by_summary_and_overview_without_operation():
    doc = {"slug": "create-a-loan", "title": "Create Loan", "type": "endpoint", "body": ""}
    r = run({"doc": doc, "oasDefinition": OAS}, entry("create-a-loan", "reference")).record
    assert r["operation"]["method"] == "post" and r["operation"]["operationId"] == "create-loan"

    doc = {"slug": "loans-overview", "title": "Loans Overview", "type": "basic", "body": "x"}
    r = run({"doc": doc, "oasDefinition": OAS}, entry("loans-overview", "reference")).record
    assert r["operation"] is None and r["oas_id"] == oas_fingerprint(_summarize_oas(OAS))

    # a page served under an alias URL still maps through the requested slug
    doc = {"slug": "renamed", "title": "Something else", "type": "endpoint", "body": ""}
    r = run({"doc": doc, "oasDefinition": OAS}, entry("delete-loan", "reference")).record
    assert r["operation"]["operationId"] == "delete-loan"


def test_reference_page_without_oas_gets_null_fields():
    doc = {"slug": "about", "title": "About", "type": "basic", "body": "x"}
    r = run({"doc": doc, "oasDefinition": {"paths": {}}}, entry("about", "reference")).record
    assert r["oas_id"] is None and r["operation"] is None and "oas" not in r


def test_ssr_props_is_parsed_once(monkeypatch):
    calls = []
    real = extract_mod.parse_ssr_props

    def counting(html):
        calls.append(1)
        return real(html)

    monkeypatch.setattr(extract_mod, "parse_ssr_props", counting)
    run({"doc": {"slug": "a", "title": "A", "body": "[b](doc:b)"}, "sidebars": {"docs": []}},
        entry("a"))
    assert len(calls) == 1


def test_html_block_guide_keeps_its_text():
    body = ('[block:html]\n{\n  "html": "<h2>Auth flow</h2><p>Step one: get a '
            '<b>token</b>.</p><a href=\\"/dc/docs/tokens\\">tokens</a>"\n}\n[/block]')
    ext = run({"doc": {"slug": "auth", "title": "Auth", "body": body}}, entry("auth"))
    md = ext.record["body_md"]
    assert "Auth flow" in md and "Step one: get a **token**." in md
    assert "[block:html]" not in md
    assert ext.html_blocks == 1
    # links inside the JSON-escaped HTML are discovered too
    assert f"{P}/docs/tokens" in {link.url for link in ext.links}


def test_auth_and_no_ssr_pages():
    ext = extract_page(LOGIN_HTML, entry("x"), base_url=BASE, section=SEC)
    assert ext.record is None and ext.error == ERR_AUTH_REQUIRED and not ext.has_ssr
    ext = extract_page("<html>%PDF</html>", entry("x"), base_url=BASE, section=SEC)
    assert ext.error == ERR_NO_SSR_PROPS
    ext = run({"sidebars": {}}, entry("x"))
    assert ext.error == ERR_NO_DOC and ext.has_ssr


# --- reusable content ---------------------------------------------------------


def test_find_reusable_content_search_shapes():
    ssr = {
        "doc": {"title": "WarnA", "body": "page body, never a component"},
        "oasDefinition": {"components": {"WarnB": "from the OAS: must be skipped"}},
        "rdmd": {"reusableContent": {"WarnA": {"body": "A text"}, "WarnB": "B text"}},
        "deep": [{"x": {"items": [{"tag": "WarnC", "source": "C text"}]}}],
        "lists": {"customBlocks": [{"name": "WarnD", "content": "D text"},
                                   {"name": "WarnE", "content": "  "}]},
    }
    found = find_reusable_content(ssr, ["WarnA", "WarnB", "WarnC", "WarnD", "WarnE"],
                                  exclude=[ssr["doc"]])
    assert found == {"WarnA": "A text", "WarnB": "B text", "WarnC": "C text",
                     "WarnD": "D text"}
    # the page doc itself is not taken as a component even when titles collide
    assert find_reusable_content({"doc": ssr["doc"]}, ["WarnA"], exclude=[ssr["doc"]]) == {}
    assert find_reusable_content(ssr, []) == {}


def test_reusable_content_is_inlined_and_unresolved_names_recorded():
    props = {
        "doc": {"slug": "w", "title": "W", "body": "Intro.\n\n<SharedNote />\n\n<Missing />",
                "reusableContent": ["SharedNote", "Missing", "DeclaredOnly"]},
        "reusableContent": {"SharedNote": {"body": "Shared note text."}},
    }
    ext = run(props, entry("w"))
    r = ext.record
    assert "Shared note text." in r["body_md"]
    assert "<Missing" not in r["body_md"]
    assert r["reusable_content"] == ["SharedNote", "Missing", "DeclaredOnly"]
    assert r["reusable_content_unresolved"] == ["Missing", "DeclaredOnly"]
    assert ext.reusable_unresolved == ["Missing", "DeclaredOnly"]


# --- custom pages -------------------------------------------------------------


def test_custom_page_falls_back_to_html():
    cp = {"_id": "cp1", "slug": "developer-forum", "title": "Developer Forum", "body": "",
          "html": "<h1>Forum</h1><p>Ask questions <a href='/dc/docs/faq'>here</a>.</p>"}
    ext = run({"custompage": cp}, entry("developer-forum", "custompage", via="link_bfs"))
    r = ext.record
    assert r["kind"] == "custompage" and r["body_source"] == "html" and ext.body_from_html
    assert "Forum" in r["body_md"] and "Ask questions" in r["body_md"]
    assert r["html_chars"] == len(cp["html"]) and r["doc_id"] == "cp1"
    assert r["discovered_via"] == "link_bfs"
    assert f"{P}/docs/faq" in {link.url for link in ext.links}
    # Converted HTML is rewritten like any body: no relative or macro links left.
    assert f"[here]({P}/docs/faq)" in r["body_md"]


def test_custom_page_html_macros_are_rewritten():
    cp = {"slug": "lender-mfa", "title": "Lender MFA", "body": "",
          "html": '<p>Read <a href="doc:mfa-setup#steps">setup</a> first.</p>'
                  "<pre><code>&lt;Loan/&gt; [x](doc:keep)</code></pre>"}
    r = run({"custompage": cp}, entry("lender-mfa", "custompage")).record
    assert f"[setup]({P}/docs/mfa-setup#steps)" in r["body_md"]
    assert "(doc:mfa-setup" not in r["body_md"]
    # code stays byte-for-byte, including what looks like a macro or a tag
    assert "<Loan/> [x](doc:keep)" in r["body_md"]


def test_custom_page_markdown_body_via_rewrite_and_htmlmode():
    body = '[block:html]\n{\n  "html": "<p>Create an <b>API key</b>.</p>"\n}\n[/block]'
    r = run({"custompage": {"slug": "api-key", "title": "API Key", "body": body}},
            entry("api-key", "custompage")).record
    assert "Create an **API key**." in r["body_md"] and r["body_source"] == "markdown"

    cp = {"slug": "x", "title": "X", "body": "stale markdown", "htmlmode": True,
          "html": "<p>Fresh html</p>"}
    r = run({"custompage": cp}, entry("x", "custompage")).record
    assert "Fresh html" in r["body_md"] and r["body_source"] == "html"


# --- changelog ----------------------------------------------------------------

REL2 = {"slug": "rel-2", "title": "Release 2", "body": "Two."}
REL1 = {"slug": "rel-1", "title": "Release 1", "body": "One."}


def test_changelog_singular_entry_is_used_when_slug_matches():
    r = run({"changelog": REL2}, entry("rel-2", "changelog", via="changelog_index")).record
    assert r["title"] == "Release 2" and r["body_md"] == "Two."
    assert r["discovered_via"] == "changelog_index" and "changelog_source" not in r


def test_changelog_mismatch_prefers_matching_list_entry():
    r = run({"changelog": REL2, "changelogs": [REL2, REL1]},
            entry("rel-1", "changelog")).record
    assert r["slug"] == "rel-1" and r["body_md"] == "One."


def test_changelog_mismatch_falls_back_to_index_entry():
    index = {"rel-0": {"slug": "rel-0", "title": "Release 0", "body": "Zero."}}
    r = run({"changelog": REL2}, entry("rel-0", "changelog"), changelog_index=index).record
    assert r["slug"] == "rel-0" and r["body_md"] == "Zero." and r["changelog_source"] == "index"


def test_changelog_mismatch_without_match_is_not_mislabelled():
    ext = run({"changelog": REL2}, entry("old-slug", "changelog"), discover=False)
    assert ext.record is None and ext.error == ERR_CHANGELOG_MISMATCH
    assert "rel-2" in ext.detail
    # the slug actually served is queued so its content is not lost
    assert [(link.url, link.discovered_via) for link in ext.links] == [
        (f"{P}/changelog/rel-2", "changelog_ref")]


# --- link discovery -----------------------------------------------------------


def test_link_discovery_sources_and_kinds():
    body = (
        "See [key](page:api-key), [sso](/dc/page/saml-sso), [r](changelog:rel-9), "
        '[t](ref:get-loan "Get a loan"), ![img](doc:not-an-image-link), '
        "[abs](https://dev.example.com/dc/docs/abs#x), [ext](https://google.com/x)"
    )
    html = ssr_html({"doc": {"slug": "a", "title": "A", "body": body}},
                    extra_body='<a href="/dc/reference/nav-only">nav</a>'
                               '<a href="/dc/changelog/rel-nav">nav</a>')
    links = extract_internal_links(html, base_url=BASE, section=SEC, source_url=f"{P}/docs/a")
    got = {(l.url, l.kind, l.discovered_via) for l in links}
    assert got == {
        (f"{P}/page/api-key", "custompage", "link_bfs"),
        (f"{P}/page/saml-sso", "custompage", "link_bfs"),
        (f"{P}/changelog/rel-9", "changelog", "changelog_ref"),
        (f"{P}/reference/get-loan", "reference", "link_bfs"),
        (f"{P}/docs/abs", "guide", "link_bfs"),
        (f"{P}/reference/nav-only", "reference", "link_bfs"),
        (f"{P}/changelog/rel-nav", "changelog", "link_bfs"),
    }
    assert {l.source_url for l in links} == {f"{P}/docs/a"}


def test_extract_record_wrapper_still_works():
    r = extract_record(ssr_html({"doc": {"slug": "a", "title": "A", "body": "x"}}),
                       entry("a"), base_url=BASE, section=SEC)
    assert r["slug"] == "a" and r["body_md"] == "x"
    assert extract_record("<html></html>", entry("a"), base_url=BASE, section=SEC) is None
