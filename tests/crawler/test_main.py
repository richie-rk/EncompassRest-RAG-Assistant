"""End-to-end crawler runs against a small synthetic ReadMe site (no network)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from scripts.crawler import __main__ as cli
from scripts.crawler.extract import _summarize_oas
from scripts.crawler.fetch import Fetcher
from scripts.crawler.oas import oas_fingerprint
from tests.crawler.fixtures.fake_site import FakeFetcher, LOGIN_HTML, ssr_html

BASE = "https://dev.example.com"
SEC = "dc"
P = f"{BASE}/{SEC}"

LOANS_OAS = {
    "openapi": "3.0.1",
    "info": {"title": "Loans", "version": "1"},
    "paths": {
        "/loans/{id}": {
            "get": {"operationId": "get-loan", "summary": "Get Loan"},
            "delete": {"operationId": "delete-loan", "summary": "Delete Loan"},
        },
        "/loans": {"post": {"operationId": "create-loan", "summary": "Create Loan"}},
    },
}
FOLDERS_OAS = {
    "openapi": "3.0.1",
    "info": {"title": "Folders", "version": "1"},
    "paths": {"/folders": {
        "get": {"operationId": "get-folders", "summary": "List Folders"},
        "post": {"operationId": "create-folder", "summary": "Create Folder"},
    }},
}

DOCS_SIDEBAR = [{
    "title": "Get Started", "slug": "get-started",
    "pages": [
        {"slug": "welcome", "title": "Welcome", "type": "basic"},
        {"slug": "html-guide", "title": "HTML Guide", "type": "basic"},
        {"slug": "partner-pdf", "title": "Partner Guide (PDF)", "type": "link",
         "link_url": "https://cdn.example.com/partner.pdf"},
        {"slug": "advanced", "title": "Advanced", "type": "basic"},
    ],
}]
WELCOME_REFS = [{"title": "Loans", "slug": "loans", "pages": [
    {"slug": "loans-overview", "title": "Loans Overview", "type": "basic"}]}]
FULL_REFS = [{"title": "Loans", "slug": "loans", "pages": [
    {"slug": "loans-overview", "title": "Loans Overview", "type": "basic", "pages": [
        {"slug": "get-loan", "title": "Get Loan", "type": "endpoint"},
        {"slug": "create-a-loan", "title": "Create Loan", "type": "endpoint"},
    ]}]}]


def _doc(slug, title, body="", **kw):
    return {"_id": f"id-{slug}", "slug": slug, "title": title, "body": body,
            "type": kw.pop("type", "basic"), **kw}


def _ref(slug, title, oas, body="", type_="endpoint"):
    return ssr_html({"sidebars": {"docs": DOCS_SIDEBAR, "refs": FULL_REFS},
                     "doc": _doc(slug, title, body, type=type_, isReference=True),
                     "oasDefinition": oas, "oasPublicUrl": f"{BASE}/oas/{oas['info']['title']}"})


def _changelog_page(*entries, single=None):
    props: Dict = {"changelogs": list(entries)}
    if single is not None:
        props["changelog"] = single
    return ssr_html(props)


REL2 = {"_id": "c2", "slug": "rel-2", "title": "Release 2", "body": "Two."}
REL1 = {"_id": "c1", "slug": "rel-1", "title": "Release 1",
        "body": "One. Fixes from [old](changelog:rel-old)."}
REL0 = {"_id": "c0", "slug": "rel-0", "title": "Release 0", "body": "Zero."}
RELOLD = {"_id": "cold", "slug": "rel-old", "title": "Old Release", "body": "Old."}

WELCOME_BODY = (
    "Welcome!\n\n<SharedNote />\n\n<MissingThing />\n\n"
    "Read the [secret guide](doc:secret-guide), the [forum](page:developer-forum) "
    "and the [old page](doc:removed-page)."
)


def site_pages() -> Dict:
    return {
        f"{P}/docs/welcome": ssr_html({
            "sidebars": {"docs": DOCS_SIDEBAR, "refs": WELCOME_REFS},
            "doc": _doc("welcome", "Welcome", WELCOME_BODY,
                        reusableContent=["SharedNote", "MissingThing"]),
            "reusableContent": {"SharedNote": {"body": "Shared note text."}},
        }),
        f"{P}/docs/html-guide": ssr_html({"doc": _doc(
            "html-guide", "HTML Guide",
            '[block:html]\n{\n  "html": "<h2>Auth flow</h2><p>Step one: get a token.</p>"'
            '\n}\n[/block]')}),
        f"{P}/docs/advanced": LOGIN_HTML,  # auth-gated
        f"{P}/docs/secret-guide": ssr_html({"doc": _doc("secret-guide", "Secret",
                                                        "Hidden text.", hidden=True)}),
        f"{P}/page/developer-forum": ssr_html({"custompage": {
            "_id": "cp-forum", "slug": "developer-forum", "title": "Developer Forum",
            "body": "", "html": "<h1>Forum</h1><p>Ask questions here.</p>"}}),
        f"{P}/reference/loans-overview": _ref("loans-overview", "Loans Overview", LOANS_OAS,
                                              body="All about loans.", type_="basic"),
        f"{P}/reference/get-loan": _ref("get-loan", "Get Loan", LOANS_OAS, body="Get one."),
        f"{P}/reference/create-a-loan": _ref("create-a-loan", "Create Loan", LOANS_OAS),
        f"{P}/reference/delete-loan": _ref("delete-loan", "Delete Loan", LOANS_OAS),
        f"{P}/reference/folders-overview": _ref("folders-overview", "Folders", FOLDERS_OAS,
                                                body="Folders.", type_="basic"),
        f"{P}/changelog": _changelog_page(REL2, REL1, single=REL2),
        f"{P}/changelog?page=2": _changelog_page(REL0, single=REL2),
        f"{P}/changelog?page=3": _changelog_page(single=REL2),
        f"{P}/changelog/rel-2": _changelog_page(single=REL2),
        # ReadMe bug the extractor must not repeat: newest entry as `changelog`
        f"{P}/changelog/rel-1": _changelog_page(REL2, REL1, single=REL2),
        f"{P}/changelog/rel-0": _changelog_page(single=REL2),
        f"{P}/changelog/rel-old": _changelog_page(single=RELOLD),
    }


NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'


def site_resources() -> Dict:
    return {
        f"{BASE}/sitemap.xml": 404,
        f"{P}/sitemap.xml": (
            f'<?xml version="1.0"?><urlset {NS}>'
            f"<url><loc>{P}/docs/welcome</loc></url>"
            f"<url><loc>{P}/reference/get-loan</loc></url>"
            f"<url><loc>{P}/reference/folders-overview</loc></url>"
            f"<url><loc>{BASE}/elsewhere/docs/x</loc></url>"
            "</urlset>"),
    }


def outputs(tmp_path: Path) -> Dict[str, Path]:
    d = tmp_path / "data"
    return {"out": d / "developer_connect.jsonl",
            "manifest": d / "developer_connect_manifest.jsonl",
            "oas": d / "developer_connect_oas.json",
            "report": d / "developer_connect_report.json"}


def argv(tmp_path: Path, *extra: str) -> List[str]:
    o = outputs(tmp_path)
    return ["--base-url", BASE, "--section", SEC, "--out", str(o["out"]),
            "--cache-dir", str(tmp_path / "cache"), *extra]


def load(tmp_path: Path):
    o = outputs(tmp_path)
    records = [json.loads(l) for l in o["out"].read_text(encoding="utf-8").splitlines()]
    manifest = [json.loads(l) for l in o["manifest"].read_text().splitlines()]
    registry = json.loads(o["oas"].read_text())
    report = json.loads(o["report"].read_text())
    return {r["url"]: r for r in records}, manifest, registry, report


# --- the full run ---------------------------------------------------------------


def test_end_to_end_crawl_of_synthetic_site(tmp_path):
    fetcher = FakeFetcher(site_pages(), site_resources())
    code = cli.main(argv(tmp_path), fetcher=fetcher)
    assert code == 0
    recs, manifest, registry, report = load(tmp_path)

    via = {url.replace(P, ""): r["discovered_via"] for url, r in recs.items()}
    assert via == {
        "/docs/welcome": "sidebar",
        "/docs/html-guide": "sidebar",
        "/reference/loans-overview": "sidebar",
        "/reference/get-loan": "reference_sidebar",      # upgraded from sitemap
        "/reference/create-a-loan": "reference_sidebar",
        "/reference/delete-loan": "oas_operation",       # linked from nowhere
        "/reference/folders-overview": "sitemap",
        "/docs/secret-guide": "link_bfs",
        "/page/developer-forum": "link_bfs",
        "/changelog/rel-2": "changelog_index",
        "/changelog/rel-1": "changelog_index",
        "/changelog/rel-0": "changelog_index",
        "/changelog/rel-old": "changelog_ref",
    }

    # hidden page included by default, flagged
    assert recs[f"{P}/docs/secret-guide"]["hidden"] is True
    # [block:html] guide keeps its text
    assert "Auth flow" in recs[f"{P}/docs/html-guide"]["body_md"]
    assert "Step one: get a token." in recs[f"{P}/docs/html-guide"]["body_md"]
    # custom page from HTML
    forum = recs[f"{P}/page/developer-forum"]
    assert forum["body_source"] == "html" and "Ask questions here." in forum["body_md"]
    # reusable content inlined / unresolved
    welcome = recs[f"{P}/docs/welcome"]
    assert "Shared note text." in welcome["body_md"]
    assert welcome["reusable_content_unresolved"] == ["MissingThing"]
    # changelog entries are never mislabelled
    assert recs[f"{P}/changelog/rel-1"]["body_md"].startswith("One.")
    assert recs[f"{P}/changelog/rel-0"]["body_md"] == "Zero."
    assert recs[f"{P}/changelog/rel-0"]["changelog_source"] == "index"
    assert recs[f"{P}/changelog/rel-old"]["title"] == "Old Release"

    # OAS: registry instead of embedded documents
    loans_id = oas_fingerprint(_summarize_oas(LOANS_OAS))
    folders_id = oas_fingerprint(_summarize_oas(FOLDERS_OAS))
    assert sorted(registry) == sorted([loans_id, folders_id])
    assert list(registry) == sorted(registry)
    assert registry[loans_id]["paths"] == LOANS_OAS["paths"]
    assert all("oas" not in r for r in recs.values())
    assert recs[f"{P}/reference/get-loan"]["operation"] == {
        "method": "get", "path": "/loans/{id}", "operationId": "get-loan",
        "summary": "Get Loan"}
    assert recs[f"{P}/reference/create-a-loan"]["operation"]["method"] == "post"
    assert recs[f"{P}/reference/delete-loan"]["operation"]["operationId"] == "delete-loan"
    assert recs[f"{P}/reference/loans-overview"]["operation"] is None
    assert recs[f"{P}/reference/loans-overview"]["oas_id"] == loans_id
    assert recs[f"{P}/reference/folders-overview"]["oas_id"] == folders_id

    # manifest: one row per fetch, provenance and error classes
    rows = {m["url"]: m for m in manifest}
    assert len(rows) == len(manifest)
    assert all("discovered_via" in m for m in manifest)
    assert rows[f"{P}/docs/secret-guide"]["source_url"] == f"{P}/docs/welcome"
    assert rows[f"{P}/reference/delete-loan"]["source_url"].startswith(f"{P}/reference/")
    assert rows[f"{P}/docs/removed-page"]["error_class"] == "broken_link"
    assert rows[f"{P}/docs/removed-page"]["unexpected"] is False
    assert rows[f"{P}/reference/create-loan"]["error_class"] == "expected_miss"
    assert rows[f"{P}/docs/advanced"]["error_class"] == "auth_required"
    assert f"{P}/docs/partner-pdf" not in rows  # external sidebar link not fetched
    assert "https://cdn.example.com/partner.pdf" not in fetcher.calls

    # report
    assert report["exit_code"] == 0
    assert report["totals"]["records"] == 13
    assert report["totals"]["oas_documents"] == 2
    assert set(report["frontier_by_discovered_via"]) == {
        "sidebar", "reference_sidebar", "sitemap", "changelog_index", "changelog_ref",
        "oas_operation", "link_bfs"}
    assert report["failures"]["expected_miss"]["count"] == 3  # create-loan, get/create-folder
    assert report["failures"]["broken_link"]["count"] == 1
    assert report["failures"]["auth_required"]["count"] == 1
    assert report["unexpected_failures"] == []
    missing = {(o["method"], o["path"]) for o in report["oas_operations_without_page"]}
    assert missing == {("get", "/folders"), ("post", "/folders")}
    assert report["reusable_content_unresolved"]["names"] == {"MissingThing": 1}
    assert report["html_conversion"]["body_from_html"] == [f"{P}/page/developer-forum"]
    assert f"{P}/docs/html-guide" in report["html_conversion"]["html_block_pages"]
    assert [l["url"] for l in report["external_links"]] == ["https://cdn.example.com/partner.pdf"]
    assert {g["url"] for g in report["navigation_gaps"]} == {f"{P}/docs/advanced"}
    assert report["cap_truncations"]["max_pages"]["not_enqueued"] == 0
    assert report["cap_truncations"]["changelog_index"]["truncated"] is False

    # no partial files left behind
    assert not list((tmp_path / "data").glob("*.partial"))


def test_skip_hidden_excludes_link_discovered_hidden_pages(tmp_path):
    code = cli.main(argv(tmp_path, "--skip-hidden"),
                    fetcher=FakeFetcher(site_pages(), site_resources()))
    assert code == 0
    recs, manifest, _, report = load(tmp_path)
    assert f"{P}/docs/secret-guide" not in recs
    assert all(not r["hidden"] for r in recs.values())
    row = next(m for m in manifest if m["url"] == f"{P}/docs/secret-guide")
    assert row["error_class"] == "skipped_hidden"
    assert report["skipped_hidden"] == [f"{P}/docs/secret-guide"]
    assert report["totals"]["skipped_hidden"] == 1
    assert report["options"]["skip_hidden"] is True
    assert report["unexpected_failures"] == []


def test_no_discover_keeps_to_welcome_sidebar_and_changelog(tmp_path):
    fetcher = FakeFetcher(site_pages(), site_resources())
    assert cli.main(argv(tmp_path, "--no-discover"), fetcher=fetcher) == 0
    recs, _, registry, report = load(tmp_path)
    assert fetcher.resource_calls == []  # no sitemap
    assert set(report["frontier_by_discovered_via"]) == {"sidebar", "changelog_index",
                                                        "changelog_ref"}
    assert f"{P}/reference/get-loan" not in recs
    assert len(registry) == 1  # loans-overview still contributes its OAS document


def test_include_hidden_is_accepted_as_noop(tmp_path):
    assert cli.main(argv(tmp_path, "--include-hidden"),
                    fetcher=FakeFetcher(site_pages(), site_resources())) == 0
    recs, *_ = load(tmp_path)
    assert recs[f"{P}/docs/secret-guide"]["hidden"] is True


def test_duplicate_urls_for_one_document_are_merged(tmp_path):
    pages = site_pages()
    # ReadMe serves the same doc at an old slug, linked from the html guide
    pages[f"{P}/docs/html-guide"] = ssr_html({"doc": _doc(
        "html-guide", "HTML Guide", "See [old](doc:html-guide-old).")})
    pages[f"{P}/docs/html-guide-old"] = pages[f"{P}/docs/html-guide"]
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources())) == 0
    recs, manifest, _, report = load(tmp_path)
    assert f"{P}/docs/html-guide-old" not in recs
    assert recs[f"{P}/docs/html-guide"]["alias_urls"] == [f"{P}/docs/html-guide-old"]
    row = next(m for m in manifest if m["url"] == f"{P}/docs/html-guide-old")
    assert row["duplicate_of"] == f"{P}/docs/html-guide"
    assert report["totals"]["duplicates_merged"] == 1


def test_unexpected_failure_sets_exit_code(tmp_path):
    pages = site_pages()
    pages[f"{P}/docs/html-guide"] = 503  # a sidebar page the server keeps failing
    code = cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources()))
    assert code == 1
    _, _, _, report = load(tmp_path)
    assert report["exit_code"] == 1
    assert [f["url"] for f in report["unexpected_failures"]] == [f"{P}/docs/html-guide"]
    assert report["failures"]["http_error"]["unexpected"] == 1


def test_max_pages_cap_is_counted(tmp_path):
    code = cli.main(argv(tmp_path, "--max-pages", "8"),
                    fetcher=FakeFetcher(site_pages(), site_resources()))
    assert code == 0
    recs, _, _, report = load(tmp_path)
    cap = report["cap_truncations"]["max_pages"]
    assert cap["cap"] == 8 and cap["not_enqueued"] > 0
    assert sum(cap["by_source"].values()) == cap["not_enqueued"]
    assert report["totals"]["frontier"] == 8


# --- never clobber good data ------------------------------------------------------


def _seed_existing(tmp_path: Path) -> Dict[Path, str]:
    contents = {}
    for name, path in outputs(tmp_path).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"PREVIOUS {name}\n", encoding="utf-8")
        contents[path] = path.read_text(encoding="utf-8")
    return contents


def test_ctrl_c_mid_crawl_leaves_previous_outputs_untouched(tmp_path):
    before = _seed_existing(tmp_path)

    def interrupt():
        raise KeyboardInterrupt

    fetcher = FakeFetcher(site_pages(), site_resources(),
                          hooks={f"{P}/reference/get-loan": interrupt})
    assert cli.main(argv(tmp_path), fetcher=fetcher) == 130
    for path, text in before.items():
        assert path.read_text(encoding="utf-8") == text
    partial = outputs(tmp_path)["out"].with_name("developer_connect.jsonl.partial")
    assert partial.exists() and "welcome" in partial.read_text(encoding="utf-8")


def test_crash_mid_crawl_leaves_previous_outputs_untouched(tmp_path, monkeypatch):
    before = _seed_existing(tmp_path)
    real = cli.walk_sidebars

    def buggy(sidebars, **kw):
        if kw.get("source_url") == f"{P}/reference/loans-overview":
            raise RuntimeError("crawler bug")
        return real(sidebars, **kw)

    monkeypatch.setattr(cli, "walk_sidebars", buggy)
    with pytest.raises(RuntimeError, match="crawler bug"):
        cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources()))
    for path, text in before.items():
        assert path.read_text(encoding="utf-8") == text
    assert outputs(tmp_path)["manifest"].with_name(
        "developer_connect_manifest.jsonl.partial").exists()


def test_one_page_extraction_error_is_isolated(tmp_path, monkeypatch):
    real = cli.extract_page

    def flaky(html, entry, **kw):
        if entry.slug == "html-guide":
            raise ValueError("converter bug")
        return real(html, entry, **kw)

    monkeypatch.setattr(cli, "extract_page", flaky)
    code = cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources()))
    assert code == 1
    recs, _, _, report = load(tmp_path)
    assert f"{P}/docs/html-guide" not in recs and f"{P}/docs/welcome" in recs
    assert report["unexpected_failures"][0]["error_class"] == "extract_error"
    assert "converter bug" in report["unexpected_failures"][0]["error"]


def test_missing_welcome_page_writes_nothing(tmp_path):
    before = _seed_existing(tmp_path)
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher({})) == 1
    for path, text in before.items():
        assert path.read_text(encoding="utf-8") == text
    assert not list((tmp_path / "data").glob("*.partial"))


# --- options and paths -------------------------------------------------------------


def test_limit_without_out_writes_sample_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(cli, "default_data_dir", lambda: data)
    code = cli.main(["--base-url", BASE, "--section", SEC, "--limit", "3",
                     "--cache-dir", str(tmp_path / "cache")],
                    fetcher=FakeFetcher(site_pages(), site_resources()))
    assert code == 0
    names = sorted(p.name for p in data.iterdir())
    assert names == ["developer_connect.sample.jsonl", "developer_connect_manifest.sample.jsonl",
                     "developer_connect_oas.sample.json", "developer_connect_report.sample.json"]
    manifest = (data / "developer_connect_manifest.sample.jsonl").read_text().splitlines()
    assert len(manifest) == 3  # 3 pages fetched; the third is auth-gated
    lines = (data / "developer_connect.sample.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_default_paths():
    args = cli.parse_args([])
    repo_data = Path(cli.__file__).resolve().parents[1] / "data"
    assert cli.default_data_dir() == repo_data  # running from the source tree
    assert args.out == repo_data / "developer_connect.jsonl"
    assert args.manifest == repo_data / "developer_connect_manifest.jsonl"
    assert args.oas_out == repo_data / "developer_connect_oas.json"
    assert args.report == repo_data / "developer_connect_report.json"
    assert args.cache_dir == repo_data / "cache"
    assert args.max_pages == 5000
    assert not args.skip_hidden and not args.offline

    custom = cli.parse_args(["--out", "/x/y/foo.jsonl", "--oas-out", "/z/o.json"])
    assert custom.manifest == Path("/x/y/foo_manifest.jsonl")
    assert custom.report == Path("/x/y/foo_report.json")
    assert custom.oas_out == Path("/z/o.json")


def test_installed_package_falls_back_to_cwd_data(tmp_path, monkeypatch):
    pkg = tmp_path / "site-packages" / "crawler"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(cli, "PKG_DIR", pkg)
    monkeypatch.chdir(tmp_path)
    assert cli.default_data_dir() == tmp_path / "data"


def test_dry_run_writes_nothing(tmp_path):
    assert cli.main(argv(tmp_path, "--dry-run"),
                    fetcher=FakeFetcher(site_pages(), site_resources())) == 0
    assert not (tmp_path / "data").exists()


def test_offline_with_empty_cache_fails_cleanly(tmp_path, monkeypatch, capsys):
    """The real Fetcher in --offline mode with nothing cached: exit 1, a clear
    message, no network, and nothing created or written."""
    def no_network(*a, **k):
        raise AssertionError("network used in --offline mode")

    monkeypatch.setattr("requests.Session.get", no_network)
    seeded = _seed_existing(tmp_path)
    cache = tmp_path / "empty-cache"
    rc = cli.main(argv(tmp_path, "--offline", "--cache-dir", str(cache)))
    assert rc == 1
    out = " ".join(capsys.readouterr().out.split())  # rich wraps long lines
    assert "offline cache miss" in out and "Nothing was written" in out
    assert "--offline only reads the HTML cache" in out
    assert not cache.exists()
    for path, text in seeded.items():
        assert path.read_text(encoding="utf-8") == text
    assert not list(tmp_path.rglob("*.partial"))


def test_offline_rerun_from_cache_without_network(tmp_path):
    """Crawl once through a real Fetcher (fake HTTP session) to fill the cache,
    then re-extract with --offline while any network call would fail."""
    pages, resources = site_pages(), site_resources()

    class Resp:
        def __init__(self, status, text, url):
            self.status_code, self.text, self.url, self.headers = status, text, url, {}

    class SiteSession:
        headers: Dict = {}

        def get(self, url, **_):
            served = resources.get(url, pages.get(url, 404))
            if isinstance(served, int):
                return Resp(served, "", url)
            return Resp(200, served, url)

    class NoNetwork:
        headers: Dict = {}

        def get(self, *a, **k):
            raise AssertionError("network used in --offline mode")

    cache = tmp_path / "cache" / SEC
    online = Fetcher(cache, delay=0, session=SiteSession(), sleep=lambda s: None)
    assert cli.main(argv(tmp_path), fetcher=online) == 0
    first, *_ = load(tmp_path)

    args = cli.parse_args(argv(tmp_path, "--offline"))
    offline = cli.make_fetcher(args)
    assert offline.offline is True
    offline.session = NoNetwork()
    assert cli.main(argv(tmp_path, "--offline"), fetcher=offline) == 0
    second, manifest, _, report = load(tmp_path)
    assert set(second) == set(first)
    assert all(m["cache_hit"] for m in manifest if m.get("http_status") == 200)
    # 404s were never cached, so offline they surface as cache misses (not failures)
    assert report["failures"]["offline_cache_miss"]["unexpected"] == 0
    assert report["options"]["offline"] is True


# --- regressions ------------------------------------------------------------------


class _SiteSession:
    """A requests-like session serving the synthetic site (for the real Fetcher)."""

    headers: Dict = {}

    def __init__(self, pages, resources, fail=lambda url: False):
        self.pages, self.resources, self.fail = pages, resources, fail
        self.calls: List[str] = []

    def get(self, url, **_):
        import requests

        self.calls.append(url)
        if self.fail(url):
            raise requests.ConnectionError("network is down")
        served = self.resources.get(url, self.pages.get(url, 404))
        resp = type("Resp", (), {})()
        resp.url, resp.headers = url, {}
        resp.status_code, resp.text = (served, "") if isinstance(served, int) else (200, served)
        return resp


def _real_fetcher(tmp_path: Path, session, **kw) -> Fetcher:
    return Fetcher(tmp_path / "cache" / SEC, delay=0, retries=1, session=session,
                   sleep=lambda s: None, **kw)


def test_changelog_entry_referenced_before_its_index_listing_keeps_its_record(tmp_path):
    """rel-3's body references rel-2, which the index lists next; /changelog/rel-2
    serves the newest entry, so only the index copy of rel-2 has its text."""
    rel3 = {"_id": "c3", "slug": "rel-3", "title": "Release 3",
            "body": "See [previous](changelog:rel-2)."}
    pages = site_pages()
    pages[f"{P}/changelog"] = _changelog_page(rel3, REL2, single=rel3)
    pages[f"{P}/changelog?page=2"] = _changelog_page(REL1, single=rel3)
    pages[f"{P}/changelog?page=3"] = _changelog_page(REL0, single=rel3)
    pages[f"{P}/changelog?page=4"] = _changelog_page(single=rel3)
    pages[f"{P}/changelog/rel-3"] = _changelog_page(single=rel3)
    pages[f"{P}/changelog/rel-2"] = _changelog_page(single=rel3)  # serves the newest
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources())) == 0
    recs, _, _, report = load(tmp_path)
    rel2 = recs[f"{P}/changelog/rel-2"]
    assert rel2["body_md"] == "Two." and rel2["changelog_source"] == "index"
    assert rel2["discovered_via"] == "changelog_index"
    assert rel2["breadcrumb"] == ["Release Notes", "Release 2"]
    assert recs[f"{P}/changelog/rel-0"]["body_md"] == "Zero."
    assert report["changelog_index"]["entries"] == 4
    # an entry only ever referenced gets its real title in the breadcrumb
    assert recs[f"{P}/changelog/rel-old"]["breadcrumb"] == ["Release Notes", "Old Release"]


def test_changelog_index_server_error_fails_the_run(tmp_path):
    pages = site_pages()
    pages[f"{P}/changelog?page=2"] = 503  # rel-0 is listed only there
    code = cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources()))
    assert code == 1
    recs, _, _, report = load(tmp_path)
    assert f"{P}/changelog/rel-0" not in recs
    assert report["changelog_index"]["incomplete"] is True
    assert [(f["url"], f["error_class"]) for f in report["unexpected_failures"]] == [
        (f"{P}/changelog?page=2", "http_error")]


def test_sitemap_server_error_fails_the_run_but_a_missing_sitemap_does_not(tmp_path):
    resources = site_resources()
    resources[f"{P}/sitemap.xml"] = 502
    code = cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), resources))
    assert code == 1
    _, _, _, report = load(tmp_path)
    assert report["sitemap"]["incomplete"] is True
    assert [f["url"] for f in report["unexpected_failures"]] == [f"{P}/sitemap.xml"]
    # the site-level sitemap.xml is a 404 in every run: tolerated
    assert report["failures"]["http_error"]["count"] == 1


class _RedirectingFetcher(FakeFetcher):
    def __init__(self, pages, resources, redirects):
        super().__init__(pages, resources)
        self.redirects = redirects

    def fetch(self, url):
        target = self.redirects.get(url)
        if target is None:
            return super().fetch(url)
        self.calls.append(url)
        res = self._serve(target, self.pages)
        res.url, res.final_url = url, target
        return res


def test_redirect_out_of_the_section_is_neither_stored_nor_harvested(tmp_path):
    foreign = ssr_html({
        "sidebars": {"docs": [{"title": "Partner", "slug": "p", "pages": [
            {"slug": "partner-a", "title": "Partner A"},
            {"slug": "partner-b", "title": "Partner B"}]}]},
        "doc": _doc("partner-a", "Partner A", "Another project's page."),
    })
    pages = site_pages()
    pages[f"{BASE}/other/docs/partner-a"] = foreign
    fetcher = _RedirectingFetcher(pages, site_resources(),
                                  {f"{P}/docs/html-guide": f"{BASE}/other/docs/partner-a"})
    assert cli.main(argv(tmp_path), fetcher=fetcher) == 0
    recs, manifest, _, report = load(tmp_path)
    assert f"{P}/docs/html-guide" not in recs
    assert not any("Another project" in r["body_md"] for r in recs.values())
    assert f"{P}/docs/partner-a" not in fetcher.calls  # foreign sidebar not walked
    row = next(m for m in manifest if m["url"] == f"{P}/docs/html-guide")
    assert row["error_class"] == "off_section_redirect" and row["unexpected"] is False
    assert row["final_url"] == f"{BASE}/other/docs/partner-a"
    assert f"{BASE}/other/docs/partner-a" in {l["url"] for l in report["external_links"]}


def test_offline_cache_miss_of_a_navigation_page_is_unexpected(tmp_path):
    pages, resources = site_pages(), site_resources()
    online = _real_fetcher(tmp_path, _SiteSession(pages, resources))
    assert cli.main(argv(tmp_path), fetcher=online) == 0
    # a sidebar page's cache file goes missing (partly filled cache)
    online.cache_path_for(f"{P}/docs/html-guide").unlink()
    (tmp_path / "data" / "developer_connect.jsonl").unlink()  # nothing to protect
    offline = _real_fetcher(tmp_path, _SiteSession({}, {}), offline=True)
    assert cli.main(argv(tmp_path, "--offline"), fetcher=offline) == 1
    _, _, _, report = load(tmp_path)
    assert [(f["url"], f["error_class"]) for f in report["unexpected_failures"]] == [
        (f"{P}/docs/html-guide", "offline_cache_miss")]
    # misses of guessed/linked URLs (404s are never cached) stay expected
    assert report["failures"]["offline_cache_miss"]["count"] == 5
    assert report["failures"]["offline_cache_miss"]["unexpected"] == 1


def test_offline_run_over_partial_cache_does_not_replace_the_dataset(tmp_path):
    pages, resources = site_pages(), site_resources()
    assert cli.main(argv(tmp_path),
                    fetcher=_real_fetcher(tmp_path, _SiteSession(pages, resources))) == 0
    before = {p: p.read_bytes() for p in outputs(tmp_path).values()}
    # The cache is wiped and refilled by a 2-page debug run.
    import shutil

    shutil.rmtree(tmp_path / "cache")
    sample = tmp_path / "sample" / "developer_connect.sample.jsonl"
    assert cli.main(argv(tmp_path, "--limit", "2", "--out", str(sample)),
                    fetcher=_real_fetcher(tmp_path, _SiteSession(pages, resources))) == 0
    offline = _real_fetcher(tmp_path, _SiteSession({}, {}), offline=True)
    assert cli.main(argv(tmp_path, "--offline"), fetcher=offline) == 1
    assert {p: p.read_bytes() for p in outputs(tmp_path).values()} == before
    partial = outputs(tmp_path)["report"].with_name("developer_connect_report.json.partial")
    report = json.loads(partial.read_text())
    assert report["published"] is False and report["exit_code"] == 1
    assert "13 already published" in report["not_published_reason"]


def test_failed_run_with_far_fewer_records_keeps_the_published_dataset(tmp_path):
    pages, resources = site_pages(), site_resources()
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, resources)) == 0
    before = {p: p.read_bytes() for p in outputs(tmp_path).values()}

    # The network drops after the seed documents: every page fetch fails.
    seeds = {f"{P}/docs/welcome", f"{P}/changelog", f"{P}/changelog?page=2",
             f"{P}/changelog?page=3"}
    session = _SiteSession(pages, resources,
                           fail=lambda url: url not in seeds and "sitemap" not in url)
    assert cli.main(argv(tmp_path, "--no-cache"), fetcher=_real_fetcher(
        tmp_path, session, force_refresh=True)) == 1
    assert {p: p.read_bytes() for p in outputs(tmp_path).values()} == before
    assert outputs(tmp_path)["out"].with_name("developer_connect.jsonl.partial").exists()

    # --force-publish replaces them anyway
    assert cli.main(argv(tmp_path, "--no-cache", "--force-publish"), fetcher=_real_fetcher(
        tmp_path, session, force_refresh=True)) == 1
    recs, _, _, report = load(tmp_path)
    assert len(recs) < 13 and report["published"] is True
    assert not list((tmp_path / "data").glob("*.partial"))


def test_publish_guard_thresholds():
    reason = cli.publish_blocked_reason
    assert reason(records=0, previous=0, exit_code=1) is None  # nothing to protect
    assert reason(records=0, previous=900, exit_code=0)  # an empty dataset never replaces one
    assert reason(records=500, previous=900, exit_code=1)
    assert reason(records=500, previous=900, exit_code=0) is None  # a clean run is trusted
    assert reason(records=890, previous=900, exit_code=1) is None  # a few failures publish


@pytest.mark.parametrize("flag", ["--max-pages", "--limit"])
def test_negative_caps_are_rejected(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args([flag, "-1"])
    assert exc.value.code == 2
    assert "must be 0 or more" in capsys.readouterr().err
    assert getattr(cli.parse_args([flag, "0"]), flag[2:].replace("-", "_")) == 0


def test_failed_publish_restores_every_previous_output(tmp_path, monkeypatch, capsys):
    before = _seed_existing(tmp_path)
    records_target = outputs(tmp_path)["out"]
    real_replace = os.replace

    def locked(src, dst):  # e.g. the records file held open by another process on Windows
        if Path(dst) == records_target and str(src).endswith(".partial"):
            raise PermissionError("file in use")
        return real_replace(src, dst)

    monkeypatch.setattr(cli.os, "replace", locked)
    with pytest.raises(PermissionError):
        cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources()))
    for path, text in before.items():  # the registry was replaced first, then restored
        assert path.read_text(encoding="utf-8") == text
    data = tmp_path / "data"
    assert not list(data.glob("*.bak"))
    partials = sorted(p.name for p in data.glob("*.partial"))
    assert len(partials) == 4
    registry = json.loads((data / "developer_connect_oas.json.partial").read_text())
    assert len(registry) == 2  # the new outputs are kept for inspection
    assert "Previous output files were left untouched" in capsys.readouterr().out


def test_dry_run_with_the_real_fetcher_writes_nothing(tmp_path, monkeypatch):
    session = _SiteSession(site_pages(), site_resources())
    monkeypatch.setattr("requests.Session.get", lambda self, url, **kw: session.get(url))
    assert cli.main(argv(tmp_path, "--dry-run", "--delay", "0")) == 0
    assert f"{P}/docs/welcome" in session.calls  # the seeds were fetched ...
    assert list(tmp_path.iterdir()) == []      # ... but nothing was written, not even the cache


def test_lone_surrogate_in_page_json_does_not_abort_the_crawl(tmp_path):
    pages = site_pages()
    pages[f"{P}/docs/html-guide"] = pages[f"{P}/docs/html-guide"].replace(
        "Step one: get a token.", "Great job \\\\ud83d")
    assert "\\ud83d" in pages[f"{P}/docs/html-guide"]
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources())) == 0
    recs, *_ = load(tmp_path)
    assert "Great job �" in recs[f"{P}/docs/html-guide"]["body_md"]


def test_lone_surrogate_in_ssr_props_json_does_not_abort_the_crawl(tmp_path):
    """The ssr-props JSON itself carries the lone surrogate escape (not a
    legacy block inside a body): parse_ssr_props must repair it, or writing
    the record as UTF-8 aborts the crawl."""
    pages = site_pages()
    pages[f"{P}/docs/secret-guide"] = pages[f"{P}/docs/secret-guide"].replace(
        '"Hidden text."', '"Hidden text \\ud83d"')
    assert '"Hidden text \\ud83d"' in pages[f"{P}/docs/secret-guide"]
    assert "[block:" not in pages[f"{P}/docs/secret-guide"]
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources())) == 0
    recs, *_ = load(tmp_path)
    assert recs[f"{P}/docs/secret-guide"]["body_md"] == "Hidden text �"


def test_section_without_a_changelog_is_not_a_failure(tmp_path):
    pages = {url: html for url, html in site_pages().items() if "/changelog" not in url}
    code = cli.main(argv(tmp_path), fetcher=FakeFetcher(pages, site_resources()))
    assert code == 0
    recs, _, _, report = load(tmp_path)
    assert not any("/changelog/" in url for url in recs)
    assert report["changelog_index"]["absent"] is True
    assert report["changelog_index"]["incomplete"] is False
    assert report["changelog_index"]["pages"][0]["note"] == "section has no changelog"
    assert report["unexpected_failures"] == [] and report["exit_code"] == 0
    # with a changelog, the flag is off
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources())) == 0
    assert load(tmp_path)[3]["changelog_index"]["absent"] is False


def test_backup_left_by_a_failed_restore_survives_later_runs(tmp_path, monkeypatch, capsys):
    """Run 1's rollback cannot restore the registry, so its previous version
    is only in <registry>.bak. Run 2 fails the same way and run 3 succeeds:
    neither may delete or overwrite that .bak."""
    before = _seed_existing(tmp_path)
    o = outputs(tmp_path)
    real_replace = os.replace
    broken = {"on": True}

    def replace(src, dst):
        if broken["on"]:
            if Path(dst) == o["out"] and str(src).endswith(".partial"):
                raise PermissionError("records file in use")
            if Path(src) == o["oas"] and str(dst).endswith(".partial"):
                raise PermissionError("registry file in use")  # rollback fails too
        return real_replace(src, dst)

    monkeypatch.setattr(cli.os, "replace", replace)
    monkeypatch.setattr(cli.console, "width", 1000)  # keep paths on one line
    kept = o["oas"].with_name(o["oas"].name + ".bak")
    for run in (1, 2):
        with pytest.raises(PermissionError):
            cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources()))
        assert kept.read_text(encoding="utf-8") == before[o["oas"]]
        out = capsys.readouterr().out
        assert "could not be restored" in out
        # the message names the backup this run made, not a glob
        made = kept if run == 1 else o["oas"].with_name(o["oas"].name + ".1.bak")
        assert f"{o['oas']}  (previous version: {made})" in out

    broken["on"] = False
    assert cli.main(argv(tmp_path), fetcher=FakeFetcher(site_pages(), site_resources())) == 0
    assert kept.read_text(encoding="utf-8") == before[o["oas"]]
    # only backups from failed restores remain; this run's own were cleaned up
    assert sorted(p.name for p in (tmp_path / "data").glob("*.bak")) == [
        "developer_connect_oas.json.1.bak", "developer_connect_oas.json.bak"]
