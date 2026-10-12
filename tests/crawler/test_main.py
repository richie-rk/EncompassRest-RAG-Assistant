"""End-to-end crawler runs against a small synthetic ReadMe site (no network)."""
from __future__ import annotations

import json
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
