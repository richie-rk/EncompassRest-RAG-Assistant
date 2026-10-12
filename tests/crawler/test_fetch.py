"""Fetcher: cache layout and atomicity, offline mode, retries, Retry-After."""
from __future__ import annotations

import os
import stat
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import requests

from scripts.crawler import fetch as fetch_mod
from scripts.crawler.fetch import (
    MAX_RETRY_AFTER_SECONDS,
    OFFLINE_CACHE_MISS,
    XML_ACCEPT,
    Fetcher,
    parse_retry_after,
)

URL = "https://dev.example.com/dc/docs/welcome"
PAGE = "<html><body>hello</body></html>"


class FakeResponse:
    def __init__(self, status: int, text: str = "", headers: Optional[Dict] = None,
                 url: str = URL):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self.url = url


class FakeSession:
    """Plays back scripted responses (or exceptions) in order."""

    def __init__(self, script: List):
        self.script = list(script)
        self.headers: Dict[str, str] = {}
        self.calls: List[Dict] = []

    def get(self, url, timeout=None, allow_redirects=True, headers=None):
        self.calls.append({"url": url, "headers": headers or {}})
        if not self.script:
            raise AssertionError(f"unexpected request for {url}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class NoNetwork:
    headers: Dict[str, str] = {}

    def get(self, *a, **k):
        raise AssertionError("network access attempted")


def make(tmp_path: Path, session, **kw) -> tuple:
    sleeps: List[float] = []
    f = Fetcher(tmp_path / "cache", delay=0, session=session, sleep=sleeps.append, **kw)
    return f, sleeps


# --- cache --------------------------------------------------------------------


def test_live_fetch_then_cache_hit(tmp_path):
    session = FakeSession([FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session)
    first = f.fetch(URL)
    assert first.http_status == 200 and not first.cache_hit
    assert first.cache_path == tmp_path / "cache" / "dc" / "docs" / "welcome.html"
    second = f.fetch(URL)  # served from disk; a second network call would fail
    assert second.cache_hit and second.html == PAGE and second.body_hash == first.body_hash
    assert len(session.calls) == 1


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    f, _ = make(tmp_path, FakeSession([FakeResponse(200, PAGE)]))
    path = f.cache_path_for(URL)
    path.parent.mkdir(parents=True)
    path.write_text("<html>old complete</html>", encoding="utf-8")

    replaced = []
    real_replace = fetch_mod.os.replace

    def boom(src, dst):
        replaced.append((Path(src), Path(dst)))
        raise OSError("disk full")

    monkeypatch.setattr(fetch_mod.os, "replace", boom)
    with pytest.raises(OSError):
        f._write_cache(path, PAGE)
    # temp file lived next to the target, the target is untouched, nothing left over
    src, dst = replaced[0]
    assert src.parent == dst.parent == path.parent and dst == path
    assert path.read_text(encoding="utf-8") == "<html>old complete</html>"
    assert sorted(p.name for p in path.parent.iterdir()) == ["welcome.html"]

    monkeypatch.setattr(fetch_mod.os, "replace", real_replace)
    f._write_cache(path, PAGE)
    assert path.read_text(encoding="utf-8") == PAGE
    assert sorted(p.name for p in path.parent.iterdir()) == ["welcome.html"]


def test_truncated_cache_file_is_refetched_online_but_used_offline(tmp_path):
    session = FakeSession([FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session)
    path = f.cache_path_for(URL)
    path.parent.mkdir(parents=True)
    path.write_text("<html><body>trunc", encoding="utf-8")  # legacy crash leftover

    offline, _ = make(tmp_path, NoNetwork(), offline=True)
    res = offline.fetch(URL)
    assert res.cache_hit and res.html.endswith("trunc")

    res = f.fetch(URL)
    assert not res.cache_hit and res.html == PAGE
    assert path.read_text(encoding="utf-8") == PAGE


def test_cache_path_stays_inside_cache_dir(tmp_path):
    f, _ = make(tmp_path, NoNetwork())
    root = (tmp_path / "cache").resolve()
    for url in (
        "https://x.com/dc/../../../../etc/passwd",
        "https://x.com/../..",
        "https://x.com/dc/docs/..%2F..%2Fsecret",
        "https://x.com/dc\\..\\..\\evil",
        "https://x.com/",
        "https://x.com",
    ):
        p = f.cache_path_for(url).resolve()
        assert root in p.parents, url
    assert f.cache_path_for("https://x.com/a/../b").name == "b.html"
    assert ".." not in f.cache_path_for("https://x.com/dc/../../etc/passwd").parts


def test_query_strings_get_their_own_cache_file(tmp_path):
    f, _ = make(tmp_path, NoNetwork())
    base = f.cache_path_for("https://x.com/dc/changelog")
    p2 = f.cache_path_for("https://x.com/dc/changelog?page=2")
    p3 = f.cache_path_for("https://x.com/dc/changelog?page=3")
    assert len({base, p2, p3}) == 3
    assert base.name == "changelog.html" and p2.name == "changelog__page=2.html"
    weird = f.cache_path_for("https://x.com/dc/changelog?q=a/../b&x=<y>")
    assert weird.parent == base.parent and "/" not in weird.name


def test_force_refresh_refetches_once_per_run(tmp_path):
    session = FakeSession([FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session, force_refresh=True)
    path = f.cache_path_for(URL)
    path.parent.mkdir(parents=True)
    path.write_text("<html>stale</html>", encoding="utf-8")
    assert f.fetch(URL).html == PAGE
    again = f.fetch(URL)
    assert again.cache_hit and again.html == PAGE
    assert len(session.calls) == 1


def test_non_200_is_returned_and_not_cached(tmp_path):
    f, sleeps = make(tmp_path, FakeSession([FakeResponse(404, "nope")]))
    res = f.fetch(URL)
    assert res.http_status == 404 and res.html == "" and res.error == "HTTP 404"
    assert not f.cache_path_for(URL).exists()
    assert sleeps == []


# --- offline ------------------------------------------------------------------


def test_offline_never_touches_the_network(tmp_path):
    f, _ = make(tmp_path, NoNetwork(), offline=True, force_refresh=True)
    miss = f.fetch(URL)
    assert miss.http_status is None and miss.error == OFFLINE_CACHE_MISS
    assert not miss.cache_hit and miss.html == ""

    path = f.cache_path_for(URL)
    path.parent.mkdir(parents=True)
    path.write_text(PAGE, encoding="utf-8")
    hit = f.fetch(URL)
    assert hit.http_status == 200 and hit.cache_hit and hit.html == PAGE

    res = f.fetch_resource("https://dev.example.com/sitemap.xml")
    assert res.error == OFFLINE_CACHE_MISS
    assert f.live_requests == 0


# --- retries ------------------------------------------------------------------


def test_retry_after_seconds_is_honoured(tmp_path):
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"}),
                           FakeResponse(200, PAGE)])
    f, sleeps = make(tmp_path, session)
    res = f.fetch(URL)
    assert res.http_status == 200 and res.attempts == 2
    assert sleeps == [7.0]


def test_retry_after_is_capped(tmp_path):
    session = FakeSession([FakeResponse(503, headers={"Retry-After": "3600"}),
                           FakeResponse(200, PAGE)])
    f, sleeps = make(tmp_path, session)
    assert f.fetch(URL).http_status == 200
    assert sleeps == [MAX_RETRY_AFTER_SECONDS]


def test_retry_after_http_date(tmp_path):
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    header = format_datetime(now + timedelta(seconds=30), usegmt=True)
    assert parse_retry_after(header, now=now) == pytest.approx(30.0)
    assert parse_retry_after(format_datetime(now - timedelta(seconds=5), usegmt=True),
                             now=now) == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after(" 12 ") == 12.0

    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20), usegmt=True)
    session = FakeSession([FakeResponse(429, headers={"Retry-After": future}),
                           FakeResponse(200, PAGE)])
    f, sleeps = make(tmp_path, session)
    assert f.fetch(URL).http_status == 200
    assert len(sleeps) == 1 and 15 <= sleeps[0] <= 20


def test_exhausted_retries_keep_last_status_and_do_not_sleep_after_last_try(tmp_path):
    session = FakeSession([FakeResponse(503), FakeResponse(502), FakeResponse(503)])
    f, sleeps = make(tmp_path, session, retries=3)
    res = f.fetch(URL)
    assert res.http_status == 503
    assert "retries exhausted" in res.error and res.attempts == 3
    assert len(sleeps) == 2  # between attempts only
    assert not f.cache_path_for(URL).exists()


def test_network_errors_are_reported_not_raised(tmp_path):
    session = FakeSession([requests.ConnectionError("reset"), requests.Timeout("slow")])
    f, sleeps = make(tmp_path, session, retries=2)
    res = f.fetch(URL)
    assert res.http_status is None and "Timeout" in res.error
    assert len(sleeps) == 1


def test_network_error_then_success(tmp_path):
    session = FakeSession([requests.ConnectionError("reset"), FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session)
    assert f.fetch(URL).html == PAGE


def test_polite_delay_between_live_fetches(tmp_path):
    clock = iter([0.0, 0.1, 0.2, 0.3]).__next__
    sleeps: List[float] = []
    session = FakeSession([FakeResponse(200, PAGE), FakeResponse(200, PAGE, url=URL + "2")])
    f = Fetcher(tmp_path / "c", delay=1.0, session=session, sleep=sleeps.append, clock=clock)
    f.fetch(URL)
    f.fetch(URL + "2")
    assert len(sleeps) == 1 and 0.9 <= sleeps[0] <= 1.0


# --- resources ----------------------------------------------------------------


def test_fetch_resource_uses_xml_accept_and_separate_cache(tmp_path):
    xml = "<?xml version='1.0'?><urlset/>"
    session = FakeSession([FakeResponse(200, xml)])
    f, _ = make(tmp_path, session)
    url = "https://dev.example.com/dc/sitemap.xml"
    res = f.fetch_resource(url)
    assert res.http_status == 200 and res.html == xml
    assert session.calls[0]["headers"]["Accept"] == XML_ACCEPT
    assert res.cache_path == tmp_path / "cache" / "_resources" / "dc" / "sitemap.xml"
    again = f.fetch_resource(url)
    assert again.cache_hit and again.html == xml


# --- regressions ----------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_cache_files_get_the_umask_mode_not_owner_only(tmp_path):
    old = os.umask(0o022)
    fetch_mod._UMASK = None  # re-read the umask set for this test
    try:
        f, _ = make(tmp_path, FakeSession([FakeResponse(200, PAGE)]))
        res = f.fetch(URL)
        assert stat.S_IMODE(res.cache_path.stat().st_mode) == 0o644
    finally:
        os.umask(old)
        fetch_mod._UMASK = None


def test_redirect_target_survives_cache_hits_and_offline_runs(tmp_path):
    moved = "https://dev.example.com/other/docs/partner"
    session = FakeSession([FakeResponse(200, PAGE, url=moved), FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session)
    assert f.fetch(URL).final_url == moved
    hit = f.fetch(URL)
    assert hit.cache_hit and hit.final_url == moved
    offline, _ = make(tmp_path, NoNetwork(), offline=True)
    assert offline.fetch(URL).final_url == moved
    # once the URL stops redirecting the marker is dropped
    refresh, _ = make(tmp_path, session, force_refresh=True)
    assert refresh.fetch(URL).final_url == URL
    assert f.fetch(URL).final_url is None
    assert sorted(p.name for p in f.cache_path_for(URL).parent.iterdir()) == ["welcome.html"]


def test_write_cache_false_fetches_without_writing_anything(tmp_path):
    session = FakeSession([FakeResponse(200, PAGE), FakeResponse(200, PAGE)])
    f, _ = make(tmp_path, session, write_cache=False)
    assert f.fetch(URL).html == PAGE
    assert f.fetch(URL).html == PAGE  # not cached, so fetched again
    assert len(session.calls) == 2
    assert not (tmp_path / "cache").exists()
