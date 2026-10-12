"""HTTP client with an on-disk cache, polite delays, and retries.

Cache layout: <cache_dir>/<relative-url-path>.html  (one file per page).
A query string becomes part of the file name (``changelog__page=2.html``) so
paginated indexes do not collide with page 1. Non-HTML resources (sitemaps)
live under ``<cache_dir>/_resources/`` so they can never shadow a page file.

Reruns hit the cache by default; pass ``force_refresh=True`` (``--no-cache``)
to refetch every URL once per run, or ``offline=True`` (``--offline``) to never
touch the network at all.

Why the extra care:
  * Cache writes are atomic (temp file in the same directory + ``os.replace``).
    A crash mid-write used to leave a truncated file that every later run
    treated as a valid cache hit. As a belt-and-braces check, a cached HTML
    file with no closing ``</html>``/``</body>`` tag is refetched when online.
    Cache files get the usual umask-based mode (``mkstemp`` alone would make
    them owner-only, unreadable to another user or container reusing them).
  * A response that was redirected keeps its final URL in a small sidecar
    file (``<cache file>.final-url``), so cache hits and ``--offline`` runs
    report the redirect too (the crawler must know a page actually came from
    elsewhere).
  * ``write_cache=False`` (``--dry-run``) fetches without writing anything.
  * 429/5xx responses honour ``Retry-After`` (seconds or HTTP-date, capped),
    and when retries run out the last status code is returned instead of
    being lost. No sleep happens after the final attempt.
  * ``fetch`` never raises for HTTP or network problems: it returns a
    ``FetchResult`` whose ``error`` says what went wrong, so the caller can
    classify every failure in the manifest.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional, Set
from urllib.parse import urlparse

import requests

USER_AGENT = (
    "EncompassRest-RAG-Assistant-Crawler/0.2 "
    "(+https://github.com/richie-rk/encompass-rag-assistant)"
)

HTML_ACCEPT = "text/html,application/xhtml+xml"
XML_ACCEPT = "application/xml,text/xml;q=0.9,*/*;q=0.5"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0

#: ``FetchResult.error`` for a URL that is not cached while ``offline=True``.
OFFLINE_CACHE_MISS = "offline cache miss"

_RESOURCE_DIR = "_resources"
_FINAL_URL_SUFFIX = ".final-url"
_SAFE_QUERY_RE = re.compile(r"[^A-Za-z0-9._=-]")
_MAX_SEGMENT = 180


@dataclass
class FetchResult:
    """Outcome of one fetch.

    ``http_status`` is None when no HTTP response exists: a network error
    after all retries, or an offline cache miss (``error == OFFLINE_CACHE_MISS``).
    ``html`` is the response text for any content type (XML for resources).
    """

    url: str
    html: str
    http_status: Optional[int]
    cache_hit: bool
    body_hash: str
    cache_path: Path
    error: Optional[str] = None
    final_url: Optional[str] = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.http_status == 200


def parse_retry_after(value: Optional[str], *, now: Optional[datetime] = None) -> Optional[float]:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date).

    Returns None when the header is absent or unparseable; never negative.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _html_looks_complete(text: str) -> bool:
    tail = text[-65536:].lower()
    return "</html>" in tail or "</body>" in tail


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_UMASK: Optional[int] = None


def _file_mode() -> int:
    """``0o666`` minus the process umask: the mode ``open()`` would give."""
    global _UMASK
    if _UMASK is None:
        _UMASK = os.umask(0)
        os.umask(_UMASK)
    return 0o666 & ~_UMASK


def _shorten(segment: str) -> str:
    if len(segment) <= _MAX_SEGMENT:
        return segment
    digest = hashlib.sha1(segment.encode("utf-8")).hexdigest()[:12]
    return f"{segment[:_MAX_SEGMENT]}_{digest}"


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        *,
        delay: float = 0.5,
        timeout: float = 30.0,
        retries: int = 3,
        force_refresh: bool = False,
        offline: bool = False,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retry_after: float = MAX_RETRY_AFTER_SECONDS,
        write_cache: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        # Offline runs only read; write_cache=False (--dry-run) fetches live
        # but must not create anything either.
        self.write_cache = write_cache and not offline
        if self.write_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.retries = max(1, int(retries))  # total attempts per URL
        self.force_refresh = force_refresh
        self.offline = offline
        self.max_retry_after = max_retry_after
        self._sleep = sleep
        self._clock = clock

        self.session = session if session is not None else requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": HTML_ACCEPT,
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_live_fetch: Optional[float] = None
        # URLs fetched live during this run: with force_refresh, a second
        # request for the same URL (welcome page, changelog index) reads the
        # fresh cache file instead of hitting the site again.
        self._refreshed: Set[str] = set()
        self.live_requests = 0

    # -- cache layout ----------------------------------------------------------

    def cache_path_for(self, url: str, *, ext: str = ".html", resource: bool = False) -> Path:
        """Map a URL to its cache file, never escaping ``cache_dir``.

        ``.``/empty segments are dropped and ``..`` segments are neutralised, so
        a hostile link like ``/a/../../etc/passwd`` stays inside the cache.
        """
        parsed = urlparse(url)
        raw_path = parsed.path.replace("\\", "/")
        segments = []
        for seg in raw_path.split("/"):
            if seg in ("", "."):
                continue
            if seg == "..":
                seg = "%2E%2E"
            segments.append(_shorten(seg))
        if not segments or raw_path.endswith("/"):
            segments.append("_index")
        if parsed.query:
            query = _SAFE_QUERY_RE.sub("_", parsed.query)
            if query != parsed.query or len(query) > 80:
                digest = hashlib.sha1(parsed.query.encode("utf-8")).hexdigest()[:10]
                query = f"{query[:80]}_{digest}"
            segments[-1] = _shorten(f"{segments[-1]}__{query}")
        if resource:
            segments.insert(0, _RESOURCE_DIR)
            if not segments[-1].endswith(ext):
                segments[-1] += ext
        else:
            segments[-1] += ext

        path = self.cache_dir.joinpath(*segments)
        root = self.cache_dir.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"cache path for {url!r} escapes cache dir: {resolved}")
        return path

    def _read_cache(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError):
            return None

    def _write_cache(self, path: Path, text: str) -> None:
        """Atomic write: temp file in the same directory, then ``os.replace``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(tmp_name, _file_mode())  # mkstemp creates 0600
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _final_url_path(path: Path) -> Path:
        return path.with_name(path.name + _FINAL_URL_SUFFIX)

    def _read_final_url(self, path: Path) -> Optional[str]:
        text = self._read_cache(self._final_url_path(path))
        return (text or "").strip() or None

    def _store(self, path: Path, text: str, *, url: str, final_url: Optional[str]) -> None:
        """Cache a 200 response together with where it was finally served from."""
        marker = self._final_url_path(path)
        if final_url and final_url != url:
            # Marker first: a crash in between must not leave a redirected
            # page in the cache looking like the requested one.
            self._write_cache(marker, final_url)
            self._write_cache(path, text)
            return
        self._write_cache(path, text)
        try:
            marker.unlink()
        except FileNotFoundError:
            pass

    # -- public API ------------------------------------------------------------

    def fetch(self, url: str) -> FetchResult:
        """Fetch an HTML page (cache first)."""
        return self._fetch(url, accept=HTML_ACCEPT, ext=".html", resource=False,
                           validate=_html_looks_complete)

    def fetch_resource(
        self, url: str, *, accept: str = XML_ACCEPT, ext: str = ".xml"
    ) -> FetchResult:
        """Fetch a non-HTML resource (e.g. sitemap XML) with the same cache,
        politeness and retry rules as pages. Cached under ``_resources/``."""
        return self._fetch(url, accept=accept, ext=ext, resource=True, validate=None)

    # -- internals -------------------------------------------------------------

    def _fetch(
        self,
        url: str,
        *,
        accept: str,
        ext: str,
        resource: bool,
        validate: Optional[Callable[[str], bool]],
    ) -> FetchResult:
        path = self.cache_path_for(url, ext=ext, resource=resource)

        use_cache = self.offline or not self.force_refresh or url in self._refreshed
        if use_cache:
            cached = self._read_cache(path)
            # Offline: any cached copy beats nothing. Online: a file that looks
            # truncated (pre-atomic-write crash) is refetched.
            if cached is not None and (self.offline or validate is None or validate(cached)):
                return FetchResult(
                    url=url, html=cached, http_status=200, cache_hit=True,
                    body_hash=_sha256(cached), cache_path=path,
                    final_url=self._read_final_url(path),
                )

        if self.offline:
            return FetchResult(
                url=url, html="", http_status=None, cache_hit=False, body_hash="",
                cache_path=path, error=OFFLINE_CACHE_MISS,
            )

        last_status: Optional[int] = None
        last_error = "no attempt made"
        attempt = 0
        for attempt in range(1, self.retries + 1):
            self._polite_wait()
            try:
                self.live_requests += 1
                resp = self.session.get(
                    url, timeout=self.timeout, allow_redirects=True,
                    headers={"Accept": accept},
                )
            except requests.RequestException as e:
                self._last_live_fetch = self._clock()
                last_status = None
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    self._sleep(self._backoff(attempt))
                continue
            self._last_live_fetch = self._clock()

            status = resp.status_code
            if status in RETRY_STATUSES:
                last_status = status
                last_error = f"HTTP {status}"
                if attempt < self.retries:
                    self._sleep(self._retry_delay(resp, attempt))
                continue

            final_url = getattr(resp, "url", None) or url
            if status != 200:
                return FetchResult(
                    url=url, html="", http_status=status, cache_hit=False,
                    body_hash="", cache_path=path, error=f"HTTP {status}",
                    final_url=final_url, attempts=attempt,
                )

            text = resp.text
            if self.write_cache:
                self._store(path, text, url=url, final_url=final_url)
                self._refreshed.add(url)
            return FetchResult(
                url=url, html=text, http_status=200, cache_hit=False,
                body_hash=_sha256(text), cache_path=path,
                final_url=final_url, attempts=attempt,
            )

        return FetchResult(
            url=url, html="", http_status=last_status, cache_hit=False, body_hash="",
            cache_path=path, attempts=attempt,
            error=f"retries exhausted after {attempt} attempt(s): {last_error}",
        )

    def _polite_wait(self) -> None:
        if self._last_live_fetch is None or self.delay <= 0:
            return
        gap = self._clock() - self._last_live_fetch
        if gap < self.delay:
            self._sleep(self.delay - gap + random.uniform(0, 0.1))

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(2 ** attempt, 30) + random.uniform(0, 1)

    def _retry_delay(self, resp: requests.Response, attempt: int) -> float:
        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
        if retry_after is not None:
            return min(retry_after, self.max_retry_after)
        return self._backoff(attempt)
