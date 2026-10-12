"""Offline test doubles for the crawler: a fake fetcher and HTML builders.

Nothing here touches the network. Pages are plain dicts keyed by URL; a value
is either the HTML string to serve (200) or an int HTTP status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from scripts.crawler.fetch import FetchResult

Served = Union[str, int]


def ssr_html(props: Dict, *, title: str = "Docs", extra_body: str = "") -> str:
    """A minimal ReadMe-like page with ``props`` embedded as ssr-props JSON."""
    payload = json.dumps(props).replace("</", "<\\/")
    return (
        f"<!DOCTYPE html><html><head><title>{title}</title></head><body>"
        f"<div id=\"app\">{extra_body}</div>"
        f'<script id="ssr-props" type="application/json">{payload}</script>'
        "</body></html>"
    )


LOGIN_HTML = (
    "<!DOCTYPE html><html><head><title>Login</title></head>"
    '<body><form class="loginForm"></form></body></html>'
)


class FakeFetcher:
    """Serves ``pages`` (HTML) and ``resources`` (XML) from dicts.

    ``hooks`` maps a URL to a callable run before serving it (e.g. to raise
    KeyboardInterrupt mid-crawl). Unknown URLs are 404s. Every call is logged.
    """

    def __init__(
        self,
        pages: Dict[str, Served],
        resources: Optional[Dict[str, Served]] = None,
        hooks: Optional[Dict[str, Callable[[], None]]] = None,
    ):
        self.pages = dict(pages)
        self.resources = dict(resources or {})
        self.hooks = dict(hooks or {})
        self.calls: List[str] = []
        self.resource_calls: List[str] = []

    def _serve(self, url: str, table: Dict[str, Served]) -> FetchResult:
        if url in self.hooks:
            self.hooks[url]()
        served = table.get(url, 404)
        if isinstance(served, int):
            return FetchResult(url=url, html="", http_status=served, cache_hit=False,
                               body_hash="", cache_path=Path("/dev/null"),
                               error=f"HTTP {served}")
        return FetchResult(url=url, html=served, http_status=200, cache_hit=False,
                           body_hash=hashlib.sha256(served.encode()).hexdigest(),
                           cache_path=Path("/dev/null"), final_url=url)

    def fetch(self, url: str) -> FetchResult:
        self.calls.append(url)
        return self._serve(url, self.pages)

    def fetch_resource(self, url: str, **_: object) -> FetchResult:
        self.resource_calls.append(url)
        return self._serve(url, self.resources)
