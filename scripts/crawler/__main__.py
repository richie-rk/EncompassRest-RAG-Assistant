"""CLI entry point for the Developer Connect crawler.

Usage:
    python -m scripts.crawler                        # full crawl, defaults
    python -m scripts.crawler --limit 10             # debug: first 10 pages -> *.sample.* files
    python -m scripts.crawler --no-cache             # force refetch all pages
    python -m scripts.crawler --offline              # re-extract from the HTML cache only
    python -m scripts.crawler --dry-run              # enumerate the seed frontier only
    python -m scripts.crawler --skip-hidden          # exclude hidden:true pages everywhere

Outputs (default: next to --out):
    developer_connect.jsonl            one record per page (see extract.py)
    developer_connect_manifest.jsonl   one row per fetch attempt
    developer_connect_oas.json         {oas_id: OpenAPI document}, each stored once
    developer_connect_report.json      completeness report

Never clobbers good data: every output is staged as ``<file>.partial`` and moved
into place with ``os.replace`` only after the crawl completes (previous files
are backed up while that happens and restored if a move fails). On an
exception or Ctrl-C the previous files stay untouched and the partial files
are kept for inspection. A run that would replace a published dataset with
no records, or that had unexpected failures and lost more than
``1 - PUBLISH_MIN_RECORD_RATIO`` of the published records, is not published
either (``--force-publish`` overrides).

Exit code: 0, or 1 when there were *unexpected* failures (see
``UNEXPECTED_CLASSES``) or the results were not published. Expected misses
(OAS operationId guesses that 404, broken author links, auth-gated pages,
offline cache misses of URLs that did not come from the site's navigation)
are reported but do not fail the run. 130 on Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Set, Tuple
from urllib.parse import urlparse

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn,
)

from .extract import PageExtraction, extract_page
from .fetch import OFFLINE_CACHE_MISS, Fetcher
from .oas import iter_operations
from .sidebar import (
    NAVIGATION_SOURCES,
    SLUG_RE,
    SOURCE_PRIORITY,
    Frontier,
    FrontierEntry,
    SeedResult,
    enumerate_seeds,
    page_url,
    url_key,
    walk_sidebars,
)

console = Console()

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://developer.icemortgagetechnology.com"
DEFAULT_MAX_PAGES = 5000

#: Failure classes that make the run exit 1, always ...
UNEXPECTED_CLASSES = frozenset({"not_found", "http_error", "network_error", "fetch_error",
                                "extract_error"})
#: ... and these only for URLs that came from the site's own navigation.
#: Offline, a navigation URL missing from the cache is a gap like a 404 online
#: (a partly filled cache must not pass as a complete re-extraction).
UNEXPECTED_IF_NAVIGATION = frozenset({"no_ssr_props", "no_doc_or_custompage",
                                      "offline_cache_miss"})

#: A run with unexpected failures that produced fewer than this share of the
#: already published records is not published over them.
PUBLISH_MIN_RECORD_RATIO = 0.9


def default_data_dir() -> Path:
    """``scripts/data`` when running from the source tree; ``./data`` (cwd)
    when the crawler is installed as a package and that directory is absent."""
    candidate = PKG_DIR.parent / "data"
    if PKG_DIR.parent.name == "scripts" and candidate.is_dir():
        return candidate
    return Path.cwd() / "data"


def derived_paths(out: Path) -> Dict[str, Path]:
    """Sibling output paths for a records file.

    ``developer_connect.jsonl``        -> ``developer_connect_manifest.jsonl``,
                                          ``developer_connect_oas.json``, ``..._report.json``
    ``developer_connect.sample.jsonl`` -> ``developer_connect_manifest.sample.jsonl`` etc.
    """
    name = out.name
    core = name[: -len(".jsonl")] if name.endswith(".jsonl") else out.stem
    suffix = ""
    if core.endswith(".sample"):
        core, suffix = core[: -len(".sample")], ".sample"
    return {
        "manifest": out.with_name(f"{core}_manifest{suffix}.jsonl"),
        "oas": out.with_name(f"{core}_oas{suffix}.json"),
        "report": out.with_name(f"{core}_report{suffix}.json"),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more (0 = no limit), got {value}")
    return value


def parse_args(argv: List[str]) -> argparse.Namespace:
    data_dir = default_data_dir()
    p = argparse.ArgumentParser(
        prog="scripts.crawler",
        description=(
            "Crawl ICE MT Developer Connect docs into JSONL. "
            f"Default data dir: {data_dir} (the repo's scripts/data when run from the "
            "source tree, otherwise ./data under the current directory)."
        ),
    )
    p.add_argument("--section", default="developer-connect",
                   help="ReadMe section/project slug (default: developer-connect)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"Site base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSONL path (default: <data dir>/developer_connect.jsonl, or "
                        "developer_connect.sample.jsonl when --limit is set)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Per-URL fetch manifest path (default: next to --out, "
                        "<name>_manifest.jsonl)")
    p.add_argument("--oas-out", type=Path, default=None,
                   help="OAS registry path (default: next to --out, "
                        "e.g. developer_connect_oas.json)")
    p.add_argument("--report", type=Path, default=None,
                   help="Completeness report path (default: next to --out, "
                        "e.g. developer_connect_report.json)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="HTML cache directory (default: <data dir>/cache)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Minimum seconds between live fetches (default: 0.5)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Per-request timeout in seconds (default: 30)")
    p.add_argument("--retries", type=int, default=3,
                   help="Attempts per URL on 429/5xx/network errors (default: 3)")
    p.add_argument("--no-cache", action="store_true",
                   help="Refetch every URL even if cached")
    p.add_argument("--offline", action="store_true",
                   help="Never touch the network: re-extract from the HTML cache "
                        "(uncached URLs are reported as offline cache misses; a miss on a "
                        "URL from the site's navigation fails the run)")
    p.add_argument("--skip-hidden", action="store_true",
                   help="Exclude pages with hidden=true (sidebar, links, sitemap and OAS "
                        "seeds alike). Hidden pages are included by default.")
    p.add_argument("--include-hidden", action="store_true",
                   help="No-op, kept for backward compatibility (hidden pages are "
                        "included by default)")
    p.add_argument("--limit", type=_non_negative_int, default=0,
                   help="Stop after N pages (debug; 0 = no limit). Disables discovery and "
                        "writes to *.sample.* files unless --out is given.")
    p.add_argument("--max-pages", type=_non_negative_int, default=DEFAULT_MAX_PAGES,
                   help="Hard ceiling on the total frontier, all sources included "
                        f"(default: {DEFAULT_MAX_PAGES}; 0 = unlimited). URLs refused by "
                        "the cap are counted in the report.")
    p.add_argument("--no-discover", action="store_true",
                   help="Disable discovery beyond the welcome sidebar and changelog index "
                        "(no sitemap, sidebar harvesting, OAS seeds or link following)")
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate the seed frontier and exit. Only the seed documents "
                        "(welcome page, changelog index, sitemaps) are fetched, from the "
                        "cache when present; nothing is written, not even the cache")
    p.add_argument("--force-publish", action="store_true",
                   help="Replace the existing outputs even when the run produced no records, "
                        "or had unexpected failures and far fewer records than the "
                        "published dataset")
    args = p.parse_args(argv)

    if args.out is None:
        args.out = data_dir / ("developer_connect.sample.jsonl" if args.limit
                               else "developer_connect.jsonl")
    derived = derived_paths(args.out)
    args.manifest = args.manifest or derived["manifest"]
    args.oas_out = args.oas_out or derived["oas"]
    args.report = args.report or derived["report"]
    args.cache_dir = args.cache_dir or data_dir / "cache"
    return args


def make_fetcher(args: argparse.Namespace) -> Fetcher:
    return Fetcher(
        cache_dir=args.cache_dir / args.section,
        delay=args.delay, timeout=args.timeout, retries=args.retries,
        force_refresh=args.no_cache, offline=args.offline,
        write_cache=not args.dry_run,
    )


# --- failure classification -----------------------------------------------------


def classify_fetch_failure(status: Optional[int], error: Optional[str], via: str) -> str:
    if error == OFFLINE_CACHE_MISS:
        return "offline_cache_miss"
    if status in (404, 410):
        if via == "oas_operation":
            return "expected_miss"
        if via in NAVIGATION_SOURCES:
            return "not_found"
        return "broken_link"
    if status is None:
        return "network_error"
    return "http_error"


def is_unexpected(error_class: str, via: str) -> bool:
    if error_class in UNEXPECTED_CLASSES:
        return True
    return error_class in UNEXPECTED_IF_NAVIGATION and via in NAVIGATION_SOURCES


# --- staged outputs ---------------------------------------------------------------


class StagedOutputs:
    """Write every output to ``<target>.partial`` and publish them together."""

    #: Publish order. Registry first: records reference it by oas_id.
    ORDER = ("oas", "records", "manifest", "report")

    def __init__(self, *, records: Path, manifest: Path, oas: Path, report: Path):
        self.targets = {"records": records, "manifest": manifest, "oas": oas, "report": report}
        self.partials = {k: v.with_name(v.name + ".partial") for k, v in self.targets.items()}
        #: Targets left torn by a failed publish that could not be rolled back.
        self.not_restored: List[Path] = []
        for path in self.targets.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.records_f: IO[str] = self.partials["records"].open("w", encoding="utf-8")
        self.manifest_f: IO[str] = self.partials["manifest"].open("w", encoding="utf-8")

    def write_record(self, record: Dict) -> None:
        self.records_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records_f.flush()

    def write_manifest(self, row: Dict) -> None:
        self.manifest_f.write(json.dumps(row) + "\n")
        self.manifest_f.flush()

    def _close(self) -> None:
        for f in (self.records_f, self.manifest_f):
            if not f.closed:
                f.close()

    def abort(self) -> List[Path]:
        self._close()
        return [p for p in self.partials.values() if p.exists()]

    def finish(self, *, records: List[Dict], registry: Dict[str, Dict], report: Dict,
               publish: bool = True) -> None:
        """Write the final partial files; move them into place if ``publish``."""
        self._close()
        # Final record list (duplicates merged), rewritten over the streamed copy.
        with self.partials["records"].open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.partials["oas"].open("w", encoding="utf-8") as f:
            json.dump({k: registry[k] for k in sorted(registry)}, f, ensure_ascii=False)
        with self.partials["report"].open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        if publish:
            self._publish()

    @staticmethod
    def _backup(target: Path) -> Optional[Path]:
        if not target.exists():
            return None
        backup = target.with_name(target.name + ".bak")
        backup.unlink(missing_ok=True)
        try:
            os.link(target, backup)  # cheap, and the target is never missing
        except OSError:
            shutil.copy2(target, backup)
        return backup

    def _publish(self) -> None:
        """Replace every target, or none: a failed move (e.g. a file held open
        on Windows) restores the targets already replaced, so records never
        end up next to a registry from another crawl."""
        replaced: List[Tuple[str, Optional[Path]]] = []
        backup: Optional[Path] = None
        try:
            for key in self.ORDER:
                backup = self._backup(self.targets[key])
                os.replace(self.partials[key], self.targets[key])
                replaced.append((key, backup))
                backup = None
        except BaseException:
            if backup is not None:
                backup.unlink(missing_ok=True)
            self._rollback(replaced)
            raise
        for _, done in replaced:
            if done is not None:
                done.unlink(missing_ok=True)

    def _rollback(self, replaced: List[Tuple[str, Optional[Path]]]) -> None:
        for key, backup in reversed(replaced):
            target, partial = self.targets[key], self.partials[key]
            try:
                os.replace(target, partial)  # the new output goes back to .partial
                if backup is not None:
                    os.replace(backup, target)
            except OSError:
                self.not_restored.append(target)


# --- the crawl ------------------------------------------------------------------


def _identity(record: Dict) -> str:
    if record.get("doc_id"):
        return f"id:{record['doc_id']}"
    return f"{record.get('kind')}:{record.get('slug')}"


def _url_tail(url: str) -> str:
    return url_key(url).rsplit("/", 1)[-1]


@dataclass
class Crawl:
    args: argparse.Namespace
    fetcher: Any
    frontier: Frontier
    seeds: SeedResult
    discover: bool
    staged: Optional[StagedOutputs] = None

    records: Dict[str, Dict] = field(default_factory=dict)       # identity -> record
    record_urls: Dict[str, List[str]] = field(default_factory=dict)  # identity -> all URLs
    registry: Dict[str, Dict] = field(default_factory=dict)
    manifest_rows: Dict[str, Dict] = field(default_factory=dict)  # url -> last row
    failures: List[Dict] = field(default_factory=list)
    duplicates: List[Dict] = field(default_factory=list)
    skipped_hidden: List[str] = field(default_factory=list)
    external_links: Dict[Tuple[str, str], Dict] = field(default_factory=dict)
    html_block_pages: Dict[str, int] = field(default_factory=dict)
    seeded_oas: Set[str] = field(default_factory=set)
    processed: int = 0
    fetched_ok: int = 0
    cache_hits: int = 0
    _cap_warned: bool = False

    def add_external(self, links: List[Dict]) -> None:
        for link in links:
            key = (link.get("title") or "", link.get("url") or "")
            self.external_links.setdefault(key, link)

    def _enqueue(self, entries: List[FrontierEntry]) -> None:
        self.frontier.add_many(entries)
        if self.frontier.cap_hit and not self._cap_warned:
            self._cap_warned = True
            console.print(f"[bold red]WARNING: --max-pages cap ({self.frontier.max_pages}) "
                          "reached; further URLs are NOT enqueued (counted in the report)."
                          "[/bold red]")

    # -- per page --------------------------------------------------------------

    def process(self, entry: FrontierEntry) -> None:
        self.processed += 1
        row: Dict[str, Any] = {
            "url": entry.url, "slug": entry.slug, "kind": entry.kind,
            "fetched_at": _now_iso(),
            "discovered_via": entry.discovered_via,
            "source_url": entry.source_url,
        }
        try:
            res = self.fetcher.fetch(entry.url)
            exc_error = None
        except Exception as e:  # Fetcher reports HTTP/network problems itself
            res, exc_error = None, f"{type(e).__name__}: {e}"

        if res is None or res.http_status != 200:
            status = getattr(res, "http_status", None)
            error = exc_error or getattr(res, "error", None) or f"HTTP {status}"
            cls = "fetch_error" if exc_error else classify_fetch_failure(
                status, error, entry.discovered_via)
            row.update({"http_status": status, "cache_hit": False, "body_hash": None,
                        "error": error[:300], "error_class": cls})
            self._fail(entry, row)
            return

        self.fetched_ok += 1
        self.cache_hits += int(bool(res.cache_hit))
        row.update({"http_status": res.http_status, "cache_hit": res.cache_hit,
                    "body_hash": res.body_hash})
        final_url = getattr(res, "final_url", None)
        if final_url and url_key(final_url) != url_key(entry.url):
            row["final_url"] = final_url
            if not self._in_section(final_url):
                # Another project's (or site's) page: storing it under this
                # URL, or walking its sidebar as ours, would be wrong.
                row.update({"error": f"redirected outside the section to {final_url}"[:300],
                            "error_class": "off_section_redirect"})
                self.add_external([{"title": entry.title, "slug": entry.slug,
                                    "url": final_url, "category": entry.category_title,
                                    "source_url": entry.url,
                                    "note": "section URL redirects outside the section"}])
                self._fail(entry, row)
                return

        try:
            ext = extract_page(
                res.html, entry, base_url=self.args.base_url, section=self.args.section,
                changelog_index=self.seeds.changelog.by_slug, discover=self.discover,
            )
        except Exception as e:
            # Page content is arbitrary; one page tripping a converter bug must
            # not cost the whole crawl. It is an unexpected failure (exit 1).
            row.update({"error": f"{type(e).__name__}: {e}"[:300],
                        "error_class": "extract_error"})
            self._fail(entry, row)
            return
        record = ext.record
        if record is None:
            row.update({"error": ext.detail or ext.error, "error_class": ext.error})
            self._fail(entry, row)
        elif self.args.skip_hidden and record.get("hidden"):
            # Hidden flag of link/sitemap/OAS-discovered pages is only known now.
            row.update({"error_class": "skipped_hidden",
                        "note": "hidden page excluded by --skip-hidden"})
            self.skipped_hidden.append(entry.url)
            self._write_row(row)
            return  # hidden pages are not used for discovery either
        else:
            if ext.oas_id and ext.oas_doc is not None:
                self.registry.setdefault(ext.oas_id, ext.oas_doc)
            if ext.html_blocks:
                self.html_block_pages[entry.url] = ext.html_blocks
            dup_of = self._store(record, entry)
            if dup_of:
                row["duplicate_of"] = dup_of
            self._write_row(row)

        self._harvest(entry, ext)

    def _in_section(self, url: str) -> bool:
        target, base = urlparse(url_key(url)), urlparse(self.args.base_url.rstrip("/"))
        return (target.netloc.lower() == base.netloc.lower()
                and target.path.startswith(f"{base.path}/{self.args.section}/"))

    def add_seed_failures(self) -> None:
        """Failed changelog index pages and sitemap documents that cost
        entries (see sidebar.enumerate_changelog/enumerate_sitemap) are
        unexpected failures: everything listed after them is missing."""
        docs = [("changelog_index", page) for page in self.seeds.changelog.pages]
        if self.seeds.sitemap is not None:
            docs += [("sitemap", doc) for doc in self.seeds.sitemap.documents]
        for via, doc in docs:
            if not doc.get("unexpected"):
                continue
            status, error = doc.get("http_status"), doc.get("error") or f"HTTP {doc.get('http_status')}"
            self.failures.append({
                "url": doc["url"], "slug": None, "kind": via, "discovered_via": via,
                "source_url": None, "http_status": status, "error": str(error)[:300],
                "error_class": classify_fetch_failure(status, error, via),
                "seed_document": True, "unexpected": True,
            })

    def _write_row(self, row: Dict) -> None:
        self.manifest_rows[row["url"]] = row
        if self.staged is not None:
            self.staged.write_manifest(row)

    def _fail(self, entry: FrontierEntry, row: Dict) -> None:
        row["unexpected"] = is_unexpected(row["error_class"], entry.discovered_via)
        self.failures.append(row)
        self._write_row(row)

    def _rank(self, record: Dict, url: str) -> Tuple[int, int]:
        entry = self.frontier.get(url)
        via = entry.discovered_via if entry else "link_bfs"
        canonical = 0 if _url_tail(url) == record.get("slug") else 1
        return canonical, SOURCE_PRIORITY.get(via, len(SOURCE_PRIORITY))

    def _store(self, record: Dict, entry: FrontierEntry) -> Optional[str]:
        """Keep one record per document; return the kept URL for a duplicate."""
        key = _identity(record)
        existing = self.records.get(key)
        if existing is None:
            self.records[key] = record
            self.record_urls[key] = [record["url"]]
            if self.staged is not None:
                self.staged.write_record(record)
            return None
        self.record_urls[key].append(record["url"])
        if self._rank(record, record["url"]) < self._rank(existing, existing["url"]):
            self.records[key] = record
            if self.staged is not None:
                self.staged.write_record(record)
            self.duplicates.append({"url": existing["url"], "duplicate_of": record["url"]})
            return None
        self.duplicates.append({"url": record["url"], "duplicate_of": existing["url"]})
        return existing["url"]

    def _harvest(self, entry: FrontierEntry, ext: PageExtraction) -> None:
        # Links the extractor insists on (e.g. the real slug behind a changelog
        # mismatch) are returned even with discovery off.
        new: List[FrontierEntry] = list(ext.links)
        if self.discover and ext.has_ssr:
            if ext.sidebars:
                walk = walk_sidebars(
                    ext.sidebars, base_url=self.args.base_url, section=self.args.section,
                    skip_hidden=self.args.skip_hidden, docs_via="sidebar",
                    refs_via="reference_sidebar", source_url=entry.url,
                )
                new = [*walk.entries, *new]
                self.add_external(walk.external_links)
            if ext.oas_id and ext.oas_id not in self.seeded_oas:
                self.seeded_oas.add(ext.oas_id)
                for op_id in ext.operation_ids:
                    if SLUG_RE.match(op_id):
                        new.append(FrontierEntry(
                            slug=op_id,
                            url=page_url(self.args.base_url, self.args.section,
                                         "reference", op_id),
                            kind="reference", title=op_id,
                            discovered_via="oas_operation", source_url=entry.url,
                        ))
        self._enqueue(new)

    # -- results ---------------------------------------------------------------

    def final_records(self) -> List[Dict]:
        out: List[Dict] = []
        for key, record in self.records.items():
            urls = list(dict.fromkeys(self.record_urls.get(key, [record["url"]])))
            entries = [e for e in (self.frontier.get(u) for u in urls) if e is not None]
            if entries:
                best = min(entries, key=lambda e: SOURCE_PRIORITY.get(e.discovered_via, 99))
                record["discovered_via"] = best.discovered_via
                own = self.frontier.get(record["url"])
                with_crumb = [e for e in ([own] if own else []) + entries if e.breadcrumb]
                if not record.get("breadcrumb") and with_crumb:
                    record["breadcrumb"] = list(with_crumb[0].breadcrumb)
                    if record.get("parent_slug") is None:
                        record["parent_slug"] = with_crumb[0].parent_slug
            aliases = [u for u in urls if u != record["url"]]
            if aliases:
                record["alias_urls"] = aliases
            out.append(record)
        return out

    def build_report(self, records: List[Dict], *, started: float) -> Dict:
        args = self.args
        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for row in self.failures:
            by_class[row["error_class"]].append({
                k: row.get(k) for k in ("url", "discovered_via", "source_url", "http_status",
                                         "error", "unexpected")
            })
        unexpected = [r for r in self.failures if r.get("unexpected")]

        record_urls = {u for urls in self.record_urls.values() for u in urls}
        skipped = set(self.skipped_hidden)
        navigation_gaps = [
            {"url": e.url, "discovered_via": e.discovered_via,
             "error_class": (self.manifest_rows.get(e.url) or {}).get("error_class"),
             "error": (self.manifest_rows.get(e.url) or {}).get("error")}
            for e in self.frontier
            if e.discovered_via in NAVIGATION_SOURCES and e.url in self.manifest_rows
            and e.url not in record_urls and e.url not in skipped
        ]

        # OAS operations no page documents (operation match, operationId == slug
        # or summary == title, mirroring oas.find_page_operation).
        documented = {(r["operation"]["method"], r["operation"]["path"])
                      for r in records if isinstance(r.get("operation"), dict)}
        slugs = {r.get("slug") for r in records}
        slugs |= {_url_tail(u) for u in record_urls}
        titles = {_norm(r.get("title")) for r in records}
        missing_ops = []
        for oas_id in sorted(self.registry):
            for method, path, op in iter_operations(self.registry[oas_id]):
                if (method, path) in documented:
                    continue
                if op.get("operationId") in slugs:
                    continue
                if op.get("summary") and _norm(op.get("summary")) in titles:
                    continue
                missing_ops.append({"oas_id": oas_id, "method": method, "path": path,
                                    "operationId": op.get("operationId"),
                                    "summary": op.get("summary")})

        unresolved_names: Counter = Counter()
        unresolved_pages = []
        for r in records:
            names = r.get("reusable_content_unresolved") or []
            if names:
                unresolved_names.update(names)
                unresolved_pages.append({"url": r["url"], "names": names})

        kept_urls = {r["url"] for r in records}
        changelog = self.seeds.changelog
        sitemap = self.seeds.sitemap
        report = {
            "generated_at": _now_iso(),
            "duration_s": round(time.monotonic() - started, 1),
            "base_url": args.base_url,
            "section": args.section,
            "options": {"offline": args.offline, "skip_hidden": args.skip_hidden,
                        "no_cache": args.no_cache, "limit": args.limit,
                        "max_pages": args.max_pages, "discover": self.discover},
            "outputs": {"records": str(args.out), "manifest": str(args.manifest),
                        "oas_registry": str(args.oas_out), "report": str(args.report)},
            "totals": {
                "frontier": len(self.frontier),
                "processed": self.processed,
                "fetched_ok": self.fetched_ok,
                "cache_hits": self.cache_hits,
                "live_requests": getattr(self.fetcher, "live_requests", None),
                "records": len(records),
                "duplicates_merged": len(self.duplicates),
                "skipped_hidden": len(self.skipped_hidden),
                "hidden_records": sum(1 for r in records if r.get("hidden")),
                "oas_documents": len(self.registry),
                "failures": len(self.failures),
                "unexpected_failures": len(unexpected),
            },
            "frontier_by_discovered_via": dict(Counter(e.discovered_via for e in self.frontier)),
            "records_by_discovered_via": dict(Counter(r.get("discovered_via") for r in records)),
            "records_by_kind": dict(Counter(r.get("kind") for r in records)),
            "failures": {
                cls: {"count": len(rows),
                      "unexpected": sum(1 for r in rows if r["unexpected"]),
                      "items": rows}
                for cls, rows in sorted(by_class.items())
            },
            "unexpected_failures": [
                {k: r.get(k) for k in ("url", "error_class", "discovered_via", "source_url",
                                        "http_status", "error")}
                for r in unexpected
            ],
            "navigation_gaps": navigation_gaps,
            "oas_operations_without_page": missing_ops,
            "reusable_content_unresolved": {
                "pages": len(unresolved_pages),
                "names": dict(unresolved_names.most_common()),
                "items": unresolved_pages,
            },
            "html_conversion": {
                "body_from_html": [r["url"] for r in records if r.get("body_source") == "html"],
                "html_block_pages": {u: n for u, n in self.html_block_pages.items()
                                     if u in kept_urls},
            },
            "external_links": list(self.external_links.values()),
            "duplicates": self.duplicates,
            "skipped_hidden": self.skipped_hidden,
            "changelog_index": {"pages": changelog.pages, "entries": len(changelog.by_slug),
                                "skipped_hidden": changelog.skipped_hidden,
                                "incomplete": changelog.incomplete},
            "sitemap": ({"documents": sitemap.documents, "page_urls": len(sitemap.entries),
                         "ignored_urls": sitemap.ignored_urls,
                         "incomplete": sitemap.incomplete}
                        if sitemap is not None else None),
            "cap_truncations": {
                "max_pages": {"cap": self.frontier.max_pages,
                              "not_enqueued": sum(self.frontier.not_enqueued.values()),
                              "by_source": dict(self.frontier.not_enqueued),
                              "examples": self.frontier.not_enqueued_examples},
                "changelog_index": {"cap": changelog.max_pages,
                                    "truncated": changelog.truncated},
                "sitemap": ({"truncated": sitemap.truncated,
                             "documents_not_fetched": sitemap.truncated_documents}
                            if sitemap is not None else None),
            },
        }
        report["exit_code"] = 1 if unexpected else 0
        return report


# --- console ----------------------------------------------------------------------


def _print_summary(report: Dict) -> None:
    t = report["totals"]
    console.rule("[bold]Summary[/bold]")
    console.print(f"  frontier        : {t['frontier']}  (processed {t['processed']})")
    for via, n in sorted(report["frontier_by_discovered_via"].items(),
                         key=lambda kv: SOURCE_PRIORITY.get(kv[0], 99)):
        console.print(f"    via {via:<17}: {n}")
    console.print(f"  fetched OK      : {t['fetched_ok']}  ({t['cache_hits']} cache hits)")
    console.print(f"  records         : {t['records']}  (duplicates merged: "
                  f"{t['duplicates_merged']}, hidden: {t['hidden_records']}, "
                  f"skipped hidden: {t['skipped_hidden']})")
    console.print(f"  OAS documents   : {t['oas_documents']}")
    for cls, info in report["failures"].items():
        flag = " [red](unexpected)[/red]" if info["unexpected"] else ""
        console.print(f"  failed {cls:<20}: {info['count']}{flag}")
    if report["navigation_gaps"]:
        console.print(f"  [yellow]navigation URLs without a record: "
                      f"{len(report['navigation_gaps'])}[/yellow]")
    if report["oas_operations_without_page"]:
        console.print(f"  [yellow]OAS operations without a page: "
                      f"{len(report['oas_operations_without_page'])}[/yellow]")
    ru = report["reusable_content_unresolved"]
    if ru["pages"]:
        console.print(f"  [yellow]pages with unresolved reusable content: {ru['pages']} "
                      f"({len(ru['names'])} distinct components)[/yellow]")
    hc = report["html_conversion"]
    console.print(f"  HTML converted  : {len(hc['html_block_pages'])} pages with "
                  f"{escape('[block:html]')}, "
                  f"{len(hc['body_from_html'])} custom pages from cp.html")
    if report["external_links"]:
        console.print(f"  external links  : {len(report['external_links'])} (not fetched)")
    cap = report["cap_truncations"]
    if cap["max_pages"]["not_enqueued"]:
        console.print(f"  [bold red]max-pages cap {cap['max_pages']['cap']} dropped "
                      f"{cap['max_pages']['not_enqueued']} URLs: "
                      f"{cap['max_pages']['by_source']}[/bold red]")
    if report["changelog_index"]["incomplete"]:
        page = [p for p in report["changelog_index"]["pages"] if p.get("unexpected")][-1]
        detail = page.get("error") or f"HTTP {page.get('http_status')}"
        console.print(f"  [bold red]changelog index incomplete: {escape(str(page['url']))} "
                      f"failed ({escape(str(detail))}); later entries are missing[/bold red]")
    if report["sitemap"] and report["sitemap"]["incomplete"]:
        failed = [d["url"] for d in report["sitemap"]["documents"] if d.get("unexpected")]
        console.print(f"  [bold red]sitemap documents failed: {escape(str(failed))}[/bold red]")
    if cap["changelog_index"]["truncated"]:
        console.print(f"  [bold red]changelog index stopped at its "
                      f"{cap['changelog_index']['cap']}-page cap[/bold red]")
    if cap["sitemap"] and cap["sitemap"]["truncated"]:
        console.print(f"  [bold red]sitemap documents not fetched (cap): "
                      f"{len(cap['sitemap']['documents_not_fetched'])}[/bold red]")


def _print_partials(paths: List[Path], staged: Optional[StagedOutputs] = None) -> None:
    torn = staged.not_restored if staged is not None else []
    if torn:
        console.print("[bold red]Publishing failed and these outputs could not be restored "
                      "(their previous version is next to them as *.bak):[/bold red]")
        for p in torn:
            console.print(f"  {escape(str(p))}")
    else:
        console.print("[yellow]Previous output files were left untouched.[/yellow]")
    console.print("[yellow]Partial results:[/yellow]")
    for p in paths:
        console.print(f"  {escape(str(p))}")


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))
    except OSError:
        return 0


def publish_blocked_reason(*, records: int, previous: int, exit_code: int) -> Optional[str]:
    """Why this run must not replace a published dataset of ``previous``
    records, or None. Guards against e.g. an --offline run over a partly
    filled cache, or a crawl during a network outage."""
    if previous <= 0:
        return None
    if records == 0:
        return f"the run produced no records ({previous} are published)"
    if exit_code and records < previous * PUBLISH_MIN_RECORD_RATIO:
        return (f"the run had unexpected failures and produced {records} records, fewer than "
                f"{PUBLISH_MIN_RECORD_RATIO:.0%} of the {previous} already published")
    return None


# --- main -------------------------------------------------------------------------


def main(argv: List[str] | None = None, *, fetcher: Any = None) -> int:
    """Run the crawler. ``fetcher`` lets tests inject a fake (anything with
    ``fetch(url)`` and ``fetch_resource(url)`` returning ``FetchResult``s)."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    started = time.monotonic()

    console.rule(f"[bold]Developer Connect crawler[/bold]  section={args.section}")
    console.print(f"  base_url   : {args.base_url}")
    console.print(f"  cache_dir  : {args.cache_dir}")
    console.print(f"  out        : {args.out}")
    console.print(f"  manifest   : {args.manifest}")
    console.print(f"  oas        : {args.oas_out}")
    console.print(f"  report     : {args.report}")
    console.print(f"  delay      : {args.delay}s   refresh={'yes' if args.no_cache else 'no'}   "
                  f"offline={'yes' if args.offline else 'no'}   "
                  f"hidden={'skip' if args.skip_hidden else 'include'}")
    if args.offline and args.no_cache:
        console.print("  [yellow]--offline wins over --no-cache: cached pages are reused[/yellow]")

    if fetcher is None:
        fetcher = make_fetcher(args)
    discover = not (args.limit or args.no_discover)

    console.print("\n[yellow]Enumerating seed frontier (welcome sidebar, changelog, "
                  "sitemap)...[/yellow]")
    t0 = time.monotonic()
    try:
        seeds = enumerate_seeds(
            fetcher, base_url=args.base_url, section=args.section,
            skip_hidden=args.skip_hidden, include_sitemap=discover,
        )
    except RuntimeError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        if args.offline:
            console.print(f"[yellow]--offline only reads the HTML cache "
                          f"({escape(str(args.cache_dir / args.section))}); run once without "
                          "--offline (or point --cache-dir at a filled cache) first.[/yellow]")
        console.print("[red]Nothing was written.[/red]")
        return 1

    frontier = Frontier(max_pages=args.max_pages)
    frontier.add_many(seeds.entries)
    console.print(f"  -> {len(frontier)} entries in {time.monotonic()-t0:.1f}s  "
                  f"{dict(Counter(e.discovered_via for e in frontier))}")
    console.print(f"  kinds: {dict(Counter(e.kind for e in frontier))}")

    if args.limit:
        del frontier.entries[args.limit:]
        console.print(f"  [dim]limited to first {len(frontier)} entries (discovery disabled)[/dim]")
    elif args.no_discover:
        console.print("  [dim]discovery disabled (--no-discover)[/dim]")
    else:
        console.print(f"  [dim]discovery enabled (cap: {args.max_pages or 'none'})[/dim]")

    if args.dry_run:
        console.print("\n[bold]Dry-run; first 15 frontier entries:[/bold]")
        for e in frontier.entries[:15]:
            line = f"  [{e.kind:10s}] {e.url}  ({e.title})  via {e.discovered_via}"
            console.print(escape(line))
        if seeds.welcome.external_links:
            console.print(f"  external sidebar links: {len(seeds.welcome.external_links)}")
        return 0

    staged = StagedOutputs(records=args.out, manifest=args.manifest, oas=args.oas_out,
                           report=args.report)
    crawl = Crawl(args=args, fetcher=fetcher, frontier=frontier, seeds=seeds,
                  discover=discover, staged=staged)
    crawl.add_external(seeds.welcome.external_links)
    crawl.add_seed_failures()

    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(), console=console,
        ) as bar:
            task = bar.add_task("Crawling", total=len(frontier))
            i = 0
            while i < len(frontier.entries):
                if args.limit and crawl.processed >= args.limit:
                    break
                entry = frontier.entries[i]
                i += 1
                bar.update(task, description=escape(f"Crawling [{entry.kind}] {entry.slug}"))
                crawl.process(entry)
                bar.update(task, total=len(frontier))
                bar.advance(task)
        records = crawl.final_records()
        report = crawl.build_report(records, started=started)
        blocked = None if args.force_publish else publish_blocked_reason(
            records=len(records), previous=_count_lines(args.out),
            exit_code=report["exit_code"])
        report["published"] = blocked is None
        if blocked:
            report["not_published_reason"] = blocked
            report["exit_code"] = 1
        staged.finish(records=records, registry=crawl.registry, report=report,
                      publish=blocked is None)
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        _print_partials(staged.abort(), staged)
        return 130
    except BaseException:
        console.print("\n[red]Crawl failed.[/red]")
        _print_partials(staged.abort(), staged)
        raise

    _print_summary(report)
    if blocked:
        console.print(f"\n  [bold red]NOT published: {escape(blocked)}.[/bold red]")
        _print_partials(staged.abort(), staged)
        console.print("  [yellow]Fix the cause and re-run (cached pages are reused), or pass "
                      "--force-publish to replace the outputs anyway.[/yellow]")
        return 1
    console.print(f"\n  [green]Wrote {args.out}[/green]")
    console.print(f"  [green]Wrote {args.manifest}[/green]")
    console.print(f"  [green]Wrote {args.oas_out}[/green]")
    console.print(f"  [green]Wrote {args.report}[/green]")
    if report["exit_code"]:
        console.print(f"  [red]{report['totals']['unexpected_failures']} unexpected "
                      "failure(s); see the report's unexpected_failures[/red]")
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
