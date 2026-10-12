"""Collapse JSONL records that are duplicate views of the same page.

ReadMe sometimes serves identical content at multiple URLs (e.g., `/docs/auth`
and `/reference/auth` both render the same `doc.body`). Our crawl picks up
both; left untreated they would create twin vectors in FAISS, surface the same
chunk twice in retrieval, and waste prompt context.

Detection: sha256(body_md), plus — for `type=endpoint` pages — the identity of
the operation the page documents. Endpoint pages often share boilerplate prose
(a "Scheduled for Sunset" banner, a stray `<br />`) while documenting
*different* operations; hashing the body alone used to drop such pages and
with them the only page-level home of their operation.

A duplicate is only dropped when that loses no API reference: if its OAS
document (embedded `oas`, or `oas_id` in the registry) has an operation that
no kept record's document has, it is kept. Pages of different API groups can
share a body (an overview boilerplate, an inlined reusable snippet, a lone
`<br />`), and the chunker only renders operations of kept records, so
dropping such a page used to drop its group's operations silently.

Preference order (which URL wins when a collision is found, and the order the
chunker processes records in — see `order_records`):
  1. How the crawler discovered the page (`discovered_via`): official
     navigation first (sidebar, reference sidebar, changelog index, sitemap),
     then derived sources (changelog refs, OAS operations), then pages found
     only by following links. Records without the field (legacy datasets)
     rank last, which keeps them deterministic among themselves.
  2. Non-hidden before hidden.
  3. Longer `body_md` wins. Hashes match so this is academic, but if any
     rendering difference creeps in, keep the richer record.
  4. Lexicographic URL — deterministic so re-runs are reproducible.

The first record per identity in that order wins.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from scripts.crawler.oas import iter_operations
from scripts.ingest.oas_synth import page_operation

#: Rank of each `discovered_via` value; lower wins. Unknown/missing rank last.
#: Same order as the crawler's `scripts.crawler.sidebar.DISCOVERY_SOURCES`
#: (kept as a literal so the ingest does not import the HTTP crawler; a test
#: pins the two together).
DISCOVERY_RANK: Dict[str, int] = {
    via: i for i, via in enumerate((
        "sidebar",
        "reference_sidebar",
        "changelog_index",
        "sitemap",
        "changelog_ref",
        "oas_operation",
        "link_bfs",
    ))
}
_UNKNOWN_RANK = len(DISCOVERY_RANK)


def preference_key(record: Mapping[str, Any]) -> Tuple[int, bool, int, str]:
    """Sort key implementing the preference order above (smaller = preferred)."""
    via = record.get("discovered_via")
    rank = DISCOVERY_RANK.get(via, _UNKNOWN_RANK) if isinstance(via, str) else _UNKNOWN_RANK
    hidden = bool(record.get("hidden"))
    body_len_neg = -len(record.get("body_md") or "")
    return (rank, hidden, body_len_neg, str(record.get("url") or ""))


def order_records(records: Sequence[Dict]) -> List[Dict]:
    """Records in preference order (stable; does not mutate the input)."""
    return sorted(records, key=preference_key)


def _content_hash(record: Mapping[str, Any]) -> str:
    h = hashlib.sha256((record.get("body_md") or "").encode("utf-8"))
    if record.get("type") == "endpoint":
        op = page_operation(record)
        if op is not None:
            h.update(f"\x00{op[0].upper()} {op[1]}".encode("utf-8"))
    return h.hexdigest()


def dedupe_by_content(
    records: List[Dict], oas_registry: Optional[Mapping[str, Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Return (kept, dropped), both in preference order. `dropped` is for logging.

    Records with empty bodies are NOT collapsed against each other on the
    body_md hash alone (every empty-body record would otherwise dedupe to one).
    Empty-body records are passed through as-is; their content uniqueness
    comes from the OAS spec, which the chunker handles separately.

    `oas_registry` resolves `oas_id`s (current crawler output); without it a
    duplicate whose `oas_id` no kept record shares is kept.
    """
    seen: Dict[str, Dict] = {}
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for r in order_records(records):
        body = r.get("body_md") or ""
        if not body.strip():
            kept.append(r)
            continue
        h = _content_hash(r)
        if h in seen:
            dropped.append(r)
            continue
        seen[h] = r
        kept.append(r)
    rescued = _keep_api_carriers(kept, dropped, oas_registry)
    if rescued:
        ids = {id(r) for r in rescued}
        kept = order_records(kept + rescued)
        dropped = [r for r in dropped if id(r) not in ids]
    return kept, dropped


def _carries_oas(record: Mapping[str, Any]) -> bool:
    oas = record.get("oas")
    return bool(record.get("oas_id")) or (isinstance(oas, dict) and bool(oas))


def _operation_keys(
    record: Mapping[str, Any], oas_registry: Optional[Mapping[str, Dict]],
) -> Optional[Set[Tuple[str, str]]]:
    """(METHOD, path) of every operation in the record's OAS document, or None
    when the document is not available."""
    oas = record.get("oas")
    if not (isinstance(oas, dict) and oas):
        oas = (oas_registry or {}).get(record.get("oas_id") or "")
    if not isinstance(oas, dict):
        return None
    return {(method.upper(), path) for method, path, _ in iter_operations(oas)}


def _keep_api_carriers(
    kept: Sequence[Mapping[str, Any]], dropped: Sequence[Dict],
    oas_registry: Optional[Mapping[str, Dict]],
) -> List[Dict]:
    """Dropped records (in preference order) whose OAS document has an
    operation that no kept record's document has."""
    carriers = [r for r in dropped if _carries_oas(r)]
    if not carriers:
        return []
    ops: Set[Tuple[str, str]] = set()
    oas_ids: Set[str] = set()

    def cover(r: Mapping[str, Any], keys: Optional[Set[Tuple[str, str]]]) -> None:
        ops.update(keys or ())
        if r.get("oas_id"):
            oas_ids.add(str(r["oas_id"]))

    for r in kept:
        if _carries_oas(r):
            cover(r, _operation_keys(r, oas_registry))
    rescued: List[Dict] = []
    for r in carriers:
        keys = _operation_keys(r, oas_registry)
        # Unknown document: fall back to comparing documents by id.
        new = (str(r.get("oas_id")) not in oas_ids) if keys is None else not keys <= ops
        if new:
            rescued.append(r)
            cover(r, keys)
    return rescued
