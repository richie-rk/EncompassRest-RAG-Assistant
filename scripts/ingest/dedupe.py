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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


def dedupe_by_content(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return (kept, dropped), both in preference order. `dropped` is for logging.

    Records with empty bodies are NOT collapsed against each other on the
    body_md hash alone (every empty-body record would otherwise dedupe to one).
    Empty-body records are passed through as-is; their content uniqueness
    comes from the OAS spec, which the chunker handles separately.
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
    return kept, dropped
