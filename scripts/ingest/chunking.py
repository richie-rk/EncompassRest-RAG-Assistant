"""Chunking strategy for the JSONL records.

Splitting:
  - If a record's text is <= PAGE_WISE_THRESHOLD (2000) chars, keep it as a
    single chunk.
  - Otherwise split via RecursiveCharacterTextSplitter at 1500/200 with
    paragraph-first separators (`\\n\\n`, `\\n`, `. `, ` `, ``), then fold
    fragments under MIN_CHUNK_SIZE into a neighbour (without duplicating the
    splitter's overlap text).
  - Every chunk is prepended with `Title: {title}\\nSection: {breadcrumb}\\n\\n`
    so retrieval has the topical anchor even on sub-page chunks.

Per-record text source:
  - `type=endpoint` -> hand-written `body_md` (when it has any text) followed
        by the full markdown of the page's own OpenAPI operation.
  - `type=basic` reference page with `body_md` shorter than
        SPARSE_BODY_THRESHOLD and an OAS document -> body + a compact API
        overview (one line per operation), not the operations themselves.
  - everything else -> `body_md` as-is.

Corpus-level rules (`chunk_corpus` / `chunk_records`):
  - ReadMe attaches the whole OAS document of an API group to every
    reference page, and several pages can document the same operation. Each
    operation, identified by (METHOD, path), is rendered in full at most once:
    records are processed in dedupe preference order (`order_records`) and a
    later page claiming an already-rendered operation contributes only its
    body. Overviews are likewise emitted once per distinct overview text, and
    a large request/response schema or example shared by several operations
    is shown in full once and referenced (with its field names) afterwards.
  - Operations present in some OAS document but documented by no page
    ("orphans") are rendered once at the end, attached to a representative
    page of a document containing them under `<page url>#<operationId>`, so
    `(url, chunk_index)` stays unique. The representative is the page most
    related to the operation's path: most path words (`/v3/loanFolders/{id}`
    -> loan, folder; words every path of the document shares do not count)
    in its slug, title or own operation's path; ties go to a `basic`
    overview page, then to preference order.

A record's OAS document is `record["oas"]` (legacy datasets embed it) or
`oas_registry[record["oas_id"]]` (current crawler output stores each distinct
document once in a registry file).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.crawler.oas import iter_operations
from scripts.ingest.dedupe import order_records
from scripts.ingest.oas_synth import (
    page_operation,
    synthesize_api_overview,
    synthesize_code_samples,
    synthesize_operation_markdown,
)


PAGE_WISE_THRESHOLD = 2000
SPLIT_CHUNK_SIZE = 1500
SPLIT_OVERLAP = 200
SPARSE_BODY_THRESHOLD = 200  # for basic/reference pages — append overview if shorter
MIN_CHUNK_SIZE = 500         # below this we merge into a neighbour to avoid tiny fragments

# Slack when looking for the splitter's overlap: whitespace stripping can shift
# chunk edges by a few characters relative to the nominal SPLIT_OVERLAP.
_OVERLAP_MARGIN = 32
# Without the source text, a suffix/prefix match shorter than this is more
# likely coincidence (a closing ``` against an opening ```json) than overlap.
_MIN_OVERLAP_MATCH = 16

OpKey = Tuple[str, str]  # (METHOD, path)


@dataclass
class Chunk:
    """One chunk's payload for embedding + indexing."""
    page_content: str
    metadata: Dict


@dataclass
class RenderLedger:
    """What has already been rendered in full during one corpus pass."""
    operations: Dict[OpKey, str] = field(default_factory=dict)  # op -> url that rendered it
    overviews: Set[str] = field(default_factory=set)            # sha256 of overview markdown
    # Large schema/example blocks shared between operations:
    # {sha256 of block: where it was rendered}. See oas_synth._render_once.
    blocks: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkReport:
    """Counters from `chunk_corpus`, surfaced in the vector store's stats.json."""
    records_in: int = 0
    records_chunked: int = 0
    records_skipped_duplicate_url: int = 0
    records_missing_oas: int = 0           # `oas_id` not found in the registry
    operations_total: int = 0              # distinct (METHOD, path) across all OAS documents
    operations_rendered: int = 0           # rendered in full (pages + orphans)
    orphan_operations_rendered: int = 0
    endpoint_pages_with_prose: int = 0     # endpoint pages whose body_md made it into chunks
    endpoint_pages_without_operation: int = 0
    overviews_rendered: int = 0


# Paragraph-first separators. Heading-preference (`\n## `, `\n### `) was tempting
# but produced 50-char fragments at every sub-heading boundary, blowing chunk
# count by 4x without semantic gain. Headings still naturally align with `\n\n`
# in our synthesized output, so we lose nothing meaningful.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=SPLIT_CHUNK_SIZE,
    chunk_overlap=SPLIT_OVERLAP,
    separators=_SEPARATORS,
    length_function=len,
    is_separator_regex=False,
)

_TAG_RE = re.compile(r"<[^>]*>")
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_VERSION_RE = re.compile(r"^v\d+$")


# --- Record inputs --------------------------------------------------------------

def record_oas(record: Mapping[str, Any], oas_registry: Optional[Mapping[str, Dict]] = None) -> Optional[Dict]:
    """The OAS document of a record: embedded (legacy) or looked up by `oas_id`."""
    oas = record.get("oas")
    if isinstance(oas, dict) and oas:
        return oas
    oas_id = record.get("oas_id")
    if oas_id and oas_registry:
        doc = oas_registry.get(oas_id)
        if isinstance(doc, dict):
            return doc
    return None


def _body_text(record: Mapping[str, Any]) -> str:
    """`body_md`, or "" when it holds no actual text (e.g. a lone `<br />`)."""
    body = record.get("body_md") or ""
    if not any(ch.isalnum() for ch in _TAG_RE.sub("", body)):
        return ""
    return body


def _join(parts: Sequence[str]) -> str:
    return "\n\n".join(p.strip("\n") for p in parts if p and p.strip())


@dataclass
class _RecordText:
    text: str
    operation: Optional[OpKey] = None       # operation rendered in full in `text`
    overview_hash: Optional[str] = None     # overview included in `text`
    prose: bool = False                     # endpoint body_md included
    no_operation: bool = False              # endpoint page with no identifiable operation
    documented: Optional[OpKey] = None      # operation the page documents (rendered or not)


def _overview_once(oas: Optional[Dict], ledger: Optional[RenderLedger]) -> Tuple[str, Optional[str]]:
    overview = synthesize_api_overview(oas)
    if not overview:
        return "", None
    digest = hashlib.sha256(overview.encode("utf-8")).hexdigest()
    if ledger is not None and digest in ledger.overviews:
        return "", None
    return overview, digest


def _build_record_text(
    record: Mapping[str, Any], oas: Optional[Dict], ledger: Optional[RenderLedger],
) -> _RecordText:
    """Decide the source text for one JSONL record (see module docstring)."""
    body = _body_text(record)
    rec_type = record.get("type")

    if rec_type == "endpoint":
        out = _RecordText(text="", prose=bool(body.strip()))
        parts = [body]
        hit = page_operation(record, oas)
        if hit is not None:
            key = (hit[0].upper(), hit[1])
            out.documented = key
            if ledger is None or key not in ledger.operations:
                md = synthesize_operation_markdown(
                    oas, hit[0], hit[1], doc_api=record.get("doc_api"),
                    seen_blocks=ledger.blocks if ledger is not None else None,
                )
                if md:
                    parts.append(md)
                    out.operation = key
        else:
            out.no_operation = True
            overview, out.overview_hash = _overview_once(oas, ledger)
            parts += [overview, synthesize_code_samples(record.get("doc_api"))]
        out.text = _join(parts)
        return out

    raw_body = record.get("body_md") or ""
    has_oas_paths = bool((oas or {}).get("paths"))
    if rec_type == "basic" and has_oas_paths and len(raw_body) < SPARSE_BODY_THRESHOLD:
        overview, digest = _overview_once(oas, ledger)
        if overview:
            return _RecordText(text=_join([body, overview]), overview_hash=digest)

    return _RecordText(text=body)


# --- Splitting ------------------------------------------------------------------

def _locate(text: str, pieces: Sequence[str]) -> Optional[List[Tuple[int, int]]]:
    """Character spans of the splitter's pieces inside `text`, or None.

    The splitter keeps separators and only strips whitespace, so every piece
    is a substring of the source; consecutive pieces overlap by at most
    SPLIT_OVERLAP characters and each one extends past its predecessor.
    """
    spans: List[Tuple[int, int]] = []
    prev_start, prev_end = -1, 0
    for p in pieces:
        idx = text.find(p, max(prev_start + 1, prev_end - SPLIT_OVERLAP - _OVERLAP_MARGIN, 0))
        while idx >= 0 and idx + len(p) <= prev_end:
            idx = text.find(p, idx + 1)
        if idx < 0:
            return None
        spans.append((idx, idx + len(p)))
        prev_start, prev_end = idx, idx + len(p)
    return spans


def _join_dropping_overlap(prev: str, nxt: str) -> str:
    """Concatenate two neighbouring pieces, dropping the splitter's overlap.

    The overlap is the longest suffix of `prev` that is also a prefix of `nxt`
    (bounded by SPLIT_OVERLAP plus a margin) and sits on whitespace boundaries
    on both sides, as the splitter's own overlaps do.
    """
    limit = min(len(prev), len(nxt), SPLIT_OVERLAP + _OVERLAP_MARGIN)
    for k in range(limit, _MIN_OVERLAP_MATCH - 1, -1):
        if not prev.endswith(nxt[:k]):
            continue
        rest = nxt[k:]
        before = prev[-k - 1] if k < len(prev) else ""
        if (not rest or rest[0].isspace()) and (not before or before.isspace()):
            return prev + rest
    return prev + "\n\n" + nxt


def _merge_small(pieces: List[str], text: Optional[str] = None) -> List[str]:
    """Walk through chunks left-to-right; merge any chunk under MIN_CHUNK_SIZE
    into its neighbour as long as the combined size stays under 2*chunk_size.

    Merging must not repeat the splitter's overlap. With the source `text` the
    merged chunk is the exact source span covering both pieces; without it
    (or if the pieces cannot be located) the overlap is found by matching the
    previous piece's suffix against the next piece's prefix.
    """
    if len(pieces) < 2:
        return list(pieces)
    cap = 2 * SPLIT_CHUNK_SIZE

    spans = _locate(text, pieces) if text is not None else None
    if spans is not None:
        assert text is not None
        merged: List[List[int]] = [list(spans[0])]
        for start, end in spans[1:]:
            m_start, m_end = merged[-1]
            small = (end - start) < MIN_CHUNK_SIZE or (m_end - m_start) < MIN_CHUNK_SIZE
            if small and max(m_end, end) - m_start < cap:
                merged[-1][1] = max(m_end, end)
            else:
                merged.append([start, end])
        return [text[s:e] for s, e in merged]

    out: List[str] = [pieces[0]]
    for p in pieces[1:]:
        if len(p) < MIN_CHUNK_SIZE or len(out[-1]) < MIN_CHUNK_SIZE:
            joined = _join_dropping_overlap(out[-1], p)
            if len(joined) < cap:
                out[-1] = joined
                continue
        out.append(p)
    return out


def _split_text(text: str) -> List[str]:
    if len(text) <= PAGE_WISE_THRESHOLD:
        return [text]
    return _merge_small(_SPLITTER.split_text(text), text)


# --- Chunk assembly ---------------------------------------------------------------

def _one_line(text: Any) -> str:
    return _NEWLINE_RUN_RE.sub(" ", str(text or "")).strip()


def _format_header(record: Mapping[str, Any]) -> str:
    title = _one_line(record.get("title")) or _one_line(record.get("slug"))
    breadcrumb = record.get("breadcrumb") or []
    section = " > ".join(_one_line(b) for b in breadcrumb if b) or "Documentation"
    return f"Title: {title}\nSection: {section}\n\n"


def strip_header(page_content: str) -> str:
    """A chunk's text without the `Title:/Section:` header added here."""
    if page_content.startswith("Title: "):
        parts = page_content.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1]
    return page_content


def _make_metadata(record: Mapping[str, Any], chunk_index: int, total_chunks: int,
                   extra: Optional[Mapping[str, Any]] = None) -> Dict:
    """Metadata travels with each chunk through FAISS load/save (pickled).

    Keep this small — every chunk pays for it. We pull URL, title, slug, kind,
    breadcrumb so the LLM can cite, and chunk_index for debugging retrieval hits.
    `operation` ("GET /path") marks chunks whose record carries that
    operation's full reference.
    """
    meta = {
        "source": "developer_connect",
        "url": record["url"],
        "title": record.get("title", ""),
        "slug": record.get("slug", ""),
        "kind": record.get("kind", ""),
        "type": record.get("type") or "",
        "breadcrumb": record.get("breadcrumb") or [],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if extra:
        meta.update(extra)
    return meta


def _chunks_for_text(text: str, record: Mapping[str, Any],
                     extra: Optional[Mapping[str, Any]] = None) -> List[Chunk]:
    if not text or not text.strip():
        return []
    header = _format_header(record)
    pieces = _split_text(text)
    total = len(pieces)
    return [
        Chunk(page_content=header + piece, metadata=_make_metadata(record, i, total, extra))
        for i, piece in enumerate(pieces)
    ]


def _chunk_one(record: Mapping[str, Any], oas: Optional[Dict],
               ledger: Optional[RenderLedger]) -> Tuple[List[Chunk], _RecordText]:
    built = _build_record_text(record, oas, ledger)
    extra = {"operation": f"{built.operation[0]} {built.operation[1]}"} if built.operation else None
    chunks = _chunks_for_text(built.text, record, extra)
    if chunks and ledger is not None:
        if built.operation:
            ledger.operations[built.operation] = record["url"]
        if built.overview_hash:
            ledger.overviews.add(built.overview_hash)
    return chunks, built


def chunk_record(
    record: Mapping[str, Any],
    oas: Optional[Dict] = None,
    *,
    oas_registry: Optional[Mapping[str, Dict]] = None,
    ledger: Optional[RenderLedger] = None,
) -> List[Chunk]:
    """Turn one JSONL record into 0..N chunks.

    `oas` defaults to the record's own document (`record_oas`). Pass a shared
    `ledger` to skip operations/overviews already rendered by earlier records
    (that is what `chunk_corpus` does); without one the record is chunked
    standalone. Returns an empty list when the record has no chunkable text.
    """
    if oas is None:
        oas = record_oas(record, oas_registry)
    return _chunk_one(record, oas, ledger)[0]


# --- Corpus ---------------------------------------------------------------------

def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _words(text: str) -> Set[str]:
    """Lowercase words of a slug, title or path segment (camelCase split,
    naive singular: "loanFolders" -> {"loan", "folder"})."""
    out: Set[str] = set()
    for word in _WORD_RE.findall(text or ""):
        word = word.lower()
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _path_words(path: str) -> Set[str]:
    words: Set[str] = set()
    for segment in path.split("/"):
        if segment and not segment.startswith("{") and not _VERSION_RE.match(segment.lower()):
            words |= _words(segment)
    return words


# (preference index, record, its OAS document, method, operation object, documented op)
_Candidate = Tuple[int, Mapping[str, Any], Dict, str, Dict, Optional[OpKey]]


def _representative(key: OpKey, candidates: Sequence[_Candidate]) -> _Candidate:
    """The record an orphan operation is attached to (see the module docstring)."""
    wanted = _path_words(key[1])
    paths = list(candidates[0][2].get("paths") or {}) if wanted else []
    if len(paths) > 1:  # words in every path of the document say nothing
        wanted -= set.intersection(*(_path_words(p) for p in paths))

    def rank(c: _Candidate) -> Tuple[int, bool, int]:
        idx, record, _, _, _, documented = c
        score = 0
        if wanted:
            words = _words(str(record.get("slug") or "")) | _words(str(record.get("title") or ""))
            if documented is not None:
                words |= _path_words(documented[1])
            score = len(wanted & words)
        return score, record.get("type") == "basic", -idx

    return max(candidates, key=rank)


def chunk_corpus(
    records: Sequence[Dict], oas_registry: Optional[Mapping[str, Dict]] = None,
) -> Tuple[List[Chunk], ChunkReport]:
    """Chunk a whole (deduplicated) corpus; see the module docstring for the rules.

    Records are processed in `order_records` preference order and chunks come
    back in that order, followed by orphan operations.
    """
    report = ChunkReport(records_in=len(records))
    ledger = RenderLedger()
    chunks: List[Chunk] = []
    seen_urls: Set[str] = set()
    # op -> every record whose OAS document has it (orphan representatives)
    op_sources: Dict[OpKey, List[_Candidate]] = {}

    for idx, record in enumerate(order_records(records)):
        url = record.get("url") or ""
        if url in seen_urls:
            report.records_skipped_duplicate_url += 1
            continue
        seen_urls.add(url)

        oas = record_oas(record, oas_registry)
        if oas is None and record.get("oas_id"):
            report.records_missing_oas += 1
        record_chunks, built = _chunk_one(record, oas, ledger)
        chunks.extend(record_chunks)
        if record_chunks:
            report.records_chunked += 1
        if built.overview_hash and record_chunks:
            report.overviews_rendered += 1
        if record.get("type") == "endpoint":
            report.endpoint_pages_with_prose += int(built.prose)
            report.endpoint_pages_without_operation += int(built.no_operation)

        if oas:
            for method, path, op in iter_operations(oas):
                op_sources.setdefault((method.upper(), path), []).append(
                    (idx, record, oas, method, op, built.documented))

    report.operations_total = len(op_sources)
    pages_rendered = sum(1 for key in op_sources if key in ledger.operations)

    for key, candidates in op_sources.items():
        if key in ledger.operations:
            continue
        _, rep, oas, method, op, _ = _representative(key, candidates)
        md = synthesize_operation_markdown(oas, method, key[1], seen_blocks=ledger.blocks)
        if not md:
            continue
        fragment = str(op.get("operationId") or "") or _slugify(f"{method}-{key[1]}")
        url = f"{rep['url']}#{fragment}"
        n = 2
        while url in seen_urls:
            url = f"{rep['url']}#{fragment}-{n}"
            n += 1
        seen_urls.add(url)
        title = _one_line(op.get("summary")) or f"{key[0]} {key[1]}"
        pseudo = {
            "url": url,
            "title": title,
            "slug": fragment,
            "kind": "reference",
            "type": "endpoint",
            "breadcrumb": list(rep.get("breadcrumb") or []) + [title],
        }
        extra = {"operation": f"{key[0]} {key[1]}", "orphan_operation": True}
        orphan_chunks = _chunks_for_text(md, pseudo, extra)
        if orphan_chunks:
            chunks.extend(orphan_chunks)
            ledger.operations[key] = url
            report.orphan_operations_rendered += 1

    report.operations_rendered = pages_rendered + report.orphan_operations_rendered
    return chunks, report


def chunk_records(records: Sequence[Dict], oas_registry: Optional[Mapping[str, Dict]] = None) -> List[Chunk]:
    """Batch helper: `chunk_corpus` without the report."""
    return chunk_corpus(records, oas_registry)[0]
