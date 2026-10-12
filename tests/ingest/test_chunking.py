"""Tests for scripts.ingest.chunking: per-record text, corpus-level rendering rules, splitting."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.ingest.chunking import (
    MIN_CHUNK_SIZE,
    SPLIT_CHUNK_SIZE,
    SPLIT_OVERLAP,
    _SEPARATORS,
    _merge_small,
    chunk_corpus,
    chunk_record,
    chunk_records,
    strip_header,
)
from tests.ingest.fixtures.synthetic import make_oas, record, to_new_contract

ITEM = "/widgets/{widgetId}"
OP_HEADINGS = {
    "GET " + ITEM, "PATCH " + ITEM, "GET /widgets", "POST /widgets", f"POST {ITEM}/archive",
}


def _corpus(oas=None) -> List[Dict]:
    oas = oas or make_oas()
    return [
        record("widgets", title="Widgets", type="basic", oas=oas, body="",
               breadcrumb=["Widgets API"], discovered_via="link_bfs"),
        record("get-widget", oas=oas, body="Usage notes: cache results for at least a minute.",
               breadcrumb=["Widgets API", "Get Widget"], discovered_via="sidebar"),
        record("update-widget", oas=oas, discovered_via="sidebar"),
        record("list-widgets", oas=oas, discovered_via="sidebar",
               doc_api={"examples": {"codes": [{"language": "curl", "code": "curl https://api.example.com/widgets"}]}}),
        record("create-widget", oas=oas, discovered_via="link_bfs"),
        record("intro", title="Introduction", kind="guide", type="basic",
               body="Welcome to the widgets guide.", discovered_via="sidebar"),
    ]


def _rendered_ops(chunks) -> Counter:
    """operation -> number of distinct URLs carrying its full reference."""
    urls = {(c.metadata["operation"], c.metadata["url"]) for c in chunks if c.metadata.get("operation")}
    return Counter(op for op, _ in urls)


def _ids(chunks):
    return [(c.metadata["url"], c.metadata["chunk_index"]) for c in chunks]


def test_legacy_and_registry_records_produce_identical_chunks():
    legacy = _corpus()
    new, registry = to_new_contract(legacy)
    assert all("oas" not in r for r in new) and len(registry) == 1
    a, report_a = chunk_corpus(legacy)
    b, report_b = chunk_corpus(new, registry)
    assert [(c.page_content, c.metadata) for c in a] == [(c.page_content, c.metadata) for c in b]
    assert report_a == report_b
    assert chunk_records(new, registry) == b


def test_endpoint_prose_is_kept_ahead_of_the_operation():
    chunks = chunk_record(record("get-widget", oas=make_oas(), body="Usage notes: paginate politely."))
    text = "".join(strip_header(c.page_content) for c in chunks)
    assert "Usage notes: paginate politely." in text
    assert text.index("Usage notes") < text.index(f"## GET {ITEM}")
    assert chunks[0].metadata["operation"] == f"GET {ITEM}"


def test_body_without_text_counts_as_empty():
    rec = record("get-widget", oas=make_oas(), body="<br />")
    chunks, report = chunk_corpus([rec])
    assert "<br />" not in "".join(c.page_content for c in chunks)
    assert report.endpoint_pages_with_prose == 0


def test_each_operation_is_rendered_in_full_once():
    oas = make_oas()
    recs = [
        record("get-widget", oas=oas, url="https://x/reference/get-widget", discovered_via="sidebar"),
        record("get-widget", oas=oas, url="https://x/docs/get-widget", discovered_via="link_bfs",
               body="Extra prose that only this copy of the page has."),
        record("get-widget", oas=oas, url="https://x/other/get-widget", discovered_via="link_bfs"),
    ]
    chunks, report = chunk_corpus(recs)
    by_url = {c.metadata["url"]: c for c in chunks}
    assert by_url["https://x/reference/get-widget"].metadata["operation"] == f"GET {ITEM}"
    later = by_url["https://x/docs/get-widget"]
    assert "operation" not in later.metadata
    assert strip_header(later.page_content) == "Extra prose that only this copy of the page has."
    assert "https://x/other/get-widget" not in by_url  # nothing of its own to add
    assert _rendered_ops(chunks)[f"GET {ITEM}"] == 1
    # Every heading appears in exactly one URL's chunks.
    for op in OP_HEADINGS:
        urls = {c.metadata["url"] for c in chunks if f"## {op}\n" in c.page_content}
        assert len(urls) == 1, op
    assert report.operations_total == report.operations_rendered == 5


def test_orphan_operations_are_rendered_once_with_unique_ids():
    chunks, report = chunk_corpus(_corpus())
    orphans = [c for c in chunks if c.metadata.get("orphan_operation")]
    # Pages exist for get/update/list/create; only the archive operation is orphaned.
    assert report.orphan_operations_rendered == 1
    assert {c.metadata["operation"] for c in orphans} == {f"POST {ITEM}/archive"}
    meta = orphans[0].metadata
    overview_url = "https://docs.example.com/developer-connect/reference/widgets"
    assert meta["url"] == f"{overview_url}#post-widgets-widgetid-archive"
    assert meta["title"] == "Archive Widget"
    assert meta["breadcrumb"] == ["Widgets API", "Archive Widget"]
    assert (meta["kind"], meta["type"]) == ("reference", "endpoint")
    assert orphans[0].page_content.startswith(
        "Title: Archive Widget\nSection: Widgets API > Archive Widget\n\n## POST")
    assert report.operations_total == report.operations_rendered == 5
    assert all(n == 1 for n in _rendered_ops(chunks).values())
    assert set(_rendered_ops(chunks)) == OP_HEADINGS
    ids = _ids(chunks)
    assert len(ids) == len(set(ids))


def test_orphans_use_operation_id_fragment_and_prefer_basic_overview_page():
    oas = make_oas()
    recs = [
        record("get-widget", oas=oas, discovered_via="sidebar"),
        record("widgets", type="basic", oas=oas, body="", discovered_via="link_bfs"),
    ]
    chunks, report = chunk_corpus(recs)
    orphan_urls = {c.metadata["url"] for c in chunks if c.metadata.get("orphan_operation")}
    base = "https://docs.example.com/developer-connect/reference/widgets"
    assert orphan_urls == {f"{base}#update-widget", f"{base}#list-widgets", f"{base}#create-widget",
                           f"{base}#post-widgets-widgetid-archive"}
    assert report.orphan_operations_rendered == 4


def test_orphans_fall_back_to_first_page_and_keep_ids_unique():
    oas = make_oas()
    # Two endpoint pages, no overview page; a page URL already uses the fragment.
    first = record("get-widget", oas=oas, discovered_via="sidebar")
    clash = record("clash", oas=oas, url=first["url"] + "#update-widget", discovered_via="link_bfs",
                   body="A page whose URL happens to collide with an orphan fragment.")
    chunks, _ = chunk_corpus([clash, first])
    orphan_urls = {c.metadata["url"] for c in chunks if c.metadata.get("orphan_operation")}
    assert first["url"] + "#update-widget-2" in orphan_urls
    assert all(u.startswith(first["url"] + "#") for u in orphan_urls)
    ids = _ids(chunks)
    assert len(ids) == len(set(ids))


def test_overview_page_gets_compact_overview_once():
    oas = make_oas()
    recs = [
        record("widgets", type="basic", oas=oas, body="Short intro.", discovered_via="sidebar"),
        record("widgets-copy", type="basic", oas=oas, body="", discovered_via="link_bfs"),
        record("widgets-long", type="basic", oas=oas, body="Long overview. " * 40, discovered_via="link_bfs"),
    ]
    chunks, report = chunk_corpus(recs)
    first = [c for c in chunks if c.metadata["slug"] == "widgets"]
    text = "".join(strip_header(c.page_content) for c in first)
    assert text.startswith("Short intro.\n\n## Widgets API (version 1.0) — API overview")
    assert f"- GET {ITEM} - Get Widget (get-widget)" in text
    assert "### Responses" not in text
    assert not [c for c in chunks if c.metadata["slug"] == "widgets-copy"]  # overview already emitted
    long_text = "".join(strip_header(c.page_content) for c in chunks if c.metadata["slug"] == "widgets-long")
    assert "API overview" not in long_text  # not sparse: body only
    assert report.overviews_rendered == 1


def test_chunk_record_standalone_and_with_registry():
    legacy = record("update-widget", oas=make_oas())
    [new], registry = to_new_contract([legacy])
    a = chunk_record(legacy)
    b = chunk_record(new, oas_registry=registry)
    c = chunk_record(new, registry[new["oas_id"]])
    assert [x.page_content for x in a] == [x.page_content for x in b] == [x.page_content for x in c]
    assert "## PATCH" in a[0].page_content
    # A record whose oas_id is unknown yields nothing rather than crashing.
    assert chunk_record(new) == []
    _, report = chunk_corpus([new], {})
    assert report.records_missing_oas == 1


def test_duplicate_urls_are_chunked_once():
    a = record("intro", kind="guide", type="basic", body="First copy.", discovered_via="sidebar")
    b = dict(a, body="Second copy, longer than the first.", discovered_via="link_bfs")
    chunks, report = chunk_corpus([b, a])
    assert [strip_header(c.page_content) for c in chunks] == ["First copy."]
    assert report.records_skipped_duplicate_url == 1


def test_long_records_are_split_with_header_and_contiguous_indices():
    body = "\n\n".join(f"Paragraph {i}: " + " ".join(f"w{i}x{j}" for j in range(60)) for i in range(30))
    chunks = chunk_record(record("guide", kind="guide", type="basic", body=body, breadcrumb=["A", "B"]))
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)
    assert all(c.page_content.startswith("Title: Guide\nSection: A > B\n\n") for c in chunks)
    joined = "\n\n".join(strip_header(c.page_content) for c in chunks)
    for i in range(30):
        assert f"Paragraph {i}:" in joined


# --- _merge_small ------------------------------------------------------------------

def _splitter():
    return RecursiveCharacterTextSplitter(chunk_size=SPLIT_CHUNK_SIZE, chunk_overlap=SPLIT_OVERLAP,
                                          separators=_SEPARATORS, length_function=len)


def _overlapping_text() -> str:
    # Short unique lines -> the splitter carries ~SPLIT_OVERLAP chars of overlap
    # between pieces, and the tail piece ends up small enough to be merged.
    lines = [f"line {i:04d} lorem ipsum dolor sit amet" for i in range(46)]
    return "\n".join(lines)


def test_merge_small_with_source_text_does_not_duplicate_overlap():
    text = _overlapping_text()
    pieces = _splitter().split_text(text)
    assert len(pieces) >= 2 and len(pieces[-1]) < MIN_CHUNK_SIZE
    assert set(pieces[-2].splitlines()) & set(pieces[-1].splitlines())  # splitter overlap exists
    merged = _merge_small(pieces, text)
    assert len(merged) == len(pieces) - 1
    for piece in merged:
        assert piece in text
        lines = piece.splitlines()
        assert len(lines) == len(set(lines))
    assert merged[-1].endswith("line 0045 lorem ipsum dolor sit amet")


def test_merge_small_without_source_text_drops_overlap():
    text = _overlapping_text()
    pieces = _splitter().split_text(text)
    merged = _merge_small(pieces)
    tail_lines = merged[-1].splitlines()
    assert len(tail_lines) == len(set(tail_lines))
    assert merged == _merge_small(pieces, text)


def test_merge_small_does_not_mistake_code_fences_for_overlap():
    prev = "x" * 600 + "\n```"
    nxt = "```json\n{\"a\": 1}\n```"
    assert _merge_small([prev, nxt]) == [prev + "\n\n" + nxt]


def test_merge_small_keeps_large_neighbours_apart():
    pieces = ["a" * 900, "b" * 900]
    assert _merge_small(pieces) == pieces
    assert _merge_small([]) == [] and _merge_small(["x"]) == ["x"]
