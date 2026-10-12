"""Tests for scripts.ingest.dedupe: preference order and duplicate detection."""
from __future__ import annotations

import random

from scripts.ingest.dedupe import DISCOVERY_RANK, dedupe_by_content, order_records, preference_key
from tests.ingest.fixtures.synthetic import make_oas, record, to_new_contract

VIAS = ["sidebar", "reference_sidebar", "changelog_index", "sitemap", "changelog_ref",
        "oas_operation", "link_bfs"]


def test_discovery_rank_order_matches_contract():
    assert sorted(DISCOVERY_RANK, key=DISCOVERY_RANK.get) == VIAS


def test_discovery_rank_matches_the_crawler_priority():
    # The crawler picks a record's discovered_via (and its duplicate winner)
    # with the same order the ingest uses to break ties.
    from scripts.crawler.sidebar import DISCOVERY_SOURCES

    assert tuple(sorted(DISCOVERY_RANK, key=DISCOVERY_RANK.get)) == DISCOVERY_SOURCES


def test_preference_order_by_discovered_via_then_unknown_last():
    recs = [record(f"p-{via}", kind="guide", type="basic", body="same body",
                   url=f"https://x/{9 - i}", discovered_via=via)
            for i, via in enumerate(VIAS)]
    recs.append(record("p-missing", kind="guide", type="basic", body="same body", url="https://x/0"))
    recs.append(record("p-odd", kind="guide", type="basic", body="same body", url="https://x/00",
                       discovered_via="carrier_pigeon"))
    shuffled = recs[:]
    random.Random(7).shuffle(shuffled)
    ordered = [r["slug"] for r in order_records(shuffled)]
    assert ordered[: len(VIAS)] == [f"p-{via}" for via in VIAS]
    assert set(ordered[len(VIAS):]) == {"p-missing", "p-odd"}

    kept, dropped = dedupe_by_content(shuffled)
    assert [r["slug"] for r in kept] == ["p-sidebar"]
    assert len(dropped) == len(recs) - 1


def test_hidden_loses_to_visible_from_the_same_source():
    visible = record("a", kind="guide", type="basic", body="x" * 10, url="https://x/z", discovered_via="link_bfs")
    hidden = record("b", kind="guide", type="basic", body="x" * 10, url="https://x/a", discovered_via="link_bfs",
                    hidden=True)
    kept, dropped = dedupe_by_content([hidden, visible])
    assert kept == [visible] and dropped == [hidden]
    # ... but discovery source outranks visibility.
    hidden["discovered_via"] = "sidebar"
    kept, _ = dedupe_by_content([hidden, visible])
    assert kept == [hidden]


def test_legacy_records_without_discovered_via_sort_deterministically():
    recs = [
        record("short", kind="guide", type="basic", body="b", url="https://x/1"),
        record("long", kind="guide", type="basic", body="bbbb", url="https://x/2"),
        record("hidden", kind="guide", type="basic", body="bbbbbbbb", url="https://x/0", hidden=True),
        record("tie-b", kind="guide", type="basic", body="bb", url="https://x/b"),
        record("tie-a", kind="guide", type="basic", body="bb", url="https://x/a"),
    ]
    expected = ["long", "tie-a", "tie-b", "short", "hidden"]
    for seed in range(5):
        shuffled = recs[:]
        random.Random(seed).shuffle(shuffled)
        assert [r["slug"] for r in order_records(shuffled)] == expected
    assert preference_key(recs[0])[0] == len(DISCOVERY_RANK)


def test_empty_bodies_are_never_collapsed():
    recs = [record(f"e{i}", body="", oas=make_oas()) for i in range(3)]
    kept, dropped = dedupe_by_content(recs)
    assert len(kept) == 3 and not dropped


def test_endpoint_pages_sharing_prose_but_not_operation_are_kept():
    banner = "> Scheduled for Sunset\n\nThis endpoint is going away."
    oas = make_oas()
    legacy = [
        record("get-widget", body=banner, oas=oas),
        record("update-widget", body=banner, oas=oas),
        record("get-widget", body=banner, oas=oas, url="https://docs.example.com/x/get-widget"),
    ]
    for recs in (legacy, to_new_contract(legacy)[0]):
        kept, dropped = dedupe_by_content(recs)
        assert sorted(r["slug"] for r in kept) == ["get-widget", "update-widget"]
        assert [r["slug"] for r in dropped] == ["get-widget"]


def test_basic_pages_still_dedupe_on_body_alone():
    a = record("a", type="basic", body="Overview text", oas=make_oas())
    b = record("b", type="basic", body="Overview text", oas=make_oas("Other API"))
    kept, dropped = dedupe_by_content([a, b])
    assert len(kept) == 1 and len(dropped) == 1


def test_body_duplicate_carrying_other_operations_is_kept():
    """Overview pages of two API groups share boilerplate. Dropping one used to
    drop every operation only its group's OAS document has."""
    widgets = record("widgets", type="basic", body="Shared boilerplate.", oas=make_oas(),
                     discovered_via="sidebar")
    gadgets = record("gadgets", type="basic", body="Shared boilerplate.",
                     oas=make_oas("Gadgets API", prefix="/gadgets"), discovered_via="link_bfs")
    no_api = record("plain", type="basic", body="Shared boilerplate.", discovered_via="link_bfs")
    legacy = [gadgets, no_api, widgets]
    new, registry = to_new_contract(legacy)
    for recs, reg in ((legacy, None), (new, registry), (new, None)):
        kept, dropped = dedupe_by_content(recs, reg)
        assert [r["slug"] for r in kept] == ["widgets", "gadgets"]
        assert [r["slug"] for r in dropped] == ["plain"]
