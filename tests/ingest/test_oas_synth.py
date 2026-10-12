"""Tests for scripts.ingest.oas_synth: one operation at a time, never the whole group."""
from __future__ import annotations

from scripts.ingest.oas_synth import (
    _type_label,
    page_operation,
    synthesize_api_overview,
    synthesize_endpoint_markdown,
    synthesize_operation_markdown,
)
from tests.ingest.fixtures.synthetic import make_oas, record, to_new_contract

ITEM = "/widgets/{widgetId}"


def _headings(md: str):
    return [line for line in md.splitlines() if line.startswith("## ")]


def test_single_operation_renders_only_that_operation():
    md = synthesize_operation_markdown(make_oas(), "get", ITEM)
    assert md.startswith(f"## GET {ITEM}\n")
    assert _headings(md) == [f"## GET {ITEM}"]
    assert "**Get Widget**" in md and "Retrieves a single widget." in md
    for other in ("Update Widget", "List Widgets", "Create Widget", "Archive Widget"):
        assert other not in md


def test_path_level_parameters_are_merged_and_overridden():
    md = synthesize_operation_markdown(make_oas(), "get", ITEM)
    # $ref'd path-level parameter is resolved.
    assert "| `widgetId` | path | yes | string | Widget identifier |" in md
    # Operation-level X-Trace overrides the path-level one with the same (name, in).
    assert md.count("`X-Trace`") == 1
    assert "operation-level trace header" in md and "path-level trace header" not in md
    # Path-level parameters come first.
    assert md.index("`widgetId`") < md.index("`X-Trace`") < md.index("`view`")
    # An operation without its own parameters still gets the path-level ones.
    patch = synthesize_operation_markdown(make_oas(), "patch", ITEM)
    assert "`widgetId`" in patch and "path-level trace header" in patch


def test_servers_precedence_operation_then_path_then_document():
    oas = make_oas()
    get_item = synthesize_operation_markdown(oas, "get", ITEM)
    assert "Base URL: `https://path.example.com`" in get_item
    assert "api.example.com" not in get_item
    patch = synthesize_operation_markdown(oas, "patch", ITEM)
    assert "Base URL: `https://op.example.com` (Op)" in patch
    listing = synthesize_operation_markdown(oas, "get", "/widgets")
    # Document-level servers; the blank ReadMe placeholder entry is dropped.
    assert "\nBase URL: `https://api.example.com` (Production)\n" in listing


def test_full_schema_for_2xx_and_one_line_for_other_statuses():
    md = synthesize_operation_markdown(make_oas(), "get", ITEM)
    assert "#### 200 — OK" in md
    assert "##### Schema" in md and '"colour"' in md
    assert "- `404` — Not Found" in md
    assert "errorCode" not in md  # the generic Error schema is not rendered
    # Only one example per block; the other is noted.
    assert "Sprocket" in md and "Cog" not in md
    assert "1 additional example(s) omitted" in md


def test_request_body_and_deprecated_notice():
    md = synthesize_operation_markdown(make_oas(), "patch", ITEM)
    assert "### Request body" in md and "_Required._" in md
    assert "marked deprecated" in md
    assert "#### 204 — No Content" in md and "- `400` — Bad Request" in md


def test_doc_api_code_samples_are_appended():
    doc_api = {"examples": {"codes": [
        {"language": "curl", "code": "curl https://api.example.com/widgets/w-1", "name": "cURL"},
        {"language": "json", "code": "{}"},  # ReadMe stub, skipped
    ]}, "results": {"codes": [{"language": "json", "code": '{"id": "w-1"}', "status": 200}]}}
    md = synthesize_operation_markdown(make_oas(), "get", ITEM, doc_api=doc_api)
    assert "### Code samples (rendered by ReadMe)" in md
    assert "curl https://api.example.com/widgets/w-1" in md
    assert "#### Response 200 — json" in md
    assert md.index("### Responses") < md.index("### Code samples")


def test_unknown_operation_renders_nothing():
    assert synthesize_operation_markdown(make_oas(), "delete", "/widgets") == ""
    assert synthesize_operation_markdown(None, "get", ITEM) == ""


def test_type_label_handles_non_string_type_members():
    assert _type_label({"type": ["string", None, 3]}) == "string|null|3"
    md = synthesize_operation_markdown(make_oas(), "get", ITEM)
    assert "| `view` | query | no | string|null | Projection |" in md


def test_ref_cycles_do_not_crash():
    oas = make_oas()
    oas["components"]["schemas"]["Widget"]["properties"]["parent"] = {"$ref": "#/components/schemas/Widget"}
    md = synthesize_operation_markdown(oas, "get", ITEM)
    assert "## GET" in md and '"parent"' in md


def test_overview_lists_every_operation_without_schemas():
    ov = synthesize_api_overview(make_oas())
    assert ov.startswith("## Widgets API (version 1.0) — API overview")
    assert "Manage widgets and their lifecycle." in ov
    assert "Base URL: `https://api.example.com` (Production)" in ov
    assert "### Operations (5)" in ov
    assert f"- GET {ITEM} - Get Widget (get-widget)" in ov
    assert f"- PATCH {ITEM} - Update Widget (update-widget) [deprecated]" in ov
    assert "- GET /widgets - List Widgets (list-widgets)" in ov
    assert "- POST /widgets - Create Widget (create-widget)" in ov
    assert f"- POST {ITEM}/archive - Archive Widget" in ov
    assert "Schema" not in ov and "```" not in ov and "### Responses" not in ov
    assert synthesize_api_overview(None) == ""


def test_endpoint_markdown_uses_record_operation_from_new_contract():
    legacy = record("some-unrelated-slug", title="Unrelated", oas=make_oas())
    [new], registry = to_new_contract([legacy])
    assert new["operation"] is None  # nothing matches this slug/title...
    new["operation"] = {"method": "patch", "path": ITEM}  # ...but the crawler said so
    md = synthesize_endpoint_markdown(new, registry[new["oas_id"]])
    assert _headings(md) == [f"## PATCH {ITEM}"]


def test_endpoint_markdown_legacy_record_finds_operation_by_slug_or_title():
    md = synthesize_endpoint_markdown(record("create-widget", oas=make_oas()))
    assert _headings(md) == ["## POST /widgets"]
    by_title = synthesize_endpoint_markdown(record("x-archive", title="Archive Widget", oas=make_oas()))
    assert _headings(by_title) == [f"## POST {ITEM}/archive"]


def test_endpoint_markdown_falls_back_to_overview_never_full_dump():
    oas = make_oas()
    rec = record("no-such-page", title="Nothing Matches", oas=oas)
    assert page_operation(rec) is None
    md = synthesize_endpoint_markdown(rec)
    assert "API overview" in md
    assert "### Responses" not in md and "##### Schema" not in md
    full = sum(len(synthesize_operation_markdown(oas, m, p)) for m, p in
               (("get", ITEM), ("patch", ITEM), ("get", "/widgets"), ("post", "/widgets")))
    assert len(md) < full / 2


def test_page_operation_prefers_valid_record_operation():
    oas = make_oas()
    rec = record("get-widget", oas=oas)
    assert page_operation(rec) == ("get", ITEM)
    rec["operation"] = {"method": "GET", "path": "/widgets"}
    assert page_operation(rec) == ("get", "/widgets")
    # An operation that is not in the document is ignored in favour of the matcher.
    rec["operation"] = {"method": "get", "path": "/nope"}
    assert page_operation(rec) == ("get", ITEM)
    # Without any OAS (dedupe on new-contract records) the field is trusted.
    bare = {"slug": "x", "operation": {"method": "post", "path": "/widgets"}}
    assert page_operation(bare) == ("post", "/widgets")


def test_shared_schema_is_rendered_once_with_a_ledger():
    oas = make_oas()
    seen: dict = {}
    first = synthesize_operation_markdown(oas, "get", ITEM, seen_blocks=seen)
    second = synthesize_operation_markdown(oas, "patch", ITEM, seen_blocks=seen)
    assert '"colour"' in first
    assert '"colour"' not in second
    assert f"_Same as the schema of the 200 response of `GET {ITEM}` (rendered there in full)._" in second
    assert "Fields: `id`, `name`, `colour`" in second
    # Without a ledger every operation is self-contained.
    assert '"colour"' in synthesize_operation_markdown(oas, "patch", ITEM)
