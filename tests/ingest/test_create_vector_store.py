"""Tests for the embedding-free half of scripts.create_vector_store (`build_documents`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pytest
from langchain_core.documents import Document

from scripts import create_vector_store as cvs
from tests.ingest.fixtures.synthetic import make_oas, record, to_new_contract


@pytest.fixture(autouse=True)
def _no_registry_env(monkeypatch):
    # A developer's .env (loaded on import) must not redirect the registry.
    monkeypatch.delenv("OAS_REGISTRY_PATH", raising=False)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _legacy_corpus():
    oas = make_oas()
    return [
        record("widgets", type="basic", oas=oas, body="", breadcrumb=["Widgets API"]),
        record("get-widget", oas=oas, body="Usage notes: be gentle."),
        record("update-widget", oas=oas),
        record("intro", kind="guide", type="basic", body="Welcome."),
        record("api-key", title="API Key", kind="custompage", type="custompage",
               body="Generate an API key from the developer portal.", url="https://x/developer-connect/page/api-key"),
        record("secret", kind="guide", type="basic", body="Hidden but useful.", hidden=True),
    ]


def _dump(documents):
    return [(d.page_content, d.metadata) for d in documents]


def test_legacy_and_new_contract_files_build_identical_documents(tmp_path):
    legacy = _legacy_corpus()
    (tmp_path / "legacy").mkdir()
    (tmp_path / "new").mkdir()
    legacy_path = _write_jsonl(tmp_path / "legacy" / "dc.jsonl", legacy)
    new, registry = to_new_contract(legacy)
    new_path = _write_jsonl(tmp_path / "new" / "dc.jsonl", new)
    (tmp_path / "new" / "dc_oas.json").write_text(json.dumps(registry), encoding="utf-8")

    docs_a, stats_a = cvs.build_documents(legacy_path)
    docs_b, stats_b = cvs.build_documents(new_path)  # registry found next to the records
    assert all(isinstance(d, Document) for d in docs_a)
    assert _dump(docs_a) == _dump(docs_b)
    assert stats_b["oas_registry_path"] == str(tmp_path / "new" / "dc_oas.json")
    assert stats_a["oas_registry_path"] is None
    for key in ("chunks", "duplicate_chunk_ratio", "operations_rendered", "orphan_operations_rendered",
                "endpoint_pages_with_prose", "max_chunks_per_url", "build_documents_seconds"):
        assert key in stats_a
    assert stats_a["operations_total"] == stats_a["operations_rendered"] == 5
    assert stats_a["orphan_operations_rendered"] == 3
    assert stats_a["endpoint_pages_with_prose"] == 1
    for key in ("chunks", "operations_rendered", "duplicate_chunk_ratio", "endpoint_pages_with_prose"):
        assert stats_a[key] == stats_b[key]


def test_custom_pages_and_hidden_pages_are_kept_by_default(tmp_path):
    path = _write_jsonl(tmp_path / "dc.jsonl", _legacy_corpus())
    docs, stats = cvs.build_documents(path)
    slugs = {d.metadata["slug"] for d in docs}
    assert {"api-key", "secret"} <= slugs
    assert stats["records_hidden"] == 1

    docs, stats = cvs.build_documents(path, skip_hidden=True)
    assert "secret" not in {d.metadata["slug"] for d in docs}
    assert stats["records_after_filter"] == stats["records_loaded"] - 1


def test_unknown_oas_id_fails_loudly(tmp_path):
    new, registry = to_new_contract(_legacy_corpus())
    path = _write_jsonl(tmp_path / "dc.jsonl", new)
    with pytest.raises(ValueError, match="missing from the OAS registry"):
        cvs.build_documents(path)
    reg = tmp_path / "elsewhere.json"
    reg.write_text(json.dumps(registry), encoding="utf-8")
    docs, _ = cvs.build_documents(path, reg)
    assert docs


def test_default_sibling_paths_match_crawler_naming():
    assert cvs.default_registry_path(Path("/d/developer_connect.jsonl")) == Path("/d/developer_connect_oas.json")
    assert cvs.default_registry_path(Path("/d/developer_connect.sample.jsonl")) == \
        Path("/d/developer_connect_oas.sample.json")
    assert cvs.default_manifest_path(Path("/d/developer_connect.sample.jsonl")) == \
        Path("/d/developer_connect_manifest.sample.jsonl")
    assert cvs.default_registry_path(cvs.JSONL_PATH) == cvs.OAS_REGISTRY_PATH


def test_registry_path_env_override(tmp_path, monkeypatch):
    new, registry = to_new_contract(_legacy_corpus())
    path = _write_jsonl(tmp_path / "dc.jsonl", new)
    reg = tmp_path / "custom_registry.json"
    reg.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("OAS_REGISTRY_PATH", str(reg))
    assert cvs.default_registry_path(path) == reg
    _, stats = cvs.build_documents(path)
    assert stats["oas_registry_path"] == str(reg)


def test_manifest_backfills_discovered_via_for_dedupe(tmp_path):
    a = record("a-page", kind="guide", type="basic", body="Same text", url="https://x/a")
    b = record("b-page", kind="guide", type="basic", body="Same text", url="https://x/b")
    path = _write_jsonl(tmp_path / "dc.jsonl", [a, b])
    docs, stats = cvs.build_documents(path)
    assert [d.metadata["url"] for d in docs] == ["https://x/a"]  # URL tie-break

    _write_jsonl(tmp_path / "dc_manifest.jsonl", [
        {"url": "https://x/a", "discovered_via": "link_bfs"},
        {"url": "https://x/b", "discovered_via": "sidebar"},
    ])
    docs, stats = cvs.build_documents(path)
    assert [d.metadata["url"] for d in docs] == ["https://x/b"]  # sidebar wins
    assert stats["records_discovered_via_backfilled"] == 2
    assert stats["dropped_dupe_urls"] == ["https://x/a"]


def test_duplicate_chunk_ids_are_rejected():
    docs = [Document(page_content="Title: A\nSection: S\n\nx", metadata={"url": "u", "chunk_index": 0}),
            Document(page_content="Title: B\nSection: S\n\ny", metadata={"url": "u", "chunk_index": 0})]
    with pytest.raises(ValueError, match="duplicate"):
        cvs._validate_unique_ids(docs)


def test_duplicate_chunk_ratio_ignores_header():
    docs = [Document(page_content=f"Title: {t}\nSection: S\n\n{body}", metadata={"url": t, "chunk_index": 0})
            for t, body in (("A", "same"), ("B", "same"), ("C", "different"), ("D", "unique"))]
    stats = cvs._document_stats(docs)
    assert stats["duplicate_chunk_ratio"] == 0.5
    assert stats["max_chunks_per_url"] == 1


def test_embedding_fingerprint_includes_backend_and_model():
    texts = ["a", "b"]
    base = cvs._texts_fingerprint(texts, backend="local", model="jinaai/jina-embeddings-v3")
    assert base == cvs._texts_fingerprint(texts, backend="local", model="jinaai/jina-embeddings-v3")
    assert base != cvs._texts_fingerprint(texts, backend="api", model="jinaai/jina-embeddings-v3")
    assert base != cvs._texts_fingerprint(texts, backend="local", model="other/model")
    assert base != cvs._texts_fingerprint(["a", "c"], backend="local", model="jinaai/jina-embeddings-v3")


def test_dry_run_cli_needs_no_embedder(tmp_path, monkeypatch):
    path = _write_jsonl(tmp_path / "dc.jsonl", _legacy_corpus())

    def _boom(*_a, **_k):
        raise AssertionError("embedder must not be constructed on --dry-run")

    monkeypatch.setattr(cvs.JinaEmbedder, "from_env", classmethod(_boom))
    assert cvs.main(["--jsonl", str(path), "--dry-run"]) == 0


def test_registry_build_keeps_operations_of_a_body_duplicate(tmp_path):
    shared = "Shared boilerplate for every settings group."
    legacy = [
        record("widgets", type="basic", oas=make_oas(), body=shared, discovered_via="sidebar"),
        record("gadgets", type="basic", oas=make_oas("Gadgets API", prefix="/gadgets"),
               body=shared, discovered_via="link_bfs"),
    ]
    new, registry = to_new_contract(legacy)
    path = _write_jsonl(tmp_path / "dc.jsonl", new)
    (tmp_path / "dc_oas.json").write_text(json.dumps(registry), encoding="utf-8")
    docs, stats = cvs.build_documents(path)
    assert stats["records_dropped_dupe"] == 0
    assert stats["operations_total"] == stats["operations_rendered"] == 10
    ops = {d.metadata.get("operation") for d in docs}
    assert "GET /gadgets" in ops and "GET /widgets" in ops
