"""Build the three-index vector store for the RAG pipeline.

Inputs:
  - scripts/data/developer_connect.jsonl   (crawled documentation records)
  - scripts/data/developer_connect_oas.json (OAS registry {oas_id: document};
        optional — legacy datasets embed each page's OAS in its record)
  - scripts/data/developer_connect_manifest.jsonl (optional; recovers
        `discovered_via` for legacy records that lack it)
  - scripts/data/Encompass_Developer_Connect_postman_collection.json

Outputs (under VECTOR_STORE_PATH, default `vector_store/`):
  - jsonl_faiss/index.faiss + index.pkl   (LangChain FAISS-IP, Jina v3 vectors)
  - jsonl_bm25.pkl + jsonl_chunks.pkl     (BM25 over the same chunks)
  - postman_bm25.pkl + postman_entries.pkl (separate BM25 for endpoint surfacing)
  - stats.json                            (chunk/index counts for sanity)

Pipeline:
  load -> filter -> dedupe -> chunk (`build_documents`, no embeddings needed)
  -> embed via Jina v3 (task=retrieval.passage) -> FAISS-IP + BM25.

Run from the repo root:
    python -m scripts.create_vector_store             # full build
    python -m scripts.create_vector_store --dry-run   # chunk + stats only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/create_vector_store.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn, MofNCompleteColumn, Progress, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from scripts.ingest.chunking import Chunk, chunk_corpus, strip_header  # noqa: E402
from scripts.ingest.dedupe import DISCOVERY_RANK, dedupe_by_content  # noqa: E402
from scripts.ingest.jina_embedder import JinaEmbedder  # noqa: E402
from scripts.ingest.postman import load_postman_entries, tokenize_query  # noqa: E402

console = Console()
load_dotenv()


# --- Paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "scripts" / "data"
JSONL_PATH = DATA_DIR / "developer_connect.jsonl"
OAS_REGISTRY_PATH = DATA_DIR / "developer_connect_oas.json"
POSTMAN_PATH = DATA_DIR / "Encompass_Developer_Connect_postman_collection.json"
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "vector_store"))


def _sibling_path(jsonl_path: Path, kind: str, ext: str) -> Path:
    """Mirror the crawler's output naming (`derived_paths` in scripts/crawler/__main__.py):
    `x.jsonl` -> `x_<kind>.<ext>`, `x.sample.jsonl` -> `x_<kind>.sample.<ext>`."""
    name = jsonl_path.name
    core = name[: -len(".jsonl")] if name.endswith(".jsonl") else jsonl_path.stem
    suffix = ""
    if core.endswith(".sample"):
        core, suffix = core[: -len(".sample")], ".sample"
    return jsonl_path.with_name(f"{core}_{kind}{suffix}.{ext}")


def default_registry_path(jsonl_path: Path) -> Path:
    """`OAS_REGISTRY_PATH` env var, else the registry the crawler wrote next to the records."""
    env = os.getenv("OAS_REGISTRY_PATH")
    if env:
        return Path(env)
    return _sibling_path(jsonl_path, "oas", "json")


def default_manifest_path(jsonl_path: Path) -> Path:
    return _sibling_path(jsonl_path, "manifest", "jsonl")


# --- Steps --------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_oas_registry(path: Optional[Path]) -> Dict[str, Dict]:
    """`{oas_id: OpenAPI document}` written by the crawler; {} when absent."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"OAS registry {path} must be a JSON object of {{oas_id: document}}")
    return data


def _check_registry(records: Sequence[Dict], registry: Dict[str, Dict], path: Path) -> None:
    """Fail loudly instead of silently dropping the API reference of pages."""
    missing = sorted({
        r["oas_id"] for r in records
        if r.get("oas_id") and not isinstance(r.get("oas"), dict) and r["oas_id"] not in registry
    })
    if missing:
        raise ValueError(
            f"{len(missing)} oas_id(s) referenced by records are missing from the OAS "
            f"registry {path} (e.g. {missing[:3]}). Use the registry written by the same "
            f"crawl, or point OAS_REGISTRY_PATH at it."
        )


def _backfill_discovered_via(records: List[Dict], manifest_path: Optional[Path]) -> int:
    """Copy `discovered_via` from the crawl manifest onto records lacking it.

    Older crawls wrote the field only to the manifest, so dedupe's
    navigation-first preference never applied. Returns the number of records
    updated (in place).
    """
    if manifest_path is None or not manifest_path.exists():
        return 0
    missing = [r for r in records if not r.get("discovered_via")]
    if not missing:
        return 0
    unknown = len(DISCOVERY_RANK)
    best: Dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            url, via = row.get("url"), row.get("discovered_via")
            if not url or not isinstance(via, str):
                continue
            if url not in best or DISCOVERY_RANK.get(via, unknown) < DISCOVERY_RANK.get(best[url], unknown):
                best[url] = via
    updated = 0
    for r in missing:
        via = best.get(r.get("url") or "")
        if via:
            r["discovered_via"] = via
            updated += 1
    return updated


def _filter_records(records: List[Dict], *, skip_hidden: bool = False) -> List[Dict]:
    """Drop records we don't ingest. Hidden pages are kept unless asked otherwise;
    records without text (e.g. empty custom pages) simply yield no chunks."""
    if not skip_hidden:
        return list(records)
    return [r for r in records if not r.get("hidden")]


def _chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    return [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]


def _validate_unique_ids(documents: Sequence[Document]) -> None:
    """The retriever identifies chunks by (metadata url, metadata chunk_index)."""
    ids = Counter((d.metadata.get("url"), d.metadata.get("chunk_index")) for d in documents)
    dupes = [cid for cid, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate (url, chunk_index) chunk id(s), e.g. {dupes[:3]}")


def _document_stats(documents: Sequence[Document]) -> Dict[str, Any]:
    sizes = sorted(len(d.page_content) for d in documents)
    bodies = Counter(strip_header(d.page_content) for d in documents)
    duplicated = sum(n for n in bodies.values() if n > 1)
    per_url = Counter(d.metadata.get("url") for d in documents)
    if not sizes:
        return {"chunks": 0}
    return {
        "chunks": len(sizes),
        "chunk_size_min": sizes[0],
        "chunk_size_median": int(statistics.median(sizes)),
        "chunk_size_p95": sizes[int(len(sizes) * 0.95)],
        "chunk_size_max": sizes[-1],
        "max_chunks_per_url": max(per_url.values()),
        # Share of chunks whose text (minus the Title/Section header) is
        # byte-identical to another chunk's.
        "duplicate_chunk_ratio": round(duplicated / len(sizes), 4),
    }


def build_documents(
    jsonl_path: Path = JSONL_PATH,
    registry_path: Optional[Path] = None,
    *,
    manifest_path: Optional[Path] = None,
    skip_hidden: bool = False,
) -> Tuple[List[Document], Dict[str, Any]]:
    """Load -> filter -> dedupe -> chunk -> Documents, without any embedding.

    `registry_path` defaults to `default_registry_path(jsonl_path)` and
    `manifest_path` to the crawl manifest next to the records. Raises
    ValueError if records reference unknown `oas_id`s or chunk ids collide.
    """
    started = time.perf_counter()
    jsonl_path = Path(jsonl_path)
    records = _load_jsonl(jsonl_path)

    reg_path = Path(registry_path) if registry_path is not None else default_registry_path(jsonl_path)
    registry = load_oas_registry(reg_path)
    _check_registry(records, registry, reg_path)

    man_path = Path(manifest_path) if manifest_path is not None else default_manifest_path(jsonl_path)
    backfilled = _backfill_discovered_via(records, man_path)

    filtered = _filter_records(records, skip_hidden=skip_hidden)
    kept, dropped = dedupe_by_content(filtered)
    chunks, report = chunk_corpus(kept, registry)
    documents = _chunks_to_documents(chunks)
    _validate_unique_ids(documents)

    stats: Dict[str, Any] = {
        "jsonl_path": str(jsonl_path),
        "oas_registry_path": str(reg_path) if reg_path.exists() else None,
        "oas_registry_documents": len(registry),
        "records_loaded": len(records),
        "records_hidden": sum(1 for r in records if r.get("hidden")),
        "records_discovered_via_backfilled": backfilled,
        "records_after_filter": len(filtered),
        "records_after_dedupe": len(kept),
        "records_dropped_dupe": len(dropped),
        "dropped_dupe_urls": [r.get("url") for r in dropped],
        "records_chunked": report.records_chunked,
        "records_skipped_duplicate_url": report.records_skipped_duplicate_url,
        **_document_stats(documents),
        "operations_total": report.operations_total,
        "operations_rendered": report.operations_rendered,
        "orphan_operations_rendered": report.orphan_operations_rendered,
        "endpoint_pages_with_prose": report.endpoint_pages_with_prose,
        "endpoint_pages_without_operation": report.endpoint_pages_without_operation,
        "overviews_rendered": report.overviews_rendered,
        "build_documents_seconds": round(time.perf_counter() - started, 2),
    }
    return documents, stats


def _texts_fingerprint(texts: Sequence[str], *, backend: str, model: str) -> str:
    """Stable hash of the embedder identity + chunk text list. If chunking or
    the embedding model changes, this changes — we use it to invalidate stale
    checkpoints (vectors from different models must never be mixed)."""
    h = hashlib.sha256()
    h.update(f"{backend}\x00{model}\x00".encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _embed_with_checkpoint(
    texts: List[str], embedder: JinaEmbedder, checkpoint: Path,
    *, save_every: int = 512,
) -> np.ndarray:
    """Embed all `texts` with on-disk resumability.

    The Jina API can rate-limit mid-build; this lets a re-run pick up where it
    left off instead of re-paying for already-embedded chunks. The checkpoint
    file holds however many vectors we've successfully computed so far (in the
    same order as `texts`); a sidecar JSON tracks the chunk-list + embedder
    fingerprint so a checkpoint from a different chunking config or model
    doesn't silently apply.
    """
    meta_path = checkpoint.with_suffix(".meta.json")
    fingerprint = _texts_fingerprint(texts, backend=embedder.backend, model=embedder.model_name)

    if checkpoint.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            existing = np.load(checkpoint)
            if existing.shape[0] >= len(texts):
                console.print(f"  [dim]checkpoint complete ({existing.shape[0]} vectors); skipping embed[/dim]")
                return existing[: len(texts)]
            console.print(f"  [dim]resuming from checkpoint ({existing.shape[0]}/{len(texts)} embedded)[/dim]")
            vectors = existing
        else:
            console.print("  [yellow]checkpoint stale (chunks or model changed) — embedding from scratch[/yellow]")
            vectors = np.zeros((0, 1024), dtype=np.float32)
    else:
        vectors = np.zeros((0, 1024), dtype=np.float32)

    def _persist(v: np.ndarray) -> None:
        np.save(checkpoint, v)
        meta_path.write_text(
            json.dumps({"fingerprint": fingerprint, "count": int(v.shape[0]),
                        "backend": embedder.backend, "model": embedder.model_name}),
            encoding="utf-8",
        )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(), console=console,
    ) as bar:
        task = bar.add_task("Embedding chunks", total=len(texts), completed=int(vectors.shape[0]))
        while vectors.shape[0] < len(texts):
            start = int(vectors.shape[0])
            end = min(start + save_every, len(texts))
            new_vecs = embedder.embed_passages(texts[start:end])
            vectors = np.vstack([vectors, new_vecs]) if vectors.size else new_vecs
            _persist(vectors)
            bar.update(task, completed=int(vectors.shape[0]))
    return vectors


def _build_jsonl_indices(
    documents: List[Document], vectors: np.ndarray, embedder: JinaEmbedder,
) -> FAISS:
    """Build FAISS-IP from precomputed vectors so we don't re-embed on retry."""
    text_emb_pairs = list(zip(
        (d.page_content for d in documents),
        vectors.tolist(),
    ))
    metadatas = [d.metadata for d in documents]
    vector_store = FAISS.from_embeddings(
        text_emb_pairs,
        embedding=embedder.as_langchain(),
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=False,  # Jina v3 returns unit vectors already.
    )
    return vector_store


def _build_jsonl_bm25(documents: List[Document]) -> tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize_query(d.page_content) for d in documents]
    return BM25Okapi(tokenized), tokenized


def _build_postman_bm25(entries) -> BM25Okapi:
    return BM25Okapi([e.tokens for e in entries])


# --- Orchestrator -------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RAG vector store from crawled docs.")
    p.add_argument("--jsonl", type=Path, default=JSONL_PATH, help="records JSONL (default: %(default)s)")
    p.add_argument("--oas-registry", type=Path, default=None,
                   help="OAS registry JSON (default: $OAS_REGISTRY_PATH or <jsonl stem>_oas.json)")
    p.add_argument("--skip-hidden", action="store_true", help="exclude pages ReadMe marks hidden")
    p.add_argument("--dry-run", action="store_true",
                   help="load, dedupe and chunk only; print stats without embedding or writing indices")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    console.rule("[bold]Encompass RAG vector store build[/bold]")
    console.print(f"  jsonl   : {args.jsonl}")
    console.print(f"  postman : {POSTMAN_PATH}")
    console.print(f"  output  : {VECTOR_STORE_PATH.resolve()}")

    if not args.jsonl.exists():
        console.print(f"[red]Missing input: {args.jsonl}[/red]")
        return 1
    if not args.dry_run and not POSTMAN_PATH.exists():
        console.print(f"[red]Missing input: {POSTMAN_PATH}[/red]")
        return 1

    # 1) Load + filter + dedupe + chunk
    documents, build_stats = build_documents(
        args.jsonl, args.oas_registry, skip_hidden=args.skip_hidden,
    )
    console.print(f"\n[cyan]Loaded {build_stats['records_loaded']} records[/cyan]"
                  f"  (OAS registry: {build_stats['oas_registry_path'] or 'none; legacy embedded OAS'})")
    console.print(f"  after filter{' (skip hidden)' if args.skip_hidden else ''}: "
                  f"{build_stats['records_after_filter']}")
    dropped_urls = build_stats["dropped_dupe_urls"]
    console.print(f"  after dedupe: {build_stats['records_after_dedupe']} "
                  f"(dropped {len(dropped_urls)} dup{'s' if len(dropped_urls) != 1 else ''})")
    for url in dropped_urls[:5]:
        console.print(f"    [dim]dropped:[/dim] {url}")
    if len(dropped_urls) > 5:
        console.print(f"    [dim]... and {len(dropped_urls) - 5} more[/dim]")

    console.print(f"\n[cyan]Chunked into {build_stats['chunks']} chunks[/cyan]")
    if not documents:
        console.print("[red]No chunks produced; aborting.[/red]")
        return 1
    for key in ("chunk_size_median", "chunk_size_p95", "chunk_size_max", "max_chunks_per_url",
                "duplicate_chunk_ratio", "operations_total", "operations_rendered",
                "orphan_operations_rendered", "endpoint_pages_with_prose", "build_documents_seconds"):
        console.print(f"  {key}: {build_stats[key]}")
    if args.dry_run:
        console.rule("[bold green]Dry run complete (no embeddings)[/bold green]")
        return 0

    # 2) Embed (with on-disk checkpoint) + FAISS index
    embedder = JinaEmbedder.from_env()
    console.print(f"\n[cyan]Jina backend: {embedder.backend}  model: {embedder.model_name}  device: {embedder.device}[/cyan]")
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint = VECTOR_STORE_PATH / "_jsonl_embeddings_checkpoint.npy"
    texts = [d.page_content for d in documents]
    vectors = _embed_with_checkpoint(texts, embedder, checkpoint)
    vector_store = _build_jsonl_indices(documents, vectors, embedder)

    # 3) JSONL BM25
    console.print("\n[yellow]Building JSONL BM25...[/yellow]")
    jsonl_bm25, _ = _build_jsonl_bm25(documents)

    # 4) Postman
    console.print("\n[yellow]Loading + indexing Postman collection...[/yellow]")
    postman_entries = load_postman_entries(POSTMAN_PATH)
    postman_bm25 = _build_postman_bm25(postman_entries)
    console.print(f"  postman entries: {len(postman_entries)}")

    # 5) Save
    console.print(f"\n[yellow]Writing artifacts to {VECTOR_STORE_PATH}/...[/yellow]")
    faiss_dir = VECTOR_STORE_PATH / "jsonl_faiss"
    vector_store.save_local(str(faiss_dir))

    with (VECTOR_STORE_PATH / "jsonl_bm25.pkl").open("wb") as f:
        pickle.dump(jsonl_bm25, f)
    with (VECTOR_STORE_PATH / "jsonl_chunks.pkl").open("wb") as f:
        # Persist chunk metadata so the retriever can map BM25 hits back to docs
        # without reloading the FAISS docstore.
        pickle.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
            f,
        )
    with (VECTOR_STORE_PATH / "postman_bm25.pkl").open("wb") as f:
        pickle.dump(postman_bm25, f)
    with (VECTOR_STORE_PATH / "postman_entries.pkl").open("wb") as f:
        pickle.dump(postman_entries, f)

    stats = {
        **{k: v for k, v in build_stats.items() if k != "dropped_dupe_urls"},
        "postman_entries": len(postman_entries),
        "embedder_backend": embedder.backend,
        "embedder_model": embedder.model_name,
        "embedder_device": embedder.device,
    }
    with (VECTOR_STORE_PATH / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    console.rule("[bold green]Build complete[/bold green]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
