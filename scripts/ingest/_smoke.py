"""Smoke-test the OAS synthesis on a handful of sentinel JSONL records.

Works with both dataset layouts: legacy records that embed their OAS document
and current records that reference the crawler's OAS registry by `oas_id`.
Each sentinel must render its own operation (not the whole API group).

Run from the repo root:  python -m scripts.ingest._smoke
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

# Re-encode stdout as UTF-8 on Windows so we don't crash on non-ASCII chars.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from scripts.ingest.chunking import record_oas  # noqa: E402
from scripts.ingest.oas_synth import page_operation, synthesize_endpoint_markdown  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL_PATH = REPO_ROOT / "scripts" / "data" / "developer_connect.jsonl"
REGISTRY_PATH = JSONL_PATH.with_name(f"{JSONL_PATH.stem}_oas.json")

SENTINELS = [
    "get-borrower-pairs",
    "assign-document-attachments",
    "create-loan",
    "get-loan",
    "view-pipeline",
]
MIN_CHARS = 500
# One operation's markdown is bounded by the schema/example caps in oas_synth;
# anything far beyond this means every operation of the group was rendered.
MAX_CHARS = 40_000


def load_records(slugs: Iterable[str]) -> Dict[str, Dict]:
    wanted = set(slugs)
    found: Dict[str, Dict] = {}
    with JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("slug") in wanted:
                found.setdefault(r["slug"], r)
                if len(found) == len(wanted):
                    break
    return found


def load_registry() -> Dict[str, Dict]:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    recs = load_records(SENTINELS)
    registry = load_registry()
    failures = []
    for slug in SENTINELS:
        rec = recs.get(slug)
        if rec is None:
            print(f"=== {slug}: NOT FOUND in jsonl ===")
            failures.append(slug)
            continue
        oas = record_oas(rec, registry)
        md = synthesize_endpoint_markdown(rec, oas)
        body = rec.get("body_md") or ""
        op = page_operation(rec, oas)
        doc_api_chars = len(json.dumps(rec.get("doc_api") or {}, ensure_ascii=False))
        print(f"=== {slug} ===")
        print(f"  body_md_chars={len(body)}  oas paths={len((oas or {}).get('paths') or {})}  "
              f"operation={op[0].upper() + ' ' + op[1] if op else None}  doc_api_chars={doc_api_chars}")
        print(f"  synthesized markdown chars: {len(md):,}")
        if md:
            print("  --- first 30 lines ---")
            for line in md.splitlines()[:30]:
                print(f"  {line}")
            print("  --- end preview ---")
        else:
            print("  (no markdown produced)")
        print()
        if op is None or not (MIN_CHARS <= len(md) <= MAX_CHARS):
            failures.append(slug)

    if failures:
        print(f"FAIL: {len(failures)} sentinel(s) missing, without an operation, "
              f"or outside {MIN_CHARS}-{MAX_CHARS} chars: {failures}")
        return 1
    print(f"PASS: all sentinels rendered their own operation ({MIN_CHARS}-{MAX_CHARS} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
